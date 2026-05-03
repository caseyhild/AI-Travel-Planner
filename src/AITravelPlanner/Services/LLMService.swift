import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Tokenizers
import CoreLocation
import MapKit

private let defaultTravelStyle = "sightseeing"

/// LLMService handles on-device inference using MLX.
/// It is an actor to ensure that only one generation runs at a time and
/// to keep the heavy model weights in memory.
actor LLMService {
    // 1B version is recommended for iPhone as it uses ~700MB - 1GB of RAM.
    // The configuration is found in LLMRegistry.
    
    // possible models:
        
    // acereason_7b_4bit
    // baichuan_m1_14b_instruct_4bit
    // bitnet_b1_58_2b_4t_4bit
    // codeLlama13b4bit
    // deepSeekR1_7B_4bit
    // deepseek_r1_4bit
    // ernie_45_0_3BPT_bf16_ft
    // exaone_4_0_1_2b_4bit
    // gemma2bQuantized
    // gemma3_1B_qat_4bit
    // gemma3n_E2B_it_lm_4bit
    // gemma3n_E2B_it_lm_bf16
    // gemma3n_E4B_it_lm_4bit
    // gemma3n_E4B_it_lm_bf16
    // gemma_2_2b_it_4bit
    // gemma_2_9b_it_4bit
    // glm4_9b_4bit
    // gpt_oss_20b_MXFP4_Q8
    // granite3_3_2b_4bit
    // granite_4_0_h_tiny_4bit_dwq
    // lfm2_1_2b_4bit
    // lfm2_8b_a1b_3bit_mlx
    // lille_130m_bf16
    // ling_mini_2_2bit
    // llama3_1_8B_4bit
    // llama3_2_1B_4bit
    // llama3_2_3B_4bit
    // llama3_8B_4bit
    // mimo_7b_sft_4bit
    // mistral7B4bit
    // mistralNeMo4bit
    // nanochat_d20_mlx
    // olmo_2_1124_7B_Instruct_4bit
    // olmoe_1b_7b_0125_instruct_4bit
    // openelm270m4bit
    // phi3_5MoE
    // phi3_5_4bit
    // phi4bit
    // qwen205b4bit
    // qwen2_5_1_5b
    // qwen2_5_7b
    // qwen3MoE_30b_a3b_4bit
    // qwen3_0_6b_4bit
    // qwen3_1_7b_4bit
    // qwen3_4b_4bit
    // qwen3_8b_4bit
    // smolLM_135M_4bit
    // smollm3_3b_4bit
    
    private let modelConfiguration = LLMRegistry.llama3_2_1B_4bit
    
    private var modelContainer: ModelContainer?
    
    private var neighborhoodByPlaceName: [String: String] = [:]
    private var briefInfoByPlaceName: [String: String] = [:]
    
    private let placesClient = MapKitPlacesClient()
    
    private var _requestedDestination: String?
    
    /// Loads the model into memory. This will download from Hugging Face on the
    /// first run and then use a local cache.
    private func loadModel() async throws -> ModelContainer {
        if let modelContainer {
            return modelContainer
        }
        
        // Use the shared LLMModelFactory to load the container.
        let container = try await LLMModelFactory.shared.loadContainer(configuration: modelConfiguration) { progress in
            // Optional: You could pipe this progress back to the UI
            print("Downloading model: \(Int(progress.fractionCompleted * 100))%")
        }
        
        self.modelContainer = container
        return container
    }

    func generateBaseTrip(request: TripRequest) async throws -> Trip {
        self._requestedDestination = request.destination.trimmingCharacters(in: .whitespacesAndNewlines)
        let days = daysBetween(request.startDate, request.endDate) + 1
        print("DEBUG [LLMService]: Starting generation for \(request.destination) trip")
        print("DEBUG [LLMService]: Travel from \(request.startDate.formatted(date: .abbreviated, time: .omitted)) to \(request.endDate.formatted(date: .abbreviated, time: .omitted)) (\(days) days)")
        print("DEBUG [LLMService]: Requested travel styles: \(request.styles)")

        let container = try await loadModel()

        // PHASE 1: LLM outline of cities and day allocations
        let outlinePrompt = """
        You are an expert travel planner. Given the destination "\(request.destination)" and total days \(days), return ONLY a JSON object with this structure:

        {
          "cities": [
            {
              "name": "City Name",
              "days": NumberOfDays
            }
          ]
        }

        HARD CONSTRAINTS:
        - The sum of all city "days" MUST equal EXACTLY \(days). Do not exceed or fall short.
        - Use only major cities within the destination region (country/state).
        - Return at most 4 cities
        - Return ONLY valid JSON, no extra text or markdown.
        
        Example: (Spain trip for 7 days)
        
        {
          "cities": [
            {
              "name": "Madrid",
              "days": 3
            },
            {
              "name": "Barcelona",
              "days": 2
            },
            {
              "name": "Seville",
              "days": 2
            }
          ]
        }

        SELF-CHECK BEFORE OUTPUT:
        - Sum(city.days) == \(days)? If not, fix before returning.
        """

        let outlineMessages: [[String: String]] = [
            ["role": "system", "content": "You provide a JSON outline of cities and days for a trip."],
            ["role": "user", "content": outlinePrompt]
        ]

        // OPTIMIZATION 1: Reduced maxTokens from 384 to 256
        let outlineResult = try await withTimeout(20.0) {
            try await container.perform { context -> MLXLMCommon.GenerateResult in
                let input = try await context.processor.prepare(input: .init(messages: outlineMessages))
                return try await MLXLMCommon.generate(
                    input: input,
                    parameters: MLXLMCommon.GenerateParameters(maxTokens: 384, temperature: 0.15),
                    context: context
                ) { (_: [Int]) -> MLXLMCommon.GenerateDisposition in
                    return .more
                }
            }
        }
        
        // OPTIMIZATION 2: Clear KV cache after generation
        MLX.GPU.clearCache()
        
        print("DEBUG [LLMService]: Phase 1 raw length=\(outlineResult.output.count)")
        print("DEBUG [LLMService]: Phase 1 LLM Output: \(outlineResult.output)")

        guard let outlineJSON = extractFirstJSON(from: outlineResult.output) else {
            print("DEBUG [LLMService]: Failed to extract JSON from Phase 1 output. Falling back.")
            do {
                let prefilledJSON = "{}"
                // fallback here should never execute, using deterministic fill below
                let filled = try await llmFillDescriptions(from: prefilledJSON, styles: request.styles, totalDays: days, container: container)
                let trip = try parseTrip(from: filled, request: request)
                await maybeReleaseModel()
                return trip
            } catch {
                throw LLMError.invalidResponse
            }
        }

        let outlineResponse: OutlineResponse
        do {
            outlineResponse = try JSONDecoder().decode(OutlineResponse.self, from: Data(outlineJSON.utf8))
        } catch {
            print("DEBUG [LLMService]: Failed to decode outline JSON: \(error). Falling back.")
            do {
                let prefilledJSON = "{}"
                let filled = try await llmFillDescriptions(from: prefilledJSON, styles: request.styles, totalDays: days, container: container)
                let trip = try parseTrip(from: filled, request: request)
                await maybeReleaseModel()
                return trip
            } catch {
                throw LLMError.invalidResponse
            }
        }

        // Normalize outline to ensure total days match exactly
        let normalizedCities = normalizeCities(outlineResponse.cities, targetDays: days, destination: request.destination)
        let normalizedTotal = normalizedCities.reduce(0) { $0 + $1.days }
        print("DEBUG [LLMService]: Normalized Outline: \(normalizedCities.map { "\($0.name) (\($0.days)d)" }.joined(separator: ", ")) [total=\(normalizedTotal)]")
        self._normalizedCities = normalizedCities

        // PHASE 2: Fetch places for each city
        var cityActivitiesList: [CityActivities] = []
        for city in normalizedCities {
            // Aim for 4 per day capped at 30 total; fetch up to min(30, days*4) per city
            let maxNeeded = min(30, city.days * 4)
            do {
                let places = await searchPlacesWithFallback(for: city.name, styles: request.styles, maxResults: maxNeeded, country: request.destination)

                // Map to PlaceSummary
                var activities: [PlaceSummary] = places.map { p in
                    PlaceSummary(
                        name: p.name,
                        latitude: p.latitude,
                        longitude: p.longitude,
                        // Primary category is the style that found it
                        categories: [p.associatedStyle].compactMap { $0 } + (p.category.map { [$0] } ?? []),
                        rating: 0.0,
                        place_id: ""
                    )
                }
                // De-duplicate by name to maximize unique options
                var seenNames = Set<String>()
                activities = activities.filter { ps in
                    let key = ps.name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                    if seenNames.contains(key) { return false }
                    seenNames.insert(key)
                    return true
                }
                print("DEBUG [LLMService]: Found \(activities.count) places for \(city.name)")
                print("DEBUG [LLMService]: Places for \(city.name): \(activities.map{ $0.name }.joined(separator: ", "))")
                
                if activities.count > maxNeeded {
                    activities = Array(activities.prefix(maxNeeded))
                }
                // Hard cap at 30 activities total regardless
                if activities.count > 30 {
                    activities = Array(activities.prefix(30))
                }
                cityActivitiesList.append(CityActivities(name: city.name, days: city.days, activities: activities))
            } catch {
                print("DEBUG [LLMService]: Error fetching places for \(city.name): \(error)")
                cityActivitiesList.append(CityActivities(name: city.name, days: city.days, activities: []))
            }
        }

        // PHASE 3: Fill missing fields only (city-chunked, truncation-safe)

        let prefilledJSON: String
        do {
            prefilledJSON = try buildPrefilledItineraryJSON(
                outline: normalizedCities,
                pools: cityActivitiesList,
                styles: request.styles
            )
            print("DEBUG [LLMService]: Prefilled JSON length=\(prefilledJSON.count)")
            print("DEBUG [LLMService]: Prefilled JSON preview=\(prefilledJSON.prefix(300)) ... \(prefilledJSON.suffix(300))")
        } catch {
            prefilledJSON = "{}"
        }

        let stylesJoined = request.styles.map { $0.lowercased() }.joined(separator: ", ")

        let prompt = """
        You are a STRICT JSON TRANSFORMER.

        You will be given a JSON document between <JSON> and </JSON>.
        You must rewrite ONLY that JSON.

        You must return:
        - The SAME structure
        - With ONLY these fields filled if empty:
          - description
          - estimatedTime
          - category

        YOU MUST NOT:
        - Add or remove objects
        - Add or remove keys
        - Rename keys
        - Reorder arrays
        - Add comments, markdown, or text

        SCHEMA (must match exactly):

        {"cities":[{"name":"STRING","days":[{"dayNumber":INTEGER,"activities":[{"name":"STRING","latitude":NUMBER,"longitude":NUMBER,"description":"STRING","estimatedTime":"STRING","category":"STRING"}]}]}]}

        FIELD RULES:
        - description: 40–80 chars, one sentence
        - estimatedTime: one of ["20 mins","30 mins","45 mins","1 hour","1 hour 30 mins","2 hours"]
        - category: one of [\(stylesJoined)]

        CRITICAL:
        - Output MUST start with '{' and end with '}'.
        - Output MUST contain ONLY the rewritten JSON.
        - Output MUST include ALL cities and ALL days.

        <JSON>
        \(prefilledJSON)
        </JSON>
        """

        let messages: [[String: String]] = [
            ["role": "system", "content": "You are a deterministic JSON rewriting engine. You never explain. You never add text. You only output valid JSON that strictly preserves schema and structure."],
            ["role": "user", "content": prompt]
        ]

        // OPTIMIZATION 1: Reduced maxTokens from 512 to 384
        let rawOutput: String
        do {
            rawOutput = try await withTimeout(40.0) {
                try await container.perform { context -> MLXLMCommon.GenerateResult in
                    let input = try await context.processor.prepare(input: .init(messages: messages))
                    return try await MLXLMCommon.generate(
                        input: input,
                        parameters: MLXLMCommon.GenerateParameters(
                            maxTokens: 512,
                            temperature: 0.05
                        ),
                        context: context
                    ) { _ in .more }
                }
            }.output
            
            // OPTIMIZATION 2: Clear KV cache after generation
            MLX.GPU.clearCache()
            
        } catch {
            print("DEBUG [LLMService]: Phase 3 LLM call failed: \(error)")
            let trip = try await deterministicFillFromPrefilled(prefilledJSON: prefilledJSON, styles: request.styles)
            await maybeReleaseModel()
            return trip
        }

        print("DEBUG [LLMService]: Phase 3 raw output preview:\n\(rawOutput.prefix(2000))")

        let sanitized = sanitizeJSONText(rawOutput)
        guard let extracted = extractFirstJSON(from: sanitized) else {
            print("DEBUG [LLMService]: Phase 3 output missing JSON. Using deterministic fallback.")
            let trip = try await deterministicFillFromPrefilled(prefilledJSON: prefilledJSON, styles: request.styles)
            await maybeReleaseModel()
            return trip
        }

        let trimmed = extracted.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.first == "{", trimmed.last == "}" else {
            print("DEBUG [LLMService]: Phase 3 output malformed JSON. Using deterministic fallback.")
            let trip = try await deterministicFillFromPrefilled(prefilledJSON: prefilledJSON, styles: request.styles)
            await maybeReleaseModel()
            return trip
        }

        // Merge overlay to guarantee structure and day count match skeleton
        let finalOutputString: String
        if let merged = try? mergeLLMOutputOntoSkeleton(
            skeletonJSON: prefilledJSON,
            llmJSON: trimmed
        ) {
            finalOutputString = merged
        } else {
            print("DEBUG [LLMService]: Phase 3 merge failed. Using deterministic fallback.")
            let trip = try await deterministicFillFromPrefilled(prefilledJSON: prefilledJSON, styles: request.styles)
            await maybeReleaseModel()
            return trip
        }

        // Validate structure of final output, fallback if invalid
        if !isValidLLMTripJSON(finalOutputString) {
            print("DEBUG [LLMService]: Phase 3 output failed structure validation. Using deterministic fallback.")
            let trip = try await deterministicFillFromPrefilled(prefilledJSON: prefilledJSON, styles: request.styles)
            await maybeReleaseModel()
            return trip
        }

        // Parse + your existing validation/correction logic
        do {
            let trip = try parseTrip(from: finalOutputString, request: request)
            
            // OPTIMIZATION 5: Clear intermediate responses
            var validationErrors: [String] = []
            
            if let jsonString = extractFirstJSON(from: finalOutputString),
               let jsonData = jsonString.data(using: .utf8) {
                let parsedResponse = try JSONDecoder().decode(LLMTripResponse.self, from: jsonData)
                validationErrors = validate(parsedResponse, expectedDays: days, requiredStyles: request.styles)
                
                if !validationErrors.isEmpty {
                    print("DEBUG [LLMService]: Validation errors: \(validationErrors.joined(separator: "; "))")
                    let correctedTrip = await correctTrip(
                        trip,
                        with: cityActivitiesList,
                        expectedDays: days,
                        styles: request.styles
                    )
                    await maybeReleaseModel()
                    return correctedTrip
                }
            }
            
            await maybeReleaseModel()
            return trip
        } catch {
            print("DEBUG [LLMService]: Failed to parse final trip: \(error). Attempting deterministic fallback.")
            let trip = try await deterministicFillFromPrefilled(prefilledJSON: prefilledJSON, styles: request.styles)
            await maybeReleaseModel()
            return trip
        }
    }
    
    // Release model if the total days exceed a threshold to reduce memory pressure after big runs
    private func maybeReleaseModel() async {
        let totalDays = _normalizedCities?.reduce(0, { $0 + $1.days }) ?? 0
        if totalDays > 10 {
            print("DEBUG [LLMService]: Releasing model container after large generation")
            modelContainer = nil
            // OPTIMIZATION 2: Clear GPU cache when releasing model
            MLX.GPU.clearCache()
        }
    }
    
    private var _normalizedCities: [OutlineCity]?
    
    private func daysBetween(_ start: Date, _ end: Date) -> Int {
        let diff = Calendar.current.dateComponents([.day], from: start, to: end).day ?? 0
        return max(1, diff)
    }
    
    private struct OutlineResponse: Codable {
        let cities: [OutlineCity]
    }
    
    private struct OutlineCity: Codable {
        let name: String
        let days: Int
    }
    
    private struct CityActivities {
        let name: String
        let days: Int
        let activities: [PlaceSummary]
    }
    
    private let styleKeywords: [String: [String]] = [
        "food": ["restaurants", "local food", "places to eat", "street food", "food spots"],
        "culture": ["cultural attractions", "museums", "art galleries", "local neighborhoods", "cultural experiences", "famous places"],
        "history": ["historic sites", "landmarks", "heritage sites", "ancient buildings", "historical tours"],
        "nightlife": ["bars", "nightlife", "live music", "clubs", "evening activities"],
        "relaxed": ["parks", "cafes", "scenic spots", "relaxing places", "hidden gems"],
        "adventure": ["outdoor activities", "hiking", "nature spots", "adventures", "excursions", "tours"]
    ]

    private let styleQueryTemplates = [
        "%@ in %@ for tourists"
    ]


    private func buildQueriesByStyle(for styles: [String], city: String) -> [String: [String]] {
        let normalized = styles.map { $0.lowercased() }
        // Fallback to default if the user provided NO styles
        let activeStyles = normalized.isEmpty ? [defaultTravelStyle] : normalized

        var result: [String: [String]] = [:]

        for style in activeStyles {
            guard let keywords = styleKeywords[style] else { continue }
            var queries: [String] = []

            for keyword in keywords {
                for template in styleQueryTemplates {
                    queries.append(String(format: template, keyword, city))
                }
            }

            result[style] = queries.shuffled()
        }

        return result
    }

    private let cityCentroids: [String: CLLocationCoordinate2D] = [
        "rome": CLLocationCoordinate2D(latitude: 41.9028, longitude: 12.4964),
        "florence": CLLocationCoordinate2D(latitude: 43.7696, longitude: 11.2558),
        "milan": CLLocationCoordinate2D(latitude: 45.4642, longitude: 9.1900),
        "venice": CLLocationCoordinate2D(latitude: 45.4408, longitude: 12.3155),
        "naples": CLLocationCoordinate2D(latitude: 40.8518, longitude: 14.2681),
        "madrid": CLLocationCoordinate2D(latitude: 40.4168, longitude: -3.7038),
        "barcelona": CLLocationCoordinate2D(latitude: 41.3874, longitude: 2.1686),
        "seville": CLLocationCoordinate2D(latitude: 37.3891, longitude: -5.9845),
        "valencia": CLLocationCoordinate2D(latitude: 39.4699, longitude: -0.3763),
        "lisbon": CLLocationCoordinate2D(latitude: 38.7223, longitude: -9.1393),
        "paris": CLLocationCoordinate2D(latitude: 48.8566, longitude: 2.3522),
        "london": CLLocationCoordinate2D(latitude: 51.5074, longitude: -0.1278)
    ]

    // Haversine distance (km) between two coordinates
    private func haversineDistanceKM(lat1: Double, lon1: Double, lat2: Double, lon2: Double) -> Double {
        let R = 6371.0
        let dLat = (lat2 - lat1) * .pi / 180
        let dLon = (lon2 - lon1) * .pi / 180
        let a = sin(dLat/2) * sin(dLat/2) + cos(lat1 * .pi / 180) * cos(lat2 * .pi / 180) * sin(dLon/2) * sin(dLon/2)
        let c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c
    }

    // Maximum plausible radius (km) for city activities
    private func maxCityRadiusKM(for city: String) -> Double {
        let lower = city.lowercased()
        if ["madrid", "barcelona", "london", "paris", "rome", "milan"].contains(lower) { return 60 }
        return 40
    }

    private func normalizeCities(_ cities: [OutlineCity], targetDays: Int, destination: String) -> [OutlineCity] {
        var result = cities
        if targetDays > 0 && result.count > targetDays {
            result = Array(result.prefix(targetDays))
        }
        let currentTotal = result.reduce(0) { $0 + $1.days }
        if currentTotal == targetDays {
            return result
        }
        if currentTotal == 0 {
            if let first = result.first {
                return [OutlineCity(name: first.name, days: max(1, targetDays))]
            } else {
                return [OutlineCity(name: destination, days: max(1, targetDays))]
            }
        }
        let scale = Double(targetDays) / Double(currentTotal)
        var scaled: [OutlineCity] = []
        var allocated = 0
        for c in result {
            let s = max(1, Int((Double(c.days) * scale).rounded()))
            scaled.append(OutlineCity(name: c.name, days: s))
            allocated += s
        }
        var diff = targetDays - allocated
        if diff != 0 && !scaled.isEmpty {
            let indicesDesc = scaled.indices.sorted { scaled[$0].days > scaled[$1].days }
            let indicesAsc = scaled.indices.sorted { scaled[$0].days < scaled[$1].days }
            var i = 0
            while diff != 0 {
                if diff > 0 {
                    let idx = indicesAsc[i % indicesAsc.count]
                    scaled[idx] = OutlineCity(name: scaled[idx].name, days: scaled[idx].days + 1)
                    diff -= 1
                } else {
                    let idx = indicesDesc[i % indicesDesc.count]
                    if scaled[idx].days > 1 || scaled.count == 1 {
                        scaled[idx] = OutlineCity(name: scaled[idx].name, days: scaled[idx].days - 1)
                        diff += 1
                    } else {
                        break
                    }
                }
                i += 1
            }
        }
        let finalTotal = scaled.reduce(0) { $0 + $1.days }
        if finalTotal != targetDays, !scaled.isEmpty {
            let delta = targetDays - finalTotal
            scaled[0] = OutlineCity(name: scaled[0].name, days: max(1, scaled[0].days + delta))
        }
        return scaled
    }

    private func prefillCityDays(name: String, days: Int, pool: [PlaceSummary]) -> (prefilledDays: [[PlaceSummary]], leftover: [PlaceSummary]) {
        guard days > 0 else { return ([], pool) }

        var seen = Set<String>()
        let uniquePool = pool.filter {
            let k = $0.name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if seen.contains(k) { return false }
            seen.insert(k)
            return true
        }

        var buckets = Array(repeating: [PlaceSummary](), count: days)
        for (i, p) in uniquePool.enumerated() {
            buckets[i % days].append(p)
        }

        // Cap at 4 per day
        for i in 0..<buckets.count {
            if buckets[i].count > 4 {
                buckets[i] = Array(buckets[i].prefix(4))
            }
        }

        let used = buckets.flatMap { $0 }.count
        let leftover = used < uniquePool.count ? Array(uniquePool[used...]) : []
        return (buckets, leftover)
    }

    
    private func buildPrefilledItineraryJSON(outline: [OutlineCity], pools: [CityActivities], styles: [String]) throws -> String {
        struct CityOut: Codable {
            let name: String
            let days: [DayOut]
        }
        struct DayOut: Codable {
            let dayNumber: Int
            let activities: [ActOut]
        }
        struct ActOut: Codable {
            let name: String
            let description: String
            let estimatedTime: String
            let category: String
            let latitude: Double
            let longitude: Double
        }
        var cities: [CityOut] = []
        let normalizedStyles = styles.map { $0.lowercased() }
        for oc in outline {
            let pool = pools.first(where: { $0.name == oc.name })?.activities ?? []
            let (buckets, _) = prefillCityDays(name: oc.name, days: oc.days, pool: pool)
            let dayObjs: [DayOut] = buckets.enumerated().map { (i, bucket) in
                let acts: [ActOut] = bucket.map { p in
                    ActOut(
                        name: p.name,
                        description: "",
                        estimatedTime: "",
                        // Use the place's associated style as the initial category
                        category: p.categories.first(where: { normalizedStyles.contains($0) }) ?? normalizedStyles.first ?? defaultTravelStyle,
                        latitude: p.latitude,
                        longitude: p.longitude
                    )
                }
                return DayOut(dayNumber: i + 1, activities: acts)
            }
            cities.append(CityOut(name: oc.name, days: dayObjs))
        }
        let root = ["cities": cities]
        let data = try JSONEncoder().encode(root)
        return String(data: data, encoding: .utf8) ?? "{}"
    }
    
    private func llmFillDescriptions(from prefilledJSON: String, styles: [String], totalDays: Int, container: ModelContainer) async throws -> String {
        let stylesJoined = styles.map { $0.lowercased() }.joined(separator: ", ")
        let prompt = """
        You are an expert travel planner. Given the prefilled itinerary JSON below, fill ONLY the missing fields (description, estimatedTime, category) for each activity. 
        
        STRICT RULES:
        1. Keep the EXACT structure provided: {"cities": [{"name": "...", "days": [...]}]}.
        2. DO NOT turn "cities" into an object where city names are keys. It MUST be an array.
        3. MANDATORY: You must return ALL cities and ALL days provided in the input. Do not stop early.
        4. MANDATORY: Copy the "latitude" and "longitude" values EXACTLY as provided. Do not set them to 0 or leave them blank.
        5. Descriptions: Unique and specific (40–80 chars). DO NOT use generic phrases like "A restaurant" or "A museum".
        6. estimatedTime: ALWAYS include logical durations like "2 hours" or "45 mins" for every activity.
        7. Categories: lowercase and MUST be one of the selected styles: [\(stylesJoined)].

        Prefilled:
        \(prefilledJSON)

        Return ONLY the completed itinerary JSON. No extra keys, no markdown, no chatter.
        """
        let messages: [[String: String]] = [
            ["role": "system", "content": "You are a precise JSON editor. You fill missing fields while maintaining schema and retaining all original data exactly."],
            ["role": "user", "content": prompt]
        ]
        
        // OPTIMIZATION 1: Reduced maxTokens from 80 to 60
        let result: MLXLMCommon.GenerateResult = try await withTimeout(45.0, operation: { () async throws -> MLXLMCommon.GenerateResult in
            try await container.perform { context -> MLXLMCommon.GenerateResult in
                let input = try await context.processor.prepare(input: .init(messages: messages))
                return try await MLXLMCommon.generate(
                    input: input,
                    parameters: MLXLMCommon.GenerateParameters(maxTokens: 80, temperature: 0.12),
                    context: context
                ) { (_: [Int]) -> MLXLMCommon.GenerateDisposition in
                    return .more
                }
            }
        })
        
        // OPTIMIZATION 2: Clear KV cache after generation
        MLX.GPU.clearCache()
        
        let cleaned = sanitizeJSONText(result.output)
        if let json = extractFirstJSON(from: cleaned) {
            return json
        }
        throw LLMError.invalidResponse
    }

    private func deterministicFillFromPrefilled(prefilledJSON: String, styles: [String]) async throws -> Trip {
        let cleaned = sanitizeJSONText(prefilledJSON)
        guard let data = extractFirstJSON(from: cleaned)?.data(using: .utf8) else { throw LLMError.invalidResponse }
        let parsed = try JSONDecoder().decode(LLMTripResponse.self, from: data)
        let normalizedStyles = styles.map { $0.lowercased() }
        
        let container = try? await loadModel()

        var filledCities: [LLMCity] = []
        for city in parsed.cities {
            var filledDays: [LLMDay] = []
            for day in city.days {
                var filledActivities: [LLMActivity] = []
                for act in day.activities {
                    let cat = normalizedStyles.contains(act.category.lowercased()) ? act.category.lowercased() : (normalizedStyles.first ?? defaultTravelStyle)
                    
                    var desc: String
                    let est: String
                    
                    if let container = container {
                        desc = await generateDescriptionWithLLM(city: city.name, name: act.name, category: cat, styles: styles, container: container)
                        est = estimatedTimeHeuristic(name: act.name, category: cat)
                        if desc.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            desc = fallbackDescription(name: act.name, city: city.name, category: cat)
                        }
                    } else {
                        est = estimatedTimeHeuristic(name: act.name, category: cat)
                        desc = fallbackDescription(name: act.name, city: city.name, category: cat)
                    }
                    
                    filledActivities.append(LLMActivity(name: act.name, description: desc, estimatedTime: est, category: cat, latitude: act.latitude, longitude: act.longitude))
                }
                filledDays.append(LLMDay(dayNumber: day.dayNumber, activities: filledActivities))
            }
            filledCities.append(LLMCity(name: city.name, days: filledDays))
        }
        
        let response = LLMTripResponse(cities: filledCities)
        
        // OPTIMIZATION 5: Encode, extract string, then nil the response
        let encoded: Data = try await MainActor.run {
            return try JSONEncoder().encode(response)
        }
        let jsonString = String(data: encoded, encoding: .utf8) ?? "{}"
        
        return try parseTrip(from: jsonString, request: TripRequest(destination: "", startDate: Date(), endDate: Date(), styles: styles))
    }
    
    private func fallbackDescription(name: String, city: String, category: String) -> String {
        switch category.lowercased() {
        case "food":
            return "A popular local spot in \(city) known for regional flavors."
        case "history":
            return "A notable historical place in \(city) reflecting its heritage."
        case "culture":
            return "A cultural attraction in \(city) offering local insight."
        case "nightlife":
            return "A lively nightlife venue in \(city) popular with locals."
        case "relaxed":
            return "A relaxing place in \(city) ideal for a quiet break."
        case "adventure":
            return "An outdoor attraction in \(city) for active travelers."
        default:
            return "A recommended place to visit in \(city)."
        }
    }

    private func correctTrip(_ trip: Trip, with cityPools: [CityActivities], expectedDays: Int, styles: [String]) async -> Trip {
        let normalizedStyles = styles.map { $0.lowercased() }
        var correctedCities: [CityPlan] = []
        var globalUsedNames = Set<String>()
        
        let containerToUse = (try? await loadModel()) ?? modelContainer
        
        for cityPlan in trip.cities {
            let pool = cityPools.first(where: { $0.name == cityPlan.name })?.activities ?? []
            _ = cityPlan.name.lowercased()
            var days = cityPlan.days
            var cityUsedNames = Set<String>()
            
            for i in 0..<days.count {
                var acts = days[i].activities
                acts = acts.filter { a in
                    let key = a.name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                    if globalUsedNames.contains(key) || cityUsedNames.contains(key) { return false }
                    return true
                }

                if acts.count < 3 {
                    var additions: [Activity] = []
                    for p in pool {
                        let k = p.name.lowercased()
                        if !globalUsedNames.contains(k) && !cityUsedNames.contains(k) {
                            if let container = containerToUse {
                                let cat = p.categories.first(where: { normalizedStyles.contains($0) }) ?? normalizedStyles.first ?? defaultTravelStyle
                                let desc = await generateDescriptionWithLLM(city: cityPlan.name, name: p.name, category: cat, styles: styles, container: container)
                                let est = estimatedTimeHeuristic(name: p.name, category: cat)

                                additions.append(Activity(id: UUID(), name: p.name, description: desc, estimatedTime: est, category: cat, latitude: p.latitude, longitude: p.longitude, imageURLs: [], sourceURL: nil))
                            }
                            if additions.count >= (3 - acts.count) { break }
                        }
                    }
                    acts.append(contentsOf: additions)
                }

                // Final fix for categories: Ensure every activity has a category from user styles
                acts = acts.map { a in
                    if normalizedStyles.contains(a.category.lowercased()) { return a }
                    let bestStyle = normalizedStyles.first ?? defaultTravelStyle
                    return Activity(id: a.id, name: a.name, description: a.description, estimatedTime: a.estimatedTime, category: bestStyle, latitude: a.latitude, longitude: a.longitude, imageURLs: a.imageURLs, sourceURL: a.sourceURL)
                }

                cityUsedNames.formUnion(acts.map { $0.name.lowercased() })
                days[i] = DayPlan(id: days[i].id, dayNumber: days[i].dayNumber, activities: Array(acts.prefix(5)))
                globalUsedNames.formUnion(acts.map { $0.name.lowercased() })
            }
            correctedCities.append(CityPlan(id: cityPlan.id, name: cityPlan.name, days: days))
        }
        
        // OPTIMIZATION 2: Clear cache after correction loop
        MLX.GPU.clearCache()
        
        return Trip(id: trip.id, destination: trip.destination, startDate: trip.startDate, endDate: trip.endDate, styles: trip.styles, cities: correctedCities)
    }
    
    private func generateDescriptionWithLLM(city: String, name: String, category: String, styles: [String], container: ModelContainer) async -> String {
        let neighborhood = self.neighborhoodByPlaceName[name] ?? ""
        let brief = self.briefInfoByPlaceName[name] ?? ""
        let neighborhoodLine = neighborhood.isEmpty ? "" : "Neighborhood: \(neighborhood)."
        let briefLine = brief.isEmpty ? "" : "Info: \(brief)."
        let prompt = """
        Write 1 unique sentence (max 160 characters) about \(name) in \(city). End the sentence with a period.
        Category (must guide the tone): \(category).
        \(neighborhoodLine) \(briefLine)
        Do NOT mention coordinates or distances. Stay within the provided info (name, neighborhood, city, Info) and the category. Avoid generic phrases.
        Return ONLY the sentence.
        """
        let messages: [[String: String]] = [
            ["role": "system", "content": "Write concise, specific descriptions."],
            ["role": "user", "content": prompt]
        ]
        do {
            // OPTIMIZATION 1: Reduced maxTokens from 80 to 60
            let result: MLXLMCommon.GenerateResult = try await withTimeout(4.0, operation: { () async throws -> MLXLMCommon.GenerateResult in
                try await container.perform { context -> MLXLMCommon.GenerateResult in
                    let input = try await context.processor.prepare(input: .init(messages: messages))
                    return try await MLXLMCommon.generate(input: input, parameters: MLXLMCommon.GenerateParameters(maxTokens: 80, temperature: 0.7), context: context) { _ in .more }
                }
            })
            
            // OPTIMIZATION 2: Clear KV cache after each description generation
            MLX.GPU.clearCache()
            
            var text = result.output.trimmingCharacters(in: .whitespacesAndNewlines).trimmingCharacters(in: CharacterSet(charactersIn: "\"'`"))
            text = cleanPOICategoryArtifacts(text)
            // Enforce max ~160, min ~30 chars with fallback if empty
            let limit = 160
            if text.count > limit {
                // Try to truncate at sentence boundary (period, exclamation, question mark)
                let sub = text.prefix(limit)
                if let lastSentenceEnd = sub.lastIndex(where: { ".!?".contains($0) }) {
                    let endIdx = sub.index(after: lastSentenceEnd)
                    text = String(sub[..<endIdx]).trimmingCharacters(in: .whitespacesAndNewlines)
                } else {
                    // Try extending search up to 40 chars more to find sentence end
                    let extendedLimit = min(text.count, limit + 40)
                    let extendedSub = text.prefix(extendedLimit)
                    if let lastSentenceEnd = extendedSub.lastIndex(where: { ".!?".contains($0) }) {
                        let endIdx = extendedSub.index(after: lastSentenceEnd)
                        text = String(extendedSub[..<endIdx]).trimmingCharacters(in: .whitespacesAndNewlines)
                    } else if let lastSpace = sub.lastIndex(of: " ") {
                        text = String(sub[..<lastSpace]).trimmingCharacters(in: .whitespacesAndNewlines)
                    } else {
                        text = String(sub)
                    }
                }
            }
            if text.count < 30 {
                if text.isEmpty {
                    // simple fallback
                    text = "\(name) is a notable place in \(city) worth visiting."
                }
            }
            return text
        } catch {
            return ""
        }
    }
    
    private func estimatedTimeHeuristic(name: String, category: String) -> String {
        let lower = name.lowercased()
        
        if lower.contains("cathedral") || lower.contains("basilica") || lower.contains("palace") {
            return "1 hour 30 mins"
        }
        if lower.contains("museum") || lower.contains("gallery") {
            return "1 hour 30 mins"
        }
        if lower.contains("market") || lower.contains("food") || category == "food" {
            return "45 mins"
        }
        if lower.contains("park") || lower.contains("garden") || category == "relaxed" {
            return "30 mins"
        }
        if lower.contains("plaza") || lower.contains("square") {
            return "20 mins"
        }
        
        switch category.lowercased() {
        case "history": return "1 hour"
        case "culture": return "1 hour"
        case "food": return "45 mins"
        case "nightlife": return "1 hour 30 mins"
        case "adventure": return "2 hours"
        default: return "1 hour"
        }
    }

    
    private func generateEstimatedTimeWithLLM(city: String, name: String, category: String, container: ModelContainer) async -> String {
        let prompt = """
        Return ONLY a realistic visit duration for \(name) in \(city).
        Vary based on venue type:
        - museums/palaces: 1–2 hours
        - churches/landmarks: 30–60 mins
        - food spots: 30–45 mins
        - parks/squares: 20–40 mins

        Return one of:
        20 mins, 30 mins, 45 mins, 1 hour, 1 hour 30 mins, 2 hours, 3 hours
        No extra text.
        """
        let messages: [[String: String]] = [
            ["role": "system", "content": "Return only short duration strings."],
            ["role": "user", "content": prompt]
        ]
        do {
            // OPTIMIZATION 1: Already at maxTokens 16, which is good
            let result: MLXLMCommon.GenerateResult = try await withTimeout(3.5, operation: { () async throws -> MLXLMCommon.GenerateResult in
                try await container.perform { context -> MLXLMCommon.GenerateResult in
                    let input = try await context.processor.prepare(input: .init(messages: messages))
                    return try await MLXLMCommon.generate(input: input, parameters: MLXLMCommon.GenerateParameters(maxTokens: 16, temperature: 0.2), context: context) { _ in .more }
                }
            })
            
            // OPTIMIZATION 2: Clear KV cache
            MLX.GPU.clearCache()
            
            let time = result.output.trimmingCharacters(in: .whitespacesAndNewlines).trimmingCharacters(in: CharacterSet(charactersIn: "\"'`."))
            let cleaned = normalizeDurationString(time)
            let diversified = diversifyDurationIfGeneric(cleaned, name: name, category: category)
            return diversified
        } catch {
            return "1 hour"
        }
    }
    
    private func diversifyDurationIfGeneric(_ duration: String, name: String, category: String) -> String {
        let d = duration.lowercased()
        let common = ["1 hour 30 mins", "1 hour 30 min", "1 hour", "60 mins"]
        if !common.contains(d) { return duration }
        // Suggest a duration based on category and hash of the place name to spread values.
        let suggestion = durationSuggestion(for: name, category: category)
        return suggestion
    }
    
    private func durationSuggestion(for name: String, category: String) -> String {
        let base: [String]
        switch category.lowercased() {
        case "food": base = ["30 mins", "45 mins", "1 hour", "1 hour 15 mins"]
        case "history": base = ["45 mins", "1 hour", "1 hour 30 mins", "2 hours"]
        case "culture": base = ["45 mins", "1 hour", "1 hour 30 mins"]
        case "adventure": base = ["1 hour", "1 hour 30 mins", "2 hours"]
        case "relaxed": base = ["30 mins", "45 mins", "1 hour"]
        case "nightlife": base = ["1 hour", "1 hour 30 mins", "2 hours"]
        default: base = ["45 mins", "1 hour", "1 hour 30 mins"]
        }
        let hash = abs(name.hashValue)
        let idx = hash % base.count
        return base[idx]
    }
    
    private func searchPlacesWithFallback(for city: String, styles: [String], maxResults: Int, country: String) async -> [MapKitPlace] {
        let queriesByStyle = buildQueriesByStyle(for: styles, city: city)
        print("DEBUG [LLMService]: Search starting for city=\(city), styles=\(styles), maxResults=\(maxResults), country=\(country)")
        var effectiveCentroid: CLLocationCoordinate2D? = nil
        // Prefer dynamic geocoding to avoid predefined-only cities.
        effectiveCentroid = await geocodeCity(city, country: country)
        let usedGeocode = (effectiveCentroid != nil)
        if effectiveCentroid == nil { effectiveCentroid = cityCentroids[city.lowercased()] }
        print("DEBUG [LLMService]: Using geocoded centroid? \(usedGeocode); fallbackListUsed? \(usedGeocode ? false : (effectiveCentroid != nil))")
        
        guard let validCentroid = effectiveCentroid else { return [] }
        print("DEBUG [LLMService]: Search region center=\(validCentroid.latitude),\(validCentroid.longitude)")
        
        var resultsByStyle: [String: [MapKitPlace]] = [:]
        
        let stylesCount = max(1, Array(queriesByStyle.keys).count)
        let perStyleQuota = max(1, maxResults / stylesCount)
        
        let minDistanceFromCenterKM = 2.0
        
        for (style, queries) in queriesByStyle {
            print("DEBUG [LLMService]: Style=\(style) has \(queries.count) queries")
            var styleResults: [MapKitPlace] = []
            
            for q in queries {
                try? await Task.sleep(nanoseconds: 1_200_000_000)
                do {
                    let span = 1.0
                    var request = MKLocalSearch.Request()
                    request.naturalLanguageQuery = q
                    request.region = MKCoordinateRegion(
                        center: validCentroid,
                        span: MKCoordinateSpan(latitudeDelta: span, longitudeDelta: span)
                    )
                    print("DEBUG [LLMService]: Query=\(q) region span=\(span)x\(span), results: ", terminator: "")
                    let response = try await MKLocalSearch(request: request).start()
                    let rawItems = Array(response.mapItems.prefix(perStyleQuota))

                    let newResults = rawItems.compactMap { item -> MapKitPlace? in
                        let name = (item.name ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                        if name.lowercased() == city.lowercased() { return nil }
                        let lat = item.placemark.coordinate.latitude
                        let lon = item.placemark.coordinate.longitude
                        let distanceKM = self.haversineDistanceKM(lat1: validCentroid.latitude, lon1: validCentroid.longitude, lat2: lat, lon2: lon)
                        let maxRadius = self.maxCityRadiusKM(for: city)
                        if distanceKM > maxRadius {
                            return nil
                        }
                        if let locality = item.placemark.locality, !locality.lowercased().contains(city.lowercased()) {
                            // Exclude places too close but locality mismatches target city to avoid nearby city bleed.
                            if distanceKM < minDistanceFromCenterKM {
                                return nil
                            }
                        }
                        let subLocal = item.placemark.subLocality
                        let local = item.placemark.locality
                        let neighborhood = subLocal ?? local
                        if let n = neighborhood, !n.isEmpty { self.neighborhoodByPlaceName[name] = n }
                        // Brief info: prefer pointOfInterestCategory or a composed subtitle
                        var brief = ""
                        if let poi = item.pointOfInterestCategory?.rawValue {
                            if poi.range(of: "MKPOI", options: .caseInsensitive) != nil {
                                brief = ""
                            } else {
                                brief = poi
                            }
                        }
                        if brief.isEmpty {
                            let parts = [item.placemark.name, item.placemark.locality, item.placemark.country].compactMap { $0 }
                            brief = parts.joined(separator: ", ")
                        }
                        if !brief.isEmpty { self.briefInfoByPlaceName[name] = cleanPOICategoryArtifacts(brief) }
                        return MapKitPlace(name: name, latitude: lat, longitude: lon, url: item.url, category: nil, associatedStyle: style)
                    }
                    // Filter out obvious low-quality fast food chains / franchises
                    let blacklist = ["kfc", "mcdonald", "burger king", "subway", "taco bell", "domino", "pizza hut", "wendy's", "starbucks"]
                    let filtered = newResults.filter { place in
                        let lower = place.name.lowercased()
                        return !blacklist.contains(where: { lower.contains($0) })
                    }
                    styleResults.append(contentsOf: filtered)
                } catch { continue }
            }
            
            print("DEBUG [LLMService]: Accumulated results for style=\(style): \(styleResults.count)")
            resultsByStyle[style] = styleResults.shuffled()
        }
        print("DEBUG [LLMService]: Styles with results: \(resultsByStyle.map { "\($0.key): \($0.value.count)" }.joined(separator: ", "))")
        
        let activeStyles = Array(resultsByStyle.keys)
        if activeStyles.isEmpty { return [] }
        
        var finalResults: [MapKitPlace] = []
        var used = Set<String>()

        // Round-robin across styles for diversity
        var queues = activeStyles.map { resultsByStyle[$0] ?? [] }
        var i = 0
        while finalResults.count < maxResults && queues.contains(where: { !$0.isEmpty }) {
            let idx = i % queues.count
            if !queues[idx].isEmpty {
                let p = queues[idx].removeFirst()
                let key = p.name.lowercased()
                if !used.contains(key) {
                    used.insert(key)
                    finalResults.append(p)
                }
            }
            i += 1
        }

        
        // Added reverse geocode pass on finalResults for missing neighborhoods
        for place in finalResults {
            if self.neighborhoodByPlaceName[place.name] == nil {
                let res = await reverseGeocode(lat: place.latitude, lon: place.longitude)
                if let n = res.neighborhood, !n.isEmpty { self.neighborhoodByPlaceName[place.name] = n }
                else if let l = res.locality, !l.isEmpty { self.neighborhoodByPlaceName[place.name] = l }
            }
        }
        
        print("DEBUG [LLMService]: Final selected results count=\(finalResults.count) for city=\(city)")
        return finalResults.shuffled()
    }

    private func geocodeCity(_ city: String, country: String) async -> CLLocationCoordinate2D? {
        try? await CLGeocoder().geocodeAddressString("\(city), \(country)").first?.location?.coordinate
    }
    
    private func reverseGeocode(lat: Double, lon: Double) async -> (neighborhood: String?, locality: String?) {
        let location = CLLocation(latitude: lat, longitude: lon)
        do {
            let placemarks = try await CLGeocoder().reverseGeocodeLocation(location)
            if let pm = placemarks.first {
                return (pm.subLocality, pm.locality)
            }
        } catch {
            // ignore
        }
        return (nil, nil)
    }

    private func extractFirstJSON(from text: String) -> String? {
        var depth = 0
        var start: String.Index? = nil
        
        for i in text.indices {
            let ch = text[i]
            if ch == "{" || ch == "[" {
                if depth == 0 { start = i }
                depth += 1
            } else if ch == "}" || ch == "]" {
                depth -= 1
                if depth == 0, let s = start {
                    return String(text[s...i])
                }
            }
        }
        return nil
    }
    
    private func validate(_ trip: LLMTripResponse, expectedDays: Int, requiredStyles: [String]) -> [String] {
        var errors: [String] = []
        let allDays = trip.cities.flatMap { $0.days }
        if allDays.count != expectedDays { errors.append("Day count mismatch.") }
        for day in allDays {
            for act in day.activities {
                if !requiredStyles.isEmpty && !requiredStyles.map({ $0.lowercased() }).contains(act.category.lowercased()) {
                    errors.append("Invalid category for '\(act.name)'.")
                }
            }
        }
        if !errors.isEmpty { print("DEBUG [LLMService]: Validation errors detail: \(errors)") }
        return errors
    }
    
    private func parseTrip(from response: String, request: TripRequest) throws -> Trip {
        let cleaned = sanitizeJSONText(response)
        print("DEBUG [LLMService]: parseTrip input length=\(cleaned.count)")
        guard let json = extractFirstJSON(from: cleaned) else {
            throw LLMError.invalidResponse
        }
        print("DEBUG [LLMService]: parseTrip JSON length=\(json.count)")
        guard let data = json.data(using: .utf8) else {
            throw LLMError.invalidResponse
        }
        let parsed = try JSONDecoder().decode(LLMTripResponse.self, from: data)
        let cities = parsed.cities.map { city in
            CityPlan(id: UUID(), name: city.name, days: city.days.map { day in
                DayPlan(id: UUID(), dayNumber: day.dayNumber, activities: day.activities.map { Activity(id: UUID(), name: $0.name, description: $0.description, estimatedTime: $0.estimatedTime, category: $0.category, latitude: $0.latitude, longitude: $0.longitude, imageURLs: [], sourceURL: nil) })
            })
        }
        
        let reqDest = request.destination.trimmingCharacters(in: .whitespacesAndNewlines)
        let destination = (!reqDest.isEmpty ? reqDest : (self._requestedDestination ?? reqDest))
        return Trip(id: UUID(), destination: destination, startDate: request.startDate, endDate: request.endDate, styles: request.styles, cities: cities)
    }

    private func withTimeout<T>(_ seconds: Double, operation: @escaping () async throws -> T) async throws -> T {
        try await withThrowingTaskGroup(of: T.self) { group in
            group.addTask { try await operation() }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
                throw LLMError.invalidResponse
            }
            let result = try await group.next()!
            group.cancelAll()
            return result
        }
    }
    
    private func sanitizeJSONText(_ text: String) -> String {
        var s = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if let startIdx = s.firstIndex(where: { $0 == "{" || $0 == "[" }),
           let endIdx = s.lastIndex(where: { $0 == "}" || $0 == "]" }) {
            s = String(s[startIdx...endIdx])
            print("DEBUG [LLMService]: sanitizeJSONText extracted length=\(s.count)")
        }
        return s
    }
    
    private func enforceRequiredFields(on jsonText: String) -> String {
        // Best-effort pass: if any activity is missing estimatedTime or has a too-short/generic description, patch them.
        // We avoid full JSON parse here to keep this resilient; operate with simple regex-like replacements.
        var text = jsonText
        // Replace obviously generic descriptions with a placeholder of acceptable length.
        let generics = ["\"description\": \"A restaurant\"", "\"description\": \"A museum\"", "\"description\": \"A park\""]
        for g in generics {
            text = text.replacingOccurrences(of: g, with: "\"description\": \"Popular spot with notable appeal; worth a short visit.\"")
        }
        // Remove any lingering MKPOI artifacts from descriptions or brief info
        text = stripMKPOIArtifactsInFinalJSON(text)
        return text
    }

    private func isValidLLMTripJSON(_ jsonText: String) -> Bool {
        let cleaned = sanitizeJSONText(jsonText)
        guard let data = cleaned.data(using: .utf8) else { return false }
        do {
            let decoded = try JSONDecoder().decode(LLMTripResponse.self, from: data)
            // Basic checks: cities non-empty, each city has name and days with dayNumber
            if decoded.cities.isEmpty { return false }
            for city in decoded.cities {
                if city.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return false }
                if city.days.isEmpty { return false }
                for day in city.days { if day.dayNumber <= 0 { return false } }
            }
            return true
        } catch {
            print("DEBUG [LLMService]: Structure validation decode error: \(error)")
            return false
        }
    }
    
    private func normalizeDurationString(_ s: String) -> String {
        let trimmed = s.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        // Accept patterns like "15 mins", "30 mins", "45 mins", "1 hour", "1 hour 30 mins", "2 hours"
        let allowedPatterns = [
            "^([1-9]|[1-9][0-9])\\s?mins$",
            "^([1-9]|[1-9][0-9])\\s?minutes$",
            "^([1-9]|[1-9][0-9])\\s?min$",
            "^[1-9]\\s?hour$",
            "^[1-9]\\s?hours$",
            "^[1-9]\\s?hour\\s([1-9]|[1-9][0-9])\\s?mins$",
            "^[1-9]\\s?hours\\s([1-9]|[1-9][0-9])\\s?mins$"
        ]
        for p in allowedPatterns { if trimmed.range(of: p, options: .regularExpression) != nil { return trimmed.replacingOccurrences(of: "minutes", with: "mins") } }
        // Fallbacks if the model included extra words: try to extract numbers and units
        if let m = trimmed.range(of: "([0-9]+)\\s?(hour|hours)", options: .regularExpression) {
            let hours = String(trimmed[m]).replacingOccurrences(of: "hours", with: "hour")
            if let m2 = trimmed.range(of: "([0-9]+)\\s?(min|mins|minutes)", options: .regularExpression) {
                let mins = String(trimmed[m2]).replacingOccurrences(of: "minutes", with: "mins")
                return hours + " " + mins
            }
            return hours
        }
        if let m = trimmed.range(of: "([0-9]+)\\s?(min|mins|minutes)", options: .regularExpression) {
            return String(trimmed[m]).replacingOccurrences(of: "minutes", with: "mins")
        }
        // Absolute fallback
        return "1 hour"
    }
    
    // New helper to merge LLM output onto skeleton to preserve exact structure and counts
    private func mergeLLMOutputOntoSkeleton(skeletonJSON: String, llmJSON: String) throws -> String {
        // Decode both into LLMTripResponse-compatible models, align by city/day/activity order, and copy only the fillable fields.
        func decode(_ text: String) throws -> LLMTripResponse {
            let cleaned = sanitizeJSONText(text)
            guard let data = extractFirstJSON(from: cleaned)?.data(using: .utf8) else { throw LLMError.invalidResponse }
            return try JSONDecoder().decode(LLMTripResponse.self, from: data)
        }
        var skeleton = try decode(skeletonJSON)
        let llm = try decode(llmJSON)
        // Merge per city/day/activity by index (order preserved).
        let cityCount = min(skeleton.cities.count, llm.cities.count)
        var mergedCities: [LLMCity] = []
        for ci in 0..<cityCount {
            var sCity = skeleton.cities[ci]
            let lCity = llm.cities[ci]
            let dayCount = min(sCity.days.count, lCity.days.count)
            var mergedDays: [LLMDay] = []
            for di in 0..<dayCount {
                var sDay = sCity.days[di]
                let lDay = lCity.days[di]
                let actCount = min(sDay.activities.count, lDay.activities.count)
                var mergedActs: [LLMActivity] = []
                for ai in 0..<actCount {
                    let sAct = sDay.activities[ai]
                    let lAct = lDay.activities[ai]
                    // Preserve structure-critical fields from skeleton; overlay fillable ones from LLM.
                    let desc = lAct.description.isEmpty ? sAct.description : lAct.description
                    let est = lAct.estimatedTime.isEmpty ? sAct.estimatedTime : lAct.estimatedTime
                    let cat = lAct.category.isEmpty ? sAct.category : lAct.category
                    mergedActs.append(LLMActivity(name: sAct.name, description: desc, estimatedTime: est, category: cat, latitude: sAct.latitude, longitude: sAct.longitude))
                }
                mergedDays.append(LLMDay(dayNumber: sDay.dayNumber, activities: mergedActs))
            }
            mergedCities.append(LLMCity(name: sCity.name, days: mergedDays))
        }
        let merged = LLMTripResponse(cities: mergedCities)
        let data = try JSONEncoder().encode(merged)
        return String(data: data, encoding: .utf8) ?? "{}"
    }
    
    /// Cleans strings from common MKPOI category artifacts and stray punctuation.
    private func cleanPOICategoryArtifacts(_ text: String) -> String {
        var cleaned = text
        let patterns = [
            #"MKPOICategoryRestaurant"#,
            #"MKPOICategoryCafe"#,
            #"MKPOICategoryBar"#,
            #"MKPOICategory"#,
            #"MKPOI"#
        ]
        for pattern in patterns {
            if let range = cleaned.range(of: pattern, options: .caseInsensitive) {
                cleaned.removeSubrange(range)
            }
        }
        // Also trim stray punctuation and whitespace
        cleaned = cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
        cleaned = cleaned.trimmingCharacters(in: CharacterSet(charactersIn: ".,;:-–—"))
        return cleaned
    }

    /// Removes MKPOI* artifacts from a JSON string by replacing them with a generic phrase.
    private func stripMKPOIArtifactsInFinalJSON(_ jsonText: String) -> String {
        var text = jsonText
        let patterns = [
            #"MKPOICategoryRestaurant"#,
            #"MKPOICategoryCafe"#,
            #"MKPOICategoryBar"#,
            #"MKPOICategory"#,
            #"MKPOI"#
        ]
        for pattern in patterns {
            // Case-insensitive replace all occurrences with generic phrase
            while let range = text.range(of: pattern, options: .caseInsensitive) {
                text.replaceSubrange(range, with: "Popular local spot")
            }
        }
        return text
    }
}

// Response Models
struct LLMTripResponse: Codable {
    var cities: [LLMCity]
}

struct LLMCity: Codable {
    let name: String
    var days: [LLMDay]
}

struct LLMDay: Codable {
    let dayNumber: Int
    var activities: [LLMActivity]
}

struct LLMActivity: Codable {
    let name: String
    let description: String
    let estimatedTime: String
    let category: String
    let latitude: Double
    let longitude: Double
    
    init(name: String, description: String, estimatedTime: String, category: String, latitude: Double, longitude: Double) {
        self.name = name
        self.description = description
        self.estimatedTime = estimatedTime
        self.category = category
        self.latitude = latitude
        self.longitude = longitude
    }
}

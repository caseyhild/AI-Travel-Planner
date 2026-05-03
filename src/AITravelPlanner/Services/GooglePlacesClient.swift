import Foundation

struct GooglePlacesClient {
    let apiKeyProvider: () -> String

    init(apiKeyProvider: @escaping () -> String) {
        self.apiKeyProvider = apiKeyProvider
    }

    enum GooglePlacesError: Error {
        case invalidURL
        case apiError(String)
        case decodingError
    }

    func searchPlaces(query: String?, latitude: Double, longitude: Double, radius: Int?, type: String?, limit: Int) async throws -> PlacesResult {
        guard let url = buildSearchURL(query: query, latitude: latitude, longitude: longitude, radius: radius, type: type) else {
            throw GooglePlacesError.invalidURL
        }

        let (data, _) = try await URLSession.shared.data(from: url)
        let decoded = try JSONDecoder().decode(GoogleTextSearchResponse.self, from: data)

        guard decoded.status == "OK" else {
            throw GooglePlacesError.apiError(decoded.status)
        }

        let places = decoded.results.prefix(limit).map { result in
            PlaceSummary(
                name: result.name,
                latitude: result.geometry.location.lat,
                longitude: result.geometry.location.lng,
                categories: result.types,
                rating: result.rating,
                place_id: result.place_id
            )
        }

        return PlacesResult(places: Array(places))
    }

    func getPlaceDetails(placeID: String) async throws -> PlaceDetails {
        guard let url = buildDetailsURL(placeID: placeID) else {
            throw GooglePlacesError.invalidURL
        }

        let (data, _) = try await URLSession.shared.data(from: url)
        let decoded = try JSONDecoder().decode(GooglePlaceDetailsResponse.self, from: data)

        guard decoded.status == "OK", let result = decoded.result else {
            throw GooglePlacesError.apiError(decoded.status)
        }

        let openingHours = result.opening_hours?.weekday_text

        return PlaceDetails(
            name: result.name,
            latitude: result.geometry.location.lat,
            longitude: result.geometry.location.lng,
            categories: result.types,
            rating: result.rating,
            website: result.website,
            openingHours: openingHours,
            place_id: result.place_id
        )
    }

    // MARK: - Private helpers

    private func buildSearchURL(query: String?, latitude: Double, longitude: Double, radius: Int?, type: String?) -> URL? {
        let apiKey = apiKeyProvider()
        let base = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        var components = URLComponents(string: base)
        var queryItems: [URLQueryItem] = []

        if let q = query, !q.isEmpty {
            queryItems.append(URLQueryItem(name: "query", value: q))
        } else {
            // query == nil: require type and radius
            guard let t = type, !t.isEmpty, let r = radius else {
                return nil
            }
            queryItems.append(URLQueryItem(name: "type", value: t))
            queryItems.append(URLQueryItem(name: "radius", value: String(r)))
        }

        queryItems.append(URLQueryItem(name: "location", value: "\(latitude),\(longitude)"))
        queryItems.append(URLQueryItem(name: "key", value: apiKey))

        // If query is provided and radius is provided, include radius and type if present
        if let q = query, !q.isEmpty {
            if let r = radius {
                queryItems.append(URLQueryItem(name: "radius", value: String(r)))
            }
            if let t = type, !t.isEmpty {
                queryItems.append(URLQueryItem(name: "type", value: t))
            }
        }

        components?.queryItems = queryItems
        return components?.url
    }

    private func buildDetailsURL(placeID: String) -> URL? {
        let apiKey = apiKeyProvider()
        let base = "https://maps.googleapis.com/maps/api/place/details/json"
        var components = URLComponents(string: base)
        let fields = "name,geometry,types,website,opening_hours,rating"
        components?.queryItems = [
            URLQueryItem(name: "place_id", value: placeID),
            URLQueryItem(name: "fields", value: fields),
            URLQueryItem(name: "key", value: apiKey)
        ]
        return components?.url
    }

    // MARK: - Internal Response Models

    private struct GoogleTextSearchResponse: Codable {
        let status: String
        let results: [GooglePlaceSummary]

        struct GooglePlaceSummary: Codable {
            let name: String
            let geometry: Geometry
            let types: [String]
            let rating: Double?
            let place_id: String

            struct Geometry: Codable {
                let location: Location

                struct Location: Codable {
                    let lat: Double
                    let lng: Double
                }
            }
        }
    }

    private struct GooglePlaceDetailsResponse: Codable {
        let status: String
        let result: GooglePlaceDetails?

        struct GooglePlaceDetails: Codable {
            let name: String
            let geometry: Geometry
            let types: [String]
            let website: String?
            let opening_hours: OpeningHours?
            let rating: Double?
            let place_id: String

            struct Geometry: Codable {
                let location: Location

                struct Location: Codable {
                    let lat: Double
                    let lng: Double
                }
            }

            struct OpeningHours: Codable {
                let weekday_text: [String]?
            }
        }
    }
}

struct PlacesResult: Codable {
    let places: [PlaceSummary]
}

struct PlaceSummary: Codable {
    let name: String
    let latitude: Double
    let longitude: Double
    let categories: [String]
    let rating: Double?
    let place_id: String
}

struct PlaceDetails: Codable {
    let name: String
    let latitude: Double
    let longitude: Double
    let categories: [String]
    let rating: Double?
    let website: String?
    let openingHours: [String]?
    let place_id: String
}

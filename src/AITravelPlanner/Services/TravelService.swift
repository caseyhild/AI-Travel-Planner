import Foundation

struct TravelService {
    private let llmService = LLMService()
    private let placesService = PlacesService()

    func generateTrip(request: TripRequest) async throws -> Trip {
        // 1. Ask on-device LLM for structured itinerary
        let rawTrip = try await llmService.generateBaseTrip(request: request)

        // 2. Enrich activities with real-world metadata
        let enrichedCities = try await rawTrip.cities.asyncMap { city in
            let enrichedDays = try await city.days.asyncMap { day in
                let enrichedActivities = try await day.activities.asyncMap { activity in
                    let enriched = try await placesService.enrich(activity: activity)
                    return enriched
                }
                return DayPlan(id: day.id, dayNumber: day.dayNumber, activities: enrichedActivities)
            }
            return CityPlan(id: city.id, name: city.name, days: enrichedDays)
        }

        return Trip(
            id: rawTrip.id,
            destination: rawTrip.destination,
            startDate: request.startDate,
            endDate: request.endDate,
            styles: request.styles,
            cities: enrichedCities
        )
    }
}

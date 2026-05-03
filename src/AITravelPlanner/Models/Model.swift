import Foundation

struct TripRequest: Codable {
    var destination: String
    var startDate: Date
    var endDate: Date
    var styles: [String]
}

struct Trip: Codable, Identifiable, Hashable {
    let id: UUID
    let destination: String
    let startDate: Date
    let endDate: Date
    let styles: [String]
    let cities: [CityPlan]
}

struct CityPlan: Codable, Identifiable, Hashable {
    let id: UUID
    let name: String
    let days: [DayPlan]
}

struct DayPlan: Codable, Identifiable, Hashable {
    let id: UUID
    let dayNumber: Int
    let activities: [Activity]
}

struct Activity: Codable, Identifiable, Hashable {
    let id: UUID
    let name: String
    let description: String
    let estimatedTime: String
    let category: String
    let latitude: Double?
    let longitude: Double?
    let imageURLs: [String]
    let sourceURL: String?
}

import Foundation
import MapKit

public struct MapKitPlace {
    public let name: String
    public let latitude: Double
    public let longitude: Double
    public let url: URL?
    public let category: String?
    /// The specific travel style (e.g., "Food", "History") that was used to find this place.
    public let associatedStyle: String?
    
    public init(name: String, latitude: Double, longitude: Double, url: URL?, category: String?, associatedStyle: String? = nil) {
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.url = url
        self.category = category
        self.associatedStyle = associatedStyle
    }
}

public actor MapKitPlacesClient {
    public init() {}
    
    public func searchPlaces(query: String, limit: Int) async throws -> [MapKitPlace] {
        let request = MKLocalSearch.Request()
        request.naturalLanguageQuery = query
        
        let search = MKLocalSearch(request: request)
        let response = try await search.start()
        
        let items = response.mapItems.prefix(limit)
        
        return items.map { item in
            let category: String?
            if #available(iOS 16.0, *) {
                category = item.pointOfInterestCategory?.rawValue.lowercased()
            } else {
                category = nil
            }
            return MapKitPlace(
                name: item.name ?? "",
                latitude: item.placemark.coordinate.latitude,
                longitude: item.placemark.coordinate.longitude,
                url: item.url,
                category: category,
                associatedStyle: nil
            )
        }
    }
}

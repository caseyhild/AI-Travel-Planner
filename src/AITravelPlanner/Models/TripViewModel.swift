import Foundation
import SwiftUI
internal import Combine

@MainActor
class TripViewModel: ObservableObject {
    @Published var isLoading = false
    @Published var trip: Trip?
    @Published var errorMessage: String?
    @Published var savedTrips: [Trip] = []

    @Published var isDatePanelPresented: Bool = false
    @Published var selectedStyles: Set<String> = []
    @Published var isNewTripPresented: Bool = false

    private let travelService = TravelService()
    private let storageKey = "saved_trips"

    init() {
        // Load trips in the background to avoid blocking app launch
        Task {
            await loadTrips()
        }
    }

    func toggleStyle(_ style: String) {
        if selectedStyles.contains(style) {
            selectedStyles.remove(style)
        } else {
            selectedStyles.insert(style)
        }
    }

    func reset() {
        isLoading = false
        errorMessage = nil
        trip = nil
        isDatePanelPresented = false
        selectedStyles = []
    }

    func generateTrip(request: TripRequest) {
        isNewTripPresented = false 
        isLoading = true
        errorMessage = nil
        trip = nil

        Task {
            do {
                let result = try await travelService.generateTrip(request: request)
                self.savedTrips.insert(result, at: 0)
                saveTrips()
                self.trip = result
            } catch {
                self.errorMessage = error.localizedDescription
            }
            self.isLoading = false
        }
    }

    func deleteTrip(at offsets: IndexSet) {
        savedTrips.remove(atOffsets: offsets)
        saveTrips()
    }

    private func saveTrips() {
        let trips = savedTrips
        // Encoding and saving to disk can be slow; move to background
        Task.detached(priority: .background) {
            if let encoded = try? JSONEncoder().encode(trips) {
                UserDefaults.standard.set(encoded, forKey: "saved_trips")
            }
        }
    }

    private func loadTrips() async {
        // Perform decoding off the main thread
        let data = UserDefaults.standard.data(forKey: storageKey)
        guard let data = data else { return }
        
        if let decoded = try? JSONDecoder().decode([Trip].self, from: data) {
            self.savedTrips = decoded
        }
    }
    
    func didSelectDate() {
        isDatePanelPresented = false
    }
}

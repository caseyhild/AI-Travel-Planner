import SwiftUI

struct TripOverviewView: View {
    @EnvironmentObject var viewModel: TripViewModel
    let trip: Trip

    var body: some View {
        List {
            Section {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Image(systemName: "calendar")
                        Text("\(trip.startDate.formatted(date: .abbreviated, time: .omitted)) - \(trip.endDate.formatted(date: .abbreviated, time: .omitted))")
                    }
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    
                    if !trip.styles.isEmpty {
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack {
                                ForEach(trip.styles, id: \.self) { style in
                                    Text(style)
                                        .font(.caption.bold())
                                        .padding(.horizontal, 10)
                                        .padding(.vertical, 4)
                                        .background(Color.accentColor.opacity(0.1))
                                        .clipShape(Capsule())
                                }
                            }
                        }
                    }
                }
                .padding(.vertical, 4)
            } header: {
                Text("Trip Details")
            }

            Section {
                ForEach(trip.cities) { city in
                    NavigationLink {
                        CityDetailView(city: city)
                    } label: {
                        VStack(alignment: .leading) {
                            Text(city.name)
                                .font(.headline)
                            Text("\(city.days.count) day\(city.days.count == 1 ? "" : "s")")
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            } header: {
                Text("Itinerary")
            }
        }
        .navigationTitle(trip.destination)
    }
}

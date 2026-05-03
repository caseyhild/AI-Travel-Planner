import SwiftUI

struct HomeView: View {
    @EnvironmentObject var viewModel: TripViewModel

    var body: some View {
        List {
            if viewModel.savedTrips.isEmpty {
                VStack(spacing: 20) {
                    Spacer()
                    Image(systemName: "airplane.departure")
                        .font(.system(size: 60))
                        .foregroundStyle(.secondary)
                    Text("No Trips Planned")
                        .font(.title3.bold())
                    Text("Tap the '+' button to start planning your next adventure.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                    Spacer()
                }
                .frame(maxWidth: .infinity)
                .listRowBackground(Color.clear)
            } else {
                ForEach(viewModel.savedTrips) { trip in
                    NavigationLink(destination: TripOverviewView(trip: trip)) {
                        TripRow(trip: trip)
                    }
                }
                .onDelete(perform: viewModel.deleteTrip)
            }
        }
        .navigationTitle("My Trips")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    viewModel.reset()
                    viewModel.isNewTripPresented = true
                } label: {
                    Image(systemName: "plus")
                }
            }
        }
        .sheet(isPresented: $viewModel.isNewTripPresented) {
            NavigationStack {
                NewTripView()
                    .toolbar {
                        ToolbarItem(placement: .cancellationAction) {
                            Button("Cancel") {
                                viewModel.isNewTripPresented = false
                            }
                        }
                    }
            }
        }
    }
}

struct TripRow: View {
    let trip: Trip
    
    private let columns = [
        GridItem(.adaptive(minimum: 70), spacing: 6)
    ]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(trip.destination)
                .font(.headline)
            
            HStack {
                Text(trip.startDate.formatted(date: .abbreviated, time: .omitted))
                Text("-")
                Text(trip.endDate.formatted(date: .abbreviated, time: .omitted))
                Spacer()
            }
            .font(.subheadline)
            .foregroundStyle(.secondary)
            
            if !trip.styles.isEmpty {
                LazyVGrid(columns: columns, alignment: .leading, spacing: 6) {
                    ForEach(trip.styles, id: \.self) { style in
                        Text(style.capitalized)
                            .font(.caption2.bold())
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.accentColor.opacity(0.12))
                            .clipShape(Capsule())
                    }
                }
                .padding(.top, 2)
            }
        }
        .padding(.vertical, 6)
    }
}


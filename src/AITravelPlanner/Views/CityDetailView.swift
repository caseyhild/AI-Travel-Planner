import SwiftUI

struct CityDetailView: View {
    let city: CityPlan

    var body: some View {
        List {
            ForEach(city.days) { day in
                Section("Day \(day.dayNumber)") {
                    ForEach(day.activities) { activity in
                        NavigationLink {
                            ActivityDetailView(activity: activity)
                        } label: {
                            VStack(alignment: .leading) {
                                Text(activity.name)
                                    .font(.headline)
                                
                                // Displaying the time here instead of a description
                                if !activity.estimatedTime.isEmpty {
                                    Text(activity.estimatedTime)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                }
            }
        }
        .navigationTitle(city.name)
    }
}

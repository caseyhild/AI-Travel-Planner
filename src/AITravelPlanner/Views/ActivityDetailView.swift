import SwiftUI
import MapKit

struct ActivityDetailView: View {
    let activity: Activity

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if !activity.imageURLs.isEmpty {
                    TabView {
                        ForEach(activity.imageURLs, id: \.self) { url in
                            AsyncImage(url: URL(string: url)) { image in
                                image.resizable().scaledToFill()
                            } placeholder: {
                                ProgressView()
                            }
                        }
                    }
                    .frame(height: 250)
                    .tabViewStyle(.page)
                }

                Text(activity.name)
                    .font(.title.bold())

                Text(activity.description)

                HStack {
                    Label(activity.estimatedTime, systemImage: "clock")
                    Spacer()
                    Text(activity.category.capitalized)
                        .foregroundStyle(.secondary)
                }

                if let lat = activity.latitude,
                   let lon = activity.longitude {
                    Map(initialPosition: .region(
                        MKCoordinateRegion(
                            center: CLLocationCoordinate2D(latitude: lat, longitude: lon),
                            span: MKCoordinateSpan(latitudeDelta: 0.01, longitudeDelta: 0.01)
                        )
                    ))
                    .frame(height: 200)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                }

                if let source = activity.sourceURL,
                   let url = URL(string: source) {
                    Link("Learn more", destination: url)
                        .font(.headline)
                }
            }
            .padding()
        }
        .navigationTitle(activity.name)
        .navigationBarTitleDisplayMode(.inline)
    }
}

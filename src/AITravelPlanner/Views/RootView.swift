import SwiftUI

struct RootView: View {
    @StateObject private var viewModel = TripViewModel()

    var body: some View {
        ZStack {
            NavigationStack {
                HomeView()
                    .navigationDestination(item: $viewModel.trip) { trip in
                        TripOverviewView(trip: trip)
                    }
            }
            .environmentObject(viewModel)
            .overlay(alignment: .center) {
                if viewModel.isLoading {
                    ZStack {
                        Color.black.opacity(0.4)
                            .ignoresSafeArea()
                        
                        ZStack {
                            RoundedRectangle(cornerRadius: 16)
                                .fill(Color(.systemBackground))
                                .frame(width: 200, height: 200)
                                .shadow(radius: 10)
                            LoadingView()
                        }
                    }
                }
            }
            
            // Hidden TextField to pre-warm keyboard on app launch
            TextField("", text: .constant(""))
                .frame(width: 0, height: 0)
                .opacity(0)
                .accessibilityHidden(true)
        }
    }
}

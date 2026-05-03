import SwiftUI

@main
struct TravelPlannerApp: App {
    init() {
        // Pre-warm keyboard subsystem
        _ = UITextField()
    }
    
    var body: some Scene {
        WindowGroup {
            RootView()
        }
    }
}

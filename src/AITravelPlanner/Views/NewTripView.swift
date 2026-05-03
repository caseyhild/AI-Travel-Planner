import SwiftUI

struct NewTripView: View {
    @EnvironmentObject var viewModel: TripViewModel

    @State private var destination = ""
    @State private var startDate = Date()
    @State private var endDate = Calendar.current.date(byAdding: .day, value: 6, to: Date())!
    @FocusState private var isDestinationFocused: Bool

    let styles = ["Food", "Culture", "History", "Nightlife", "Relaxed", "Adventure"]

    var body: some View {
        Form {
            Section("Destination") {
                TextField("Japan, Italy, Peru...", text: $destination)
                    .autocorrectionDisabled()
                    .textInputAutocapitalization(.words)
                    .focused($isDestinationFocused)
            }
            .contentShape(Rectangle())
            .onTapGesture {
                isDestinationFocused = true
            }

            Section("Dates") {
                HStack {
                    VStack(alignment: .leading) {
                        Text("Start")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                        Text(startDate.formatted(date: .abbreviated, time: .omitted))
                            .foregroundStyle(.blue)
                    }
                    Spacer()
                    VStack(alignment: .trailing) {
                        Text("End")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                        Text(endDate.formatted(date: .abbreviated, time: .omitted))
                            .foregroundStyle(.blue)
                    }
                }
                .contentShape(Rectangle())
                .onTapGesture {
                    isDestinationFocused = false
                    viewModel.isDatePanelPresented = true
                }
            }

            Section("Travel Style") {
                VStack(spacing: 8) {
                    HStack(spacing: 8) {
                        ForEach(styles.prefix(3), id: \.self) { style in
                            styleButton(style)
                        }
                    }
                    HStack(spacing: 8) {
                        ForEach(styles.suffix(3), id: \.self) { style in
                            styleButton(style)
                        }
                    }
                }
                .onTapGesture {
                    isDestinationFocused = false
                }
            }

            Section {
                Button {
                    isDestinationFocused = false
                    let request = TripRequest(
                        destination: destination,
                        startDate: startDate,
                        endDate: endDate,
                        styles: Array(viewModel.selectedStyles)
                    )
                    viewModel.generateTrip(request: request)
                } label: {
                    Text("Generate Trip")
                        .font(.headline)
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .background(
                            (destination.isEmpty || viewModel.selectedStyles.isEmpty) ?
                            Color.gray.opacity(0.3) : Color.blue
                        )
                        .cornerRadius(16)
                }
                .buttonStyle(.plain)
                .disabled(destination.isEmpty || viewModel.selectedStyles.isEmpty)
                .listRowInsets(EdgeInsets())
                .listRowBackground(Color.clear)
                .allowsHitTesting(true)
            }
            .allowsHitTesting(true)
        }
        .navigationTitle("New Trip")
        .scrollDismissesKeyboard(.interactively)
        .simultaneousGesture(
            TapGesture().onEnded { _ in
                if isDestinationFocused {
                    isDestinationFocused = false
                }
            }
        )
        .sheet(isPresented: $viewModel.isDatePanelPresented) {
            NavigationStack {
                DateRangePickerView(startDate: $startDate, endDate: $endDate) {
                    viewModel.didSelectDate()
                }
                .toolbar {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button("Done") {
                            viewModel.isDatePanelPresented = false
                        }
                    }
                }
            }
        }
    }
    
    @ViewBuilder
    private func styleButton(_ style: String) -> some View {
        let isSelected = viewModel.selectedStyles.contains(style)

        Button {
            isDestinationFocused = false
            viewModel.toggleStyle(style)
        } label: {
            Text(style)
                .font(.body)
                .padding(.vertical, 8)
                .frame(maxWidth: .infinity)
                .background(isSelected ? Color.blue.opacity(0.1) : Color.gray.opacity(0.1))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(isSelected ? Color.blue : Color.clear, lineWidth: 2)
                )
                .foregroundColor(isSelected ? .blue : .primary)
                .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        .buttonStyle(.plain)
    }
}

import SwiftUI

private struct DateRange: Equatable {
    var start: Date?
    var end: Date?
}

struct DateRangePickerView: View {
    @Binding var startDate: Date
    @Binding var endDate: Date
    var onComplete: () -> Void

    @State private var dateRange: DateRange
    @State private var selectedMonth: Int
    @State private var selectedYear: Int
    @State private var isShowingMonthPicker: Bool = false

    private let calendar = Calendar.current
    private let cellHeight: CGFloat = 40
    private let circleSize: CGFloat = 36
    private let connectorHeight: CGFloat = 30

    init(startDate: Binding<Date>, endDate: Binding<Date>, onComplete: @escaping () -> Void) {
        self._startDate = startDate
        self._endDate = endDate
        self.onComplete = onComplete
        self._dateRange = State(initialValue: DateRange(start: startDate.wrappedValue, end: endDate.wrappedValue))
        self._selectedMonth = State(initialValue: calendar.component(.month, from: startDate.wrappedValue))
        self._selectedYear = State(initialValue: calendar.component(.year, from: startDate.wrappedValue))
    }

    var body: some View {
        VStack(spacing: 12) {
            let displayedMonth = calendar.date(from: DateComponents(year: selectedYear, month: selectedMonth)) ?? Date()

            HStack {
                Button { decrementMonth() } label: { Image(systemName: "chevron.left") }
                Spacer()
                Button { isShowingMonthPicker = true } label: {
                    Text(monthTitle(for: displayedMonth)).font(.headline)
                }
                Spacer()
                Button { incrementMonth() } label: { Image(systemName: "chevron.right") }
            }
            .padding(.horizontal)

            monthGrid(for: displayedMonth)

            Spacer(minLength: 0)
        }
        .padding(.vertical)
        .navigationTitle("Select Dates")
        .sheet(isPresented: $isShowingMonthPicker) {
            monthYearPickerView
        }
    }

    @ViewBuilder
    private func monthGrid(for month: Date) -> some View {
        let days = daysInMonthGrid(for: month)
        let columns = Array(repeating: GridItem(.flexible(), spacing: 0), count: 7)

        LazyVGrid(columns: columns, spacing: 0) {
            ForEach(weekdaySymbols(), id: \.self) { symbol in
                Text(symbol)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, minHeight: cellHeight)
            }

            ForEach(days.indices, id: \.self) { index in
                if let date = days[index] {
                    dayCell(for: date)
                } else {
                    Color.clear.frame(height: cellHeight)
                }
            }
        }
        .padding(.horizontal)
        // This is the absolute animation killer for the grid
        .transaction { transaction in
            transaction.animation = nil
            transaction.disablesAnimations = true
        }
    }

    @ViewBuilder
    private func dayCell(for date: Date) -> some View {
        let dStart = dateRange.start?.startOfDay(calendar)
        let dEnd = dateRange.end?.startOfDay(calendar)
        let dCurrent = date.startOfDay(calendar)

        let isStart = dStart == dCurrent
        let isEnd = dEnd == dCurrent
        let isInRange = (dStart != nil && dEnd != nil) && (dCurrent >= dStart! && dCurrent <= dEnd!)

        Button {
            handleTap(on: dCurrent)
        } label: {
            ZStack {
                Rectangle()
                    .fill(isInRange ? Color.accentColor.opacity(0.25) : Color.clear)
                    .frame(height: connectorHeight)
                    .mask {
                        HStack(spacing: 0) {
                            Rectangle().fill(isStart ? Color.clear : Color.black)
                            Rectangle().fill(isEnd ? Color.clear : Color.black)
                        }
                    }

                Text("\(calendar.component(.day, from: date))")
                    .font(.system(size: 16, weight: (isStart || isEnd) ? .bold : .regular))
                    .frame(maxWidth: .infinity, minHeight: cellHeight)
                    .background(
                        Circle()
                            .fill((isStart || isEnd) ? Color.accentColor : Color.clear)
                            .frame(width: circleSize, height: circleSize)
                    )
                    .foregroundStyle((isStart || isEnd) ? .white : .primary)
            }
        }
        .buttonStyle(.plain)
        // Force every single cell update to be non-animatable
        .transaction { transaction in
            transaction.animation = nil
            transaction.disablesAnimations = true
        }
    }

    private func handleTap(on date: Date) {
        // Use an immediate transaction for the state change
        var transaction = Transaction()
        transaction.disablesAnimations = true
        transaction.animation = nil
        
        withTransaction(transaction) {
            if dateRange.start == nil || (dateRange.start != nil && dateRange.end != nil) {
                dateRange = DateRange(start: date, end: nil)
            } else if let start = dateRange.start {
                let finalStart = min(start, date)
                let finalEnd = max(start, date)
                dateRange = DateRange(start: finalStart, end: finalEnd)
                
                startDate = finalStart
                endDate = finalEnd
                
                // Binding updates often trigger implicit animations in parent views
                // This 0.5s delay allows the UI to "snap" before the sheet closes
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                    onComplete()
                }
            }
        }
    }

    private var monthYearPickerView: some View {
        VStack {
            HStack { Spacer(); Button("Done") { isShowingMonthPicker = false }.padding() }
            HStack {
                Picker("Month", selection: $selectedMonth) {
                    ForEach(1...12, id: \.self) { m in Text(DateFormatter().monthSymbols[m - 1]).tag(m) }
                }.pickerStyle(.wheel)
                Picker("Year", selection: $selectedYear) {
                    ForEach(Array((selectedYear-50)...(selectedYear+50)), id: \.self) { y in Text(verbatim: "\(y)").tag(y) }
                }.pickerStyle(.wheel)
            }.frame(height: 180)
            Spacer()
        }
    }

    private func monthTitle(for date: Date) -> String {
        let formatter = DateFormatter(); formatter.dateFormat = "LLLL yyyy"; return formatter.string(from: date)
    }

    private func daysInMonthGrid(for month: Date) -> [Date?] {
        let start = calendar.date(from: calendar.dateComponents([.year, .month], from: month))!
        let range = calendar.range(of: .day, in: .month, for: start)!
        let firstWeekday = (calendar.component(.weekday, from: start) - calendar.firstWeekday + 7) % 7
        var grid: [Date?] = Array(repeating: nil, count: firstWeekday)
        for day in 0..<range.count { grid.append(calendar.date(byAdding: .day, value: day, to: start)) }
        while grid.count % 7 != 0 { grid.append(nil) }
        return grid
    }

    private func weekdaySymbols() -> [String] {
        let symbols = calendar.shortWeekdaySymbols
        let first = calendar.firstWeekday - 1
        return Array(symbols[first...] + symbols[..<first])
    }

    private func incrementMonth() {
        if let next = calendar.date(byAdding: .month, value: 1, to: calendar.date(from: DateComponents(year: selectedYear, month: selectedMonth))!) {
            selectedYear = calendar.component(.year, from: next); selectedMonth = calendar.component(.month, from: next)
        }
    }

    private func decrementMonth() {
        if let prev = calendar.date(byAdding: .month, value: -1, to: calendar.date(from: DateComponents(year: selectedYear, month: selectedMonth))!) {
            selectedYear = calendar.component(.year, from: prev); selectedMonth = calendar.component(.month, from: prev)
        }
    }
}

private extension Date {
    func startOfDay(_ calendar: Calendar) -> Date { calendar.startOfDay(for: self) }
}

import Foundation

enum LLMError: Error, LocalizedError {
    case invalidResponse
    case decodingFailed(underlying: Error)
    case networkFailure
    case cancelled

    var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "The language model returned an invalid or incomplete response."
        case .decodingFailed(let underlying):
            return "Failed to decode response: \(underlying.localizedDescription)"
        case .networkFailure:
            return "A network error occurred while contacting the language model."
        case .cancelled:
            return "The request was cancelled."
        }
    }
}

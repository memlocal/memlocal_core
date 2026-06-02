/// Generates a completion from a system + user prompt. Implement with OpenAI, etc.
abstract class LlmProvider {
  /// Returns the model's completion for the given [system] and [user] prompts.
  Future<String> complete(String system, String user);
}

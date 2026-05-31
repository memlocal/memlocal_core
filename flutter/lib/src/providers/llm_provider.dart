/// Generates a completion from a system + user prompt. Implement with OpenAI, etc.
abstract class LlmProvider {
  Future<String> complete(String system, String user);
}

/// Turns text into an embedding vector. Implement with OpenAI, ONNX, etc.
abstract class EmbeddingProvider {
  Future<List<double>> embedOne(String text);
  int get dimensions;
}

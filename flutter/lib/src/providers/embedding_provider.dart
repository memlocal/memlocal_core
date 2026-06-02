/// Turns text into an embedding vector. Implement with OpenAI, ONNX, etc.
///
/// The engine stores caller-supplied embeddings, so the [dimensions] reported
/// here must match the `dimensions` the engine was opened with.
abstract class EmbeddingProvider {
  /// Embeds a single [text] into a dense vector of length [dimensions].
  Future<List<double>> embedOne(String text);

  /// The length of the vectors produced by [embedOne].
  int get dimensions;
}

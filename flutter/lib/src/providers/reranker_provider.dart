/// Reorders candidate documents by relevance to a query (cross-encoder reranking).
abstract class RerankerProvider {
  /// Returns [documents] indices paired with relevance scores, most-relevant
  /// first, limited to [topN].
  Future<List<RerankResult>> rerank(
    String query,
    List<String> documents, {
    int topN,
  });
}

/// One reranked result: the index into the original documents list + its score.
class RerankResult {
  const RerankResult(this.index, this.score);
  final int index;
  final double score;
}

/// Reorders candidate documents by relevance to a query (cross-encoder reranking).
abstract class RerankerProvider {
  /// Scores each of [documents] against [query] and returns results
  /// most-relevant first, limited to [topN]. Each result references its
  /// document by index into the input list.
  Future<List<RerankResult>> rerank(
    String query,
    List<String> documents, {
    int topN,
  });
}

/// One reranked result: the index into the original documents list + its score.
class RerankResult {
  /// Pairs a document [index] (into the original list) with its relevance [score].
  const RerankResult(this.index, this.score);

  /// Index of this document in the list passed to [RerankerProvider.rerank].
  final int index;

  /// Relevance score; higher is more relevant.
  final double score;
}

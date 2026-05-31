import 'dart:convert';
import 'package:http/http.dart' as http;
import 'reranker_provider.dart';

/// [RerankerProvider] backed by the Jina AI rerank API (mirrors memlocal_core).
class JinaReranker implements RerankerProvider {
  JinaReranker(
    this.apiKey, {
    this.model = 'jina-reranker-v2-base-multilingual',
    this.baseUrl = 'https://api.jina.ai/v1',
  });

  final String apiKey;
  final String model;
  final String baseUrl;

  @override
  Future<List<RerankResult>> rerank(
    String query,
    List<String> documents, {
    int topN = 5,
  }) async {
    if (documents.isEmpty) return const [];
    final res = await http.post(
      Uri.parse('$baseUrl/rerank'),
      headers: {
        'Authorization': 'Bearer $apiKey',
        'Content-Type': 'application/json',
      },
      body: jsonEncode({
        'model': model,
        'query': query,
        'documents': documents,
        'top_n': topN,
      }),
    );
    if (res.statusCode != 200) {
      throw Exception('Jina rerank ${res.statusCode}: ${res.body}');
    }
    final data = jsonDecode(res.body) as Map<String, dynamic>;
    final results = data['results'] as List;
    return results
        .map((r) => RerankResult(
              (r['index'] as num).toInt(),
              (r['relevance_score'] as num).toDouble(),
            ))
        .toList();
  }
}

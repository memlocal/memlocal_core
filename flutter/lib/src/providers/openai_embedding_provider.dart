import 'dart:convert';
import 'package:http/http.dart' as http;
import 'embedding_provider.dart';

/// [EmbeddingProvider] backed by the OpenAI embeddings API.
class OpenAIEmbeddingProvider implements EmbeddingProvider {
  OpenAIEmbeddingProvider(
    this.apiKey, {
    this.model = 'text-embedding-3-small',
    this.dimensions = 1536,
    this.baseUrl = 'https://api.openai.com/v1',
  });

  final String apiKey;
  final String model;
  @override
  final int dimensions;
  final String baseUrl;

  @override
  Future<List<double>> embedOne(String text) async {
    final res = await http.post(
      Uri.parse('$baseUrl/embeddings'),
      headers: {
        'Authorization': 'Bearer $apiKey',
        'Content-Type': 'application/json',
      },
      body: jsonEncode({'model': model, 'input': text, 'dimensions': dimensions}),
    );
    if (res.statusCode != 200) {
      throw Exception('OpenAI embeddings ${res.statusCode}: ${res.body}');
    }
    final data = jsonDecode(res.body) as Map<String, dynamic>;
    final embedding = (data['data'] as List).first['embedding'] as List;
    return embedding.map((e) => (e as num).toDouble()).toList();
  }
}

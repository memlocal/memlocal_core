import 'dart:convert';
import 'package:http/http.dart' as http;
import 'embedding_provider.dart';

/// [EmbeddingProvider] backed by the OpenAI embeddings API.
class OpenAIEmbeddingProvider implements EmbeddingProvider {
  /// Creates a provider that calls the OpenAI embeddings API with [apiKey].
  ///
  /// [model] is the embedding model and [dimensions] the requested vector size
  /// (these must be consistent with each other and with the engine's
  /// `dimensions`). [baseUrl] can point at an OpenAI-compatible endpoint.
  OpenAIEmbeddingProvider(
    this.apiKey, {
    this.model = 'text-embedding-3-small',
    this.dimensions = 1536,
    this.baseUrl = 'https://api.openai.com/v1',
  });

  /// OpenAI API key sent as a bearer token. The caller owns this secret.
  final String apiKey;

  /// The OpenAI embedding model to use.
  final String model;

  /// Length of the produced vectors; sent as the API's `dimensions` parameter.
  @override
  final int dimensions;

  /// Base URL of the (OpenAI-compatible) API.
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

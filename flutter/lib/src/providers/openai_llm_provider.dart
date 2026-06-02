import 'dart:convert';
import 'package:http/http.dart' as http;
import 'llm_provider.dart';

/// [LlmProvider] backed by the OpenAI Chat Completions API.
class OpenAILlmProvider implements LlmProvider {
  /// Creates a provider that calls the OpenAI Chat Completions API with
  /// [apiKey], using [model]. [baseUrl] can point at an OpenAI-compatible
  /// endpoint.
  OpenAILlmProvider(
    this.apiKey, {
    this.model = 'gpt-5.4-nano',
    this.baseUrl = 'https://api.openai.com/v1',
  });

  /// OpenAI API key sent as a bearer token. The caller owns this secret.
  final String apiKey;

  /// The chat model to use.
  final String model;

  /// Base URL of the (OpenAI-compatible) API.
  final String baseUrl;

  @override
  Future<String> complete(String system, String user) async {
    final res = await http.post(
      Uri.parse('$baseUrl/chat/completions'),
      headers: {
        'Authorization': 'Bearer $apiKey',
        'Content-Type': 'application/json',
      },
      // Minimal params for maximum compatibility with the gpt-5 family
      // (some newer models reject custom temperature / max_tokens).
      body: jsonEncode({
        'model': model,
        'messages': [
          {'role': 'system', 'content': system},
          {'role': 'user', 'content': user},
        ],
      }),
    );
    if (res.statusCode != 200) {
      throw Exception('OpenAI chat ${res.statusCode}: ${res.body}');
    }
    final data = jsonDecode(res.body) as Map<String, dynamic>;
    return (data['choices'] as List).first['message']['content'] as String;
  }
}

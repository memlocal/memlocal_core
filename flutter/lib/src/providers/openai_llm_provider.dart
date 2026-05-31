import 'dart:convert';
import 'package:http/http.dart' as http;
import 'llm_provider.dart';

/// [LlmProvider] backed by the OpenAI Chat Completions API.
class OpenAILlmProvider implements LlmProvider {
  OpenAILlmProvider(
    this.apiKey, {
    this.model = 'gpt-5.4-nano',
    this.baseUrl = 'https://api.openai.com/v1',
  });

  final String apiKey;
  final String model;
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

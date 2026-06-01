import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:memlocal/memlocal.dart';
import 'package:path_provider/path_provider.dart';

const _dimensions = 1536;

/// System prompt for the LLM extraction step: decides what is worth storing,
/// splits into atomic memories, and classifies each into one of the engine's
/// 8 memory types. The model must return ONLY a JSON array.
const _extractionSystemPrompt = '''
You extract durable, atomic memories from a user's message for a long-term memory system.
Return ONLY a JSON array (no prose, no markdown fences). Each element must be:
{"content": "<one atomic fact, written in third person, self-contained>", "type": "<one of: episodic, factual, semantic, procedural, social, spatial, prospective, affective>"}

Rules:
- One fact per element. Preserve proper nouns exactly.
- Only include information worth remembering long-term. Greetings, small talk, acknowledgements, and pure questions contain nothing to remember -> return [].
- Pick the best type: episodic=events/experiences; factual=stable personal facts/preferences; semantic=general knowledge; procedural=how-to/workflows; social=people/relationships; spatial=places/locations; prospective=reminders/future intentions; affective=feelings/emotions.
Return [] when nothing is worth storing.
''';

/// Reads a key from the loaded `.env`, treating empty/whitespace as absent.
String? _envKey(String name) {
  final v = dotenv.maybeGet(name)?.trim();
  return (v == null || v.isEmpty) ? null : v;
}

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await RustLib.init();
  try {
    await dotenv.load(fileName: '.env');
  } catch (_) {
    // .env missing or unreadable — keys will be absent; the UI will prompt.
  }
  runApp(const MemoryChatApp());
}

class MemoryChatApp extends StatelessWidget {
  const MemoryChatApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'memlocal chat',
        theme: ThemeData(
          colorSchemeSeed: Colors.indigo,
          useMaterial3: true,
        ),
        home: const ChatScreen(),
      );
}

/// The kind of item rendered in the transcript.
enum ChatRole { user, recalled, stored, assistant, system, error }

/// One renderable entry in the chat transcript.
class ChatItem {
  ChatItem.user(this.text)
      : role = ChatRole.user,
        recalled = const [],
        scores = const [],
        rerankedByJina = false,
        stored = const [];
  ChatItem.assistant(this.text)
      : role = ChatRole.assistant,
        recalled = const [],
        scores = const [],
        rerankedByJina = false,
        stored = const [];
  ChatItem.system(this.text)
      : role = ChatRole.system,
        recalled = const [],
        scores = const [],
        rerankedByJina = false,
        stored = const [];
  ChatItem.error(this.text)
      : role = ChatRole.error,
        recalled = const [],
        scores = const [],
        rerankedByJina = false,
        stored = const [];
  ChatItem.recalled(
    this.recalled, {
    required this.scores,
    required this.rerankedByJina,
  })  : role = ChatRole.recalled,
        text = '',
        stored = const [];
  ChatItem.stored(this.stored)
      : role = ChatRole.stored,
        text = '',
        recalled = const [],
        scores = const [],
        rerankedByJina = false;

  final ChatRole role;
  final String text;
  final List<RecalledMemory> recalled;

  /// The score to display per recalled memory (parallel to [recalled]): the
  /// Jina relevance score when [rerankedByJina], otherwise the semantic score.
  final List<double?> scores;

  /// Whether [recalled] was reordered by the Jina reranker (vs. semantic order).
  final bool rerankedByJina;

  /// The (content, type) memories extracted and stored for this turn. Empty
  /// means nothing was worth storing (chit-chat).
  final List<({String content, String type})> stored;
}

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final _input = TextEditingController();
  final _scroll = ScrollController();
  final List<ChatItem> _items = [];

  Memlocal? _engine;
  EmbeddingProvider? _embeddingProvider;
  LlmProvider? _llmProvider;
  RerankerProvider? _reranker;

  bool _initializing = true;
  bool _sending = false;
  String? _initError;

  bool get _ready =>
      _engine != null && _embeddingProvider != null && _llmProvider != null;

  @override
  void initState() {
    super.initState();
    _bootstrap();
  }

  @override
  void dispose() {
    _input.dispose();
    _scroll.dispose();
    super.dispose();
  }

  Future<void> _bootstrap() async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final engine = await Memlocal.open(
        dbPath: '${dir.path}/memlocal_demo.db',
        dimensions: _dimensions,
      );
      // Keys come solely from the `.env` file (see flutter/example/.env).
      final apiKey = _envKey('OPENAI_API_KEY');
      final jinaKey = _envKey('JINA_API_KEY');
      if (!mounted) return;
      setState(() {
        _engine = engine;
        _initializing = false;
        if (apiKey != null) {
          _embeddingProvider = OpenAIEmbeddingProvider(apiKey);
          _llmProvider = OpenAILlmProvider(apiKey);
        }
        // Jina is optional: when absent, retrieval stays semantic-only.
        if (jinaKey != null) {
          _reranker = JinaReranker(jinaKey);
        }
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _initializing = false;
        _initError = '$e';
      });
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scroll.hasClients) return;
      _scroll.animateTo(
        _scroll.position.maxScrollExtent,
        duration: const Duration(milliseconds: 250),
        curve: Curves.easeOut,
      );
    });
  }

  Future<void> _send() async {
    final text = _input.text.trim();
    if (text.isEmpty || _sending || !_ready) return;

    final engine = _engine!;
    final embeddingProvider = _embeddingProvider!;
    final llmProvider = _llmProvider!;
    final reranker = _reranker;

    setState(() {
      _items.add(ChatItem.user(text));
      _input.clear();
      _sending = true;
    });
    _scrollToBottom();

    try {
      // b. Embed the new message.
      final embedding = await embeddingProvider.embedOne(text);
      // c. Recall PRIOR memories: pull a larger candidate pool BEFORE storing
      //    the current message, then optionally rerank it down to the top 5.
      final pool = await engine.searchSemantic(embedding: embedding, k: 20);

      List<RecalledMemory> recalled;
      List<double?> scores;
      bool rerankedByJina;
      String? rerankNote;

      if (reranker != null && pool.isNotEmpty) {
        try {
          final ranked = await reranker.rerank(
            text,
            pool.map((m) => m.content).toList(),
            topN: 5,
          );
          recalled = ranked.map((r) => pool[r.index]).toList();
          scores = ranked.map<double?>((r) => r.score).toList();
          rerankedByJina = true;
        } catch (e) {
          // Reranking is best-effort: fall back to semantic order, note it,
          // but never abort the turn.
          recalled = pool.take(5).toList();
          scores = recalled.map((m) => m.score).toList();
          rerankedByJina = false;
          rerankNote = '(rerank failed: $e, using semantic order)';
        }
      } else {
        recalled = pool.take(5).toList();
        scores = recalled.map((m) => m.score).toList();
        rerankedByJina = false;
      }

      // d. Store step: run an LLM extraction over the message to decide what's
      //    worth keeping, split it into atomic memories, and classify each into
      //    one of the engine's memory types. Embeddings for stored items are
      //    computed from each extracted memory's own content (the RAW message
      //    embedding above is only the retrieval query, and is unchanged).
      List<({String content, String type})> stored = [];
      String? storeNote;
      try {
        final extracted = await _extractMemories(text);
        for (final m in extracted) {
          final emb = await embeddingProvider.embedOne(m.content);
          await engine.addMemory(content: m.content, kind: m.type, embedding: emb);
        }
        stored = extracted;
      } catch (e) {
        // Extraction failed -> don't lose the message: store it verbatim as factual.
        final emb = await embeddingProvider.embedOne(text);
        await engine.addMemory(content: text, kind: 'factual', embedding: emb);
        stored = [(content: text, type: 'factual')];
        storeNote = 'extraction failed (${e.toString()}); stored raw as factual';
      }

      // e. Build the memory-grounded system prompt + single LLM call.
      final system =
          'You are a helpful assistant with long-term memory of this user. '
          'Relevant memories you have recalled:\n'
          '${recalled.isEmpty ? "(none yet)" : recalled.map((m) => "- ${m.content}").join("\n")}'
          '\nUse them when relevant; if none apply, just answer normally.';
      final reply = await llmProvider.complete(system, text);

      // f. Show recalled context, then what was stored, then the assistant reply.
      if (!mounted) return;
      setState(() {
        _items.add(ChatItem.recalled(
          recalled,
          scores: scores,
          rerankedByJina: rerankedByJina,
        ));
        if (rerankNote != null) _items.add(ChatItem.system(rerankNote));
        _items.add(ChatItem.stored(stored));
        if (storeNote != null) _items.add(ChatItem.system(storeNote));
        _items.add(ChatItem.assistant(reply));
        _sending = false;
      });
      _scrollToBottom();
    } catch (e) {
      // g. Surface the error (incl. OpenAI 4xx body) without crashing.
      if (!mounted) return;
      setState(() {
        _items.add(ChatItem.error('$e'));
        _sending = false;
      });
      _scrollToBottom();
    }
  }

  /// Runs the LLM extraction step over [text]: asks the model what is worth
  /// remembering, split into atomic memories and classified by type. Returns
  /// the extracted (content, type) pairs (possibly empty — e.g. for chit-chat).
  ///
  /// Parses the model output robustly (tolerating ```json fences and prose
  /// around the array). RETHROWS on any parse failure so the caller can fall
  /// back to storing the raw message.
  Future<List<({String content, String type})>> _extractMemories(
    String text,
  ) async {
    final llmProvider = _llmProvider!;
    final raw = await llmProvider.complete(_extractionSystemPrompt, text);

    // Strip whitespace and any markdown code fences the model may have added.
    var cleaned = raw.trim();
    if (cleaned.startsWith('```')) {
      cleaned = cleaned.replaceFirst(RegExp(r'^```(?:json)?'), '');
      if (cleaned.endsWith('```')) {
        cleaned = cleaned.substring(0, cleaned.length - 3);
      }
      cleaned = cleaned.trim();
    }
    // Take the substring from the first '[' to the last ']'.
    final start = cleaned.indexOf('[');
    final end = cleaned.lastIndexOf(']');
    if (start == -1 || end == -1 || end < start) {
      throw FormatException('no JSON array in model output: $raw');
    }
    final decoded = jsonDecode(cleaned.substring(start, end + 1));
    if (decoded is! List) {
      throw const FormatException('extraction output was not a JSON array');
    }

    final out = <({String content, String type})>[];
    for (final element in decoded) {
      if (element is! Map) continue;
      final content = (element['content'] as String?)?.trim() ?? '';
      // Allowed types are the 8 documented ones; pass others through anyway
      // (the Rust side defaults unknown stored-names to semantic).
      final type =
          (element['type'] as String?)?.trim().toLowerCase() ?? 'semantic';
      if (content.isEmpty) continue;
      out.add((content: content, type: type));
    }
    return out;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('memlocal chat'),
      ),
      body: _buildBody(),
    );
  }

  Widget _buildBody() {
    if (_initializing) {
      return const Center(child: CircularProgressIndicator());
    }
    if (_initError != null) {
      return Padding(
        padding: const EdgeInsets.all(24),
        child: Center(
          child: Text(
            'Failed to open the memory engine:\n$_initError',
            textAlign: TextAlign.center,
          ),
        ),
      );
    }
    return Column(
      children: [
        if (!_ready) const _ApiKeyBanner(),
        Expanded(child: _buildTranscript()),
        if (_sending) const LinearProgressIndicator(minHeight: 2),
        _buildComposer(),
      ],
    );
  }

  Widget _buildTranscript() {
    if (_items.isEmpty) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Text(
            _ready
                ? 'Say hello — every message becomes a memory.'
                : 'Set OPENAI_API_KEY in flutter/example/.env and restart the app.',
            textAlign: TextAlign.center,
            style: TextStyle(color: Theme.of(context).hintColor),
          ),
        ),
      );
    }
    return ListView.builder(
      controller: _scroll,
      padding: const EdgeInsets.symmetric(vertical: 12),
      itemCount: _items.length,
      itemBuilder: (context, i) => _ChatItemView(item: _items[i]),
    );
  }

  Widget _buildComposer() {
    final canSend = _ready && !_sending;
    return SafeArea(
      top: false,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(12, 8, 12, 8),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Expanded(
              child: TextField(
                controller: _input,
                enabled: canSend,
                minLines: 1,
                maxLines: 5,
                textInputAction: TextInputAction.newline,
                keyboardType: TextInputType.multiline,
                decoration: InputDecoration(
                  hintText: _ready
                      ? 'Message'
                      : 'Set OPENAI_API_KEY in .env to start…',
                  border: const OutlineInputBorder(),
                  isDense: true,
                ),
                onSubmitted: (_) => _send(),
              ),
            ),
            const SizedBox(width: 8),
            IconButton.filled(
              onPressed: canSend ? _send : null,
              icon: const Icon(Icons.send),
            ),
          ],
        ),
      ),
    );
  }
}

/// Inline banner shown when no OpenAI API key is configured. Keys come solely
/// from `flutter/example/.env`, so this is informational (not tappable).
class _ApiKeyBanner extends StatelessWidget {
  const _ApiKeyBanner();

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Material(
      color: scheme.secondaryContainer,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Row(
          children: [
            Icon(Icons.key, color: scheme.onSecondaryContainer),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                'Set OPENAI_API_KEY in flutter/example/.env and restart the app.',
                style: TextStyle(color: scheme.onSecondaryContainer),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Renders a single transcript entry based on its [ChatRole].
class _ChatItemView extends StatelessWidget {
  const _ChatItemView({required this.item});

  final ChatItem item;

  @override
  Widget build(BuildContext context) {
    switch (item.role) {
      case ChatRole.user:
        return _Bubble(
          text: item.text,
          alignment: Alignment.centerRight,
          color: Theme.of(context).colorScheme.primary,
          textColor: Theme.of(context).colorScheme.onPrimary,
        );
      case ChatRole.assistant:
        return _Bubble(
          text: item.text,
          alignment: Alignment.centerLeft,
          color: Theme.of(context).colorScheme.surfaceContainerHighest,
          textColor: Theme.of(context).colorScheme.onSurface,
        );
      case ChatRole.system:
        return Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
          child: Text(
            item.text,
            textAlign: TextAlign.center,
            style: TextStyle(
              color: Theme.of(context).hintColor,
              fontStyle: FontStyle.italic,
              fontSize: 12,
            ),
          ),
        );
      case ChatRole.error:
        return _Bubble(
          text: item.text,
          alignment: Alignment.centerLeft,
          color: Theme.of(context).colorScheme.errorContainer,
          textColor: Theme.of(context).colorScheme.onErrorContainer,
        );
      case ChatRole.recalled:
        return _RecalledSection(
          memories: item.recalled,
          scores: item.scores,
          rerankedByJina: item.rerankedByJina,
        );
      case ChatRole.stored:
        return _StoredSection(stored: item.stored);
    }
  }
}

/// A left/right aligned chat bubble.
class _Bubble extends StatelessWidget {
  const _Bubble({
    required this.text,
    required this.alignment,
    required this.color,
    required this.textColor,
  });

  final String text;
  final Alignment alignment;
  final Color color;
  final Color textColor;

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: alignment,
      child: Container(
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.78,
        ),
        margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: color,
          borderRadius: BorderRadius.circular(16),
        ),
        child: SelectableText(text, style: TextStyle(color: textColor)),
      ),
    );
  }
}

/// The "🧠 recalled" section listing retrieved memories (shown above the reply).
class _RecalledSection extends StatelessWidget {
  const _RecalledSection({
    required this.memories,
    required this.scores,
    required this.rerankedByJina,
  });

  final List<RecalledMemory> memories;

  /// Score to show per memory (parallel to [memories]): Jina relevance when
  /// [rerankedByJina], otherwise the semantic score.
  final List<double?> scores;

  /// Whether [memories] were reordered by the Jina reranker.
  final bool rerankedByJina;

  @override
  Widget build(BuildContext context) {
    final hintStyle = TextStyle(
      color: Theme.of(context).hintColor,
      fontStyle: FontStyle.italic,
      fontSize: 12,
    );

    if (memories.isEmpty) {
      return Align(
        alignment: Alignment.centerLeft,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 6, 16, 2),
          child: Text('🧠 no relevant memories yet', style: hintStyle),
        ),
      );
    }

    return Align(
      alignment: Alignment.centerLeft,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 2),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Padding(
              padding: const EdgeInsets.only(bottom: 6),
              child: Text(
                '🧠 recalled ${memories.length} '
                '${rerankedByJina ? "(reranked by Jina)" : "(semantic)"}',
                style: TextStyle(
                  color: Theme.of(context).hintColor,
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
            Wrap(
              spacing: 6,
              runSpacing: 6,
              children: [
                for (var i = 0; i < memories.length; i++)
                  Chip(
                    visualDensity: VisualDensity.compact,
                    materialTapTargetSize:
                        MaterialTapTargetSize.shrinkWrap,
                    label: Text(
                      _label(memories[i], i < scores.length ? scores[i] : null),
                      style: const TextStyle(fontSize: 12),
                    ),
                  ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  String _label(RecalledMemory m, double? score) => score != null
      ? '[${m.kind}] ${m.content}  (${score.toStringAsFixed(2)})'
      : '[${m.kind}] ${m.content}';
}

/// The "💾 stored" section showing what the extraction step decided to persist
/// for this turn — one chip per atomic memory as `[<type>] <content>`. When the
/// list is empty, the message was deemed chit-chat and nothing was stored.
///
/// Rendered distinctly from the 🧠 recalled section (tinted chips) so the
/// classification and selectivity are visible at a glance.
class _StoredSection extends StatelessWidget {
  const _StoredSection({required this.stored});

  final List<({String content, String type})> stored;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;

    if (stored.isEmpty) {
      return Align(
        alignment: Alignment.centerLeft,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 2, 16, 2),
          child: Text(
            '💾 nothing worth storing',
            style: TextStyle(
              color: Theme.of(context).hintColor,
              fontStyle: FontStyle.italic,
              fontSize: 12,
            ),
          ),
        ),
      );
    }

    return Align(
      alignment: Alignment.centerLeft,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 2, 16, 6),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Padding(
              padding: const EdgeInsets.only(bottom: 6),
              child: Text(
                '💾 stored ${stored.length}',
                style: TextStyle(
                  color: Theme.of(context).hintColor,
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
            Wrap(
              spacing: 6,
              runSpacing: 6,
              children: [
                for (final m in stored)
                  Chip(
                    visualDensity: VisualDensity.compact,
                    materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                    backgroundColor: scheme.tertiaryContainer,
                    side: BorderSide(color: scheme.tertiary.withValues(alpha: 0.4)),
                    label: Text(
                      '[${m.type}] ${m.content}',
                      style: TextStyle(
                        fontSize: 12,
                        color: scheme.onTertiaryContainer,
                      ),
                    ),
                  ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

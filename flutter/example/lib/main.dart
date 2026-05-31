import 'package:flutter/material.dart';
import 'package:memlocal/memlocal.dart';
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

const _apiKeyPref = 'openai_api_key';
const _jinaKeyPref = 'jina_api_key';
const _dimensions = 1536;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await RustLib.init();
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
enum ChatRole { user, recalled, assistant, system, error }

/// One renderable entry in the chat transcript.
class ChatItem {
  ChatItem.user(this.text)
      : role = ChatRole.user,
        recalled = const [],
        scores = const [],
        rerankedByJina = false;
  ChatItem.assistant(this.text)
      : role = ChatRole.assistant,
        recalled = const [],
        scores = const [],
        rerankedByJina = false;
  ChatItem.system(this.text)
      : role = ChatRole.system,
        recalled = const [],
        scores = const [],
        rerankedByJina = false;
  ChatItem.error(this.text)
      : role = ChatRole.error,
        recalled = const [],
        scores = const [],
        rerankedByJina = false;
  ChatItem.recalled(
    this.recalled, {
    required this.scores,
    required this.rerankedByJina,
  })  : role = ChatRole.recalled,
        text = '';

  final ChatRole role;
  final String text;
  final List<RecalledMemory> recalled;

  /// The score to display per recalled memory (parallel to [recalled]): the
  /// Jina relevance score when [rerankedByJina], otherwise the semantic score.
  final List<double?> scores;

  /// Whether [recalled] was reordered by the Jina reranker (vs. semantic order).
  final bool rerankedByJina;
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
  String? _apiKey;
  String? _jinaKey;
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
      final prefs = await SharedPreferences.getInstance();
      final key = prefs.getString(_apiKeyPref);
      final jinaKey = prefs.getString(_jinaKeyPref);
      if (!mounted) return;
      setState(() {
        _engine = engine;
        _initializing = false;
        _applyKey(key);
        _applyJinaKey(jinaKey);
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _initializing = false;
        _initError = '$e';
      });
    }
  }

  /// Builds (or clears) the OpenAI providers from a key. Call inside setState.
  void _applyKey(String? key) {
    final trimmed = key?.trim();
    if (trimmed == null || trimmed.isEmpty) {
      _apiKey = null;
      _embeddingProvider = null;
      _llmProvider = null;
      return;
    }
    _apiKey = trimmed;
    _embeddingProvider = OpenAIEmbeddingProvider(trimmed);
    _llmProvider = OpenAILlmProvider(trimmed);
  }

  /// Builds (or clears) the optional Jina reranker from a key. Call inside
  /// setState. When absent, [_reranker] stays null and the chat falls back to
  /// plain semantic top-5.
  void _applyJinaKey(String? key) {
    final trimmed = key?.trim();
    if (trimmed == null || trimmed.isEmpty) {
      _jinaKey = null;
      _reranker = null;
      return;
    }
    _jinaKey = trimmed;
    _reranker = JinaReranker(trimmed);
  }

  Future<void> _openSettings() async {
    final openAiController = TextEditingController(text: _apiKey ?? '');
    final jinaController = TextEditingController(text: _jinaKey ?? '');
    final saved = await showDialog<({String openAi, String jina})>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Settings'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: openAiController,
              obscureText: true,
              autofocus: true,
              decoration: const InputDecoration(
                labelText: 'OpenAI API key',
                hintText: 'sk-...',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: jinaController,
              obscureText: true,
              decoration: const InputDecoration(
                labelText: 'Jina API key (optional — enables reranking)',
                hintText: 'jina_...',
                border: OutlineInputBorder(),
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(
              ctx,
              (openAi: openAiController.text, jina: jinaController.text),
            ),
            child: const Text('Save'),
          ),
        ],
      ),
    );
    openAiController.dispose();
    jinaController.dispose();
    if (saved == null) return; // dialog cancelled

    final prefs = await SharedPreferences.getInstance();
    final openAi = saved.openAi.trim();
    if (openAi.isEmpty) {
      await prefs.remove(_apiKeyPref);
    } else {
      await prefs.setString(_apiKeyPref, openAi);
    }
    final jina = saved.jina.trim();
    if (jina.isEmpty) {
      await prefs.remove(_jinaKeyPref);
    } else {
      await prefs.setString(_jinaKeyPref, jina);
    }
    if (!mounted) return;
    setState(() {
      _applyKey(openAi);
      _applyJinaKey(jina);
    });
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

      // d. Store the current message for future turns.
      await engine.addMemory(content: text, embedding: embedding);
      // e. Build the memory-grounded system prompt + single LLM call.
      final system =
          'You are a helpful assistant with long-term memory of this user. '
          'Relevant memories you have recalled:\n'
          '${recalled.isEmpty ? "(none yet)" : recalled.map((m) => "- ${m.content}").join("\n")}'
          '\nUse them when relevant; if none apply, just answer normally.';
      final reply = await llmProvider.complete(system, text);

      // f. Show recalled context, then the assistant reply.
      if (!mounted) return;
      setState(() {
        _items.add(ChatItem.recalled(
          recalled,
          scores: scores,
          rerankedByJina: rerankedByJina,
        ));
        if (rerankNote != null) _items.add(ChatItem.system(rerankNote));
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('memlocal chat'),
        actions: [
          IconButton(
            tooltip: 'Settings',
            icon: const Icon(Icons.settings),
            onPressed: _initializing ? null : _openSettings,
          ),
        ],
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
        if (!_ready) _ApiKeyBanner(onTap: _openSettings),
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
                : 'Add your OpenAI API key to start.',
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
                  hintText:
                      _ready ? 'Message' : 'Add an API key to start…',
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

/// Inline banner shown when no API key is configured.
class _ApiKeyBanner extends StatelessWidget {
  const _ApiKeyBanner({required this.onTap});

  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Material(
      color: scheme.secondaryContainer,
      child: InkWell(
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          child: Row(
            children: [
              Icon(Icons.key, color: scheme.onSecondaryContainer),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  'Add your OpenAI API key to start',
                  style: TextStyle(color: scheme.onSecondaryContainer),
                ),
              ),
              Icon(Icons.chevron_right, color: scheme.onSecondaryContainer),
            ],
          ),
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
      ? '${m.content}  (${score.toStringAsFixed(2)})'
      : m.content;
}

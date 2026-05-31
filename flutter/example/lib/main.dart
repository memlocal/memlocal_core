import 'package:flutter/material.dart';
import 'package:memlocal/memlocal.dart';
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

const _apiKeyPref = 'openai_api_key';
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
enum ChatRole { user, recalled, assistant, error }

/// One renderable entry in the chat transcript.
class ChatItem {
  ChatItem.user(this.text)
      : role = ChatRole.user,
        recalled = const [];
  ChatItem.assistant(this.text)
      : role = ChatRole.assistant,
        recalled = const [];
  ChatItem.error(this.text)
      : role = ChatRole.error,
        recalled = const [];
  ChatItem.recalled(this.recalled)
      : role = ChatRole.recalled,
        text = '';

  final ChatRole role;
  final String text;
  final List<RecalledMemory> recalled;
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
  EmbeddingProvider? _embeddingProvider;
  LlmProvider? _llmProvider;

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
      if (!mounted) return;
      setState(() {
        _engine = engine;
        _initializing = false;
        _applyKey(key);
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

  Future<void> _openSettings() async {
    final controller = TextEditingController(text: _apiKey ?? '');
    final saved = await showDialog<String?>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('OpenAI API key'),
        content: TextField(
          controller: controller,
          obscureText: true,
          autofocus: true,
          decoration: const InputDecoration(
            labelText: 'sk-...',
            hintText: 'Paste your OpenAI API key',
            border: OutlineInputBorder(),
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, controller.text),
            child: const Text('Save'),
          ),
        ],
      ),
    );
    controller.dispose();
    if (saved == null) return; // dialog cancelled

    final prefs = await SharedPreferences.getInstance();
    final trimmed = saved.trim();
    if (trimmed.isEmpty) {
      await prefs.remove(_apiKeyPref);
    } else {
      await prefs.setString(_apiKeyPref, trimmed);
    }
    if (!mounted) return;
    setState(() => _applyKey(trimmed));
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

    setState(() {
      _items.add(ChatItem.user(text));
      _input.clear();
      _sending = true;
    });
    _scrollToBottom();

    try {
      // b. Embed the new message.
      final embedding = await embeddingProvider.embedOne(text);
      // c. Recall PRIOR memories (search BEFORE storing the current message).
      final recalled = await engine.searchSemantic(embedding: embedding, k: 5);
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
        _items.add(ChatItem.recalled(recalled));
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
            tooltip: 'OpenAI API key',
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
      case ChatRole.error:
        return _Bubble(
          text: item.text,
          alignment: Alignment.centerLeft,
          color: Theme.of(context).colorScheme.errorContainer,
          textColor: Theme.of(context).colorScheme.onErrorContainer,
        );
      case ChatRole.recalled:
        return _RecalledSection(memories: item.recalled);
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
  const _RecalledSection({required this.memories});

  final List<RecalledMemory> memories;

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
                '${memories.length == 1 ? "memory" : "memories"}',
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
                for (final m in memories)
                  Chip(
                    visualDensity: VisualDensity.compact,
                    materialTapTargetSize:
                        MaterialTapTargetSize.shrinkWrap,
                    label: Text(
                      m.score != null
                          ? '${m.content}  (${m.score!.toStringAsFixed(2)})'
                          : m.content,
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
}

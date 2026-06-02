# memlocal

**Local-first cognitive memory for AI agents — on-device, no server, no data leaving the device.**

`memlocal` is an open-source, local-first **cognitive memory engine for AI agents**
that runs entirely on-device (Rust + an embedded [CozoDB](https://www.cozodb.org/)
database). This package is its **Flutter/Dart binding**: it compiles the Rust
engine from source and exposes it to Dart over
[flutter_rust_bridge](https://pub.dev/packages/flutter_rust_bridge), so your
Flutter app can store and recall memories without a backend.

- Website: <https://memlocal.dev>
- Core engine: <https://github.com/memlocal/memlocal_core>

## Why memlocal

LLM apps forget everything between turns. The usual fix is a hosted vector
database, which means a server, a network round-trip, and your users' data
leaving their device. `memlocal` takes the opposite approach:

- **On-device and private.** The engine and its database run inside your app.
  No server, works offline, low latency, and nothing leaves the device unless
  *you* choose a cloud provider for embeddings or LLM calls.
- **Structured, not just a blob store.** Memories are typed across **8 cognitive
  memory types** (episodic, factual, semantic, procedural, social, spatial,
  prospective, affective), so recall can be richer than flat similarity.
- **One embedded database, multiple modes.** CozoDB gives you vector, full-text,
  and graph queries in a single embedded store.

As a credibility signal: on the **LoCoMo** (Long Conversation Memory) benchmark
the core engine reaches an **80% pass rate** (see the numbers and methodology at
<https://memlocal.dev> and in the [core repo](https://github.com/memlocal/memlocal_core)).

## Features

What ships in this binding today:

- **FFI to the Rust engine** — open a persistent or in-memory engine, store
  memories, and run semantic (HNSW) vector search, all from Dart.
- **Typed memories** — every stored memory carries a type (a stored-name string
  such as `factual`, `episodic`, `spatial`, …).
- **Bring-your-own providers** — embedding, LLM, and reranker abstractions you
  can implement for any backend, with ready-made implementations for **OpenAI**
  (embeddings + chat) and **Jina** (reranking) included.
- **An interactive example** — a memory-chat app that, on each turn, recalls
  relevant memories, stores what's worth keeping, and answers grounded in what
  it has recalled.

## Status

Early release (`0.1.0`). The FFI surface exposed here is a **focused subset** of
the full engine, and the API **will evolve**. Higher-level conveniences (batch
ingestion, automatic context assembly, deduplication, multi-channel retrieval,
etc.) live in the Rust core and are **not yet** exposed through this binding —
this package currently provides open / store / semantic-search plus the Dart
providers. See the roadmap in the
[core repo](https://github.com/memlocal/memlocal_core) for what's next.

## Platform support

| Platform | Status     | Native build               |
| -------- | ---------- | -------------------------- |
| Android  | Supported  | built from source (NDK)    |
| iOS      | Supported  | built from source (Xcode)  |

The native engine is **built from source** on your machine via
`flutter_rust_bridge` + [cargokit](https://github.com/irondash/cargokit) as part
of your normal `flutter build` / `flutter run`. There are no pre-built binaries.

### Prerequisites

Because consumers compile the Rust, you need a working Rust toolchain **and** the
relevant targets installed:

- A [Rust toolchain](https://rustup.rs/) (`rustup`).
- Android: the Android NDK, plus the Android Rust targets:

  ```bash
  rustup target add aarch64-linux-android armv7-linux-androideabi \
    x86_64-linux-android i686-linux-android
  ```

- iOS: Xcode (with command-line tools), plus the iOS Rust targets:

  ```bash
  rustup target add aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios
  ```

The first build compiles the Rust core and will take noticeably longer than a
pure-Dart package; subsequent builds are incremental.

## Install

Add the dependency to your app's `pubspec.yaml`:

```yaml
dependencies:
  memlocal: ^0.1.0
```

Then run `flutter pub get`.

## Quick start

Initialize the bridge, open an engine, wire up your providers, then store and
search:

```dart
import 'package:memlocal/memlocal.dart';
import 'package:path_provider/path_provider.dart';

Future<void> main() async {
  // 1. Initialize the Rust bridge before using any engine API.
  await RustLib.init();

  // 2. Open a persistent engine. `dimensions` must match your embedding model.
  final dir = await getApplicationDocumentsDirectory();
  final engine = await Memlocal.open(
    dbPath: '${dir.path}/memlocal.db',
    dimensions: 1536,
  );
  // (Or, for a throwaway engine: await Memlocal.openInMemory(dimensions: 1536);)

  // 3. Construct providers. Keys are your app's responsibility (see "Providers").
  final embeddings = OpenAIEmbeddingProvider(openAiApiKey); // 1536-dim by default
  final llm = OpenAILlmProvider(openAiApiKey);
  final reranker = JinaReranker(jinaApiKey); // optional

  // 4. Store a memory: embed the text yourself, then add it with a type.
  final content = 'Sirsho is building memlocal, an on-device memory engine.';
  final embedding = await embeddings.embedOne(content);
  final id = await engine.addMemory(
    content: content,
    kind: 'factual',
    embedding: embedding,
  );

  // 5. Recall: embed the query, then run semantic search for the top-k matches.
  final query = await embeddings.embedOne('What is Sirsho working on?');
  final results = await engine.searchSemantic(embedding: query, k: 5);
  for (final RecalledMemory m in results) {
    print('[${m.kind}] ${m.content}  (score: ${m.score})');
  }
}
```

### The store → recall → reply pattern

The example app wires these primitives into a chat loop. On each user message it:

1. **embeds** the message and **recalls** a candidate pool with `searchSemantic`,
   optionally reranking it down with the `JinaReranker`;
2. uses the LLM to **extract** what's worth keeping, splits it into atomic
   memories, classifies each into one of the 8 types, and **stores** them with
   `addMemory`; and
3. **replies** with the LLM, grounding the system prompt in the recalled
   memories.

See [`example/`](example/) for the full implementation.

## Providers

`memlocal` separates *storage/recall* (the on-device engine) from *intelligence*
(embeddings, generation, reranking). You supply the latter by implementing small
abstractions, so you can use any backend — cloud or local:

- `EmbeddingProvider` — `embedOne(text)` returns a vector; `dimensions` reports
  its size.
- `LlmProvider` — `complete(system, user)` returns a completion.
- `RerankerProvider` — `rerank(query, documents, {topN})` reorders candidates by
  relevance and returns `RerankResult`s (index into the input list + score).

Ready-made implementations are included:

- `OpenAIEmbeddingProvider` (`text-embedding-3-small`, 1536-dim by default).
- `OpenAILlmProvider` (`gpt-5.4-nano` by default).
- `JinaReranker` (`jina-reranker-v2-base-multilingual`).

API keys for any cloud provider are **the app's responsibility** — `memlocal`
never stores or transmits keys on your behalf. (The engine itself needs no keys
and makes no network calls.)

## Example

[`example/`](example/) is an **interactive memory chat**: it stores typed
memories from your messages, recalls relevant ones (with optional Jina
reranking), and answers grounded in that memory using `gpt-5.4-nano`.

To run it:

```bash
cd example
cp .env.example .env   # then fill in OPENAI_API_KEY (JINA_API_KEY is optional)
flutter run
```

Make sure the [prerequisites](#prerequisites) above are installed — the first
run compiles the native engine.

## Links

- Website: <https://memlocal.dev>
- Core engine (Rust): <https://github.com/memlocal/memlocal_core>
- License: [Apache-2.0](LICENSE)

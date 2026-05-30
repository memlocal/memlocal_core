# memlocal Flutter package — Design Spec

- **Date:** 2026-05-30
- **Author:** Sirsha Chakraborty (sirsho)
- **Status:** Draft for review
- **Topic:** A from-scratch Flutter/Dart package (`memlocal`) that binds the `memlocal_core` Rust engine for Android + iOS, publishable to pub.dev, with an example app.

---

## 1. Summary

`memlocal_core` is a local-first cognitive memory engine written in Rust (CozoDB-backed). Today it is consumable only from Rust. This project adds a new `flutter/` directory inside the `memlocal_core` repository containing a complete Flutter package named **`memlocal`** that exposes the engine to Dart via [flutter_rust_bridge](https://github.com/fzyzcjy/flutter_rust_bridge) (FRB), plus an example app demonstrating real usage.

The design is **layered**: a faithful FFI binding to the engine primitives, plus an optional higher-level Dart layer with pluggable AI providers. **All engine functionality ships in v1**, delivered through a **phased** implementation plan executed with Claude Code agent orchestration. The external AI services integrated in v1 are **OpenAI** (embeddings + LLM) and **Jina** (reranking, mirroring the core engine); the provider abstractions remain open for others.

## 2. Goals & non-goals

### Goals
- A self-contained, pub.dev-publishable Dart package `memlocal` under `memlocal_core/flutter/`.
- Full parity with the `MemlocalEngine` public API (CRUD, all search modes, context pipelines, tool-calling, consolidation, graph, time-travel, profiles, prospective, buffers).
- Pluggable Dart-side AI providers (embedding / LLM / reranker); OpenAI implementations shipped.
- Android + iOS support, built from source via flutter_rust_bridge + cargokit.
- A polished example app exercising the full surface.
- Tests that run offline/CI (mock embedder) and clean publish-readiness.

### Non-goals (v1)
- Prebuilt binary distribution (fast-follow; build-from-source for v1).
- Publishing `memlocal_core` to crates.io (use a git dependency at publish time).
- Desktop/web/WASM targets (engine may support them later; not in scope here).
- On-device/ONNX embeddings, or shipped provider implementations beyond OpenAI (embedding/LLM) and Jina (reranker) — other seams exist but their impls are out of scope for v1.
- Replacing the engine's algorithms; we bind, we do not redesign.

## 3. Context — what exists in `memlocal_core`

- `crate-type = ["lib", "cdylib", "staticlib"]`; already depends on `flutter_rust_bridge = "=2.11.1"`.
- `MemlocalEngine` (`src/api/bridge.rs`) is the single entry point — a `Send + Sync` struct (internally `Arc`/`Mutex`) whose methods take `&self`. The bridge doc states it is "held as an opaque pointer by the platform layer" and "all complex types cross the FFI boundary as JSON strings; `Vec<f32>` is passed as native float arrays."
- The engine makes **no network calls**. Embedding/LLM/reranker are pluggable **synchronous** Rust traits:
  - `EmbeddingProvider::embed_one(&self, text) -> Result<Vec<f32>>`
  - `LlmProvider::complete(&self, system, user) -> Result<String>`
  - `RerankerProvider::rerank(&self, query, docs, top_k) -> Result<Vec<(usize, f64)>>`
- Most engine methods accept a **pre-computed embedding** (`&[f32]`); only `execute_tool` and `prepare_context_iterative` require provider trait objects directly.
- CozoDB uses the SQLite backend (mobile-friendly). The optional `http` feature (reqwest) is **not** used by this package.
- Note: the README's reference to a separate `memlocal_dart` SDK is obsolete; this package is built fresh and does not depend on it.

## 4. Naming & repo layout

| Name | Kind | Location |
|---|---|---|
| `memlocal_core` | existing Rust core crate | `memlocal_core/` (repo root) |
| `memlocal_flutter` | Rust FFI wrapper crate | `memlocal_core/flutter/rust/` |
| `memlocal` | Dart/pub.dev package | `memlocal_core/flutter/` (package root) |

```
memlocal_core/
└── flutter/                         # pub.dev package root (name: memlocal)
    ├── pubspec.yaml
    ├── analysis_options.yaml
    ├── lib/
    │   ├── memlocal.dart             # public barrel (exports)
    │   └── src/
    │       ├── rust/                 # FRB-generated Dart (frb_generated.dart, api/…)
    │       ├── memlocal_base.dart    # Memlocal high-level wrapper
    │       ├── models/               # MemoryItem, MemoryType, MemoryEdge, configs, …
    │       └── providers/
    │           ├── embedding_provider.dart        # abstract
    │           ├── openai_embedding_provider.dart
    │           ├── llm_provider.dart              # abstract
    │           ├── openai_llm_provider.dart
    │           ├── reranker_provider.dart         # abstract
    │           └── jina_reranker.dart
    ├── rust/                         # crate: memlocal_flutter
    │   ├── Cargo.toml                # depends on memlocal_core (path in dev → git at publish)
    │   └── src/{lib.rs, api.rs, frb_generated.rs}
    ├── android/                      # cargokit Gradle hook (.so per ABI)
    ├── ios/                          # cargokit podspec + build-phase script
    ├── example/                      # demo app
    ├── flutter_rust_bridge.yaml
    ├── cargokit/                     # vendored build glue
    ├── CHANGELOG.md
    ├── README.md
    └── LICENSE                       # Apache-2.0 (matches core)
```

## 5. Architecture

```
┌── Dart package (memlocal) ─────────────────────────────────────────┐
│  Memlocal (high-level: addMemory/addMemories/search/prepareContext) │
│  EmbeddingProvider · LlmProvider · RerankerProvider  ── OpenAI HTTP  │  ← all networking in Dart
└───────────────────────────────┬────────────────────────────────────┘
                                 │ flutter_rust_bridge
                                 │ (opaque handle · Float32List · JSON · Dart closures)
┌── memlocal_flutter (rust/) ────┴────────────────────────────────────┐
│  api.rs — FRB-friendly wrappers + DartXProvider closure-adapters     │
└───────────────────────────────┬────────────────────────────────────┘
                                 │ Rust call (unchanged engine)
┌── memlocal_core ───────────────┴────────────────────────────────────┐
│  MemlocalEngine · CozoDB · EmbeddingProvider/LlmProvider/Reranker    │
└─────────────────────────────────────────────────────────────────────┘
```

Three units, each with one clear purpose and a well-defined interface:
- **`memlocal_core`** — the engine. Unchanged except for a few thin public pass-through methods (see §8.1).
- **`memlocal_flutter`** — the FFI seam. Translates between FRB-friendly signatures and the engine; hosts the provider closure-adapters. No business logic.
- **`memlocal` (Dart)** — ergonomic API, data models, providers, all HTTP/keys.

## 6. Provider model & the FFI bridge (the heart of the design)

The engine's provider traits are **synchronous Rust**; Dart providers are **asynchronous** (HTTP). Bridging them is the core engineering. Two complementary mechanisms:

### 6.1 Dart-closure adapter (primary)
Dart passes provider closures across FRB. The wrapper crate defines adapter structs implementing the engine traits by invoking those closures and bridging the async Dart result back to the synchronous signature (blocking the FRB **worker** thread — never the Dart UI isolate — on a channel until Dart replies):

```rust
// memlocal_flutter/src/api.rs (sketch)
struct DartEmbeddingProvider { embed: /* Dart: String -> Future<Vec<f32>> */ }
impl EmbeddingProvider for DartEmbeddingProvider {
    fn embed_one(&self, text: &str) -> Result<Vec<f32>> { /* call Dart, await result on worker thread */ }
}
// → lets the FULL existing pipeline run unchanged:
engine.prepare_context_iterative(query, &dart_embed, &dart_llm, user_id, max)
```

This is what makes the multi-round **iterative retrieval**, **reranking**, **extraction**, and the **tool-calling** loop work with Dart-side AI. It is the riskiest piece and is validated in Phase 0.

### 6.2 Precomputed-embedding fast path (optimization)
For operations that embed exactly one known text (e.g., `addMemory`, single-query search/context), Dart computes the embedding first and passes a `Float32List`; the wrapper injects a trivial in-memory embedder. No callback round-trips. Used where it simplifies without losing capability.

### 6.3 Threading & safety
- The engine is held as an FRB **opaque handle** wrapping `Arc<MemlocalEngine>` (already `Send + Sync`).
- Long-running FFI calls are exposed as FRB **async** functions so the Dart UI isolate never blocks.
- Adapter blocking happens only on FRB worker threads; the Dart closure executes on the Dart isolate, so no deadlock. This invariant is asserted by a Phase 0 test.

## 7. Reranking (Jina)

Reranking is central to the engine's accuracy — the cross-encoder reranking the *entire* candidate pool was the single highest-impact accuracy fix in core. The core uses **Jina Reranker v2** (`jina-reranker-v2-base-multilingual`, `src/http/jina_reranker.rs`). OpenAI has no rerank endpoint, so to preserve the benchmarked behavior we mirror core:
- Ship the abstract `RerankerProvider` seam in Dart, plus a concrete **`JinaReranker`** Dart provider (calls `https://api.jina.ai/v1/rerank`).
- OpenAI remains the embedding + LLM provider; Jina is the reranker. This relaxes "OpenAI-only" solely for reranking.
- Requires a Jina API key (generous free tier: 10M tokens). Reranking stays **optional** — `prepareContext` runs without it; `prepareContextReranked`/`iterative` use it only when invoked.
- The abstraction lets users drop in Cohere/Voyage/FlashRank later with no SDK change.

## 8. Rust FFI surface (`memlocal_flutter/src/api.rs`)

Complete coverage of `MemlocalEngine`. Complex types cross as JSON (engine convention) unless FRB-mirrored (§10). `Vec<f32>` ↔ `Float32List`.

### 8.1 Thin additions to `memlocal_core`
Surface the full context family currently private behind the engine:
- `MemlocalEngine::prepare_context(...)` (simple)
- `MemlocalEngine::prepare_context_reranked(...)`
(These wrap existing `ToolExecutor` methods; `prepare_context_iterative` already exists.)

### 8.2 Exposed functions
- **Lifecycle:** `open(config_json) -> MemlocalHandle`, `close(h)`
- **CRUD:** `put_memory(h, item_json, Float32List)`, `get_memory`, `get_memories`, `delete_memory`, `invalidate_memory`, `memory_count`
- **Search (embedding passed in):** `search_semantic`, `search_text`, `search_hybrid`, `search_graph`, `search_graph_recursive`, `search_at_time`
- **Higher-level (provider-driven, via §6 adapters):**
  - `add_memory(h, content, Float32List, type?, confidence?, user_id?)` — full dedup + store + auto-link (precomputed-embedding fast path, §6.2)
  - `add_memories(h, text, temporal_json, embed_cb, llm_cb)` — extraction → store
  - `prepare_context(h, query, user_id?, max?, embed_cb)`
  - `prepare_context_reranked(h, query, user_id?, max?, embed_cb, rerank_cb)`
  - `prepare_context_iterative(h, query, user_id?, max?, embed_cb, llm_cb, rerank_cb?)`
  - `execute_tool(h, tool_call_json, embed_cb, llm_cb) -> tool_result_json`
  - `get_tool_definitions() -> Vec<json>`
- **Consolidation:** `find_consolidation_clusters(h, user_id?, embeddings, min_age_secs, min_cluster_size) -> Vec<Vec<MemoryItem>>` (Dart performs LLM summarization on returned clusters)
- **Graph/Profile/Prospective/Buffers/WorkingMemory:** `put_edge`, `get_edges`, `put_profile`, `get_profile`, `put_prospective`, `get_pending_prospective`, `complete_prospective`, `append_message`, `get_messages`, `sensory_add/items/clear`, `working_memory_*`, `get_important_memories`, `export_relations`

## 9. Dart public API

```dart
final mem = await Memlocal.open(
  config: MemlocalConfig(dbPath: dbPath, embeddingDimensions: 1536),
  embeddingProvider: OpenAIEmbeddingProvider(apiKey: key),     // enables embedding-driven methods
  llmProvider:       OpenAILlmProvider(apiKey: key),           // enables extraction / iterative / tools
  rerankerProvider:  JinaReranker(apiKey: jinaKey),            // optional; enables reranked/iterative context
);

// High-level (providers do the AI work in Dart)
await mem.addMemory('User prefers dark mode', type: MemoryType.factual);
final res  = await mem.addMemories(rawConversationText);       // OpenAI extraction → store
final ctx  = await mem.prepareContext('What UI prefs?', mode: ContextMode.reranked); // -> String
final hits = await mem.search('UI preferences', k: 5, mode: SearchMode.hybrid);
final clusters = await mem.consolidate(userId: 'u1');          // clusters → Dart summarize → store

// Low-level (no providers; you pass embeddings)
await mem.putMemory(item, embedding);                          // Float32List
final hits2 = await mem.searchSemantic(embedding, k: 5);

await mem.close();
```

Provider abstractions:
```dart
abstract class EmbeddingProvider { Future<List<double>> embedOne(String text); int get dimensions; }
abstract class LlmProvider       { Future<String> complete(String system, String user); }
abstract class RerankerProvider  { Future<List<RankedDoc>> rerank(String query, List<String> docs, int topK); }
```
Shipped: `OpenAIEmbeddingProvider` (text-embedding-3-small, 1536-d), `OpenAILlmProvider` (gpt-5.4-nano default, configurable), `JinaReranker` (jina-reranker-v2-base-multilingual).

## 10. Data types & serialization

- **FRB-mirrored Dart classes** for the high-traffic, stable types: `MemoryItem`, `MemoryType` (enum), `MemoryEdge`, `MemlocalConfig`/`StorageConfig`, `Message`, `ToolDefinition`. Type-safe, no manual JSON.
- **JSON strings** for rarer/deeply-nested or fast-moving shapes (`ToolResult`, `export_relations` output, consolidation payloads), decoded by hand-written Dart models where ergonomics warrant.
- Embeddings always travel as `Float32List`; provider results (`List<double>` from JSON) are converted at the boundary.
- Enum string mapping mirrors the engine's `from_stored_name`/stored-name conventions exactly.

## 11. Example app (OpenAI)

A small but complete app that doubles as copy-paste reference:
- **Setup:** API-key fields — OpenAI (required) and Jina (optional, enables reranked/iterative recall) — held in memory / `shared_preferences`; DB path via `path_provider`.
- **Capture:** add a typed memory; "add from text" runs OpenAI extraction → multiple memories.
- **Recall:** an "Ask" box runs `prepareContext` (toggle simple / reranked / iterative) and shows the assembled context block alongside raw `search` hits with scores and channels.
- **Browse:** memory list with type chips, importance, age; delete/invalidate.
- **Extras:** user profile view; prospective reminders; a "consolidate" action.
- Clear separation so the provider classes and `Memlocal` calls are obvious to readers.

## 12. Build, dev & publish workflow

- **Codegen:** `flutter_rust_bridge_codegen generate` → `frb_generated.rs` + `lib/src/rust/`.
- **Native build:** cargokit compiles `rust/` during `flutter build` — Android `.so` per ABI (arm64-v8a, armeabi-v7a, x86_64), iOS via build-phase script (device arm64 + simulator).
- **Core dependency:** `memlocal_core = { path = "../../" }` for local dev and the example (works in-repo); flip to `{ git = "https://github.com/Sirsho29/memlocal_core", tag = "vX.Y.Z" }` at publish time (documented one-line switch; optional helper script).
- **Publish:** `flutter pub publish --dry-run` must pass; pub.dev metadata (description, homepage, repository, topics) set in `pubspec.yaml`.
- **Toolchain prerequisites** (documented in README): Rust stable + targets, Android NDK, Xcode; `flutter_rust_bridge_codegen`.

## 13. Testing

- **Dart unit tests** with a deterministic mock `EmbeddingProvider` (hash-based) so suites run offline/CI with no key. Mock is a test aid, not shipped.
- **Rust smoke test** in the wrapper crate: open in-memory engine, put/search round-trip, and a callback-bridge test (Rust invokes a Dart/stub closure and blocks correctly).
- **Integration:** run the example on one Android emulator + one iOS simulator each CI run (build) and at least once manually on physical Android + iOS.
- **Golden parity (selective):** for a small fixture set, assert Dart results match the Rust engine's direct results.

## 14. Phased delivery plan & agent management

Detailed task breakdown is produced by the writing-plans step; this is the agreed shape. Execution uses Claude Code orchestration: each task is TDD-first, gated by verification-before-completion, with a code-review checkpoint per phase; independent tasks fan out to parallel subagents.

| Phase | Focus | Exit criteria | Agents |
|---|---|---|---|
| **0 — Walking skeleton** | Scaffold `flutter/` + `memlocal_flutter`; FRB+cargokit wired; prove CozoDB cross-compiles; prove the §6.1 callback bridge | `memoryCount()` returns over FFI on Android device + iOS sim; a Rust→Dart callback round-trips | Sequential, single agent |
| **1 — Core FFI + models** | CRUD, all searches, edges, profile, prospective, buffers, working memory, time-travel, graph-recursive; FRB-mirrored models; low-level `Memlocal` + mock-embedder tests | All pure methods callable + tested from Dart | Parallel subagents |
| **2 — Providers + bridge** | Dart provider abstractions + OpenAI impls; Rust closure-adapters; `addMemory` | addMemory + semantic search work end-to-end with OpenAI | Small parallel set |
| **3 — Pipelines** | addMemories (extraction), prepareContext (simple/reranked/iterative), consolidation, executeTool + tool defs | Full pipeline parity verified against engine | Parallel where independent |
| **4 — Example app** | Full OpenAI demo across the surface | App runs on Android + iOS, exercises all features | Single agent |
| **5 — Polish & publish-readiness** | README, dartdoc, CHANGELOG, lints, CI, dep flip, `pub publish --dry-run` | Dry-run clean; CI green | Single agent |

Phase 0 is strictly sequential (it discovers the build/threading unknowns). Phases 1 and 3 are the main fan-out points.

## 15. Risks & mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| CozoDB + SQLite won't cross-compile cleanly for all Android ABIs / iOS | High | Phase 0 walking skeleton proves it before any breadth work |
| Sync-trait → async-Dart callback bridge deadlock/threading | High | Phase 0 dedicated bridge test; block only FRB worker threads; document the invariant |
| FRB 2.11.1 codegen quirks vs engine types | Med | Pin FRB version; mirror only stable types, JSON for the rest |
| Binary size with CozoDB | Med | No `http`/reqwest in the lib; `opt-level="z"` + LTO already set in core |
| iOS min version / NDK minSdk specifics | Med | Pin and document during Phase 0 (candidate: iOS 13+, Android minSdk 23+) |
| Reranker is a separate service + key (Jina) | Low | Reranking optional (pipeline runs without it); free tier; pluggable for other rerankers |
| Publish-time path→git dep drift | Low | Helper script + `--dry-run` gate in CI |

## 16. Open questions

1. ~~Reranker default~~ — **Decided (Option A):** ship `JinaReranker` (`jina-reranker-v2-base-multilingual`), mirroring core; OpenAI for embeddings + LLM. Needs a Jina API key (free tier).
2. ~~Default OpenAI LLM model~~ — **Decided:** `gpt-5.4-nano` ([docs](https://developers.openai.com/api/docs/models/gpt-5.4-nano)), configurable. To confirm during Phase 3: exact API surface (Chat Completions vs Responses API) and structured-output/JSON mode for the extraction call. (Embeddings remain `text-embedding-3-small`.)
3. Minimum platform versions (iOS / Android `minSdk`) — to pin in Phase 0.
4. License file: Apache-2.0 to match core — assumed.

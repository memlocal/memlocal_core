/// Local-first cognitive memory for AI agents — a Flutter/Dart binding to the
/// Rust [memlocal](https://memlocal.dev) engine (an embedded CozoDB store).
///
/// The engine runs entirely on-device: there is no server and no data leaves
/// the device. This library exposes it to Dart over `flutter_rust_bridge`.
///
/// Call [RustLib.init] once before using any engine API, then open an engine
/// with [Memlocal.open] (persistent) or [Memlocal.openInMemory]. Store memories
/// with [Memlocal.addMemory] and recall them with [Memlocal.searchSemantic].
///
/// Embeddings, LLM completion, and reranking are *bring-your-own*: implement
/// [EmbeddingProvider], [LlmProvider], and [RerankerProvider] for any backend,
/// or use the included [OpenAIEmbeddingProvider], [OpenAILlmProvider], and
/// [JinaReranker]. See the package README and `example/` for the full
/// store → recall → reply pattern.
library;

export 'src/rust/api/skeleton.dart';
export 'src/rust/frb_generated.dart' show RustLib;
export 'src/providers/embedding_provider.dart';
export 'src/providers/openai_embedding_provider.dart';
export 'src/providers/llm_provider.dart';
export 'src/providers/openai_llm_provider.dart';
export 'src/providers/reranker_provider.dart';
export 'src/providers/jina_reranker.dart';

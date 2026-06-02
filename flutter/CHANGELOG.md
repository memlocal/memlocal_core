# Changelog

## 0.1.0

Initial release of the Flutter/Dart binding to the [memlocal](https://memlocal.dev)
cognitive memory engine.

- **FFI binding to the Rust engine** via `flutter_rust_bridge`, built from source
  with cargokit:
  - Open a persistent engine (`Memlocal.open`) backed by a database file, or an
    in-memory engine (`Memlocal.openInMemory`).
  - Store a memory with a typed classification and a caller-supplied embedding
    (`addMemory`).
  - Semantic (HNSW) vector search over stored memories (`searchSemantic`).
  - Count stored memories (`memoryCount`).
- **Bring-your-own provider model** with abstractions for embeddings, LLM
  completion, and reranking, plus shipped implementations:
  - `OpenAIEmbeddingProvider` and `OpenAILlmProvider` (OpenAI HTTP APIs).
  - `JinaReranker` (Jina AI rerank API).
- **Interactive memory-chat example** that stores typed memories, recalls them
  with optional Jina reranking, and answers grounded in recalled memory.
- **Platforms:** Android and iOS, built from source via `flutter_rust_bridge` +
  cargokit (a Rust toolchain with the relevant targets is required).

This is an early release: the FFI surface is a focused subset of the engine and
the API will evolve in subsequent versions.

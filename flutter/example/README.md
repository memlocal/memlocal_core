# memlocal example — memory chat

An interactive **memory chat** built on the [`memlocal`](https://pub.dev/packages/memlocal)
Flutter binding. On every message it:

1. **recalls** relevant memories with semantic search (and reranks them with
   Jina when a `JINA_API_KEY` is set);
2. **stores** what's worth keeping — an LLM extracts atomic memories from your
   message and classifies each into one of the engine's 8 cognitive memory
   types; and
3. **replies** with `gpt-5.4-nano`, grounded in the recalled memories.

The transcript shows what was recalled and what was stored on each turn, so you
can watch the memory build up.

## Running

```bash
cp .env.example .env   # in this directory
# then edit .env:
#   OPENAI_API_KEY=...   (required — embeddings + replies)
#   JINA_API_KEY=...     (optional — enables reranking)
flutter run
```

## Prerequisites

This package builds the Rust engine **from source**, so you need a working Rust
toolchain (via [rustup](https://rustup.rs/)) with the Android/iOS targets
installed, plus the Android NDK or Xcode for your target platform. See the
[package README](../README.md#prerequisites) for the exact `rustup target add`
commands. The first build will take longer while the native core compiles.

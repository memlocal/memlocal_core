# memlocal_core

Local-first AI memory engine modeled on human cognitive architecture. Manages sensory, short-term, and long-term memory backed by an embedded CozoDB database with vector search, full-text search, and graph algorithms.

## Tech Stack

- **Rust 2021** edition, crate types: lib, cdylib, staticlib
- **CozoDB 0.7** (embedded, SQLite backend) -- vector (HNSW), BM25 FTS, LSH, graph algorithms
- **flutter_rust_bridge 2.11.1** for FFI to Dart/Flutter
- **Key deps:** serde/serde_json, uuid v4, sha2 (SHA-256), chrono, thiserror, rayon, log

## Module Map

| Module | Purpose |
|---|---|
| `api::bridge` | FFI bridge -- `MemlocalEngine` is the main opaque handle held by platform layers |
| `storage::memory_store` | Low-level CozoDB operations: CRUD, search (semantic/text/LSH/hybrid/graph), scoring, edges, profiles, prospective |
| `storage::schema` | CozoScript DDL for all 5 relations + 3 indices |
| `models` | 15 data types: MemoryItem, MemoryType, MemoryEdge, MemoryRelation, Message, UserProfile, ProspectiveItem, CoreConfig, StorageConfig, SearchMode, MemoryCategory, MemorySearchResult, ContextResult, AddResult, ConsolidationResult, MemoryDelta |
| `shortterm` | SensoryBuffer (in-memory VecDeque + TTL), ConversationBuffer (CozoDB-backed sliding window), WorkingMemory (transient context assembly) |
| `longterm` | 8 subsystem wrappers: Episodic, Semantic, Factual, Procedural, Social, Spatial, Prospective, Affective |
| `consolidation` | MemoryConsolidator -- greedy cosine clustering of episodic memories for summarization |
| `tools` | 10 tool definitions, ToolExecutor (dispatch + prepare_context), LLM extraction prompts |
| `error` | MemlocalError enum (7 variants) + Result type alias |
| `http` (feature-gated) | Optional HTTP client for LLM/embedding APIs |

## Key Types

- `MemlocalEngine` -- the main entry point; wraps store, buffers, tools, consolidator, 8 LT subsystems
- `MemoryItem` -- single memory with id, content, type, hash, timestamps, metadata, optional score
- `MemoryType` -- 12-variant enum (SensoryBuffer, WorkingMemory, AttentionContext, ConversationBuffer, Episodic, Semantic, Factual, Procedural, Social, Spatial, Prospective, Affective)
- `MemoryEdge` -- directed graph edge with from_id, to_id, relation, weight
- `MemoryRelation` -- 10-variant enum (RelatesTo, Contradicts, Supersedes, CausedBy, PartOf, PrefersOver, Follows, InstanceOf, BelongsTo, SimilarTo)
- `SearchMode` -- Semantic, Text, Graph, Hybrid
- `ProspectiveItem` -- future reminder with TriggerType (TopicMention, TimeBased, UserPresence, SemanticMatch)
- `UserProfile` -- static_facts + dynamic_context as BTreeMap<String, String>
- `CoreConfig` / `StorageConfig` -- all tunable parameters with defaults

## 10 Memory Tools

| Tool | Required Args | Description |
|---|---|---|
| `add_memory` | content | Store a memory with hash + semantic dedup |
| `add_memories` | text | LLM-driven extraction from raw conversation text |
| `search_memory` | query | Search by semantic/text/hybrid/graph mode |
| `get_memories` | -- | List memories with optional type/user filters |
| `delete_memory` | memory_id | Soft-delete via CozoDB RETRACT |
| `get_user_profile` | user_id | Retrieve user profile summary |
| `add_relationship` | from_id, to_id, relation | Create a knowledge graph edge |
| `get_relationships` | memory_id | Get all edges for a memory node |
| `add_reminder` | content, trigger_type, trigger_condition | Create a prospective trigger |
| `get_context` | query | Assemble context block for LLM injection |

## Key Commands

```bash
cargo build                    # Debug build
cargo build --release          # Optimized (LTO, opt-level z, strip)
cargo test                     # Run tests
cargo build --features http    # Include reqwest-based HTTP client
```

## Architecture Pointers

- [.claude/architecture.md](.claude/architecture.md) -- Memory taxonomy, ingestion/search pipeline, data flow
- [.claude/conventions.md](.claude/conventions.md) -- Rust patterns, module structure, FFI rules
- [.claude/storage.md](.claude/storage.md) -- CozoDB schema, indices, Datalog query patterns
- [.claude/memory-subsystems.md](.claude/memory-subsystems.md) -- Deep guide per subsystem (sensory through affective)
- [.claude/scoring.md](.claude/scoring.md) -- Time-decay, RRF, importance, confidence formulas
- [.claude/gotchas.md](.claude/gotchas.md) -- Known issues and non-obvious behaviors

## Critical Invariants

1. **All complex types cross FFI as JSON strings.** `Vec<f32>` (embeddings) is the exception -- FRB handles native float arrays efficiently. See `api/bridge.rs` header comment.

2. **`Arc<MemoryStore>` is the shared ownership pattern.** Every subsystem (episodic, semantic, ...), the ToolExecutor, and the MemoryConsolidator each hold an `Arc::clone(&store)`. The MemlocalEngine itself holds the original `Arc<MemoryStore>`.

3. **WorkingMemory is transient.** It is assembled per query via `prepare_context()` and never persisted. It lives behind a `Mutex<WorkingMemory>` on the engine for thread safety.

4. **Embeddings default to 1536 dimensions** (matching OpenAI text-embedding-3-small). Configured via `StorageConfig::embedding_dimensions`, default in `config.rs` line 34.

5. **CozoDB Validity (time-travel)** is used for soft-delete (RETRACT) and history. The `vld: Validity` key column on `mem_items` enables `@ "NOW"` queries to see only current versions.

6. **Dedup is two-layer:** exact hash (SHA-256 of content) then semantic similarity (cosine > 0.85 or > 0.70 with conflicting specifics).

7. **`MemoryType::from_stored_name` defaults to `Semantic`** for unknown strings. `MemoryRelation::from_stored_name` defaults to `RelatesTo`. `TriggerType::from_stored_name` defaults to `TopicMention`.

8. **`min_confidence_to_store` defaults to 0.3** (in StorageConfig), but the `add_memories` tool uses a harder threshold of 0.5 to skip low-confidence extractions.

9. **delete_memory() is a soft-delete.** It uses CozoDB RETRACT validity -- the memory becomes invisible at `@ "NOW"` but its history is preserved.

10. **10 memory tools** are defined in `tools/tool_definitions.rs`. All are dispatched through `ToolExecutor::dispatch()`. The `add_memories` tool requires `LlmProvider` in the execution context; all others only need `EmbeddingProvider`.

## Default Configuration Values

| Parameter | Default | Location |
|---|---|---|
| `embedding_dimensions` | 1536 | `StorageConfig` |
| `hnsw_m` | 16 | `StorageConfig` |
| `hnsw_ef_construction` | 100 | `StorageConfig` |
| `min_confidence_to_store` | 0.3 | `StorageConfig` |
| `enable_time_decay` | true | `StorageConfig` |
| `conversation_buffer_size` | 20 | `CoreConfig` |
| `sensory_buffer_capacity` | 100 | `CoreConfig` |
| `sensory_ttl_ms` | 5000 | `CoreConfig` |
| RRF constant K | 60.0 | `memory_store.rs` (hardcoded) |
| Consolidation cluster threshold | 0.65 | `consolidator.rs` (hardcoded) |
| Semantic dedup threshold | 0.85 | `tool_executor.rs` (hardcoded) |
| Hybrid dedup word overlap | 0.60 | `memory_store.rs` (hardcoded) |
| Graph search hops | 2 | `bridge.rs` (hardcoded) |

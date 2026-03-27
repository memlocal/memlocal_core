# Conventions

## Error Handling

All errors use `MemlocalError` (defined in `error.rs`) with `thiserror` derive:

| Variant | Wraps | When |
|---|---|---|
| `Database(String)` | Manual | CozoDB instance open/init failures |
| `Schema(String)` | Manual | Schema creation issues |
| `Query(String)` | Manual | CozoScript execution failures (maps from `cozo::Error`) |
| `NotFound(String)` | Manual | Missing entities |
| `Serialization(serde_json::Error)` | `#[from]` | JSON ser/de failures (automatic conversion) |
| `InvalidArgument(String)` | Manual | Bad tool call arguments, missing required fields |
| `Internal(String)` | Manual | Catch-all for unexpected errors (e.g., LLM response parse failure) |

Convenience alias: `pub type Result<T> = std::result::Result<T, MemlocalError>;`

CozoDB errors are converted to strings at the boundary -- the crate does not expose CozoDB types publicly.

## FFI Boundary (api/bridge.rs)

- **All complex types cross FFI as JSON strings.** This is stated in the module doc comment. The platform layer serializes/deserializes on both sides.
- **`Vec<f32>` (embeddings) are native float arrays** -- flutter_rust_bridge handles these efficiently without JSON conversion.
- **`MemlocalEngine` is the single opaque handle.** The platform layer holds one instance. It wraps:
  - `Arc<MemoryStore>` -- the database
  - `Mutex<SensoryBuffer>` -- thread-safe sensory buffer
  - `Mutex<HashMap<String, ConversationBuffer>>` -- per-session conversation buffers
  - `Mutex<WorkingMemory>` -- thread-safe working memory
  - `ToolExecutor` -- tool dispatch
  - `MemoryConsolidator` -- clustering
  - `CoreConfig` -- configuration snapshot
  - 8 long-term subsystem structs (public fields)

## Ownership Pattern

`Arc<MemoryStore>` is cloned and distributed to every component that needs database access:

```
MemlocalEngine
  store: Arc<MemoryStore>  <-- original
  tool_executor: ToolExecutor(Arc::clone(&store))
  consolidator: MemoryConsolidator(Arc::clone(&store))
  episodic: EpisodicMemory(Arc::clone(&store))
  semantic: SemanticMemory(Arc::clone(&store))
  factual: FactualMemory(Arc::clone(&store))
  procedural: ProceduralMemory(Arc::clone(&store))
  social: SocialMemory(Arc::clone(&store))
  spatial: SpatialMemory(Arc::clone(&store))
  prospective: ProspectiveMemory(Arc::clone(&store))
  affective: AffectiveMemory(Arc::clone(&store))
```

ConversationBuffer also holds `Arc<MemoryStore>` -- created lazily per session in `append_message()`.

## Trait Abstractions

### EmbeddingProvider (tools/tool_executor.rs)

```rust
pub trait EmbeddingProvider: Send + Sync {
    fn embed_one(&self, text: &str) -> Result<Vec<f32>>;
}
```

Platform layer implements this. Required for all search operations and memory storage.

### LlmProvider (tools/tool_executor.rs)

```rust
pub trait LlmProvider: Send + Sync {
    fn complete(&self, system: &str, user: &str) -> Result<String>;
}
```

Only needed for `add_memories` tool (LLM-driven extraction/classification). Optional in `ExecutionContext`.

### Summarizer (consolidation/consolidator.rs)

```rust
pub trait Summarizer: Send + Sync {
    fn summarize(&self, contents: &[String]) -> Result<String>;
}
```

For consolidation LLM calls. Defined but not used internally -- the platform layer drives consolidation.

## Module Organization

```
src/
  lib.rs              -- pub mod declarations (7 modules + 1 feature-gated)
  error.rs            -- MemlocalError + Result alias
  api/
    mod.rs            -- re-exports MemlocalEngine
    bridge.rs         -- MemlocalEngine struct + all public API methods
  models/
    mod.rs            -- 15 sub-modules, all glob re-exported
    config.rs         -- CoreConfig, StorageConfig with defaults
    memory_type.rs    -- MemoryType enum (12 variants)
    memory_category.rs -- MemoryCategory enum (3 variants)
    memory_item.rs    -- MemoryItem struct, SHA-256 hashing, CozoDB serialization
    memory_edge.rs    -- MemoryEdge struct
    memory_relation.rs -- MemoryRelation enum (10 variants)
    message.rs        -- Message struct (role, content, timestamp)
    user_profile.rs   -- UserProfile with static_facts/dynamic_context BTreeMaps
    prospective_item.rs -- ProspectiveItem, TriggerType enum (4 variants)
    search_mode.rs    -- SearchMode enum (4 variants)
    memory_search_result.rs -- MemorySearchResult container
    context_result.rs -- ContextResult container
    add_result.rs     -- AddResult container
    consolidation_result.rs -- ConsolidationResult container
    memory_delta.rs   -- MemoryDelta, MemoryAction enum
  storage/
    mod.rs            -- re-exports MemoryStore, MemorySchema
    schema.rs         -- CozoScript DDL generation
    memory_store.rs   -- All database operations (~1275 lines)
  shortterm/
    mod.rs            -- re-exports SensoryBuffer, ConversationBuffer, WorkingMemory
    sensory_buffer.rs -- In-memory VecDeque with TTL eviction
    conversation_buffer.rs -- CozoDB-backed sliding window
    working_memory.rs -- Transient context assembly
  longterm/
    mod.rs            -- re-exports all 8 subsystem structs
    episodic.rs       -- EpisodicMemory
    semantic.rs       -- SemanticMemory
    factual.rs        -- FactualMemory
    procedural.rs     -- ProceduralMemory
    social.rs         -- SocialMemory (+ graph algorithms)
    spatial.rs        -- SpatialMemory
    prospective.rs    -- ProspectiveMemory (+ trigger checking)
    affective.rs      -- AffectiveMemory
  consolidation/
    mod.rs            -- re-exports MemoryConsolidator, Summarizer
    consolidator.rs   -- Greedy cosine clustering
  tools/
    mod.rs            -- re-exports TemporalContext, all definitions, all executor types
    tool_definitions.rs -- 10 tool definitions + tool_names constants
    tool_executor.rs  -- ToolExecutor, EmbeddingProvider, LlmProvider, ToolCall, ToolResult
    prompts.rs        -- EXTRACTION_SYSTEM, DEDUP_SYSTEM, CONSOLIDATION_SYSTEM prompts, TemporalContext
  http/              -- (feature-gated, `--features http`) reqwest-based HTTP client
```

## Feature Gates

| Feature | Dependency | Purpose |
|---|---|---|
| `http` | `dep:reqwest` (0.12, json+blocking) | HTTP client for LLM/embedding APIs |

Default features: none. The `http` feature adds `pub mod http` to `lib.rs`.

## Crate Output Types

`crate-type = ["lib", "cdylib", "staticlib"]` -- supports Rust library, C dynamic library (for FFI), and C static library.

## Release Profile

```toml
[profile.release]
lto = true
opt-level = "z"     # Optimize for size
strip = true
codegen-units = 1
```

## Naming Conventions

- **Relation names:** prefixed with `mem_` (mem_items, mem_edges, mem_conversations, mem_profiles, mem_prospective)
- **Index names:** prefixed with `mem_` (mem_vec_idx, mem_fts_idx, mem_lsh_idx)
- **Stored names:** snake_case strings for all enums (memory types, relations, trigger types, search modes)
- **Default fallbacks:** Unknown stored names silently default to a safe variant rather than erroring (Semantic for types, RelatesTo for relations, TopicMention for triggers, Hybrid for search modes)

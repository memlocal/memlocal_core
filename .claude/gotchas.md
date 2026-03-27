# Gotchas

Non-obvious behaviors discovered by reading the source code.

## 1. LSH Index Is Used in Primary Search (But Silently Fails)

The LSH index (`mem_lsh_idx`) IS used in hybrid search -- it is the third signal alongside semantic and BM25. However, `search_lsh()` catches all errors and returns an empty vec on failure (`Err(_) => Ok(vec![])`). Short queries or queries with only common words may produce LSH errors that are silently swallowed. This means hybrid search results can vary between 2-signal and 3-signal RRF without any indication.

## 2. SemanticMatch Trigger Does NOT Use Embeddings

In `longterm/prospective.rs`, `check_triggers()` accepts a `_query_embedding: Option<&[f32]>` parameter but never uses it. The `SemanticMatch` trigger type falls back to the same keyword substring matching as `TopicMention`. The comment says "The platform layer can do a proper semantic comparison." The constant `_SEMANTIC_TRIGGER_THRESHOLD = 0.75` is defined but unused (underscore-prefixed to suppress dead code warning).

## 3. Embedding Dimension Mismatch Is Silent

If you change `StorageConfig::embedding_dimensions` after initial schema creation, the HNSW index was already created with the original dimension. CozoDB will likely error on vector operations but `try_run()` swallows index creation errors. You could end up with a working database but broken vector search with no clear error message about dimension mismatch.

## 4. WorkingMemory Is Never Persisted

`WorkingMemory` is purely transient -- it exists only in memory behind `Mutex<WorkingMemory>` on the engine. Despite having `MemoryType::WorkingMemory` and `MemoryType::AttentionContext` defined with TTLs (300s and 60s respectively), these types are never actually stored to CozoDB by the core library. They exist for the platform layer to use if desired.

## 5. min_confidence_to_store Is Defined But Not Enforced in Core

`StorageConfig::min_confidence_to_store` (default 0.3) is defined in the config but is NOT checked anywhere in the core crate's storage or tool executor code. The `add_memories` tool has its own hardcoded threshold of 0.5 for skipping low-confidence extractions. The config value exists for the platform layer to respect.

## 6. Validity Column Is Auto-Set to ASSERT on Every Write

When storing a memory via `put_memory()`, the `vld` column is always set to the literal string `'ASSERT'`. CozoDB interprets this as "assert this version is valid from now on." You cannot manually set a specific validity timestamp from the Rust API -- all writes are "as of now." The Validity mechanism is only used for two operations: ASSERT (create/update) and RETRACT (soft-delete).

## 7. Graph Search Depth Is Hardcoded to 2

`MemlocalEngine::search_graph()` calls `store.search_graph(embedding, k, user_id, memory_type, 2)` with `hops = 2` hardcoded. The internal method supports variable depth, but the public API does not expose it. Depth-2 neighbor scores are further dampened by a 0.5 multiplier: `hop2_score = hop1_score * weight * 0.5`.

## 8. Hash Field Is for Dedup, Not Security

`MemoryItem::compute_hash()` uses SHA-256, but the hash is a **content-addressable dedup key**, not a security mechanism. It is checked via `store.find_by_hash()` in the `add_memory` tool to detect exact duplicate content before embedding generation (which is expensive). It is not used for integrity verification.

## 9. delete_memory() Is Actually invalidate_memory()

`MemoryStore::delete_memory()` calls `invalidate_memory()` which uses `vld = 'RETRACT'`. This means "deleted" memories are actually soft-deleted -- their history is preserved and they could theoretically be queried at a past time-travel point. There is no true hard-delete operation exposed.

## 10. Semantic Dedup Threshold Is Asymmetric

In `add_memory()`, there are two dedup thresholds:
- Score > 0.85: near-duplicate, always updates existing
- Score > 0.70 AND conflicting specifics: contradiction detected, updates existing

The `has_conflicting_specifics()` function only checks for differing numeric tokens (digits). Non-numeric contradictions (e.g., "likes cats" vs "likes dogs") are NOT detected.

## 11. ConversationBuffer.clear() Does Not Delete from DB

Calling `clear()` on a ConversationBuffer only empties the in-memory `Vec<Message>`. The persisted messages in `mem_conversations` remain. There is no method to delete conversation history from CozoDB through the ConversationBuffer API.

## 12. Graph Seed Count Is Clamped

In `search_graph()`, the number of seed nodes is `k.clamp(1, 5)` -- even if you request k=100 results, only 1-5 seeds are used for the graph traversal. This limits the graph search radius regardless of the requested result count.

## 13. from_stored_name Defaults Silently

All enum types default silently on unknown strings rather than erroring:
- `MemoryType::from_stored_name("garbage")` -> `MemoryType::Semantic`
- `MemoryRelation::from_stored_name("garbage")` -> `MemoryRelation::RelatesTo`
- `TriggerType::from_stored_name("garbage")` -> `TriggerType::TopicMention`
- `SearchMode::from_str_lossy("garbage")` -> `SearchMode::Hybrid`

This can mask bugs where an incorrect type string is passed through.

## 14. Timestamps Are Epoch Seconds with Millisecond Precision

CozoDB stores timestamps as `Float` (f64) representing epoch seconds: `dt.timestamp_millis() as f64 / 1000.0`. Conversion back uses `(epoch_secs * 1000.0).round() as i64` to recover milliseconds. This means sub-millisecond precision is lost in the round-trip.

## 15. Auto-Edge Creation Is Best-Effort

When `add_memory` stores a new memory, it searches for semantically similar existing memories and creates `RelatesTo` edges for those with score > 0.5. Edge creation errors are silently ignored (`let _ = self.store.put_edge(&edge)`). If edge creation fails, the memory is still stored successfully -- you just lose the graph connection.

## 16. FTS Query Sanitization Strips All Special Characters

`sanitize_fts_query()` replaces every non-alphanumeric, non-whitespace character with a space. This means queries containing hyphens, apostrophes, or special symbols (e.g., "don't", "state-of-the-art", "C++") will be tokenized differently from how the content was indexed. The FTS index uses the `Simple` tokenizer which may preserve some of these.

## 17. prepare_context Query Decomposition Is Simple

`decompose_query()` splits on " and ", " also ", " as well as ", plus periods and question marks. Sub-parts shorter than 6 characters are dropped. This is a heuristic that can mis-split complex queries or fail to split when conjunctions are phrased differently ("what about X, also Y" works; "tell me X along with Y" does not).

## 18. add_memories Confidence Default Differs from add_memory

- `add_memory`: confidence defaults to 0.9 when not provided
- `add_memories` (LLM extraction): confidence defaults to 0.8 per extracted item when the LLM omits it

This means LLM-extracted memories start with lower confidence than manually added ones.

## 19. word_overlap Uses Minimum Set Size

The `word_overlap()` function divides intersection count by the minimum of the two sets' sizes (not the union). This makes it asymmetric in a sense: a short phrase will show high overlap with a long text that contains all its words. This is intentional for dedup (a short update to a long existing memory should be detected as overlapping).

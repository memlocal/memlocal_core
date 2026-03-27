# Architecture

## 12-Type Memory Taxonomy

Defined in `models/memory_type.rs` as `enum MemoryType` with categories from `models/memory_category.rs`:

### Sensory
| Type | Stored Name | TTL | Description |
|---|---|---|---|
| SensoryBuffer | `sensory_buffer` | 5000ms | Ultra-short buffer for raw perception before interpretation |

### Short-Term
| Type | Stored Name | TTL | Description |
|---|---|---|---|
| WorkingMemory | `working_memory` | 300,000ms (5 min) | Temporary scratchpad for calculations and partial plans |
| AttentionContext | `attention_context` | 60,000ms (1 min) | What the model is actively reasoning about right now |
| ConversationBuffer | `conversation_buffer` | 300,000ms (5 min) | Recent messages needed to stay coherent |

### Long-Term (all persistent, no TTL)
| Type | Stored Name | Description |
|---|---|---|
| Episodic | `episodic` | Past conversations, completed tasks, user journeys |
| Semantic | `semantic` | General knowledge facts and relationships |
| Factual | `factual` | User preferences, account details, constraints |
| Procedural | `procedural` | Agent skills library, tool-use policies, SOPs |
| Social | `social` | Contact graph, team relationships, interaction history |
| Spatial | `spatial` | Location history, route memory, spatial relationships |
| Prospective | `prospective` | Reminders, follow-ups, scheduled actions |
| Affective | `affective` | User sentiment, tone preferences, emotional salience |

## Ingestion Pipeline

### Single Memory (add_memory tool)

1. **Content arrives** via `ToolExecutor::add_memory()` with content, optional type (defaults to "factual"), optional user_id, optional confidence (defaults to 0.9).
2. **Exact dedup** -- SHA-256 hash of content checked via `store.find_by_hash()`. Returns `"duplicate"` status if match found.
3. **Embedding generation** -- `EmbeddingProvider::embed_one(content)` called (platform-layer trait).
4. **Semantic dedup** -- `store.search_semantic(&embedding, 3, user_id, None)` finds top-3 similar. If best match score > 0.85 (near-duplicate) or > 0.70 with conflicting numeric specifics (`has_conflicting_specifics()`), the existing memory is updated in-place rather than creating a new one.
5. **Store as new** -- `MemoryItem` created with `Uuid::new_v4()`, stored via `store.put_memory()` which uses `vld: 'ASSERT'` for CozoDB time-travel.
6. **Auto-link** -- Post-store, searches top-3 similar memories again. Any with score > 0.5 get a `RelatesTo` edge added (best-effort, errors ignored).

### Batch Extraction (add_memories tool)

1. **Raw text** submitted to the LLM via `LlmProvider::complete()` with the `EXTRACTION_SYSTEM` prompt.
2. **TemporalContext** provided so the LLM can resolve relative dates ("tomorrow", "next Saturday") to absolute UTC timestamps.
3. **JSON array parsed** -- each item has content, type, confidence, optional valid_at/invalid_at.
4. **Confidence filter** -- items with confidence < 0.5 are skipped.
5. **Source annotation** -- if input text <= 500 chars, appended as `[Source: ...]` for FTS recall.
6. **Delegated to add_memory** -- each extracted item goes through the full single-memory pipeline (hash dedup, semantic dedup, store, auto-link).
7. **Result** -- returns counts of extracted, stored, updated, and skipped.

## Search Pipeline

All search methods are in `storage/memory_store.rs`.

### Search Methods

The `SearchMode` enum has 4 variants: Semantic, Text, Graph, Hybrid. LSH is a search capability used internally by hybrid search, not a standalone mode.

- **Semantic** -- HNSW vector search via `~mem_items:mem_vec_idx`. Distance = cosine. Score = `(1 - distance) * sqrt(confidence)`. ef = k * 2.
- **Text** -- BM25 FTS via `~mem_items:mem_fts_idx`. Query is sanitized (non-alphanumeric chars replaced with spaces). Score = raw BM25 score.
- **LSH** -- Jaccard similarity via `~mem_items:mem_lsh_idx`. Used in hybrid search. Silently returns empty on failure (short queries may fail).
- **Graph** -- Seeds from semantic search (top min(k, 5) results), then BFS 2-hop traversal over `mem_edges`. Depth-1 neighbors get `seed_score * edge_weight`. Depth-2 neighbors get `hop1_score * edge_weight * 0.5`. Invalid nodes filtered out.
- **Hybrid** -- Runs semantic + text + LSH in sequence, merges with RRF. See scoring.md for formula details.

### Unified Dispatch

`MemoryStore::search()` dispatches to the correct mode. Falls back to text search when embedding is None for Graph/Hybrid modes.

### search_hybrid_deduped (used by prepare_context)

Fetches `k + 15` results from hybrid search, then clusters by word overlap. Two results with >60% word overlap are considered about the same topic -- the older one is dropped, keeping only the most recent version. Truncates to k.

## Context Assembly

Defined in `shortterm/working_memory.rs` -- `WorkingMemory::to_context_block()`.

### Assembly Order (tiered priority)

1. **Triggered Reminders** (highest) -- `=== Triggered Reminders ===`, each prefixed with `!`, showing trigger type and condition.
2. **User Profile** -- `=== User Profile ===`, rendered via `UserProfile::to_summary()` showing static facts and dynamic context.
3. **Important Memories** -- `=== Important Memories ===`, deduplicated against the relevant set (by ID). Shows `[TypeName] content`.
4. **Relevant Memories** -- `=== Relevant Memories ===`, grouped by type name in a BTreeMap (alphabetical). Each item shows `[age] content (relevance: score)` where age is human-readable ("today", "3 days ago", "2 weeks ago", etc.).
5. **Focused Items** (lowest) -- `=== Focused Items ===`, showing attention context items.

### prepare_context (ToolExecutor)

The main pre-LLM-call optimization. Instead of Claude making tool calls:
1. **Query decomposition** -- splits on conjunctions ("and", "also", "as well as"), periods, and question marks. Sub-queries must be >5 chars.
2. **Per-sub-query search** -- each sub-query gets embedded and run through `search_hybrid_deduped(k=15)`, deduped by ID across sub-queries.
3. **Sort and truncate** -- all results sorted by score descending, top 20 kept.
4. **Profile fetch** -- `store.get_profile(user_id)`.
5. **Pending reminders** -- `store.get_pending_prospective(user_id)`.
6. **Important memories** -- `store.get_important_memories(user_id, 5, 0.6)` (top 5 with importance >= 0.6).
7. **WorkingMemory assembled** and `to_context_block()` returned as a string for system prompt injection.

## Consolidation

Defined in `consolidation/consolidator.rs`.

### Flow

1. Fetch up to 200 episodic memories for the user.
2. Filter to unconsolidated items older than `min_episodic_age_secs` (checked via `created_at < cutoff` and `metadata.consolidated != true`).
3. Match provided embeddings to eligible items by ID.
4. **Greedy clustering** -- iterate items in order; for each unvisited item, find all unvisited items with cosine similarity >= `CLUSTER_THRESHOLD` (0.65) to the seed. Mark matched items as visited.
5. Discard clusters smaller than `min_cluster_size`.
6. Return clusters as `Vec<Vec<MemoryItem>>`. **LLM summarization is NOT done in Rust** -- the platform layer implements the `Summarizer` trait and calls the LLM to produce a semantic summary.

### Key Constants

- `CLUSTER_THRESHOLD = 0.65` (cosine similarity)
- Max episodic memories scanned: 200

# Storage (CozoDB)

## Relations

All relations are defined in `storage/schema.rs` via `MemorySchema::create_statements()`. Prefixed with `mem_` to avoid namespace collisions. Uses `:create` with idempotent error handling (already-exists errors are silently ignored via `try_run`).

### mem_items (Core Memory)

Uses CozoDB Validity for native time-travel.

| Column | Type | Key? | Default | Purpose |
|---|---|---|---|---|
| `id` | String | **Key** | -- | UUID v4 |
| `vld` | Validity | **Key** | -- | Time-travel version marker (ASSERT/RETRACT) |
| `content` | String | Value | -- | Memory text content |
| `type` | String | Value | -- | Memory type stored name (e.g., "factual", "episodic") |
| `hash` | String | Value | -- | SHA-256 of content for dedup |
| `user_id` | String | Value | `''` | Owner user ID |
| `agent_id` | String | Value | `''` | Agent that created the memory |
| `session_id` | String | Value | `''` | Session context |
| `metadata_json` | String | Value | `'{}'` | JSON-encoded metadata (confidence, access_count, consolidated, etc.) |
| `embedding` | `<F32; N>` | Value | -- | Vector embedding (N = embedding_dimensions, default 1536) |
| `created_at` | Float | Value | -- | Epoch seconds (millisecond precision) |
| `updated_at` | Float | Value | -- | Epoch seconds |
| `event_start` | Float | Value | `0.0` | When the described event begins (maps from MemoryItem.valid_at) |
| `event_end` | Float | Value | `0.0` | When the described event ends (maps from MemoryItem.invalid_at) |

### mem_edges (Knowledge Graph)

| Column | Type | Key? | Default | Purpose |
|---|---|---|---|---|
| `from_id` | String | **Key** | -- | Source memory ID |
| `to_id` | String | **Key** | -- | Target memory ID |
| `relation` | String | **Key** | -- | Relation type stored name |
| `weight` | Float | Value | `1.0` | Edge strength 0.0-1.0 |
| `created_at` | Float | Value | -- | Epoch seconds |

### mem_conversations (Messages)

| Column | Type | Key? | Default | Purpose |
|---|---|---|---|---|
| `session_id` | String | **Key** | -- | Conversation session ID |
| `seq` | Int | **Key** | -- | Sequence number within session |
| `role` | String | Value | -- | "user", "assistant", or "system" |
| `content` | String | Value | -- | Message text |
| `timestamp` | Float | Value | -- | Epoch seconds |
| `metadata_json` | String | Value | `'{}'` | JSON metadata |

### mem_profiles (User Profiles)

| Column | Type | Key? | Default | Purpose |
|---|---|---|---|---|
| `user_id` | String | **Key** | -- | User identifier |
| `static_facts_json` | String | Value | `'{}'` | JSON-encoded BTreeMap of long-lived facts |
| `dynamic_context_json` | String | Value | `'{}'` | JSON-encoded BTreeMap of changing context |
| `updated_at` | Float | Value | -- | Epoch seconds |

### mem_prospective (Reminders/Triggers)

| Column | Type | Key? | Default | Purpose |
|---|---|---|---|---|
| `id` | String | **Key** | -- | UUID v4 |
| `content` | String | Value | -- | What to remember |
| `trigger_type` | String | Value | -- | "topic_mention", "time_based", "user_presence", "semantic_match" |
| `trigger_condition` | String | Value | -- | Topic keyword, ISO datetime, user ID, or semantic query |
| `user_id` | String | Value | `''` | Owner |
| `completed` | Int | Value | `0` | 0 = pending, 1 = completed |
| `created_at` | Float | Value | -- | Epoch seconds |
| `completed_at` | Float | Value | `0.0` | Epoch seconds when completed |

## Indices

### HNSW Vector Index (mem_vec_idx)

```cozoscript
::hnsw create mem_items:mem_vec_idx {
    dim: <embedding_dimensions>,  -- default 1536
    m: <hnsw_m>,                  -- default 16
    dtype: F32,
    fields: [embedding],
    distance: Cosine,
    ef_construction: <hnsw_ef_construction>  -- default 100
}
```

Used by `search_semantic()`. Query ef = k * 2. Supports optional filter expressions on `user_id` and `type`.

### FTS Index (mem_fts_idx)

```cozoscript
::fts create mem_items:mem_fts_idx {
    extractor: content,
    tokenizer: Simple,
    filters: [Lowercase, AlphaNumOnly, Stemmer('english')]
}
```

Uses English stemmer for better recall ("running" matches "run"). Query is sanitized: non-alphanumeric chars replaced with spaces, whitespace collapsed.

### LSH Index (mem_lsh_idx)

```cozoscript
::lsh create mem_items:mem_lsh_idx {
    extractor: content,
    tokenizer: Simple,
    filters: [Lowercase],
    n_perm: 200,
    target_threshold: 0.7,
    n_gram: 3
}
```

Near-duplicate detection via Jaccard similarity on 3-grams. Used as a third signal in hybrid search. No stemmer (unlike FTS) -- uses raw lowercase tokens.

## Validity / Time-Travel

CozoDB's native Validity system is used on `mem_items`:

- **Insert/Update:** `vld = 'ASSERT'` -- creates a new version visible at all future time points.
- **Soft-delete (invalidate):** `vld = 'RETRACT'` -- makes the item invisible at `"NOW"` but preserves history.
- **Queries:** All reads use `@ "NOW"` to see only current versions. Example: `*mem_items{id, content, type, ..., @ "NOW"}`.
- **Hard delete:** `delete_memory()` calls `invalidate_memory()` -- there is no true hard delete, only RETRACT.

The `event_start` and `event_end` columns are separate from Validity -- they track when the described real-world event occurs, not when the database row was created/retracted.

## Common Datalog Patterns

### Get memory by ID (current version)

```datalog
?[id, content, type, hash, ...] :=
    *mem_items{id, content, type, hash, ..., @ "NOW"}, id == $id
```

### List memories with filters

```datalog
?[id, content, type, ...] :=
    *mem_items{id, content, type, ..., @ "NOW"},
    user_id == $uid, type == $mtype
:order -updated_at
:limit 20
```

### HNSW vector search

```datalog
?[id, content, ..., distance] :=
    ~mem_items:mem_vec_idx{id, content, ... |
        query: vec($q), k: 10, ef: 20, bind_distance: distance,
        filter: user_id == "some_user"}
```

### BM25 full-text search

```datalog
?[id, content, ..., score] :=
    ~mem_items:mem_fts_idx{id, content, ... |
        query: $q, k: 10, bind_score: score}
```

### LSH near-duplicate search

```datalog
?[id, content, ..., score] :=
    ~mem_items:mem_lsh_idx{id, content, ... |
        query: $q, k: 10, bind_score: score}
```

### Graph traversal (neighbors)

```datalog
?[to_id, weight] := *mem_edges{from_id, to_id, weight}, from_id == $id
```

### PageRank

```datalog
?[node, score] <~ PageRank(*mem_edges{from_id: from, to_id: to}, iters: 20)
```

### Community Detection (Louvain)

```datalog
?[node, community] <~ CommunityDetectionLouvain(*mem_edges{from_id: from, to_id: to})
```

### Shortest Path (BFS)

```datalog
?[path] <~ ShortestPathBFS(*mem_edges{from_id: from, to_id: to},
    starting: [$from], goals: [$to])
```

### RETRACT a memory (soft-delete)

```datalog
?[id, vld, content, type, ...] :=
    *mem_items{id, content, type, ..., @ "NOW"},
    id == $id, vld = 'RETRACT'
:put mem_items {id, vld => content, type, ...}
```

### Count memories

```datalog
?[count(id)] := *mem_items{id, type, @ "NOW"}, type == $mtype
```

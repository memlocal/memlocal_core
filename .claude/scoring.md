# Scoring

All scoring logic lives in `storage/memory_store.rs`.

## Time-Decay Factor

**Method:** `MemoryStore::time_decay_factor(&self, item: &MemoryItem) -> f64`

### Formula

```
decay = exp(-lambda * days_since_update)
```

Where `days_since_update = (Utc::now() - item.updated_at).num_days() as f64`.

### Lambda Values by Type

| Memory Type | Lambda | Approximate Half-Life |
|---|---|---|
| Factual, Semantic, Procedural | 0.002 | ~346 days |
| Social | 0.003 | ~231 days |
| Episodic, Spatial, Affective | 0.005 | ~138 days |
| Prospective | 0.02 | ~35 days |
| All others (short-term types) | 0.005 | ~138 days |

### Gating

Time decay is only applied when `self.enable_time_decay` is true (set from `StorageConfig::enable_time_decay`, default: `true`). When disabled, returns 1.0 (no decay).

## Confidence Multiplier

**Method:** `MemoryStore::apply_confidence(item: &MemoryItem, raw_score: f64) -> f64`

### Formula

```
adjusted_score = raw_score * sqrt(confidence)
```

Where `confidence = item.metadata["confidence"]` (f64), defaulting to 1.0 if absent.

The sqrt dampening means:
- confidence 1.0 -> multiplier 1.0
- confidence 0.81 -> multiplier 0.9
- confidence 0.5 -> multiplier ~0.707
- confidence 0.25 -> multiplier 0.5

## Importance Score

**Method:** `MemoryStore::compute_importance(&self, item: &MemoryItem) -> f64`

### Formula

```
importance = 0.3 * confidence + 0.3 * access_factor + 0.4 * recency_factor
```

Where:
- `confidence = item.metadata["confidence"]` (default 0.8 if absent)
- `access_count = item.metadata["access_count"]` (default 0)
- `access_factor = ln(1 + access_count) / ln(1 + 100)` (log-scaled, normalized against 100 accesses)
- `days_since = (Utc::now() - item.updated_at).num_days() as f64`
- `recency_factor = exp(-0.0115 * days_since)` (half-life ~60 days, fixed lambda regardless of type)

### Coefficients

| Component | Weight | Source |
|---|---|---|
| Confidence | 0.3 | From metadata, default 0.8 |
| Access frequency | 0.3 | Log-scaled access_count |
| Recency | 0.4 | Fixed lambda 0.0115 |

### Usage

Called by `get_important_memories(user_id, limit, min_importance)`:
1. Fetches up to 500 memories for the user
2. Computes importance for each, sets as score
3. Filters by `min_importance` threshold
4. Sorts descending, truncates to `limit`

In `prepare_context()`, called as `get_important_memories(user_id, 5, 0.6)`.

## Reciprocal Rank Fusion (RRF)

**Used in:** `MemoryStore::search_hybrid()`

### Formula

```
RRF_score(item) = SUM over lists: 1 / (K + rank + 1)
```

Where:
- **K = 60.0** (constant, hardcoded as `K_RRF`)
- **rank** = 0-indexed position in each result list
- Lists: semantic results, text (BM25) results, LSH results

### Post-RRF Adjustments

After RRF scoring, each item's score is further adjusted:

```
final_score = apply_confidence(item, rrf_score * time_decay_factor(item))
            = rrf_score * time_decay_factor * sqrt(confidence)
```

Results are sorted by final_score descending, top k returned.

### Score Ranges

With 3 lists, the theoretical maximum RRF score for an item at rank 0 in all three lists:
```
3 * (1 / (60 + 0 + 1)) = 3 / 61 = ~0.0492
```

## Hybrid Dedup (search_hybrid_deduped)

**Used by:** `ToolExecutor::prepare_context()`

### Parameters

- Fetches `k + 15` results from `search_hybrid()` (extra headroom for dedup)
- **Word overlap threshold: 0.60** (>60% shared words = same topic)
- **Selection strategy: keep newest** -- when two items overlap, the one with the more recent `updated_at` survives
- **Overlap metric:** Jaccard-like ratio: `|intersection| / min(|words_a|, |words_b|)`. Words are split by whitespace, trimmed of non-alphanumeric chars, filtered to length > 2.

### Algorithm

For each item in score-ordered results:
1. Check if any kept item has >60% overlap AND has `updated_at >= item.updated_at` (i.e., the kept item is newer). If so, the new item is "dominated" and skipped.
2. If not dominated, remove any kept items that this new item dominates (>60% overlap AND this item is newer).
3. Add the new item to the kept list.
4. Truncate to k.

## Semantic Search Scoring

In `search_semantic()`:
```
raw_score = 1.0 - cosine_distance
adjusted_score = raw_score * sqrt(confidence)
```

The HNSW index returns cosine distance (0 = identical, 2 = opposite). Score is converted to similarity.

## Text Search Scoring

In `search_text()`:
```
score = raw BM25 score (from CozoDB FTS index)
```

No confidence or time-decay adjustment is applied to pure text search results. These adjustments only happen in hybrid search where text results are merged via RRF.

## LSH Search Scoring

In `search_lsh()`:
```
score = raw Jaccard similarity score (from CozoDB LSH index)
```

Same as text -- no post-processing. Only adjusted when merged in hybrid search.

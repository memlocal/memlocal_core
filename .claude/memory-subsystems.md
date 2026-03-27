# Memory Subsystems

## Sensory Buffer

**File:** `shortterm/sensory_buffer.rs`

In-memory ultra-short buffer for raw perception. NOT persisted to CozoDB.

### Data Structure

- `VecDeque<TimedItem>` where `TimedItem = { message: Message, added_at: Instant }`
- Bounded by capacity (default: 100, from `CoreConfig::sensory_buffer_capacity`)
- TTL eviction (default: 5000ms, from `CoreConfig::sensory_ttl_ms`)

### Behavior

- **add():** Evicts expired items first, then pops front if at capacity, then pushes new item to back.
- **items():** Evicts expired first, returns references to all remaining messages.
- **recent(n):** Returns last n non-expired items in chronological order.
- **clear():** Drops all items.
- Eviction is lazy -- only happens on add/items/recent/len/is_empty calls.
- Uses `std::time::Instant` (monotonic clock), not wall-clock time.

### Thread Safety

Wrapped in `Mutex<SensoryBuffer>` on `MemlocalEngine`.

## Conversation Buffer

**File:** `shortterm/conversation_buffer.rs`

CozoDB-backed sliding window over conversation messages. Persists to `mem_conversations` relation.

### Configuration

- `max_messages`: sliding window size (default: 20, from `CoreConfig::conversation_buffer_size`)
- One buffer per session, stored in `Mutex<HashMap<String, ConversationBuffer>>` on the engine.

### Behavior

- **Lazy loading:** `load()` called on first `append()`. Reads from CozoDB via `store.get_messages(session_id, Some(max_messages))`. Also reads `message_count()` to set sequence counter.
- **append(message):** Loads if needed, increments `seq`, sets `session_id` on the message, calls `store.put_message(&msg, seq)`, pushes to in-memory vec, trims excess from front if over window size.
- **messages():** Returns the in-memory slice (may be stale if not loaded).
- **recent(n):** Returns the last n messages from the in-memory vec.
- **clear():** Clears in-memory buffer only. Does NOT delete from CozoDB -- persisted messages remain.

### Persistence

Messages are stored in CozoDB's `mem_conversations` relation with `(session_id, seq)` as composite key. Ordered by `seq` on retrieval.

## Working Memory

**File:** `shortterm/working_memory.rs`

Transient context scratchpad assembled per LLM call. NEVER persisted.

### Fields

| Field | Type | Purpose |
|---|---|---|
| `relevant_memories` | `Vec<MemoryItem>` | Search results relevant to current query |
| `important_memories` | `Vec<MemoryItem>` | High-importance memories regardless of query |
| `triggered_reminders` | `Vec<ProspectiveItem>` | Prospective items whose triggers fired |
| `user_profile` | `Option<UserProfile>` | Loaded user profile |
| `attention_items` | `Vec<MemoryItem>` | Manually focused items (via focus/unfocus) |

### to_context_block() Assembly Order

Returns a formatted string for system prompt injection:

1. `=== Triggered Reminders ===` -- each as `! content\n  (trigger: type -- condition)`
2. `=== User Profile ===` -- static facts and dynamic context as bullet lists
3. `=== Important Memories ===` -- deduplicated against relevant_memories by ID, shown as `[TypeName] content`
4. `=== Relevant Memories ===` -- grouped by type (BTreeMap, alphabetical), each as `[age] content (relevance: score)` where age is human-readable
5. `=== Focused Items ===` -- attention context as `[TypeName] content`

Sections are only emitted if non-empty. Output is trimmed at the end.

### has_context()

Returns true if any of: relevant_memories, important_memories, triggered_reminders, non-empty profile, or attention_items is present.

### Lifecycle

Created via `WorkingMemory::new()`, populated by setters, consumed via `to_context_block()`, then `clear()` resets all fields.

## Long-Term: Episodic Memory

**File:** `longterm/episodic.rs`

Past conversations, completed tasks, user journeys with temporal context.

### API

- `record(item, embedding)` -- stores via `store.put_memory()`
- `get_recent(user_id, limit)` -- `store.get_memories(user_id, Some(Episodic), limit)` ordered by `-updated_at`
- `search(query_embedding, k, user_id)` -- `store.search_semantic()` filtered to Episodic type

### Time Decay

Lambda = 0.005 (half-life ~138 days). Episodic memories age moderately -- old events become less relevant.

## Long-Term: Semantic Memory

**File:** `longterm/semantic.rs`

General knowledge facts and relationships. Also the target type for consolidation summaries.

### API

- `record(item, embedding)` -- stores via `store.put_memory()`
- `get_facts(user_id, limit)` -- `store.get_memories()` filtered to Semantic type
- `search(query_embedding, k, user_id)` -- semantic search filtered to Semantic type

### Time Decay

Lambda = 0.002 (half-life ~346 days). Facts persist longest.

## Long-Term: Factual Memory

**File:** `longterm/factual.rs`

Stable personal facts: user preferences, account details, constraints. Display name: "Factual / Profile".

### API

- `record(item, embedding)` -- stores via `store.put_memory()`
- `get_user_facts(user_id, limit)` -- filtered to Factual type
- `search(query_embedding, k, user_id)` -- semantic search filtered to Factual type

### Time Decay

Lambda = 0.002 (half-life ~346 days). Same as semantic -- personal facts are long-lived.

## Long-Term: Procedural Memory

**File:** `longterm/procedural.rs`

Skills, workflows, routines, how-to knowledge, SOPs.

### API

- `record(item, embedding)` -- stores via `store.put_memory()`
- `get_procedures(user_id, limit)` -- filtered to Procedural type
- `search(query_embedding, k, user_id)` -- semantic search filtered to Procedural type

### Time Decay

Lambda = 0.002 (half-life ~346 days). Skills persist.

## Long-Term: Social Memory

**File:** `longterm/social.rs`

Contact graph, team relationships, interaction patterns. The only subsystem that directly uses graph algorithms.

### API

- `record(item, embedding)` -- stores memory item
- `record_relationship(edge)` -- stores an edge in the knowledge graph
- `get_contacts(user_id, limit)` -- filtered to Social type
- `get_relationships(memory_id)` -- fetches both incoming and outgoing edges
- `detect_communities()` -- runs CozoDB Louvain community detection on `mem_edges`
- `page_rank(iterations)` -- runs CozoDB PageRank on `mem_edges`
- `shortest_path(from_id, to_id)` -- BFS shortest path between two memory nodes

### Time Decay

Lambda = 0.003 (half-life ~231 days). Slightly faster decay than facts, slower than events.

## Long-Term: Spatial Memory

**File:** `longterm/spatial.rs`

Location-tagged memories and spatial relationships.

### API

- `record(item, embedding)` -- stores via `store.put_memory()`
- `get_all(user_id, limit)` -- filtered to Spatial type
- `search(query_embedding, k, user_id)` -- semantic search filtered to Spatial type

### Time Decay

Lambda = 0.005 (half-life ~138 days). Same as episodic.

## Long-Term: Prospective Memory

**File:** `longterm/prospective.rs`

Future intentions, reminders, planned actions. Uses both `mem_items` (for searchable memory) and `mem_prospective` (for trigger logic).

### API

- `record(item, embedding)` -- stores the memory item (searchable)
- `add_trigger(item)` -- stores a ProspectiveItem in `mem_prospective`
- `get_pending(user_id)` -- fetches uncompleted prospective items (completed == 0)
- `complete_trigger(trigger_id)` -- marks as completed, sets completed_at
- `get_all(user_id, limit)` -- filtered to Prospective type
- `check_triggers(query, query_embedding, user_id)` -- evaluates which triggers fire

### Trigger Types and Evaluation

| TriggerType | Evaluation Logic |
|---|---|
| `TopicMention` | Case-insensitive substring match: `query.to_lowercase().contains(condition.to_lowercase())` |
| `TimeBased` | Parses condition as RFC 3339 datetime, fires if `now >= trigger_time` |
| `UserPresence` | Exact match: `user_id == trigger_condition` |
| `SemanticMatch` | **Falls back to keyword matching** (same as TopicMention). The `_query_embedding` param is unused. Comment says "platform layer can do proper semantic comparison." |

### Constants

- `_SEMANTIC_TRIGGER_THRESHOLD = 0.75` (defined but unused -- prefixed with underscore)

### Time Decay

Lambda = 0.02 (half-life ~35 days). Reminders expire fastest.

## Long-Term: Affective Memory

**File:** `longterm/affective.rs`

Emotional context, sentiment, mood tracking.

### API

- `record(item, embedding)` -- stores via `store.put_memory()`
- `get_sentiment_history(user_id, limit)` -- filtered to Affective type
- `search(query_embedding, k, user_id)` -- semantic search filtered to Affective type

### Time Decay

Lambda = 0.005 (half-life ~138 days). Same as episodic.

## Consolidation

**File:** `consolidation/consolidator.rs`

Converts clusters of related episodic memories into higher-level semantic summaries.

### Algorithm

1. Fetch up to 200 episodic memories for user
2. Filter: `created_at < (now - min_episodic_age_secs)` AND `metadata.consolidated != true`
3. Match provided embeddings by memory ID
4. Greedy clustering: iterate items; for each unvisited seed, collect all unvisited items with cosine similarity >= 0.65 to that seed
5. Discard clusters smaller than `min_cluster_size`
6. Return `Vec<Vec<MemoryItem>>` -- the platform layer handles LLM summarization

### Constants

- `CLUSTER_THRESHOLD = 0.65` (cosine similarity)
- Max items scanned: 200

### Cosine Similarity

Implemented inline in `MemoryConsolidator::cosine_similarity()` -- standard dot product / (norm_a * norm_b), returns 0.0 on dimension mismatch or empty vectors.

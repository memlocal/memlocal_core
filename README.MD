<p align="center">
  <h1 align="center">memlocal_core</h1>
  <p align="center">
    <strong>Local-first cognitive memory engine for AI agents</strong>
  </p>
  <p align="center">
    <a href="https://memlocal.dev">Website</a> · <a href="https://github.com/memlocal/memlocal_core">GitHub</a> · <a href="https://github.com/memlocal/memlocal_dart">Flutter SDK</a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/language-Rust-orange" alt="Rust" />
        <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="Apache 2.0 License" />
    <img src="https://img.shields.io/badge/database-CozoDB-green" alt="CozoDB" />
    <img src="https://img.shields.io/badge/LoCoMo-80.0%25_pass-brightgreen" alt="LoCoMo Benchmark" />
  </p>
</p>

---

**memlocal** is an open-source, local-first memory layer that gives AI agents persistent, structured recall — designed to run entirely on a smartphone. It draws from cognitive psychology's multi-store model to organise memories into eight types, stores them in an embedded CozoDB database (graph + vector + full-text in one process), and retrieves them through a multi-channel pipeline with cross-encoder reranking.

On the [LoCoMo benchmark](https://snap-research.github.io/locomo/), memlocal achieves an **80.0% pass rate** and **4.21/5 average LLM score** — outperforming [Letta](https://github.com/letta-ai/letta) (74.0%), [Mem0](https://mem0.ai/research) (66.9%), [PAM](https://manager.harmix.ai/blog/pam-locomo-benchmark-results) (74.35%), and [OpenAI ChatGPT Memory](https://openai.com/index/memory-and-new-controls-for-chatgpt/) (52.9%), while keeping all data on-device.

```
LoCoMo → 80.0% pass rate
```

---

## Table of Contents

- [Why memlocal](#why-memlocal)
- [Architecture](#architecture)
- [Memory Model](#memory-model)
- [Storage Layer — CozoDB](#storage-layer--cozodb)
- [Ingestion Pipeline](#ingestion-pipeline)
- [Retrieval Pipeline](#retrieval-pipeline)
- [Scoring Functions](#scoring-functions)
- [Benchmark Results](#benchmark-results)
- [Comparison with Other Systems](#comparison-with-other-systems)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Provider Traits](#provider-traits)
- [Cloud APIs and Local Alternatives](#cloud-apis-and-local-alternatives)
- [Platform SDKs](#platform-sdks)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Why memlocal

LLMs have no persistent memory. When a conversation ends, everything is lost. Existing solutions — Mem0, Letta, Zep — address this, but they all require cloud infrastructure, server-side vector databases, or external graph stores. They can't run on a phone.

memlocal is built around three constraints:

1. **Privacy** — Personal memories should never leave the device. No cloud sync, no server-side storage, no third-party data retention.
2. **Latency** — On-device retrieval eliminates network round-trips. Context is assembled in milliseconds, not seconds.
3. **Offline** — The system works without an internet connection. The database, vector indices, and knowledge graph all run in-process.

The core engine is written in Rust and compiles to a ~4MB shared library. A typical memory store for 1,000 memories with 1536-dimensional embeddings occupies ~25MB on disk. The entire system — database, HNSW index, BM25 index, knowledge graph — runs inside a single embedded process.

---

## Architecture

memlocal implements a three-layer architecture mirroring the Atkinson-Shiffrin multi-store model of human memory:

```
┌─────────────────────────────────────────────────────────────────┐
│                        SENSORY BUFFER                          │
│  Ring buffer • TTL: 5s • Capacity: 100 items • Noise filter    │
└──────────────────────────┬──────────────────────────────────────┘
                           │ LLM extraction
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       WORKING MEMORY                           │
│  Context assembler • Flat (SingleHop) / Sectioned (MultiHop)   │
│  KEY FACTS → TOP EVIDENCE → RAW EXCERPTS → SESSION CONTEXT     │
└──────────────────────────┬──────────────────────────────────────┘
                           │ store / retrieve
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LONG-TERM MEMORY                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Episodic │  │ Factual  │  │ Semantic │  │Procedural│       │
│  ├──────────┤  ├──────────┤  ├──────────┤  ├──────────┤       │
│  │  Social  │  │ Spatial  │  │Prospective│ │Affective │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│                                                                 │
│  CozoDB: HNSW vectors + BM25 full-text + Datalog graph          │
│  + LSH dedup + Triples FTS + Session summaries                  │
└─────────────────────────────────────────────────────────────────┘
```

**How a query is answered:**

1. The raw query enters the retrieval pipeline.
2. A **query classifier** determines complexity (SingleHop / MultiHop / Temporal / OpenEnded).
3. **Six retrieval channels** generate candidates in parallel: per-type semantic search, BM25 keyword matching, recursive Datalog graph traversal (2 hops), triple FTS, session-window expansion, and speaker-filtered search.
4. All candidates are pooled, deduplicated, and passed to a **cross-encoder reranker** (Jina Reranker v2).
5. The top-ranked items are assembled into a **context block** — flat for simple queries, sectioned for complex ones.
6. The context block is injected into the LLM's system prompt. Most questions are answered in a **single LLM call** with no tool-calling round-trips.

---

## Memory Model

Each extracted fact is classified into one of eight types inspired by Tulving's memory taxonomy:

| Type            | What it stores                                | Decay λ | Half-life |
| --------------- | --------------------------------------------- | ------- | --------- |
| **Episodic**    | Events, experiences, meetings, trips          | 0.005   | ~139 days |
| **Factual**     | Preferences, personal details, stable facts   | 0.002   | ~347 days |
| **Semantic**    | General knowledge, learned concepts           | 0.002   | ~347 days |
| **Procedural**  | Workflows, skills, routines, how-to knowledge | 0.002   | ~347 days |
| **Social**      | Relationships, people, team dynamics          | 0.003   | ~231 days |
| **Spatial**     | Locations, places, addresses                  | 0.005   | ~139 days |
| **Prospective** | Reminders, future intentions, follow-ups      | 0.020   | ~35 days  |
| **Affective**   | Emotions, moods, feelings, sentiment          | 0.005   | ~139 days |

Additionally, three short-term types handle transient state:

| Type                    | Purpose                       | TTL       |
| ----------------------- | ----------------------------- | --------- |
| **Sensory Buffer**      | Raw input before processing   | 5 seconds |
| **Working Memory**      | Current reasoning scratchpad  | 5 minutes |
| **Conversation Buffer** | Recent messages for coherence | 5 minutes |

Each memory item carries: content, type classification, speaker attribution, confidence score, semantic triple (subject-predicate-object), embedding vector (1536d), temporal validity markers (valid_at / invalid_at), and metadata (access count, reinforcement count).

---

## Storage Layer — CozoDB

### Why CozoDB

On a smartphone, you can't run Pinecone + Elasticsearch + Neo4j. CozoDB collapses all three into a single embedded library:

| Capability                | What memlocal uses it for                                 | Alternative would require       |
| ------------------------- | --------------------------------------------------------- | ------------------------------- |
| **HNSW vector indices**   | Semantic similarity search over 1536-d embeddings         | Pinecone, Qdrant, or Chroma     |
| **BM25 full-text search** | Keyword and entity retrieval for exact name/date matching | Elasticsearch or Tantivy        |
| **Recursive Datalog**     | Multi-hop graph traversal in a single query               | Neo4j or manual BFS             |
| **MinHash LSH**           | Near-duplicate detection during deduplication             | Custom implementation           |
| **Validity schema**       | Native time-travel queries (point-in-time memory state)   | Temporal tables or soft deletes |
| **SQLite backend**        | Mobile-friendly persistent storage                        | —                               |
| **C FFI**                 | Embedding in Rust → Flutter/Kotlin/Swift                  | —                               |

CozoDB is written in Rust, compiles for iOS/Android/desktop/WASM, and its SQLite backend requires no additional dependencies on mobile.

### Schema

Seven relations form the core schema:

| Relation            | Purpose                            | Key columns                                                                  |
| ------------------- | ---------------------------------- | ---------------------------------------------------------------------------- |
| `mem_items`         | Core memory items with time-travel | id, vld (Validity), content, type, embedding[1536], speaker, event_start/end |
| `mem_edges`         | Knowledge graph edges              | from_id, to_id, relation, weight                                             |
| `mem_conversations` | Raw conversation messages          | session_id, seq, role, content, timestamp                                    |
| `mem_profiles`      | User profiles                      | user_id, data fields                                                         |
| `mem_prospective`   | Reminders and future intentions    | id, trigger_type, trigger_condition                                          |
| `mem_triples`       | Subject-predicate-object facts     | memory_id, subject, predicate, object, speaker                               |
| `mem_summaries`     | Per-session narrative summaries    | session_id, summary, speakers, key_topics, embedding                         |

### Indices

Five indices are maintained automatically:

| Index               | Type                                          | Purpose                               |
| ------------------- | --------------------------------------------- | ------------------------------------- |
| `mem_vec_idx`       | HNSW (M=16, ef_construction=100, cosine)      | Semantic similarity search            |
| `mem_fts_idx`       | BM25                                          | Full-text keyword search over content |
| `mem_lsh_idx`       | MinHash (n_perm=200, n_gram=3, threshold=0.5) | Near-duplicate detection              |
| `mem_triples_fts`   | BM25                                          | Structured fact lookup by entity name |
| `mem_summaries_vec` | HNSW                                          | Session-level semantic search         |

---

## Ingestion Pipeline

```
Raw text (conversation / document)
        │
        ▼
┌──────────────────────────┐
│  LLM Extraction (v2)     │  Claude Haiku 4.5 / Gemma 3 4B
│  - Typed memories         │  - Proper noun preservation
│  - Observations           │  - Speaker attribution
│  - Semantic triples       │  - Temporal anchoring
│  - Session summary        │  - Contradiction detection
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Deduplication            │  LSH near-duplicate detection
│  - ADD / UPDATE / SKIP    │  + LLM dedup classification
│  - CONTRADICTION          │  Reinforcement counting
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Storage + Graph Build    │  CozoDB put_memory
│  - Embed (1536d)          │  - Auto-create edges (cos > 0.5)
│  - Store triples          │  - Store session summary
│  - Index (HNSW + BM25)    │
└──────────────────────────┘
```

The extraction prompt (v2) enforces critical constraints:

- **Proper noun preservation**: "Nothing is Impossible" must not become "a book"; "Matt Patterson" must not become "a performer"
- **Atomic memories**: One fact per item
- **Speaker identity**: Each fact attributed to the person who stated it
- **Observations**: Ultra-literal factual statements preserved exactly as spoken
- **Semantic triples**: Every memory includes a (subject, predicate, object) triple with named entities, never pronouns

---

## Retrieval Pipeline

### Step 1 — Query Classification

```rust
fn classify_retrieval_query(query: &str) -> RetrievalQueryComplexity {
    // SingleHop  → pool: 80,  context: 15  (default / simple factual)
    // MultiHop   → pool: 120, context: 18  (when/how long/both)
    // Temporal   → pool: 140, context: 24  (before/after/during/date)
    // OpenEnded  → pool: 100, context: 15  (would/could/should)
}
```

### Step 2 — Multi-Channel Candidate Generation

Six channels retrieve candidates in parallel for each query variant:

| Channel               | Method                                             | Budget                                                                                  |
| --------------------- | -------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **Per-type semantic** | HNSW search filtered by each of 8 memory types     | 15 episodic, 10 factual, 8 semantic, 5 social, 5 affective, 5 prospective, 3 procedural |
| **Hybrid catch-all**  | BM25 + semantic (untyped)                          | 10 per variant                                                                          |
| **Graph traversal**   | Recursive Datalog, 2-hop from top-5 semantic seeds | 5 seeds × 2 hops                                                                        |
| **Triple FTS**        | BM25 over subject+predicate+object fields          | 10 per entity                                                                           |
| **Session windows**   | ±2 adjacent turns from same session as any hit     | Expands from conversation hits                                                          |
| **Speaker-filtered**  | Semantic search scoped to detected speaker         | 10 results                                                                              |

All candidates are deduplicated by memory ID. Each candidate's provenance (which channels found it) is tracked for diagnostics.

### Step 3 — Cross-Encoder Reranking

The full candidate pool (80–140 items depending on query complexity) is scored by a cross-encoder:

```
Candidates (80-140) → Jina Reranker v2 → Relevance scores [0, 1] → Sort → Top-K
```

**Critical design decision**: The reranker sees the _entire_ pool, not just the top-K. This was the single highest-impact fix during development — when only 8 candidates were reranked (a pool-size bug), strong evidence at ranks 10–50 was silently discarded before the cross-encoder could evaluate it. Fixing this one line moved the pass rate from 66.3% to ~72%.

### Step 4 — Context Assembly

The top-ranked items are assembled into a context block whose format adapts to query complexity:

**SingleHop** → Flat ranked list. No headers, no type labels, no scores. ~1,000–1,500 characters. Prevents the LLM from over-including tangentially relevant information.

**MultiHop / Temporal / OpenEnded** → Sectioned context with:

- KEY FACTS (BM25 exact matches)
- KEY FACTS (structured triples, grouped by speaker)
- TOP EVIDENCE (reranked candidates)
- RAW CONVERSATION EXCERPTS
- SESSION CONTEXT (session summaries)
- Important Memories

### Step 5 — Iterative Retrieval (optional)

For complex queries, a two-round retrieval pattern runs:

1. Round 1: Standard retrieval
2. LLM evaluates sufficiency → generates 1–3 refined queries if gaps exist
3. Round 2: Retrieval with refined queries
4. Merge round 1 + round 2 candidates → final ranking

---

## Scoring Functions

### Exponential Time Decay

Each memory type decays at a rate λ calibrated to its expected persistence:

```
D(t) = e^(−λt)
```

where t is days since last update. Factual memories (λ=0.002, half-life ~347 days) persist far longer than prospective reminders (λ=0.020, half-life ~35 days).

### Confidence Dampening

The extraction LLM assigns each memory a confidence c ∈ [0, 1]. During retrieval, confidence is applied with square-root dampening:

```
S(s, c) = s · √c
```

This avoids over-penalising moderate-confidence items while still down-ranking speculative extractions.

### Reinforcement

When a memory is mentioned again (duplicate or update), its reinforcement count r is incremented. The reinforcement factor uses logarithmic normalisation:

```
R(r) = ln(1 + r) / ln(1 + r_max)
```

where r_max = 10. First few reinforcements matter most; further repetitions provide diminishing boosts.

### Composite Importance

```
I = 0.25·c + 0.25·R(r) + 0.20·A(a) + 0.30·D(t)
```

where c = confidence, R(r) = reinforcement, A(a) = ln(1+access_count)/ln(101), D(t) = time decay. Weights prioritise recency (30%), balanced with reliability and usage patterns.

---

## Benchmark Results

Evaluated on the **LoCoMo** (Long Conversation Memory) benchmark.

### Overall Results

| Metric                | Value              |
| --------------------- | ------------------ |
| **Pass rate (≥3/5)**  | **80.0%**          |
| **Average LLM score** | **4.21/5**         |
| Average keyword F1    | 0.146              |

### Per-Category Breakdown

| Category        | Avg LLM Score | Pass Rate |
| --------------- | ------------- | --------- |
| Single-hop      | 3.00          | 45.5%     |
| **Multi-hop**   | 4.65          | **96.2%** |
| Temporal        | 4.14          | 77.3%     |
| **Adversarial** | 4.42          | **83.3%** |

### Performance Trajectory

| Version | Key Changes                                                            | Pass Rate |
| ------- | ---------------------------------------------------------------------- | --------- |
| v1      | Baseline: BM25 + semantic search                                       | 55.8%     |
| v2      | + Speaker attribution, triples, session summaries, iterative retrieval | 62.8%     |
| v3      | + Jina reranker, query classification, adaptive context                | 66.3%     |
| **v4**  | **+ Pool fix, flat SingleHop context, session windows, extraction v2** | **80.0%** |

### Failure Patterns

| Pattern      | Description                                       |
| ------------ | ------------------------------------------------- |
| SPEAKER_SWAP | LLM attributes a fact to the wrong speaker        |
| WRONG_FACT   | LLM includes incorrect details from noisy context |
| INCOMPLETE   | Evidence ranked just outside the context window   |
| LLM_REFUSE   | LLM declines to answer despite evidence present   |

---

## Comparison with Other Systems

### memlocal vs. Letta (MemGPT v2)

| Dimension           | memlocal                              | Letta                                                                         |
| ------------------- | ------------------------------------- | ----------------------------------------------------------------------------- |
| **LoCoMo accuracy** | **80.0%**                             | **74.0%** ([source](https://www.letta.com/blog/benchmarking-ai-agent-memory)) |
| Architecture        | Embedded Rust library, single-process | REST API server, multi-service                                                |
| Database            | CozoDB (embedded, SQLite backend)     | PostgreSQL + pgvector (server-required)                                       |
| Deployment          | On-device (smartphone, desktop)       | Cloud / self-hosted server                                                    |
| Memory model        | 8 cognitive types with typed decay    | Flat archival + recall memory                                                 |
| Graph support       | Native Datalog (recursive, multi-hop) | None built-in                                                                 |
| Full-text search    | Native BM25                           | Via PostgreSQL FTS                                                            |
| Reranking           | Cross-encoder (Jina / FlashRank)      | None                                                                          |
| Privacy             | All data on-device                    | Data on server                                                                |
| Offline support     | Full                                  | Requires server connection                                                    |

Letta pioneered the virtual context management paradigm (paging information between a fixed context window and external storage), but its REST API architecture requires a running server with PostgreSQL. memlocal achieves a **6-point higher accuracy** while running as an embedded library with no server dependency.

### memlocal vs. Mem0

| Dimension           | memlocal                                   | Mem0                                            |
| ------------------- | ------------------------------------------ | ----------------------------------------------- |
| **LoCoMo accuracy** | **80.0%**                                  | **66.9%** ([source](https://mem0.ai/research))  |
| Architecture        | Embedded Rust library                      | Python server + multiple backends               |
| Database            | CozoDB (graph + vector + FTS, one process) | Neo4j + Qdrant + Elasticsearch (three services) |
| Deployment          | On-device                                  | Cloud / self-hosted (requires Neo4j, Qdrant)    |
| Memory types        | 8 cognitive types with typed decay         | Untyped facts + graph variant (Mem0ᵍ)           |
| Memory extraction   | LLM-driven with typed classification       | LLM-driven (untyped)                            |
| Graph support       | Native CozoDB Datalog                      | Neo4j (separate service)                        |
| Reranking           | Cross-encoder (Jina / FlashRank)           | None                                            |
| Time-travel         | Native CozoDB Validity schema              | Manual version tracking                         |
| Binary size         | ~4MB shared library                        | Python + dependencies                           |
| Privacy             | All data on-device                         | Data on server(s)                               |

Mem0 achieves 66.9% on LoCoMo (LLM-as-a-Judge accuracy), with their graph-enhanced variant Mem0ᵍ reaching 68.4%. memlocal outperforms both by **13+ percentage points** while running as a single embedded process — Mem0 requires three separate database services (Neo4j, Qdrant, Elasticsearch), making smartphone deployment impractical.

### memlocal vs. PAM (Harmix)

| Dimension           | memlocal                                             | PAM                                                                                |
| ------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **LoCoMo accuracy** | **80.0%**                                            | **74.35%** ([source](https://manager.harmix.ai/blog/pam-locomo-benchmark-results)) |
| Architecture        | Embedded Rust library                                | Cloud agent with file-first memory                                                 |
| Database            | CozoDB (embedded)                                    | Local files + LLM file tools                                                       |
| Deployment          | On-device                                            | Cloud (requires LLM API)                                                           |
| Retrieval           | Multi-channel (6 channels) + cross-encoder reranking | File-based (ls, grep, cat)                                                         |
| Multi-hop accuracy  | **96.2%**                                            | 80.1%                                                                              |

PAM uses a "File-First" architecture where the LLM navigates structured files using tools like `grep` and `cat`. While innovative, memlocal achieves **5.65 points higher overall accuracy** with a multi-channel retrieval pipeline and cross-encoder reranking. Notably, memlocal achieves **96.2% on multi-hop** vs PAM's 80.1%.

### Full LoCoMo Leaderboard

| Rank  | System         | LoCoMo Score | Local-first?          | Embedded? |
| ----- | -------------- | ------------ | --------------------- | --------- |
| 1     | SmartSearch    | 93.5%        | No (CPU server)       | No        |
| 2     | Mem-U          | 92.09%       | No (cloud)            | No        |
| 3     | MemoryMachine  | 84.87%       | No (cloud LLM loop)   | No        |
| 4     | Engram         | 80.0%        | No (Node.js + SQLite) | No        |
| **4** | **memlocal**   | **80.0%**    | **Yes**               | **Yes**   |
| 6     | Memobase       | 75.8%        | No (cloud)            | No        |
| 7     | PAM            | 74.35%       | No (cloud)            | No        |
| 8     | Letta          | 74.0%        | No (REST server)      | No        |
| 9     | Mem0ᵍ (graph)  | 68.4%        | No (3 services)       | No        |
| 10    | Mem0           | 66.9%        | No (3 services)       | No        |
| 11    | RAG baseline   | 61.0%        | —                     | —         |
| 12    | Zep            | 58.4%        | No (Neo4j + Docker)   | No        |
| 13    | OpenAI ChatGPT | 52.9%        | No (cloud)            | No        |

> **Note**: Scores are LLM-as-a-Judge accuracy from published results. Mem0/Memobase/Zep scores from [Mem0 research](https://mem0.ai/research) and [Memobase evaluation](https://github.com/memodb-io/memobase/blob/main/docs/experiments/locomo-benchmark/README.md). PAM scores from [Harmix blog](https://manager.harmix.ai/blog/pam-locomo-benchmark-results). Letta scores from [Letta blog](https://www.letta.com/blog/benchmarking-ai-agent-memory). SmartSearch from arXiv:2603.15599. Engram from [engram.fyi](https://www.engram.fyi/research). Direct comparison requires caution as evaluation details (judge LLM, prompt, category inclusion) vary across publications.

memlocal is the **only system** that achieves ≥80% on LoCoMo while running as an embedded, on-device library with no server dependencies.

---

## Project Structure

```
memlocal_core/
├── Cargo.toml                    # Crate config: Apache-2.0, lib + cdylib + staticlib
├── src/
│   ├── lib.rs                    # Crate root — re-exports all modules
│   ├── api/
│   │   ├── mod.rs
│   │   └── bridge.rs             # MemlocalEngine — main FFI entry point
│   ├── models/
│   │   ├── mod.rs
│   │   ├── memory_type.rs        # 12-variant MemoryType enum
│   │   ├── memory_category.rs    # Sensory / ShortTerm / LongTerm
│   │   ├── memory_item.rs        # Core MemoryItem struct
│   │   ├── memory_relation.rs    # 14-variant MemoryRelation enum
│   │   ├── config.rs             # CoreConfig, StorageConfig
│   │   └── ...                   # Triple, UserProfile, Message, etc.
│   ├── storage/
│   │   ├── mod.rs
│   │   ├── memory_store.rs       # MemoryStore — all CozoDB operations
│   │   └── schema.rs             # DDL for 7 relations + 5 indices
│   ├── tools/
│   │   ├── mod.rs
│   │   ├── tool_executor.rs      # Retrieval pipeline, reranking, context assembly
│   │   ├── tool_definitions.rs   # 11 tool definitions (add_memory, search, etc.)
│   │   └── prompts.rs            # Extraction v2, dedup, consolidation prompts
│   ├── shortterm/
│   │   ├── mod.rs
│   │   ├── sensory_buffer.rs     # Ring buffer with TTL eviction
│   │   ├── conversation_buffer.rs# Session message buffer
│   │   └── working_memory.rs     # Context block assembly (flat + sectioned)
│   ├── longterm/
│   │   ├── mod.rs
│   │   ├── episodic.rs           # Per-type memory modules
│   │   ├── semantic.rs
│   │   ├── factual.rs
│   │   ├── procedural.rs
│   │   ├── social.rs
│   │   ├── spatial.rs
│   │   ├── prospective.rs
│   │   └── affective.rs
│   ├── consolidation/
│   │   └── mod.rs                # MemoryConsolidator — episodic → semantic
│   ├── error.rs                  # MemlocalError enum, Result type
│   └── http/                     # Optional (feature = "http")
│       ├── mod.rs
│       ├── anthropic_client.rs   # Claude API (tool calling)
│       ├── openai_client.rs      # OpenAI-compatible completions
│       ├── gemini_client.rs      # Google Gemini API
│       ├── openai_embeddings.rs  # text-embedding-3-small
│       ├── jina_reranker.rs      # Jina Reranker v2
│       └── llm_client.rs         # run_with_tools orchestrator
```

---

## Getting Started

### Prerequisites

- Rust 1.75+ (2021 edition)
- For the `http` feature: network access for API calls

### Add to your project

```toml
[dependencies]
memlocal_core = { git = "https://github.com/memlocal/memlocal_core" }
```

With built-in HTTP clients:

```toml
[dependencies]
memlocal_core = { git = "https://github.com/memlocal/memlocal_core", features = ["http"] }
```

### Basic Usage

```rust
use memlocal_core::api::MemlocalEngine;
use memlocal_core::models::{CoreConfig, StorageConfig};
use memlocal_core::tools::{EmbeddingProvider, ExecutionContext};

// 1. Open the engine
let config = CoreConfig {
    storage: StorageConfig {
        db_path: Some("my_memories.db".into()),
        ..Default::default()
    },
    ..Default::default()
};
let engine = MemlocalEngine::open(config)?;

// 2. Implement your embedding provider
struct MyEmbeddings; // wraps OpenAI, ONNX, etc.
impl EmbeddingProvider for MyEmbeddings {
    fn embed_one(&self, text: &str) -> memlocal_core::error::Result<Vec<f32>> {
        // Your embedding logic here
        todo!()
    }
}
let embeddings = MyEmbeddings;

// 3. Store a memory
use memlocal_core::models::{MemoryItem, MemoryType};
let item = MemoryItem::new("User prefers dark mode", MemoryType::Factual);
let embedding = embeddings.embed_one(&item.content)?;
engine.put_memory(&item, &embedding)?;

// 4. Search memories
let query_embedding = embeddings.embed_one("What UI preferences does the user have?")?;
let results = engine.search_semantic(&query_embedding, 5, None, None)?;

// 5. Prepare context for an LLM (the main retrieval entry point)
let context = engine.prepare_context(
    "What UI preferences does the user have?",
    &embeddings,
    None,    // user_id
    Some(15), // max results
    false,   // bm25_only
)?;
// Inject `context` into your LLM's system prompt
```

### Using Built-in HTTP Clients

With the `http` feature enabled:

```rust
use memlocal_core::http::{
    OpenAiEmbeddingProvider,
    AnthropicClient,
    JinaReranker,
};

// Embeddings
let embeddings = OpenAiEmbeddingProvider::new("sk-...")
    .with_model("text-embedding-3-small")
    .with_dimensions(1536);

// LLM (for memory extraction)
let llm = AnthropicClient::new("sk-ant-...")
    .with_model("claude-haiku-4-5-20251001");

// Reranker
let reranker = JinaReranker::new("jina-...")
    .with_model("jina-reranker-v2-base-multilingual");

// Full extraction + storage
use memlocal_core::tools::ExecutionContext;
use memlocal_core::tools::prompts::TemporalContext;

let temporal = TemporalContext::utc();
let ctx = ExecutionContext::full(&embeddings, &llm, &temporal)
    .with_reranker(&reranker);

// Extract and store memories from conversation text
let result = engine.execute_tool("add_memories", &serde_json::json!({
    "text": "Alice: I just got back from Tokyo! The ramen at Ichiran was incredible.\nBob: Nice! I've been wanting to visit Japan.",
}), &ctx)?;
```

---

## Configuration

```rust
CoreConfig {
    storage: StorageConfig {
        in_memory: false,                     // true for tests
        db_path: Some("memlocal.db".into()),  // SQLite file path
        hnsw_m: 16,                           // HNSW max edges per node
        hnsw_ef_construction: 100,            // HNSW construction quality
        embedding_dimensions: 1536,           // Must match your embedding model
        min_confidence_to_store: 0.3,         // Extraction confidence threshold
        enable_time_decay: true,              // Type-aware decay scoring
    },
    conversation_buffer_size: 20,             // Messages per session
    sensory_buffer_capacity: 100,             // Raw input buffer
    sensory_ttl_ms: 5000,                     // Sensory buffer TTL
}
```

---

## API Reference

### MemlocalEngine

The main entry point. All methods are synchronous (blocking) for FFI compatibility.

**Memory CRUD:**

- `put_memory(item, embedding)` — Store a memory with its embedding vector
- `get_memory(id)` — Retrieve by ID
- `get_memories(user_id, memory_type, limit)` — List memories with optional filters
- `delete_memory(id)` — Hard delete
- `invalidate_memory(id)` — Soft invalidate via CozoDB Validity
- `memory_count(memory_type)` — Count memories

**Search:**

- `search_semantic(embedding, k, user_id, memory_type)` — HNSW vector search
- `search_text(query, k)` — BM25 full-text search
- `search_hybrid(query, embedding, k, user_id, memory_type)` — Combined BM25 + semantic
- `search_graph(embedding, k, user_id, memory_type)` — 2-hop graph traversal from semantic seeds

**Context Preparation:**

- `prepare_context(query, embedding_provider, user_id, max_results, bm25_only)` — Full retrieval pipeline → context block string
- `prepare_context_reranked(...)` — Same with cross-encoder reranking
- `prepare_context_iterative_reranked(...)` — Two-round iterative retrieval with LLM sufficiency check

**Conversation:**

- `append_message(message, session_id)` — Add to conversation buffer
- `get_messages(session_id, limit)` — Retrieve conversation history

**Knowledge Graph:**

- `put_edge(edge)` — Add a typed edge between memories
- `get_edges(memory_id)` — Get all edges for a memory

**Profile & Prospective:**

- `put_profile(profile)` / `get_profile(user_id)` — User profile CRUD
- `put_prospective(item)` / `get_pending_prospective(user_id)` / `complete_prospective(id)` — Reminders

**11 Tool Definitions** (for LLM tool-calling integration): `add_memory`, `search_memory`, `get_memories`, `delete_memory`, `get_user_profile`, `add_relationship`, `get_relationships`, `add_reminder`, `get_context`, `add_memories`, `request_more_context`

---

## Provider Traits

The core engine makes no network calls. All I/O is delegated to three pluggable traits:

```rust
/// Implement with OpenAI, ONNX, or any embedding service
pub trait EmbeddingProvider: Send + Sync {
    fn embed_one(&self, text: &str) -> Result<Vec<f32>>;
}

/// Implement with Anthropic, OpenAI, Gemini, or local llama.cpp
pub trait LlmProvider: Send + Sync {
    fn complete(&self, system: &str, user: &str) -> Result<String>;
}

/// Implement with Jina, Cohere, Voyage, or local FlashRank
pub trait RerankerProvider: Send + Sync {
    fn rerank(&self, query: &str, documents: &[String], top_k: usize)
        -> Result<Vec<(usize, f64)>>;
}
```

This design means switching between cloud and local inference is a configuration change, not a code change.

---

## Cloud APIs and Local Alternatives

| Function       | Cloud API (current)               | Local Alternative                | Trade-off                         |
| -------------- | --------------------------------- | -------------------------------- | --------------------------------- |
| **LLM**        | Claude Haiku 4.5                  | Gemma 3 4B via llama.cpp         | Lower extraction quality          |
| **Embeddings** | text-embedding-3-small (1536d)    | all-MiniLM-L6-v2 via ONNX (384d) | Smaller embedding, reduced recall |
| **Reranker**   | Jina Reranker v2 ($0.02/M tokens) | FlashRank ONNX (~33MB)           | 400-800ms latency on mobile       |

### Reranker API Pricing

| Provider          | Price          | Free tier        |
| ----------------- | -------------- | ---------------- |
| Jina AI           | $0.02/M tokens | 10M tokens free  |
| Voyage AI         | $0.05/M tokens | 200M tokens free |
| Mixedbread        | $0.15/M tokens | —                |
| Cohere            | $2/1K searches | —                |
| FlashRank (local) | Free           | ~33MB ONNX model |

---

## Platform SDKs

| Platform             | SDK                                                        | Status       |
| -------------------- | ---------------------------------------------------------- | ------------ |
| **Flutter/Dart**     | [memlocal_dart](https://github.com/memlocal/memlocal_dart) | ✅ Available |
| **Kotlin (Android)** | memlocal-android                                           | 🔜 Planned   |
| **Swift (iOS)**      | memlocal-swift                                             | 🔜 Planned   |
| **Python**           | memlocal-py                                                | 🔜 Planned   |

The Rust core compiles to `cdylib` and `staticlib` targets. Platform SDKs implement the provider traits using their native HTTP stacks and call into the core via FFI (flutter_rust_bridge for Dart, JNI for Kotlin, C FFI for Swift).

---

## Roadmap

- [x] Core memory engine (Rust + CozoDB)
- [x] Eight cognitive memory types with typed decay
- [x] Multi-channel retrieval (semantic + BM25 + graph + triples + session windows)
- [x] Cross-encoder reranking (Jina Reranker v2)
- [x] Query-complexity-adaptive context assembly
- [x] Iterative two-round retrieval
- [x] LLM-driven extraction with proper noun preservation (v2)
- [x] Flutter/Dart SDK
- [x] LoCoMo benchmark: 80.0% pass rate
- [ ] Reciprocal Rank Fusion (RRF) score combination
- [ ] Fully local inference (Gemma 3 4B + MiniLM + FlashRank)
- [ ] Kotlin/Swift native SDK bindings
- [ ] MCP (Model Context Protocol) server integration
- [ ] Multi-agent shared memory with access controls
- [ ] LaTeX white paper for arXiv submission

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing expectations, and pull request guidelines. Review [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for collaboration standards and [SECURITY.md](SECURITY.md) for vulnerability reporting.

The project follows standard Rust conventions:

```bash
# Clone
git clone https://github.com/memlocal/memlocal_core.git
cd memlocal_core

# Build
cargo build

# Build with HTTP clients
cargo build --features http

# Test
cargo test

# Release build (optimised for size: LTO + strip)
cargo build --release
```

The release profile is configured for minimum binary size:

```toml
[profile.release]
lto = true
opt-level = "z"
strip = true
codegen-units = 1
```

---

## Citation

If you use memlocal in your research, please cite:

```bibtex
@misc{chakraborty2026memlocal,
  title     = {memlocal: A Local-First Cognitive Memory Architecture for On-Device AI Agents},
  author    = {Sirsha Chakraborty},
  year      = {2026},
  url       = {https://github.com/memlocal/memlocal_core},
  note      = {White paper available at https://memlocal.dev}
}
```

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full text.

---

<p align="center">
  Built with 🧠 by <a href="https://memlocal.dev">memlocal.dev</a>
</p>

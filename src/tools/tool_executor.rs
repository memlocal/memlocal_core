use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use chrono::{DateTime, Datelike, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::prompts::{self, TemporalContext};
use super::tool_definitions::tool_names;
use crate::error::{MemlocalError, Result};
use crate::models::*;
use crate::storage::MemoryStore;

/// Trait for embedding generation — implemented by the platform layer.
pub trait EmbeddingProvider: Send + Sync {
    fn embed_one(&self, text: &str) -> Result<Vec<f32>>;
}

/// Trait for LLM completions — used by `add_memories` for extraction/classification.
/// Platform layers implement this with their HTTP client (Anthropic, OpenAI, etc.).
pub trait LlmProvider: Send + Sync {
    fn complete(&self, system: &str, user: &str) -> Result<String>;
}

/// Trait for reranking candidate memories after recall.
/// Implemented by cloud reranker providers such as Jina.
pub trait RerankerProvider: Send + Sync {
    /// Returns `(document_index, score)` pairs sorted by descending relevance.
    fn rerank(&self, query: &str, documents: &[String], top_k: usize) -> Result<Vec<(usize, f64)>>;
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub tool_name: String,
    pub content: String,
    pub duration_ms: u64,
    pub success: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct RetrievedMemoryDiagnostic {
    pub id: String,
    pub memory_type: String,
    pub rank: usize,
    pub score: Option<f64>,
    pub pre_rerank_rank: Option<usize>,
    pub pre_rerank_score: Option<f64>,
    pub rerank_score: Option<f64>,
    pub sources: Vec<String>,
    pub included_in_context: bool,
    pub content: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct PrepareContextDiagnostics {
    pub query: String,
    pub sub_queries: Vec<String>,
    pub total_unique_retrieved: usize,
    pub context_limit: usize,
    pub truncated_count: usize,
    pub source_counts: BTreeMap<String, usize>,
    pub ranked_memories: Vec<RetrievedMemoryDiagnostic>,
    pub reserved_keyword_match_ids: Vec<String>,
    pub context_char_count: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct PreparedContext {
    pub context_block: String,
    pub diagnostics: PrepareContextDiagnostics,
}

/// Optional providers passed to `execute`. Only `add_memories` needs the LLM provider.
pub struct ExecutionContext<'a> {
    pub embedding_provider: &'a dyn EmbeddingProvider,
    pub llm_provider: Option<&'a dyn LlmProvider>,
    pub reranker_provider: Option<&'a dyn RerankerProvider>,
    pub temporal: Option<&'a TemporalContext>,
}

impl<'a> ExecutionContext<'a> {
    /// Minimal context (no LLM, no temporal) — sufficient for all tools except `add_memories`.
    pub fn new(embedding_provider: &'a dyn EmbeddingProvider) -> Self {
        Self {
            embedding_provider,
            llm_provider: None,
            reranker_provider: None,
            temporal: None,
        }
    }

    /// Full context with LLM + temporal — required for `add_memories`.
    pub fn full(
        embedding_provider: &'a dyn EmbeddingProvider,
        llm_provider: &'a dyn LlmProvider,
        temporal: &'a TemporalContext,
    ) -> Self {
        Self {
            embedding_provider,
            llm_provider: Some(llm_provider),
            reranker_provider: None,
            temporal: Some(temporal),
        }
    }

    pub fn with_reranker(mut self, reranker_provider: &'a dyn RerankerProvider) -> Self {
        self.reranker_provider = Some(reranker_provider);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RetrievalQueryComplexity {
    SingleHop,
    MultiHop,
    Temporal,
    OpenEnded,
}

pub struct ToolExecutor {
    store: Arc<MemoryStore>,
}

impl ToolExecutor {
    pub fn new(store: Arc<MemoryStore>) -> Self {
        Self { store }
    }

    /// Pre-compute context for a user query BEFORE sending to the LLM.
    ///
    /// This is the key latency optimization: instead of Claude making tool calls
    /// (each costing an LLM round-trip), we fetch relevant context upfront and
    /// inject it into the system prompt. Most questions can then be answered
    /// in a single LLM call.
    ///
    /// Returns a formatted context block ready for system prompt injection.
    pub fn prepare_context(
        &self,
        query: &str,
        embedding_provider: &dyn EmbeddingProvider,
        user_id: Option<&str>,
        max_results: Option<usize>,
        bm25_only: bool,
    ) -> Result<String> {
        self.prepare_context_reranked(
            query,
            embedding_provider,
            user_id,
            max_results,
            bm25_only,
            None,
        )
    }

    pub fn prepare_context_reranked(
        &self,
        query: &str,
        embedding_provider: &dyn EmbeddingProvider,
        user_id: Option<&str>,
        max_results: Option<usize>,
        bm25_only: bool,
        reranker: Option<&dyn RerankerProvider>,
    ) -> Result<String> {
        Ok(self
            .prepare_context_with_diagnostics_reranked(
                query,
                embedding_provider,
                user_id,
                max_results,
                bm25_only,
                reranker,
            )?
            .context_block)
    }

    pub fn prepare_context_with_diagnostics(
        &self,
        query: &str,
        embedding_provider: &dyn EmbeddingProvider,
        user_id: Option<&str>,
        max_results: Option<usize>,
        bm25_only: bool,
    ) -> Result<PreparedContext> {
        self.prepare_context_with_diagnostics_reranked(
            query,
            embedding_provider,
            user_id,
            max_results,
            bm25_only,
            None,
        )
    }

    pub fn prepare_context_with_diagnostics_reranked(
        &self,
        query: &str,
        embedding_provider: &dyn EmbeddingProvider,
        user_id: Option<&str>,
        max_results: Option<usize>,
        bm25_only: bool,
        reranker: Option<&dyn RerankerProvider>,
    ) -> Result<PreparedContext> {
        // Query shaping: search the original question plus compact, person-focused,
        // and temporal-aware variants so retrieval does not depend on the exact wording.
        let query_complexity = classify_retrieval_query(query);
        let sub_queries = build_query_variants(query);
        let temporal_focus = is_temporal_query(query);
        let mut all_memories = Vec::new();
        let mut seen_ids = HashSet::new();
        let mut memory_sources: HashMap<String, BTreeSet<String>> = HashMap::new();

        let mut register_item = |item: &MemoryItem, source: &str| {
            memory_sources
                .entry(item.id.clone())
                .or_default()
                .insert(source.to_string());
        };

        // Speaker-aware query routing
        let known_speakers = self.store.get_known_speakers(user_id).unwrap_or_default();
        let target_speaker = detect_query_speaker(query, &known_speakers);

        // If a specific speaker is detected, add speaker-filtered retrieval
        if let Some(ref speaker) = target_speaker {
            if !bm25_only {
                let emb = embedding_provider.embed_one(query)?;
                let speaker_results = self.store.search_by_speaker(
                    &emb, speaker, 10, user_id
                ).unwrap_or_default();
                for item in speaker_results {
                    register_item(&item, &format!("speaker_filter:{}", speaker));
                    if seen_ids.insert(item.id.clone()) {
                        all_memories.push(item);
                    }
                }
            }
        }

        for sq in &sub_queries {
            let mut session_seed_ids: Vec<String> = Vec::new();
            if !bm25_only {
                let emb = embedding_provider.embed_one(sq)?;

                // 1. Per-type retrieval (Engram-style: each type retrieves independently)
                let type_searches = [
                    (MemoryType::Episodic, 15),
                    (MemoryType::Factual, 10),
                    (MemoryType::Semantic, 8),
                    (MemoryType::Social, 5),
                    (MemoryType::Affective, 5),
                    (MemoryType::Prospective, 5),
                    (MemoryType::Procedural, 3),
                ];

                for (mem_type, k) in &type_searches {
                    let results =
                        self.store
                            .search_hybrid_deduped(sq, &emb, *k, user_id, Some(*mem_type))?;
                    for item in results {
                        if temporal_focus
                            && matches!(item.memory_type, MemoryType::Episodic | MemoryType::Prospective)
                        {
                            if let Some(session_id) = item.session_id.as_ref() {
                                if !session_seed_ids.iter().any(|existing| existing == session_id) {
                                    session_seed_ids.push(session_id.clone());
                                }
                            }
                        }
                        register_item(&item, &format!("type:{}", mem_type.stored_name()));
                        if seen_ids.insert(item.id.clone()) {
                            all_memories.push(item);
                        }
                    }
                }

                // 2. Untyped hybrid search as catch-all
                let results =
                    self.store
                        .search_hybrid_deduped(sq, &emb, 10, user_id, None)?;
                for item in results {
                    register_item(&item, "hybrid:catch_all");
                    if seen_ids.insert(item.id.clone()) {
                        all_memories.push(item);
                    }
                }

                // 2b. Graph search — 2-hop edge traversal from semantic seeds
                let graph_results =
                    self.store.search_graph(&emb, 5, user_id, None, 2)?;
                for item in graph_results {
                    if temporal_focus {
                        if let Some(session_id) = item.session_id.as_ref() {
                            if !session_seed_ids.iter().any(|existing| existing == session_id) {
                                session_seed_ids.push(session_id.clone());
                            }
                        }
                    }
                    register_item(&item, "graph:2hop");
                    if seen_ids.insert(item.id.clone()) {
                        all_memories.push(item);
                    }
                }

                if temporal_focus {
                    for session_id in session_seed_ids.iter().take(3) {
                        let session_results =
                            self.store.get_memories_by_session(session_id, user_id, 10)?;
                        for item in session_results {
                            register_item(&item, &format!("session:{session_id}"));
                            if seen_ids.insert(item.id.clone()) {
                                all_memories.push(item);
                            }
                        }
                    }
                }

                // 2c. Triple-based structured retrieval
                let query_entities = extract_entities(sq);
                for entity in &query_entities {
                    let triples = self.store.search_triples(Some(entity), None, None)
                        .unwrap_or_default();
                    for triple in &triples {
                        if let Ok(Some(memory)) = self.store.get_memory(&triple.memory_id) {
                            register_item(&memory, &format!("triple:{}", entity));
                            if seen_ids.insert(memory.id.clone()) {
                                all_memories.push(memory);
                            }
                        }
                    }
                }

                // 2d. Triple FTS search
                let triple_fts_results = self.store.search_triples_fts(sq, 10)
                    .unwrap_or_default();
                for triple in &triple_fts_results {
                    if let Ok(Some(memory)) = self.store.get_memory(&triple.memory_id) {
                        register_item(&memory, "triple:fts");
                        if seen_ids.insert(memory.id.clone()) {
                            all_memories.push(memory);
                        }
                    }
                }
            }

            // 3. Entity-focused BM25 search (always runs — primary for bm25_only mode)
            let entities = extract_entities(sq);
            for entity in &entities {
                let k = if bm25_only { 20 } else { 15 }; // more BM25 results when it's the only source
                let results = self.store.search_text(entity, k)?;
                for item in results {
                    if let Some(uid) = user_id {
                        if item.user_id.as_deref() != Some(uid) {
                            continue;
                        }
                    }
                    register_item(&item, &format!("bm25:entity:{entity}"));
                    if seen_ids.insert(item.id.clone()) {
                        all_memories.push(item);
                    }
                }
            }

            // 4. Focused keyword BM25 — query stripped of stopwords
            let keywords = extract_keywords(sq);
            if !keywords.is_empty() {
                let k = if bm25_only { 15 } else { 10 };
                let results = self.store.search_text(&keywords, k)?;
                for item in results {
                    if let Some(uid) = user_id {
                        if item.user_id.as_deref() != Some(uid) {
                            continue;
                        }
                    }
                    register_item(&item, &format!("bm25:keywords:{keywords}"));
                    if seen_ids.insert(item.id.clone()) {
                        all_memories.push(item);
                    }
                }
            }
        }

        // 5. Session summary retrieval (helps multi-hop and temporal queries)
        if !bm25_only {
            let query_emb = embedding_provider.embed_one(query)?;
            let summaries = self.store.search_summaries(&query_emb, 3)
                .unwrap_or_default();
            for summary in &summaries {
                let session_memories = self.store.get_memories_by_session(
                    &summary.session_id,
                    user_id,
                    15,
                ).unwrap_or_default();
                for item in session_memories {
                    register_item(&item, &format!("summary_session:{}", summary.session_id));
                    if seen_ids.insert(item.id.clone()) {
                        all_memories.push(item);
                    }
                }
            }
        }

        let raw_session_seeds: Vec<(String, String)> = all_memories
            .iter()
            .filter(|item| is_raw_conversation_item(item))
            .filter_map(|item| {
                item.session_id
                    .as_ref()
                    .map(|session_id| (session_id.clone(), item.content.clone()))
            })
            .collect();
        let mut expanded_sessions = HashSet::new();
        for (session_id, anchor_content) in raw_session_seeds.into_iter().take(5) {
            if !expanded_sessions.insert(session_id.clone()) {
                continue;
            }

            let adjacent = self
                .store
                .get_adjacent_turns(&session_id, &anchor_content, 2, user_id)
                .unwrap_or_default();
            for item in adjacent {
                register_item(&item, "session_window");
                if seen_ids.insert(item.id.clone()) {
                    all_memories.push(item);
                }
            }
        }

        // --- Fix 1: Reserve BM25 slots (guaranteed keyword hits) ---
        // Collect the top BM25-only hits per entity so they can't be displaced by semantic results
        let mut bm25_reserved: Vec<MemoryItem> = Vec::new();
        let mut bm25_reserved_ids = std::collections::HashSet::new();
        for sq in &sub_queries {
            let entities = extract_entities(sq);
            for entity in &entities {
                let results = self.store.search_text(entity, 5)?;
                for item in results {
                    if let Some(uid) = user_id {
                        if item.user_id.as_deref() != Some(uid) {
                            continue;
                        }
                    }
                    register_item(&item, &format!("bm25:reserved:{entity}"));
                    if bm25_reserved_ids.insert(item.id.clone()) {
                        bm25_reserved.push(item);
                    }
                }
            }
        }
        // Cap reserved slots at 15 to avoid flooding context
        bm25_reserved.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        bm25_reserved.truncate(15);

        // Apply is_latest penalty: demote superseded facts
        for item in &mut all_memories {
            if !item.is_latest() {
                item.score = Some(item.score.unwrap_or(0.0) * 0.5);
            }
        }

        // --- Fix 3: Keyword overlap boost ---
        // Boost scores of memories containing exact query keywords
        let query_terms: Vec<String> = extract_keywords(query)
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        if !query_terms.is_empty() {
            for item in &mut all_memories {
                let content_lower = item.content.to_lowercase();
                let matched = query_terms
                    .iter()
                    .filter(|t| content_lower.contains(t.as_str()))
                    .count();
                if matched > 0 {
                    let boost = 1.0 + (matched as f64 / query_terms.len() as f64) * 0.5;
                    item.score = Some(item.score.unwrap_or(0.0) * boost);
                }
            }
        }

        apply_query_aware_boosts(&mut all_memories, query);

        // Sort by boosted score before optional reranking.
        all_memories.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let limit = resolve_context_limit(query, max_results);
        let total_unique_retrieved = all_memories.len();
        let truncated_count = total_unique_retrieved.saturating_sub(limit);
        let pre_rerank_positions: HashMap<String, (usize, Option<f64>)> = all_memories
            .iter()
            .enumerate()
            .map(|(index, item)| (item.id.clone(), (index + 1, item.score)))
            .collect();

        let mut rerank_scores_by_id: HashMap<String, f64> = HashMap::new();
        let rerank_pool_size = resolve_rerank_pool_size(query).min(all_memories.len());
        if let Some(reranker) = reranker {
            if rerank_pool_size > 1 && limit > 0 {
                let rerank_candidates: Vec<MemoryItem> = all_memories
                    .iter()
                    .take(rerank_pool_size)
                    .cloned()
                    .collect();
                let documents: Vec<String> = rerank_candidates
                    .iter()
                    .map(|item| item.content.clone())
                    .collect();
                let rerank_top_k = rerank_candidates.len();

                match reranker.rerank(query, &documents, rerank_top_k) {
                    Ok(reranked) => {
                        let mut reordered = Vec::with_capacity(all_memories.len());
                        let mut selected_ids = HashSet::new();

                        for (index, rerank_score) in reranked {
                            if let Some(mut item) = rerank_candidates.get(index).cloned() {
                                rerank_scores_by_id.insert(item.id.clone(), rerank_score);
                                selected_ids.insert(item.id.clone());
                                item.score = Some(rerank_score);
                                reordered.push(item);
                            }
                        }

                        for item in all_memories {
                            if selected_ids.contains(&item.id) {
                                continue;
                            }
                            reordered.push(item);
                        }

                        all_memories = reordered;
                    }
                    Err(err) => {
                        log::warn!("Reranking failed for query {:?}: {}", query, err);
                    }
                }
            }
        }

        let ranked_before_truncation: Vec<RetrievedMemoryDiagnostic> = all_memories
            .iter()
            .enumerate()
            .map(|(index, item)| {
                let (pre_rerank_rank, pre_rerank_score) = pre_rerank_positions
                    .get(&item.id)
                    .copied()
                    .unwrap_or((index + 1, item.score));

                RetrievedMemoryDiagnostic {
                    id: item.id.clone(),
                    memory_type: item.memory_type.stored_name().to_string(),
                    rank: index + 1,
                    score: item.score,
                    pre_rerank_rank: Some(pre_rerank_rank),
                    pre_rerank_score,
                    rerank_score: rerank_scores_by_id.get(&item.id).copied(),
                    sources: memory_sources
                        .get(&item.id)
                        .map(|sources| sources.iter().cloned().collect())
                        .unwrap_or_default(),
                    included_in_context: index < limit,
                    content: item.content.clone(),
                }
            })
            .collect();
        all_memories.truncate(limit);

        // --- Option C: BM25 reserved hits go into a separate priority section ---
        // These are displayed as "KEY FACTS" at the top of the context block
        let keyword_matches = bm25_reserved;
        let context_block = if matches!(query_complexity, RetrievalQueryComplexity::SingleHop) {
            let mut wm = crate::shortterm::WorkingMemory::new();
            wm.set_relevant(all_memories);
            wm.to_flat_context_block()
        } else {
            // Get user profile
            let profile = match user_id {
                Some(uid) => self.store.get_profile(uid)?,
                None => None,
            };

            // Get pending reminders
            let prospective = self
                .store
                .get_pending_prospective(user_id)
                .unwrap_or_default();

            // Get important memories
            let important = self
                .store
                .get_important_memories(user_id, 10, 0.5)
                .unwrap_or_default();

            // Collect triples relevant to the query for structured context
            let mut key_triples = Vec::new();
            {
                let query_entities = extract_entities(query);
                let mut seen_triples = HashSet::new();
                for entity in &query_entities {
                    let triples = self.store.search_triples(Some(entity), None, None)
                        .unwrap_or_default();
                    for t in triples {
                        let key = format!("{}|{}|{}", t.subject, t.predicate, t.object);
                        if seen_triples.insert(key) {
                            key_triples.push(t);
                        }
                    }
                }
                // Also add triples from FTS
                let fts_triples = self.store.search_triples_fts(query, 15)
                    .unwrap_or_default();
                for t in fts_triples {
                    let key = format!("{}|{}|{}", t.subject, t.predicate, t.object);
                    if seen_triples.insert(key) {
                        key_triples.push(t);
                    }
                }
                key_triples.truncate(20); // Cap to avoid flooding context
            }

            // Collect session summaries for narrative context
            let context_summaries = if !bm25_only {
                let q_emb = embedding_provider.embed_one(query)?;
                self.store.search_summaries(&q_emb, 5).unwrap_or_default()
            } else {
                Vec::new()
            };

            // Assemble using WorkingMemory
            let mut wm = crate::shortterm::WorkingMemory::new();
            wm.set_keyword_matches(keyword_matches);
            wm.set_key_triples(key_triples);
            wm.set_session_summaries(context_summaries);
            wm.set_relevant(all_memories);
            wm.set_important(important);
            wm.set_profile(profile);
            wm.set_triggered_reminders(prospective);
            wm.to_context_block()
        };
        let mut source_counts = BTreeMap::new();
        for sources in memory_sources.values() {
            for source in sources {
                *source_counts.entry(source.clone()).or_insert(0) += 1;
            }
        }

        Ok(PreparedContext {
            diagnostics: PrepareContextDiagnostics {
                query: query.to_string(),
                sub_queries,
                total_unique_retrieved,
                context_limit: limit,
                truncated_count,
                source_counts,
                ranked_memories: ranked_before_truncation,
                reserved_keyword_match_ids: bm25_reserved_ids.into_iter().collect(),
                context_char_count: context_block.chars().count(),
            },
            context_block,
        })
    }

    /// Execute a single tool call (backward-compatible — no LLM/temporal).
    pub fn execute(
        &self,
        tool_call: &ToolCall,
        embedding_provider: &dyn EmbeddingProvider,
    ) -> ToolResult {
        let ctx = ExecutionContext::new(embedding_provider);
        self.execute_with_context(tool_call, &ctx)
    }

    /// Execute a single tool call with full context (LLM + temporal).
    pub fn execute_with_context(&self, tool_call: &ToolCall, ctx: &ExecutionContext) -> ToolResult {
        let start = std::time::Instant::now();
        let result = self.dispatch(&tool_call.name, &tool_call.arguments, ctx);
        let duration_ms = start.elapsed().as_millis() as u64;

        match result {
            Ok(value) => ToolResult {
                tool_call_id: tool_call.id.clone(),
                tool_name: tool_call.name.clone(),
                content: serde_json::to_string(&value).unwrap_or_default(),
                duration_ms,
                success: true,
            },
            Err(e) => ToolResult {
                tool_call_id: tool_call.id.clone(),
                tool_name: tool_call.name.clone(),
                content: serde_json::json!({"error": e.to_string()}).to_string(),
                duration_ms,
                success: false,
            },
        }
    }

    /// Execute multiple tool calls in sequence.
    pub fn execute_all(
        &self,
        tool_calls: &[ToolCall],
        embedding_provider: &dyn EmbeddingProvider,
    ) -> Vec<ToolResult> {
        tool_calls
            .iter()
            .map(|tc| self.execute(tc, embedding_provider))
            .collect()
    }

    fn dispatch(
        &self,
        name: &str,
        args: &serde_json::Value,
        ctx: &ExecutionContext,
    ) -> Result<serde_json::Value> {
        match name {
            tool_names::ADD_MEMORY => self.add_memory(args, ctx),
            tool_names::ADD_MEMORIES => self.add_memories(args, ctx),
            tool_names::SEARCH_MEMORY => self.search_memory(args, ctx.embedding_provider),
            tool_names::GET_MEMORIES => self.get_memories(args),
            tool_names::DELETE_MEMORY => self.delete_memory(args),
            tool_names::GET_PROFILE => self.get_profile(args),
            tool_names::ADD_RELATIONSHIP => self.add_relationship(args),
            tool_names::GET_RELATIONSHIPS => self.get_relationships(args),
            tool_names::ADD_REMINDER => self.add_reminder(args, ctx.embedding_provider),
            tool_names::GET_CONTEXT => self.get_context(args, ctx.embedding_provider),
            tool_names::REQUEST_MORE_CONTEXT => self.request_more_context(args, ctx),
            _ => Err(MemlocalError::InvalidArgument(format!(
                "Unknown tool: {name}"
            ))),
        }
    }

    // ─────────── add_memory (with semantic dedup) ───────────

    fn add_memory(
        &self,
        args: &serde_json::Value,
        ctx: &ExecutionContext,
    ) -> Result<serde_json::Value> {
        let content = args["content"]
            .as_str()
            .ok_or_else(|| MemlocalError::InvalidArgument("missing 'content'".into()))?;
        let type_str = args["memory_type"].as_str().unwrap_or("factual");
        let user_id = args["user_id"].as_str().map(String::from);
        let session_id = args["session_id"].as_str().map(String::from);
        let confidence = args["confidence"].as_f64().unwrap_or(0.9);

        let memory_type = MemoryType::from_stored_name(type_str);

        // Parse optional temporal fields
        let valid_at = args["valid_at"]
            .as_str()
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc));
        let invalid_at = args["invalid_at"]
            .as_str()
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc));
        let extra_metadata = args.get("metadata").and_then(|value| value.as_object());

        // 1. Exact dedup by content hash
        let hash = MemoryItem::compute_hash(content);
        if let Some(existing) = self.store.find_by_hash(&hash, user_id.as_deref())? {
            // Reinforcement: bump count even for exact duplicates
            let mut metadata = existing.metadata.clone();
            let current_count = metadata.get("reinforcement_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(1);
            if let Some(obj) = metadata.as_object_mut() {
                obj.insert("reinforcement_count".to_string(), serde_json::json!(current_count + 1));
                obj.insert("last_reinforced_at".to_string(), serde_json::json!(Utc::now().to_rfc3339()));
                obj.insert("confidence".to_string(), serde_json::json!(confidence));
                if let Some(extra) = extra_metadata {
                    for (key, value) in extra {
                        obj.insert(key.clone(), value.clone());
                    }
                }
            }
            if let Some(speaker) = args.get("speaker").and_then(|v| v.as_str()) {
                if !speaker.is_empty() {
                    if let Some(obj) = metadata.as_object_mut() {
                        obj.insert("speaker".to_string(), serde_json::json!(speaker));
                    }
                }
            }
            // Re-embed and update (to persist the metadata change)
            let embedding = ctx.embedding_provider.embed_one(content)?;
            let updated = MemoryItem {
                metadata,
                updated_at: Utc::now(),
                ..existing.clone()
            };
            self.store.put_memory(&updated, &embedding)?;

            return Ok(serde_json::json!({
                "status": "reinforced",
                "memory_id": existing.id,
                "reason": "exact content match, reinforcement count incremented"
            }));
        }

        // 2. Semantic dedup — check for highly similar existing memories
        let embedding = ctx.embedding_provider.embed_one(content)?;
        let similar = self
            .store
            .search_semantic(&embedding, 3, user_id.as_deref(), None)?;

        if let Some(existing) = similar.first() {
            let sim_score = existing.score.unwrap_or(0.0);
            // v5: Lower threshold (0.70) + conflict detection for contradicting specifics
            let is_near_duplicate = sim_score > 0.85;
            let is_conflicting =
                sim_score > 0.70 && has_conflicting_specifics(&existing.content, content);
            if is_near_duplicate || is_conflicting {
                // Update the existing memory instead of adding a duplicate
                let now = Utc::now();
                // Increment reinforcement count
                let mut metadata = existing.metadata.clone();
                let current_count = metadata.get("reinforcement_count")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1);
                if let Some(obj) = metadata.as_object_mut() {
                    obj.insert("reinforcement_count".to_string(), serde_json::json!(current_count + 1));
                    obj.insert("last_reinforced_at".to_string(), serde_json::json!(now.to_rfc3339()));
                    obj.insert("confidence".to_string(), serde_json::json!(confidence));
                }
                // Preserve speaker if provided
                if let Some(speaker) = args.get("speaker").and_then(|v| v.as_str()) {
                    if !speaker.is_empty() {
                        if let Some(obj) = metadata.as_object_mut() {
                            obj.insert("speaker".to_string(), serde_json::json!(speaker));
                        }
                    }
                }
                if let Some(extra) = extra_metadata {
                    if let Some(obj) = metadata.as_object_mut() {
                        for (key, value) in extra {
                            obj.insert(key.clone(), value.clone());
                        }
                    }
                }
                let updated = MemoryItem {
                    id: existing.id.clone(),
                    content: content.to_string(),
                    memory_type,
                    hash: hash.clone(),
                    user_id: user_id.clone(),
                    agent_id: None,
                    session_id: session_id.clone().or_else(|| existing.session_id.clone()),
                    metadata,
                    created_at: existing.created_at,
                    updated_at: now,
                    valid_at,
                    invalid_at,
                    score: None,
                };
                self.store.put_memory(&updated, &embedding)?;

                return Ok(serde_json::json!({
                    "status": "updated",
                    "memory_id": existing.id,
                    "reason": "semantically similar memory updated"
                }));
            }
        }

        // 3. Store as new
        let id = Uuid::new_v4().to_string();
        let now = Utc::now();
        let mut metadata = serde_json::json!({"confidence": confidence});
        if let Some(extra) = extra_metadata {
            if let Some(obj) = metadata.as_object_mut() {
                for (key, value) in extra {
                    obj.insert(key.clone(), value.clone());
                }
            }
        }
        if let Some(speaker) = args.get("speaker").and_then(|v| v.as_str()) {
            if !speaker.is_empty() {
                metadata["speaker"] = serde_json::json!(speaker);
            }
        }
        let item = MemoryItem {
            id: id.clone(),
            content: content.to_string(),
            memory_type,
            hash,
            user_id,
            agent_id: None,
            session_id,
            metadata,
            created_at: now,
            updated_at: now,
            valid_at,
            invalid_at,
            score: None,
        };

        self.store.put_memory(&item, &embedding)?;

        // Auto-create edges to semantically related memories (graph intelligence)
        let similar = self
            .store
            .search_semantic(&embedding, 3, item.user_id.as_deref(), None)?;
        for sim in &similar {
            if sim.id != id && sim.score.unwrap_or(0.0) > 0.5 {
                let edge = MemoryEdge::new(id.clone(), sim.id.clone(), MemoryRelation::RelatesTo);
                let _ = self.store.put_edge(&edge); // best-effort, don't fail on edge error
            }
        }

        Ok(serde_json::json!({
            "status": "stored",
            "memory_id": id,
            "type": memory_type.stored_name(),
        }))
    }

    // ─────────── add_memories (LLM-driven extraction + classification) ───────────

    fn add_memories(
        &self,
        args: &serde_json::Value,
        ctx: &ExecutionContext,
    ) -> Result<serde_json::Value> {
        let text = args["text"]
            .as_str()
            .ok_or_else(|| MemlocalError::InvalidArgument("missing 'text'".into()))?;
        let user_id = args["user_id"].as_str();
        let batch_session_id = args["session_id"].as_str();
        let preserve_source = args["preserve_source"].as_bool().unwrap_or(false);

        let llm = ctx.llm_provider.ok_or_else(|| {
            MemlocalError::InvalidArgument(
                "add_memories requires an LlmProvider in ExecutionContext".into(),
            )
        })?;

        let temporal = ctx.temporal.cloned().unwrap_or_else(TemporalContext::ist);

        // Call LLM with extraction prompt
        let user_msg = prompts::build_extraction_user(text, &temporal);
        let response = llm.complete(prompts::EXTRACTION_SYSTEM, &user_msg)?;

        // Parse JSON response
        let json_text = response.trim();
        let json_text = if json_text.starts_with("```") {
            json_text
                .lines()
                .skip(1)
                .take_while(|l| !l.starts_with("```"))
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            json_text.to_string()
        };

        let response_obj: serde_json::Value = serde_json::from_str(&json_text).map_err(|e| {
            MemlocalError::Internal(format!("Failed to parse extraction response: {e}"))
        })?;

        // Handle both old format (array) and new format (object) for backwards compat
        let (extracted, observations, session_summary, speakers_detected) = if response_obj.is_array() {
            // Old format: flat array
            let arr: Vec<serde_json::Value> = serde_json::from_value(response_obj).unwrap_or_default();
            (arr, Vec::new(), None, Vec::new())
        } else {
            // New format: structured object
            let memories = response_obj.get("memories")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            let observations = response_obj.get("observations")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            let summary = response_obj.get("session_summary")
                .and_then(|v| v.as_str())
                .map(String::from);
            let speakers: Vec<String> = response_obj.get("speakers_detected")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
            (memories, observations, summary, speakers)
        };

        let mut stored = Vec::new();
        let mut skipped = 0;
        let mut updated = 0;
        let mut extracted_memory_ids = Vec::new();
        let mut observation_keys = HashSet::new();

        for item in &extracted {
            let content = match item["content"].as_str() {
                Some(c) if !c.is_empty() => c,
                _ => continue,
            };
            let type_str = item["type"].as_str().unwrap_or("factual");
            let confidence = item["confidence"].as_f64().unwrap_or(0.8);

            // Extract new structured fields
            let speaker = item.get("speaker").and_then(|v| v.as_str()).unwrap_or("");
            let triple_obj = item.get("triple");
            let contradicts_pattern = item.get("contradicts_pattern").and_then(|v| v.as_str());

            // Skip low-confidence extractions
            if confidence < 0.5 {
                skipped += 1;
                continue;
            }

            // Append source text so FTS can match original phrasing too
            let content_with_source = if text.len() <= 500 {
                format!("{content}\n[Source: {text}]")
            } else {
                // For large texts, just use the extracted content
                content.to_string()
            };

            // Build add_memory args with extracted fields
            let mut add_args = serde_json::json!({
                "content": content_with_source,
                "memory_type": type_str,
                "confidence": confidence,
            });
            if let Some(uid) = user_id {
                add_args["user_id"] = serde_json::Value::String(uid.to_string());
            }
            if let Some(session_id) = batch_session_id {
                add_args["session_id"] = serde_json::Value::String(session_id.to_string());
            }
            if let Some(va) = item.get("valid_at") {
                add_args["valid_at"] = va.clone();
            }
            if let Some(ia) = item.get("invalid_at") {
                add_args["invalid_at"] = ia.clone();
            }
            if !speaker.is_empty() {
                add_args["speaker"] = serde_json::json!(speaker);
            }

            // Delegate to add_memory (which handles dedup)
            let result = self.add_memory(&add_args, ctx)?;
            if let Some(memory_id) = result["memory_id"].as_str() {
                extracted_memory_ids.push(memory_id.to_string());
            }

            // Store triple if present and memory was stored/updated/reinforced
            if let (Some(memory_id), Some(triple_val)) = (result["memory_id"].as_str(), triple_obj) {
                if let (Some(subject), Some(predicate), Some(object)) = (
                    triple_val.get("subject").and_then(|v| v.as_str()),
                    triple_val.get("predicate").and_then(|v| v.as_str()),
                    triple_val.get("object").and_then(|v| v.as_str()),
                ) {
                    // Check if this triple already exists
                    let existing_triples = self.store.search_triples(
                        Some(subject), Some(predicate), Some(object)
                    ).unwrap_or_default();

                    if existing_triples.is_empty() {
                        let triple = Triple {
                            subject: subject.to_string(),
                            predicate: predicate.to_string(),
                            object: object.to_string(),
                            memory_id: memory_id.to_string(),
                            speaker: speaker.to_string(),
                            mention_count: 1,
                            last_mentioned: chrono::Utc::now().timestamp_millis() as f64 / 1000.0,
                            session_id: batch_session_id.unwrap_or("").to_string(),
                            confidence,
                        };
                        let _ = self.store.put_triple(&triple);
                    } else {
                        // Reinforcement: increment mention count
                        let _ = self.store.increment_triple_mention(
                            subject, predicate, object,
                            chrono::Utc::now().timestamp_millis() as f64 / 1000.0,
                        );
                    }
                }
            }

            // Handle contradiction detection
            if let Some(pattern) = contradicts_pattern {
                let parts: Vec<&str> = pattern.splitn(2, ' ').collect();
                if parts.len() == 2 {
                    let old_triples = self.store.search_triples(Some(parts[0]), Some(parts[1]), None)
                        .unwrap_or_default();

                    for old_triple in &old_triples {
                        if old_triple.memory_id != result["memory_id"].as_str().unwrap_or("") {
                            // Mark old memory as not latest + create Updates edge
                            if let Ok(Some(old_memory)) = self.store.get_memory(&old_triple.memory_id) {
                                // Write is_latest: false into old memory's metadata
                                let mut old_meta = old_memory.metadata.clone();
                                if let Some(obj) = old_meta.as_object_mut() {
                                    obj.insert("is_latest".to_string(), serde_json::json!(false));
                                }
                                let updated_old = MemoryItem {
                                    metadata: old_meta,
                                    updated_at: old_memory.updated_at, // preserve original timestamp
                                    ..old_memory
                                };
                                // Re-embed and persist (need embedding for put_memory)
                                if let Ok(old_emb) = ctx.embedding_provider.embed_one(&updated_old.content) {
                                    let _ = self.store.put_memory(&updated_old, &old_emb);
                                }

                                if let Some(new_id) = result["memory_id"].as_str() {
                                    let edge = MemoryEdge::new(
                                        new_id.to_string(),
                                        old_triple.memory_id.clone(),
                                        MemoryRelation::Updates,
                                    );
                                    let _ = self.store.put_edge(&edge);
                                }
                            }
                        }
                    }
                }
            }

            match result["status"].as_str() {
                Some("stored") => stored.push(result),
                Some("updated") => updated += 1,
                Some("duplicate") | Some("reinforced") => skipped += 1,
                _ => stored.push(result),
            }
        }

        for item in &observations {
            let raw_content = match item.get("content").and_then(|value| value.as_str()) {
                Some(content) => content,
                None => {
                    skipped += 1;
                    continue;
                }
            };
            let content = raw_content.split_whitespace().collect::<Vec<_>>().join(" ");
            if content.is_empty() {
                skipped += 1;
                continue;
            }

            let confidence = item.get("confidence").and_then(|value| value.as_f64()).unwrap_or(0.8);
            if confidence < 0.5 {
                skipped += 1;
                continue;
            }

            let speaker = item.get("speaker").and_then(|value| value.as_str()).unwrap_or("");
            let observation_key = format!(
                "{}::{}",
                speaker.trim().to_lowercase(),
                content.to_lowercase()
            );
            if !observation_keys.insert(observation_key) {
                skipped += 1;
                continue;
            }

            let mut add_args = serde_json::json!({
                "content": content,
                "memory_type": "factual",
                "confidence": confidence,
                "metadata": {
                    "observation": true,
                    "extraction_kind": "observation"
                }
            });
            if let Some(uid) = user_id {
                add_args["user_id"] = serde_json::Value::String(uid.to_string());
            }
            if let Some(session_id) = batch_session_id {
                add_args["session_id"] = serde_json::Value::String(session_id.to_string());
            }
            if let Some(va) = item.get("valid_at") {
                add_args["valid_at"] = va.clone();
            }
            if let Some(ia) = item.get("invalid_at") {
                add_args["invalid_at"] = ia.clone();
            }
            if !speaker.is_empty() {
                add_args["speaker"] = serde_json::json!(speaker);
            }

            let result = self.add_memory(&add_args, ctx)?;
            if let Some(memory_id) = result["memory_id"].as_str() {
                extracted_memory_ids.push(memory_id.to_string());
            }

            match result["status"].as_str() {
                Some("stored") => stored.push(result),
                Some("updated") => updated += 1,
                Some("duplicate") | Some("reinforced") => skipped += 1,
                _ => stored.push(result),
            }
        }

        // Dual-layer: also store raw text segments as Episodic memories
        let mut raw_stored = 0;
        let mut raw_memory_ids = Vec::new();
        if preserve_source {
            // Split on double-newlines to get segments (dialog turns, paragraphs, etc.)
            let segments: Vec<&str> = text
                .split("\n\n")
                .map(|s| s.trim())
                .filter(|s| !s.is_empty() && s.len() > 10) // skip empty/tiny segments
                .collect();

            for segment in &segments {
                let seg_hash = MemoryItem::compute_hash(segment);
                // Skip if exact duplicate already exists
                if self
                    .store
                    .find_by_hash(&seg_hash, user_id)
                    .ok()
                    .flatten()
                    .is_some()
                {
                    continue;
                }

                let seg_embedding = ctx.embedding_provider.embed_one(segment)?;
                let seg_id = Uuid::new_v4().to_string();
                let now = Utc::now();

                // Parse valid_at from [Session N, datetime] header if present
                let valid_at = parse_session_datetime(segment);

                let seg_item = MemoryItem {
                    id: seg_id.clone(),
                    content: segment.to_string(),
                    memory_type: MemoryType::Episodic,
                    hash: seg_hash,
                    user_id: user_id.map(String::from),
                    agent_id: None,
                    session_id: batch_session_id.map(String::from),
                    metadata: serde_json::json!({
                        "source": "raw_conversation",
                        "session_id": batch_session_id,
                    }),
                    created_at: now,
                    updated_at: now,
                    valid_at,
                    invalid_at: None,
                    score: None,
                };

                self.store.put_memory(&seg_item, &seg_embedding)?;

                // Create edges to similar memories (like extracted memories get)
                let similar = self.store.search_semantic(&seg_embedding, 3, user_id, None)?;
                for sim in &similar {
                    if sim.id == seg_id {
                        continue;
                    }
                    let weight = sim.score.unwrap_or(0.5);
                    if weight > 0.5 {
                        let edge = MemoryEdge::new(seg_id.clone(), sim.id.clone(), MemoryRelation::RelatesTo);
                        let _ = self.store.put_edge(&edge);
                    }
                }

                raw_memory_ids.push(seg_id);
                raw_stored += 1;
            }
        }

        if let Some(session_id) = batch_session_id {
            for raw_id in &raw_memory_ids {
                for extracted_id in &extracted_memory_ids {
                    if raw_id == extracted_id {
                        continue;
                    }

                    let mut raw_to_extracted = MemoryEdge::new(
                        raw_id.clone(),
                        extracted_id.clone(),
                        MemoryRelation::DuringSession,
                    );
                    raw_to_extracted.weight = 1.0;
                    raw_to_extracted.metadata = serde_json::json!({"session_id": session_id});
                    let _ = self.store.put_edge(&raw_to_extracted);

                    let mut extracted_to_raw = MemoryEdge::new(
                        extracted_id.clone(),
                        raw_id.clone(),
                        MemoryRelation::DuringSession,
                    );
                    extracted_to_raw.weight = 1.0;
                    extracted_to_raw.metadata = serde_json::json!({"session_id": session_id});
                    let _ = self.store.put_edge(&extracted_to_raw);
                }
            }
        }

        // Store session summary in mem_summaries
        if let Some(summary) = &session_summary {
            if let Some(sid) = batch_session_id {
                if let Ok(summary_embedding) = ctx.embedding_provider.embed_one(summary) {
                    let doc_date = ctx.temporal.map(|t| {
                        t.now_utc.timestamp_millis() as f64 / 1000.0
                    }).unwrap_or(0.0);
                    let _ = self.store.put_summary(
                        sid,
                        summary,
                        &summary_embedding,
                        &speakers_detected,
                        &[], // topics extracted from summary could be added later
                        doc_date,
                    );
                }
            }
        }

        Ok(serde_json::json!({
            "status": "extracted",
            "extracted": extracted.len(),
            "observations_extracted": observations.len(),
            "stored": stored.len(),
            "updated": updated,
            "skipped": skipped,
            "raw_preserved": raw_stored,
            "session_summary": session_summary,
            "speakers_detected": speakers_detected,
            "memories": stored,
        }))
    }

    // ─────────── Remaining tools (unchanged) ───────────

    fn search_memory(
        &self,
        args: &serde_json::Value,
        embedding_provider: &dyn EmbeddingProvider,
    ) -> Result<serde_json::Value> {
        let query = args["query"]
            .as_str()
            .ok_or_else(|| MemlocalError::InvalidArgument("missing 'query'".into()))?;
        let mode_str = args["mode"].as_str().unwrap_or("hybrid");
        let limit = args["limit"].as_u64().unwrap_or(20) as usize;
        let user_id = args["user_id"].as_str();
        let type_str = args["memory_type"].as_str();

        let mode = SearchMode::from_str_lossy(mode_str);
        let memory_type = type_str.map(MemoryType::from_stored_name);

        let start = std::time::Instant::now();

        let needs_embedding = matches!(
            mode,
            SearchMode::Semantic | SearchMode::Hybrid | SearchMode::Graph
        );
        let embedding = if needs_embedding {
            Some(embedding_provider.embed_one(query)?)
        } else {
            None
        };

        let mut items = self.store.search(
            query,
            embedding.as_deref(),
            mode,
            limit,
            user_id,
            memory_type,
        )?;

        // Temporal search: if date params provided, run temporal search and merge results
        let date_from = args["date_from"].as_str();
        let date_to = args["date_to"].as_str();

        if date_from.is_some() || date_to.is_some() {
            // Default date_from to epoch, date_to to far-future if only one is provided
            let df = date_from.unwrap_or("1970-01-01T00:00:00Z");
            let dt = date_to.unwrap_or("2099-12-31T23:59:59Z");

            if let Ok(temporal_items) = self.store.search_temporal(df, dt, limit, user_id) {
                let mut seen_ids: std::collections::HashSet<String> =
                    items.iter().map(|m| m.id.clone()).collect();
                for item in temporal_items {
                    if seen_ids.insert(item.id.clone()) {
                        items.push(item);
                    }
                }
            }
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(serde_json::json!({
            "results": items.iter().map(|m| serde_json::json!({
                "id": m.id,
                "content": m.content,
                "type": m.memory_type.stored_name(),
                "score": m.score,
                "created_at": m.created_at.to_rfc3339(),
            })).collect::<Vec<_>>(),
            "total": items.len(),
            "search_time_ms": duration_ms,
            "mode": mode_str,
        }))
    }

    fn get_memories(&self, args: &serde_json::Value) -> Result<serde_json::Value> {
        let user_id = args["user_id"].as_str();
        let type_str = args["memory_type"].as_str();
        let limit = args["limit"].as_u64().unwrap_or(20) as usize;

        let memory_type = type_str.map(MemoryType::from_stored_name);
        let items = self.store.get_memories(user_id, memory_type, limit)?;

        Ok(serde_json::json!({
            "memories": items.iter().map(|m| serde_json::json!({
                "id": m.id,
                "content": m.content,
                "type": m.memory_type.stored_name(),
                "created_at": m.created_at.to_rfc3339(),
            })).collect::<Vec<_>>(),
            "total": items.len(),
        }))
    }

    fn delete_memory(&self, args: &serde_json::Value) -> Result<serde_json::Value> {
        let memory_id = args["memory_id"]
            .as_str()
            .ok_or_else(|| MemlocalError::InvalidArgument("missing 'memory_id'".into()))?;
        self.store.delete_memory(memory_id)?;
        Ok(serde_json::json!({
            "status": "deleted",
            "memory_id": memory_id,
        }))
    }

    fn get_profile(&self, args: &serde_json::Value) -> Result<serde_json::Value> {
        let user_id = args["user_id"]
            .as_str()
            .ok_or_else(|| MemlocalError::InvalidArgument("missing 'user_id'".into()))?;
        let profile = self.store.get_profile(user_id)?;
        match profile {
            Some(p) => Ok(serde_json::json!({
                "user_id": p.user_id,
                "summary": p.to_summary(),
                "static_facts": p.static_facts,
                "dynamic_context": p.dynamic_context,
            })),
            None => Ok(serde_json::json!({
                "status": "not_found",
                "user_id": user_id,
            })),
        }
    }

    fn add_relationship(&self, args: &serde_json::Value) -> Result<serde_json::Value> {
        let from_id = args["from_id"]
            .as_str()
            .ok_or_else(|| MemlocalError::InvalidArgument("missing 'from_id'".into()))?;
        let to_id = args["to_id"]
            .as_str()
            .ok_or_else(|| MemlocalError::InvalidArgument("missing 'to_id'".into()))?;
        let relation_str = args["relation"]
            .as_str()
            .ok_or_else(|| MemlocalError::InvalidArgument("missing 'relation'".into()))?;
        let weight = args["weight"].as_f64().unwrap_or(1.0);

        let relation = MemoryRelation::from_stored_name(relation_str);
        let edge = MemoryEdge {
            from_id: from_id.to_string(),
            to_id: to_id.to_string(),
            relation,
            weight,
            metadata: serde_json::Value::Object(Default::default()),
            created_at: Utc::now(),
        };

        self.store.put_edge(&edge)?;

        Ok(serde_json::json!({
            "status": "created",
            "from_id": from_id,
            "to_id": to_id,
            "relation": relation.stored_name(),
        }))
    }

    fn get_relationships(&self, args: &serde_json::Value) -> Result<serde_json::Value> {
        let memory_id = args["memory_id"]
            .as_str()
            .ok_or_else(|| MemlocalError::InvalidArgument("missing 'memory_id'".into()))?;

        let from = self.store.get_edges_from(memory_id)?;
        let to = self.store.get_edges_to(memory_id)?;

        Ok(serde_json::json!({
            "memory_id": memory_id,
            "outgoing": from.iter().map(|e| serde_json::json!({
                "to": e.to_id,
                "relation": e.relation.stored_name(),
                "weight": e.weight,
            })).collect::<Vec<_>>(),
            "incoming": to.iter().map(|e| serde_json::json!({
                "from": e.from_id,
                "relation": e.relation.stored_name(),
                "weight": e.weight,
            })).collect::<Vec<_>>(),
        }))
    }

    fn add_reminder(
        &self,
        args: &serde_json::Value,
        embedding_provider: &dyn EmbeddingProvider,
    ) -> Result<serde_json::Value> {
        let content = args["content"]
            .as_str()
            .ok_or_else(|| MemlocalError::InvalidArgument("missing 'content'".into()))?;
        let trigger_type_str = args["trigger_type"]
            .as_str()
            .ok_or_else(|| MemlocalError::InvalidArgument("missing 'trigger_type'".into()))?;
        let trigger_condition = args["trigger_condition"]
            .as_str()
            .ok_or_else(|| MemlocalError::InvalidArgument("missing 'trigger_condition'".into()))?;
        let user_id = args["user_id"].as_str().map(String::from);

        let trigger_type = TriggerType::from_stored_name(trigger_type_str);
        let id = Uuid::new_v4().to_string();

        let item = ProspectiveItem {
            id: id.clone(),
            content: content.to_string(),
            trigger_type,
            trigger_condition: trigger_condition.to_string(),
            user_id: user_id.clone(),
            completed: false,
            created_at: Some(Utc::now()),
            completed_at: None,
        };

        self.store.put_prospective(&item)?;

        let memory_text = if trigger_type == TriggerType::SemanticMatch {
            trigger_condition.to_string()
        } else {
            format!("Reminder: {content} (trigger: {trigger_condition})")
        };

        let memory_item = MemoryItem {
            id: Uuid::new_v4().to_string(),
            content: format!("Reminder: {content} (trigger: {trigger_condition})"),
            memory_type: MemoryType::Prospective,
            hash: MemoryItem::compute_hash(content),
            user_id,
            agent_id: None,
            session_id: None,
            metadata: serde_json::json!({
                "reminder_id": id,
                "trigger_type": trigger_type_str,
                "trigger_condition": trigger_condition,
            }),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            valid_at: None,
            invalid_at: None,
            score: None,
        };

        let embedding = embedding_provider.embed_one(&memory_text)?;
        self.store.put_memory(&memory_item, &embedding)?;

        Ok(serde_json::json!({
            "status": "created",
            "reminder_id": id,
            "trigger_type": trigger_type_str,
        }))
    }

    fn get_context(
        &self,
        args: &serde_json::Value,
        embedding_provider: &dyn EmbeddingProvider,
    ) -> Result<serde_json::Value> {
        let query = args["query"]
            .as_str()
            .ok_or_else(|| MemlocalError::InvalidArgument("missing 'query'".into()))?;
        let user_id = args["user_id"].as_str();

        let start = std::time::Instant::now();

        let embedding = embedding_provider.embed_one(query)?;
        let memories = self
            .store
            .search_hybrid(query, &embedding, 20, user_id, None)?;

        let profile = if let Some(uid) = user_id {
            self.store.get_profile(uid)?
        } else {
            None
        };

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(serde_json::json!({
            "relevant_memories": memories.iter().map(|m| serde_json::json!({
                "id": m.id,
                "content": m.content,
                "type": m.memory_type.stored_name(),
                "score": m.score,
            })).collect::<Vec<_>>(),
            "user_profile": profile.as_ref().map(|p| p.to_summary()),
            "retrieval_time_ms": duration_ms,
        }))
    }

    // ─────────── request_more_context ───────────

    fn request_more_context(
        &self,
        args: &serde_json::Value,
        ctx: &ExecutionContext,
    ) -> Result<serde_json::Value> {
        let query = args["refined_query"]
            .as_str()
            .ok_or_else(|| MemlocalError::InvalidArgument("missing 'refined_query'".into()))?;
        let reason = args.get("reason").and_then(|v| v.as_str()).unwrap_or("");

        let context = self.prepare_context(
            query,
            ctx.embedding_provider,
            None,
            Some(15),
            false,
        )?;

        let context = if let Some(reranker) = ctx.reranker_provider {
            self.prepare_context_reranked(
                query,
                ctx.embedding_provider,
                None,
                Some(15),
                false,
                Some(reranker),
            )?
        } else {
            context
        };

        Ok(serde_json::json!({
            "status": "additional_context_retrieved",
            "query": query,
            "reason": reason,
            "context": context,
        }))
    }

    // ─────────── prepare_context_iterative ───────────

    /// Iterative retrieval: runs 1-2 rounds, using LLM to assess context sufficiency.
    /// MemMachine's agent-mode pattern: if first retrieval is insufficient, refine queries.
    pub fn prepare_context_iterative(
        &self,
        query: &str,
        embedding_provider: &dyn EmbeddingProvider,
        llm_provider: &dyn LlmProvider,
        user_id: Option<&str>,
        max_results: Option<usize>,
    ) -> Result<PreparedContext> {
        self.prepare_context_iterative_reranked(
            query,
            embedding_provider,
            llm_provider,
            user_id,
            max_results,
            None,
        )
    }

    pub fn prepare_context_iterative_reranked(
        &self,
        query: &str,
        embedding_provider: &dyn EmbeddingProvider,
        llm_provider: &dyn LlmProvider,
        user_id: Option<&str>,
        max_results: Option<usize>,
        reranker: Option<&dyn RerankerProvider>,
    ) -> Result<PreparedContext> {
        // Round 1: Standard retrieval
        let round1 = self.prepare_context_with_diagnostics_reranked(
            query,
            embedding_provider,
            user_id,
            max_results,
            false,
            reranker,
        )?;

        // Ask the LLM: is the context sufficient?
        let sufficiency_prompt = format!(
            "Query: {}\n\nRetrieved context:\n{}\n\n\
             Is this sufficient to answer the query? \
             If NOT, output a JSON array of 1-3 follow-up search queries \
             that would help fill gaps. If sufficient, output an empty array [].",
            query,
            // Truncate context to avoid blowing up token count
            &round1.context_block[..round1.context_block.len().min(3000)]
        );

        let refinement = llm_provider.complete(
            "You are a retrieval sufficiency checker. Output ONLY a JSON array of strings. \
             No explanation, no markdown.",
            &sufficiency_prompt
        )?;

        // Parse refinement queries
        let refinement_text = refinement.trim();
        let refinement_text = if refinement_text.starts_with("```") {
            refinement_text
                .lines()
                .skip(1)
                .take_while(|l| !l.starts_with("```"))
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            refinement_text.to_string()
        };

        let sub_queries: Vec<String> = serde_json::from_str(&refinement_text)
            .unwrap_or_default();

        if sub_queries.is_empty() {
            return Ok(round1);
        }

        // Round 2: Search with refined queries
        let mut combined_ids: HashSet<String> = round1.diagnostics.ranked_memories
            .iter()
            .map(|m| m.id.clone())
            .collect();
        let mut additional_memories = Vec::new();

        for sq in &sub_queries {
            let round2 = self.prepare_context_with_diagnostics_reranked(
                sq,
                embedding_provider,
                user_id,
                Some(10),
                false,
                reranker,
            )?;
            for mem in round2.diagnostics.ranked_memories {
                if combined_ids.insert(mem.id.clone()) {
                    additional_memories.push(mem);
                }
            }
        }

        // If round 2 found new stuff, rebuild context with merged results
        if additional_memories.is_empty() {
            return Ok(round1);
        }

        // Merge: combine round1 memories + additional round2 memories
        let mut all_ranked = round1.diagnostics.ranked_memories;
        all_ranked.extend(additional_memories);

        // Re-sort by score and truncate
        all_ranked.sort_by(|a, b| {
            b.score.unwrap_or(f64::MIN)
                .partial_cmp(&a.score.unwrap_or(f64::MIN))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let limit = max_results.unwrap_or(30);
        all_ranked.truncate(limit);

        // Rebuild context block using the existing pipeline
        // (Simplest approach: run prepare_context one more time with a combined query)
        let combined_query = format!("{} {}", query, sub_queries.join(" "));
        let final_context = self.prepare_context_with_diagnostics_reranked(
            &combined_query,
            embedding_provider,
            user_id,
            max_results,
            false,
            reranker,
        )?;

        Ok(PreparedContext {
            context_block: final_context.context_block,
            diagnostics: PrepareContextDiagnostics {
                query: query.to_string(),
                sub_queries,
                total_unique_retrieved: all_ranked.len(),
                context_limit: limit,
                truncated_count: 0,
                source_counts: final_context.diagnostics.source_counts,
                ranked_memories: all_ranked,
                reserved_keyword_match_ids: final_context.diagnostics.reserved_keyword_match_ids,
                context_char_count: final_context.diagnostics.context_char_count,
            },
        })
    }
}

fn detect_query_speaker(query: &str, known_speakers: &[String]) -> Option<String> {
    let query_lower = query.to_lowercase();
    for speaker in known_speakers {
        if query_lower.contains(&speaker.to_lowercase()) {
            return Some(speaker.clone());
        }
    }
    None
}

fn classify_retrieval_query(query: &str) -> RetrievalQueryComplexity {
    let lower = query.to_lowercase();

    if lower.starts_with("would ")
        || lower.starts_with("could ")
        || lower.starts_with("should ")
        || lower.contains(" likely ")
        || lower.contains("consider")
        || lower.contains("want to")
    {
        return RetrievalQueryComplexity::OpenEnded;
    }

    if is_temporal_query(query) {
        return RetrievalQueryComplexity::Temporal;
    }

    if lower.starts_with("when ")
        || lower.starts_with("how long")
        || lower.contains(" before ")
        || lower.contains(" after ")
        || lower.contains(" during ")
        || lower.contains(" both ")
    {
        return RetrievalQueryComplexity::MultiHop;
    }

    RetrievalQueryComplexity::SingleHop
}

fn retrieval_strategy(query: &str) -> (usize, usize) {
    match classify_retrieval_query(query) {
        RetrievalQueryComplexity::SingleHop => (80, 15),
        RetrievalQueryComplexity::MultiHop => (120, 18),
        RetrievalQueryComplexity::Temporal => (140, 24),
        RetrievalQueryComplexity::OpenEnded => (100, 15),
    }
}

fn resolve_rerank_pool_size(query: &str) -> usize {
    retrieval_strategy(query).0
}

fn resolve_context_limit(query: &str, max_results: Option<usize>) -> usize {
    let default_limit = retrieval_strategy(query).1;
    max_results.map(|limit| limit.min(default_limit)).unwrap_or(default_limit)
}

fn build_query_variants(query: &str) -> Vec<String> {
    let mut variants = Vec::new();
    push_query_variant(&mut variants, query.to_string());

    for part in decompose_query(query) {
        push_query_variant(&mut variants, part);
    }

    if let Some(distilled) = distill_query(query) {
        push_query_variant(&mut variants, distilled);
    }

    if let Some(subject_focus) = build_subject_focus_query(query) {
        push_query_variant(&mut variants, subject_focus);
    }

    if let Some(temporal_focus) = build_temporal_focus_query(query) {
        push_query_variant(&mut variants, temporal_focus);
    }

    for temporal_window in build_temporal_window_queries(query) {
        push_query_variant(&mut variants, temporal_window);
    }

    variants
}

fn push_query_variant(variants: &mut Vec<String>, candidate: String) {
    let trimmed = candidate.trim();
    if trimmed.len() < 4 {
        return;
    }

    let normalized = normalize_query_for_dedup(trimmed);
    if variants
        .iter()
        .any(|existing| normalize_query_for_dedup(existing) == normalized)
    {
        return;
    }

    variants.push(trimmed.to_string());
}

fn normalize_query_for_dedup(query: &str) -> String {
    query
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Decompose a multi-topic query into sub-queries for broader retrieval.
/// "What's Rahul's job and his fitness goal?" → ["Rahul's job", "Rahul's fitness goal"]
fn decompose_query(query: &str) -> Vec<String> {
    let stem = query
        .split(['?', '.'])
        .next()
        .unwrap_or(query);

    let parts: Vec<&str> = stem
        .split(" and ")
        .flat_map(|s| s.split(" also "))
        .flat_map(|s| s.split(" as well as "))
        .map(|s| s.trim())
        .filter(|s| s.len() > 5)
        .collect();

    if parts.len() > 1 {
        parts.into_iter().map(String::from).collect()
    } else {
        vec![query.to_string()]
    }
}

fn distill_query(query: &str) -> Option<String> {
    let distilled = query
        .split_whitespace()
        .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|word| !word.is_empty())
        .filter(|word| {
            let lower = word.to_lowercase();
            word.len() > 2 && !QUERY_STOPWORDS.contains(&lower.as_str())
        })
        .collect::<Vec<_>>()
        .join(" ");

    if distilled.is_empty() {
        None
    } else {
        Some(distilled)
    }
}

fn build_subject_focus_query(query: &str) -> Option<String> {
    let names = extract_name_entities(query);
    let keywords = extract_keywords(query);

    if names.is_empty() || keywords.is_empty() {
        return None;
    }

    Some(format!("{} {}", names.join(" "), keywords))
}

fn build_temporal_focus_query(query: &str) -> Option<String> {
    if !is_temporal_query(query) {
        return None;
    }

    let mut parts = extract_name_entities(query);
    let keywords = extract_keywords(query);
    if !keywords.is_empty() {
        parts.push(keywords);
    }

    if is_future_leaning_query(query) {
        parts.push("future plan upcoming reminder".to_string());
    } else {
        parts.push("date time event timeline".to_string());
    }

    Some(parts.join(" "))
}

fn build_temporal_window_queries(query: &str) -> Vec<String> {
    if !is_temporal_query(query) {
        return Vec::new();
    }

    const MONTHS: &[(&str, &str)] = &[
        ("January", "january"),
        ("February", "february"),
        ("March", "march"),
        ("April", "april"),
        ("May", "may"),
        ("June", "june"),
        ("July", "july"),
        ("August", "august"),
        ("September", "september"),
        ("October", "october"),
        ("November", "november"),
        ("December", "december"),
    ];

    let lower = query.to_lowercase();
    let years: Vec<String> = query
        .split_whitespace()
        .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|word| word.len() == 4 && word.chars().all(|ch| ch.is_ascii_digit()))
        .map(|word| word.to_string())
        .collect();

    let mut variants = Vec::new();

    for (month_title, month_lower) in MONTHS {
        if lower.contains(month_lower) {
            if years.is_empty() {
                variants.push(format!("{} date event timeline", month_title));
            } else {
                for year in &years {
                    variants.push(format!("{} {} date event timeline", month_title, year));
                }
            }
        }
    }

    if lower.contains("summer") {
        if years.is_empty() {
            variants.push("June July August summer timeline".to_string());
        } else {
            for year in &years {
                variants.push(format!("June July August {} summer timeline", year));
            }
        }
    }

    if lower.contains("spring") {
        if years.is_empty() {
            variants.push("March April May spring timeline".to_string());
        } else {
            for year in &years {
                variants.push(format!("March April May {} spring timeline", year));
            }
        }
    }

    if lower.contains("winter") {
        if years.is_empty() {
            variants.push("December January February winter timeline".to_string());
        } else {
            for year in &years {
                variants.push(format!("December January February {} winter timeline", year));
            }
        }
    }

    if lower.contains("fall") || lower.contains("autumn") {
        if years.is_empty() {
            variants.push("September October November fall timeline".to_string());
        } else {
            for year in &years {
                variants.push(format!("September October November {} fall timeline", year));
            }
        }
    }

    for year in &years {
        variants.push(format!("{} date event timeline", year));
    }

    variants
}

/// Check if two similar memories have conflicting specific values (numbers, dates, times).
/// Used for contradiction detection at storage time.
fn has_conflicting_specifics(old: &str, new: &str) -> bool {
    let extract_nums = |s: &str| -> Vec<String> {
        s.split_whitespace()
            .filter(|w| w.chars().any(|c| c.is_ascii_digit()))
            .map(|w| w.to_string())
            .collect()
    };
    let old_nums = extract_nums(old);
    let new_nums = extract_nums(new);
    // Both have numeric specifics but they differ → likely a contradiction
    !old_nums.is_empty() && !new_nums.is_empty() && old_nums != new_nums
}

/// Standard English stopwords for query analysis. Used to identify meaningful
/// terms when constructing BM25 search phrases. This is NOT a BM25 filter
/// (CozoDB handles that via IDF) — it's for extracting action words and
/// compound phrases from user questions.
const QUERY_STOPWORDS: &[&str] = &[
    "what", "when", "where", "who", "how", "did", "does", "was", "were", "is",
    "are", "the", "a", "an", "in", "on", "at", "to", "for", "of", "with",
    "and", "or", "but", "not", "do", "has", "have", "had", "will", "would",
    "could", "should", "that", "this", "from", "been", "being", "her", "his",
    "their", "its", "she", "he", "they", "you", "any", "some", "many", "much",
    "more", "most", "very", "also", "just", "about", "into", "than", "then",
    "there", "here",
];

/// Parse a datetime from a `[Session N, H:MM am/pm on D Month, YYYY]` header.
/// Returns `Some(DateTime<Utc>)` if the pattern matches.
fn parse_session_datetime(text: &str) -> Option<DateTime<Utc>> {
    // Look for pattern: [Session N, TIME on DATE]
    let bracket_start = text.find('[')?;
    let bracket_end = text.find(']')?;
    let header = &text[bracket_start + 1..bracket_end];

    // Split on ", " to get "Session N" and "TIME on DATE"
    let comma_pos = header.find(", ")?;
    let datetime_part = &header[comma_pos + 2..];

    // Split "H:MM am/pm on D Month, YYYY"
    let on_pos = datetime_part.find(" on ")?;
    let time_part = &datetime_part[..on_pos].trim();
    let date_part = &datetime_part[on_pos + 4..].trim();

    // Parse time
    let time_tokens: Vec<&str> = time_part.split_whitespace().collect();
    if time_tokens.len() != 2 {
        return None;
    }
    let hm: Vec<&str> = time_tokens[0].split(':').collect();
    if hm.len() != 2 {
        return None;
    }
    let mut hour: u32 = hm[0].parse().ok()?;
    let minute: u32 = hm[1].parse().ok()?;
    match time_tokens[1].to_lowercase().as_str() {
        "am" => {
            if hour == 12 {
                hour = 0;
            }
        }
        "pm" => {
            if hour != 12 {
                hour += 12;
            }
        }
        _ => return None,
    }

    // Parse date "D Month, YYYY"
    let date_clean = date_part.replace(',', "");
    let date_tokens: Vec<&str> = date_clean.split_whitespace().collect();
    if date_tokens.len() != 3 {
        return None;
    }
    let day: u32 = date_tokens[0].parse().ok()?;
    let month: u32 = match date_tokens[1].to_lowercase().as_str() {
        "january" | "jan" => 1,
        "february" | "feb" => 2,
        "march" | "mar" => 3,
        "april" | "apr" => 4,
        "may" => 5,
        "june" | "jun" => 6,
        "july" | "jul" => 7,
        "august" | "aug" => 8,
        "september" | "sep" => 9,
        "october" | "oct" => 10,
        "november" | "nov" => 11,
        "december" | "dec" => 12,
        _ => return None,
    };
    let year: i32 = date_tokens[2].parse().ok()?;

    use chrono::NaiveDate;
    let date = NaiveDate::from_ymd_opt(year, month, day)?;
    let time = chrono::NaiveTime::from_hms_opt(hour, minute, 0)?;
    let naive = chrono::NaiveDateTime::new(date, time);
    Some(DateTime::from_naive_utc_and_offset(naive, Utc))
}

/// Extract entity terms from a query for targeted BM25 search.
/// Produces compound phrases (proper noun + action word) for precise matching.
/// E.g., "What did Melanie paint recently?" → ["Melanie", "Melanie paint", "Melanie recently", "paint", "recently"]
fn extract_entities(query: &str) -> Vec<String> {
    let words: Vec<&str> = query
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.len() >= 2)
        .collect();

    let mut entities = Vec::new();
    let mut proper_nouns = Vec::new();
    let mut significant_words = Vec::new();

    for (i, word) in words.iter().enumerate() {
        let lower = word.to_lowercase();
        if QUERY_STOPWORDS.contains(&lower.as_str()) {
            continue;
        }

        // Proper nouns: capitalized, not first word
        if i > 0 && word.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
            proper_nouns.push(word.to_string());
        }

        // Significant words: >3 chars, not stopword
        if lower.len() > 3 {
            significant_words.push(lower);
        }
    }

    // 1. Compound phrases: each proper noun + each significant word
    for noun in &proper_nouns {
        entities.push(noun.clone()); // standalone proper noun
        for sig in &significant_words {
            if sig.to_lowercase() != noun.to_lowercase() {
                entities.push(format!("{} {}", noun, sig)); // "Melanie paint"
            }
        }
    }

    // 2. Significant words as standalone BM25 terms
    for sig in &significant_words {
        if !entities.iter().any(|e| e.to_lowercase() == *sig) {
            entities.push(sig.clone());
        }
    }

    entities
}

fn extract_name_entities(query: &str) -> Vec<String> {
    const NON_ENTITY_CAPS: &[&str] = &[
        "What", "When", "Where", "Who", "How", "Why", "Would", "Did", "Does", "Do",
        "Is", "Are", "Was", "Were", "Can", "Could", "Should", "May", "If", "In",
        "On", "At", "The", "A", "An",
    ];
    const MONTHS: &[&str] = &[
        "January", "February", "March", "April", "May", "June", "July", "August",
        "September", "October", "November", "December",
    ];

    let mut names = Vec::new();
    for (index, word) in query.split_whitespace().enumerate() {
        let cleaned = word.trim_matches(|c: char| !c.is_alphanumeric());
        if cleaned.len() < 2 {
            continue;
        }
        let first = match cleaned.chars().next() {
            Some(ch) => ch,
            None => continue,
        };
        if !first.is_uppercase() {
            continue;
        }
        if index == 0 && NON_ENTITY_CAPS.contains(&cleaned) {
            continue;
        }
        if NON_ENTITY_CAPS.contains(&cleaned) || MONTHS.contains(&cleaned) {
            continue;
        }
        if !names.iter().any(|name| name == cleaned) {
            names.push(cleaned.to_string());
        }
    }
    names
}

/// Extract keywords from a query by removing stopwords.
/// Returns a clean BM25 query string. E.g., "What did Melanie paint recently?" → "melanie paint recently"
fn extract_keywords(query: &str) -> String {
    query
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
        .filter(|w| w.len() > 2 && !QUERY_STOPWORDS.contains(&w.as_str()))
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_temporal_query(query: &str) -> bool {
    let lower = query.to_lowercase();
    lower.starts_with("when ")
        || lower.starts_with("how long")
        || lower.contains(" before ")
        || lower.contains(" after ")
        || lower.contains(" recently")
        || lower.contains(" last ")
        || lower.contains(" next ")
        || lower.contains(" soon")
        || lower.contains(" ago")
        || lower.contains(" summer")
        || lower.contains(" winter")
        || lower.contains(" spring")
        || lower.contains(" fall")
}

fn is_future_leaning_query(query: &str) -> bool {
    let lower = query.to_lowercase();
    lower.starts_with("would ")
        || lower.contains(" will ")
        || lower.contains(" going to ")
        || lower.contains(" planning ")
        || lower.contains(" plan ")
        || lower.contains(" soon")
        || lower.contains(" next ")
}

fn apply_query_aware_boosts(items: &mut [MemoryItem], query: &str) {
    let temporal_focus = is_temporal_query(query);
    let future_focus = is_future_leaning_query(query);
    let target_names = extract_name_entities(query);
    let query_lower = query.to_lowercase();

    for item in items {
        let mut boost = 1.0;
        let content_lower = item.content.to_lowercase();

        if temporal_focus {
            if item.valid_at.is_some() {
                boost *= 1.2;
            }
            if matches!(item.memory_type, MemoryType::Episodic) {
                boost *= 1.15;
            }
            if future_focus && matches!(item.memory_type, MemoryType::Prospective) {
                boost *= 1.25;
            }
            if is_raw_conversation_item(item) {
                boost *= 1.05;
            }

            if let Some(valid_at) = item.valid_at {
                let month_name = valid_at.format("%B").to_string().to_lowercase();
                let short_month = valid_at.format("%b").to_string().to_lowercase();
                let year = valid_at.format("%Y").to_string();

                if query_lower.contains(&month_name) || query_lower.contains(&short_month) {
                    boost *= 1.2;
                }
                if query_lower.contains(&year) {
                    boost *= 1.15;
                }

                if query_lower.contains("summer")
                    && matches!(valid_at.month(), 6 | 7 | 8)
                {
                    boost *= 1.15;
                }
                if query_lower.contains("spring")
                    && matches!(valid_at.month(), 3 | 4 | 5)
                {
                    boost *= 1.15;
                }
                if (query_lower.contains("fall") || query_lower.contains("autumn"))
                    && matches!(valid_at.month(), 9 | 10 | 11)
                {
                    boost *= 1.15;
                }
                if query_lower.contains("winter")
                    && matches!(valid_at.month(), 12 | 1 | 2)
                {
                    boost *= 1.15;
                }
            }
        }

        let target_match_count = target_names
            .iter()
            .filter(|name| content_lower.contains(&name.to_lowercase()))
            .count();
        if target_match_count > 0 {
            boost *= 1.0 + target_match_count as f64 * 0.15;
        }

        if boost != 1.0 {
            item.score = Some(item.score.unwrap_or(0.0) * boost);
        }
    }
}

fn is_raw_conversation_item(item: &MemoryItem) -> bool {
    item.metadata
        .get("source")
        .and_then(|value| value.as_str())
        .map(|value| value == "raw_conversation")
        .unwrap_or(false)
}

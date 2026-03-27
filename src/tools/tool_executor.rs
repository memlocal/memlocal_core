use std::sync::Arc;

use chrono::{DateTime, Utc};
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

/// Optional providers passed to `execute`. Only `add_memories` needs the LLM provider.
pub struct ExecutionContext<'a> {
    pub embedding_provider: &'a dyn EmbeddingProvider,
    pub llm_provider: Option<&'a dyn LlmProvider>,
    pub temporal: Option<&'a TemporalContext>,
}

impl<'a> ExecutionContext<'a> {
    /// Minimal context (no LLM, no temporal) — sufficient for all tools except `add_memories`.
    pub fn new(embedding_provider: &'a dyn EmbeddingProvider) -> Self {
        Self {
            embedding_provider,
            llm_provider: None,
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
            temporal: Some(temporal),
        }
    }
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
        // v5: Query decomposition — split multi-topic queries and search each
        let sub_queries = decompose_query(query);
        let mut all_memories = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();

        for sq in &sub_queries {
            if !bm25_only {
                let emb = embedding_provider.embed_one(sq)?;

                // 1. Per-type retrieval (Engram-style: each type retrieves independently)
                let type_searches = [
                    (MemoryType::Episodic, 15),
                    (MemoryType::Factual, 10),
                    (MemoryType::Semantic, 8),
                    (MemoryType::Social, 5),
                ];

                for (mem_type, k) in &type_searches {
                    let results =
                        self.store
                            .search_hybrid_deduped(sq, &emb, *k, user_id, Some(*mem_type))?;
                    for item in results {
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
                    if seen_ids.insert(item.id.clone()) {
                        all_memories.push(item);
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
                    if seen_ids.insert(item.id.clone()) {
                        all_memories.push(item);
                    }
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
                let results = self.store.search_text(entity, 3)?;
                for item in results {
                    if let Some(uid) = user_id {
                        if item.user_id.as_deref() != Some(uid) {
                            continue;
                        }
                    }
                    if bm25_reserved_ids.insert(item.id.clone()) {
                        bm25_reserved.push(item);
                    }
                }
            }
        }
        // Cap reserved slots at 10 to avoid flooding context
        bm25_reserved.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        bm25_reserved.truncate(10);

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

        // Sort by boosted score, keep top results
        all_memories.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let limit = max_results.unwrap_or(30);
        all_memories.truncate(limit);

        // --- Option C: BM25 reserved hits go into a separate priority section ---
        // These are displayed as "KEY FACTS" at the top of the context block
        let keyword_matches: Vec<MemoryItem> = bm25_reserved
            .into_iter()
            .filter(|item| !seen_ids.contains(&item.id))
            .collect();

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

        // Assemble using WorkingMemory
        let mut wm = crate::shortterm::WorkingMemory::new();
        wm.set_keyword_matches(keyword_matches);
        wm.set_relevant(all_memories);
        wm.set_important(important);
        wm.set_profile(profile);
        wm.set_triggered_reminders(prospective);

        Ok(wm.to_context_block())
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

        // 1. Exact dedup by content hash
        let hash = MemoryItem::compute_hash(content);
        if let Some(existing) = self.store.find_by_hash(&hash, user_id.as_deref())? {
            return Ok(serde_json::json!({
                "status": "duplicate",
                "memory_id": existing.id,
                "reason": "exact content match"
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
                let updated = MemoryItem {
                    id: existing.id.clone(),
                    content: content.to_string(),
                    memory_type,
                    hash: hash.clone(),
                    user_id: user_id.clone(),
                    agent_id: None,
                    session_id: None,
                    metadata: serde_json::json!({"confidence": confidence}),
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
        let item = MemoryItem {
            id: id.clone(),
            content: content.to_string(),
            memory_type,
            hash,
            user_id,
            agent_id: None,
            session_id: None,
            metadata: serde_json::json!({"confidence": confidence}),
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

        let extracted: Vec<serde_json::Value> = serde_json::from_str(&json_text).map_err(|e| {
            MemlocalError::Internal(format!("Failed to parse extraction response: {e}"))
        })?;

        let mut stored = Vec::new();
        let mut skipped = 0;
        let mut updated = 0;

        for item in &extracted {
            let content = match item["content"].as_str() {
                Some(c) if !c.is_empty() => c,
                _ => continue,
            };
            let type_str = item["type"].as_str().unwrap_or("factual");
            let confidence = item["confidence"].as_f64().unwrap_or(0.8);

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
            if let Some(va) = item.get("valid_at") {
                add_args["valid_at"] = va.clone();
            }
            if let Some(ia) = item.get("invalid_at") {
                add_args["invalid_at"] = ia.clone();
            }

            // Delegate to add_memory (which handles dedup)
            let result = self.add_memory(&add_args, ctx)?;
            match result["status"].as_str() {
                Some("stored") => stored.push(result),
                Some("updated") => updated += 1,
                Some("duplicate") => skipped += 1,
                _ => stored.push(result),
            }
        }

        // Dual-layer: also store raw text segments as Episodic memories
        let mut raw_stored = 0;
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

                let seg_item = MemoryItem {
                    id: seg_id,
                    content: segment.to_string(),
                    memory_type: MemoryType::Episodic,
                    hash: seg_hash,
                    user_id: user_id.map(String::from),
                    agent_id: None,
                    session_id: None,
                    metadata: serde_json::json!({"source": "raw_conversation"}),
                    created_at: now,
                    updated_at: now,
                    valid_at: None,
                    invalid_at: None,
                    score: None,
                };

                self.store.put_memory(&seg_item, &seg_embedding)?;
                raw_stored += 1;
            }
        }

        Ok(serde_json::json!({
            "status": "extracted",
            "extracted": extracted.len(),
            "stored": stored.len(),
            "updated": updated,
            "skipped": skipped,
            "raw_preserved": raw_stored,
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
}

/// Decompose a multi-topic query into sub-queries for broader retrieval.
/// "What's Rahul's job and his fitness goal?" → ["Rahul's job", "Rahul's fitness goal"]
fn decompose_query(query: &str) -> Vec<String> {
    let lower = query.to_lowercase();
    // Split on common conjunctions
    let parts: Vec<&str> = lower
        .split(['?', '.'])
        .next()
        .unwrap_or(&lower)
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

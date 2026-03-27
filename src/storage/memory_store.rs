use std::collections::{BTreeMap, HashMap};

use cozo::{DataValue, DbInstance, NamedRows, ScriptMutability};
use log::debug;

use super::schema::MemorySchema;
use crate::error::{MemlocalError, Result};
use crate::models::*;

/// Low-level storage layer backed by CozoDB.
pub struct MemoryStore {
    db: DbInstance,
    embedding_dim: u32,
    enable_time_decay: bool,
    initialized: bool,
}

impl MemoryStore {
    // ──────────── Lifecycle ────────────

    /// Open and initialize the memory store.
    pub fn open(config: &StorageConfig, embedding_dim: u32) -> Result<Self> {
        let db = if config.in_memory {
            debug!("[MemoryStore] Opening in-memory database");
            DbInstance::new("mem", "", "")
                .map_err(|e| MemlocalError::Database(format!("Failed to open in-memory DB: {e}")))?
        } else {
            let path = config.db_path.as_deref().unwrap_or("memlocal.db");
            debug!("[MemoryStore] Opening SQLite database at: {path}");
            DbInstance::new("sqlite", path, "")
                .map_err(|e| MemlocalError::Database(format!("Failed to open SQLite DB: {e}")))?
        };

        let mut store = Self {
            db,
            embedding_dim,
            enable_time_decay: config.enable_time_decay,
            initialized: false,
        };
        store.ensure_schema(config)?;
        debug!("[MemoryStore] Database ready");
        Ok(store)
    }

    fn ensure_schema(&mut self, config: &StorageConfig) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        // Smoke test
        self.run_immutable("?[x] <- [[1]]", BTreeMap::new())?;
        debug!("[MemoryStore] DB smoke-test passed");

        // Create relations
        let statements = MemorySchema::create_statements(
            self.embedding_dim,
            config.hnsw_m,
            config.hnsw_ef_construction,
        );
        for (i, stmt) in statements.iter().enumerate() {
            debug!("[MemoryStore] Running statement {i} ({} chars)", stmt.len());
            self.try_run(stmt);
        }

        // Create indices
        self.try_run(&MemorySchema::create_vector_index(
            self.embedding_dim,
            config.hnsw_m,
            config.hnsw_ef_construction,
        ));
        self.try_run(&MemorySchema::create_fts_index());
        self.try_run(&MemorySchema::create_lsh_index());
        self.try_run(&MemorySchema::create_triples_fts_index());
        self.try_run(&MemorySchema::create_summaries_vector_index(
            self.embedding_dim,
            config.hnsw_m,
            config.hnsw_ef_construction,
        ));
        debug!("[MemoryStore] Indices ensured");

        // Verify
        let verify_script = format!("?[count(id)] := *{}{{id}}", MemorySchema::MEMORIES);
        self.run_immutable(&verify_script, BTreeMap::new())?;
        debug!("[MemoryStore] Schema verified");

        self.initialized = true;
        Ok(())
    }

    fn try_run(&self, script: &str) {
        if let Err(e) = self
            .db
            .run_script(script, BTreeMap::new(), ScriptMutability::Mutable)
        {
            debug!("[MemoryStore] try_run ignored: {e}");
        }
    }

    fn run_mutable(&self, script: &str, params: BTreeMap<String, DataValue>) -> Result<NamedRows> {
        self.db
            .run_script(script, params, ScriptMutability::Mutable)
            .map_err(|e| MemlocalError::Query(e.to_string()))
    }

    fn run_immutable(
        &self,
        script: &str,
        params: BTreeMap<String, DataValue>,
    ) -> Result<NamedRows> {
        self.db
            .run_script(script, params, ScriptMutability::Immutable)
            .map_err(|e| MemlocalError::Query(e.to_string()))
    }

    /// Close the database.
    pub fn close(&self) -> Result<()> {
        // CozoDB DbInstance drops automatically; nothing explicit needed.
        Ok(())
    }

    // ──────────── Memory CRUD ────────────

    /// Insert or update a memory item with its embedding.
    pub fn put_memory(&self, item: &MemoryItem, embedding: &[f32]) -> Result<()> {
        let m = item.to_map();
        let embedding_dv: Vec<DataValue> = embedding
            .iter()
            .map(|&f| DataValue::from(f as f64))
            .collect();

        // v4: Use ASSERT validity for time-travel support
        let script = format!(
            "?[id, vld, content, type, hash, user_id, agent_id, session_id, \
             speaker, document_date, \
             metadata_json, embedding, created_at, updated_at, event_start, event_end] <- \
             [[$id, 'ASSERT', $content, $type, $hash, $user_id, $agent_id, $session_id, \
             $speaker, $document_date, \
             $metadata_json, vec($embedding), $created_at, $updated_at, $event_start, $event_end]]\n\
             :put {} {{id, vld => content, type, hash, user_id, agent_id, session_id, \
             speaker, document_date, \
             metadata_json, embedding, created_at, updated_at, event_start, event_end}}",
            MemorySchema::MEMORIES
        );

        let params = BTreeMap::from([
            ("id".into(), DataValue::Str(json_str(&m["id"]).into())),
            (
                "content".into(),
                DataValue::Str(json_str(&m["content"]).into()),
            ),
            ("type".into(), DataValue::Str(json_str(&m["type"]).into())),
            ("hash".into(), DataValue::Str(json_str(&m["hash"]).into())),
            (
                "user_id".into(),
                DataValue::Str(json_str(&m["user_id"]).into()),
            ),
            (
                "agent_id".into(),
                DataValue::Str(json_str(&m["agent_id"]).into()),
            ),
            (
                "session_id".into(),
                DataValue::Str(json_str(&m["session_id"]).into()),
            ),
            ("speaker".into(), DataValue::Str(
                item.metadata.get("speaker")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .into()
            )),
            ("document_date".into(), DataValue::from(
                item.metadata.get("document_date")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0)
            )),
            (
                "metadata_json".into(),
                DataValue::Str(json_str(&m["metadata_json"]).into()),
            ),
            ("embedding".into(), DataValue::List(embedding_dv)),
            (
                "created_at".into(),
                DataValue::from(m["created_at"].as_f64().unwrap_or(0.0)),
            ),
            (
                "updated_at".into(),
                DataValue::from(m["updated_at"].as_f64().unwrap_or(0.0)),
            ),
            (
                "event_start".into(),
                DataValue::from(m["valid_at"].as_f64().unwrap_or(0.0)),
            ),
            (
                "event_end".into(),
                DataValue::from(m["invalid_at"].as_f64().unwrap_or(0.0)),
            ),
        ]);

        self.run_mutable(&script, params)?;
        Ok(())
    }

    /// Insert or update a batch of memory items.
    pub fn put_memories(&self, items: &[(&MemoryItem, &[f32])]) -> Result<()> {
        for (item, embedding) in items {
            self.put_memory(item, embedding)?;
        }
        Ok(())
    }

    /// Get a memory item by ID (current version via time-travel).
    pub fn get_memory(&self, id: &str) -> Result<Option<MemoryItem>> {
        let script = format!(
            "?[id, content, type, hash, user_id, agent_id, session_id, \
             speaker, document_date, \
             metadata_json, created_at, updated_at, valid_at, invalid_at] := \
             *{}{{id, content, type, hash, user_id, agent_id, session_id, \
             speaker, document_date, \
             metadata_json, created_at, updated_at, \
             event_start: valid_at, event_end: invalid_at, @ \"NOW\"}}, id == $id",
            MemorySchema::MEMORIES
        );
        let params = BTreeMap::from([("id".into(), DataValue::Str(id.into()))]);
        let result = self.run_immutable(&script, params)?;

        if result.rows.is_empty() {
            return Ok(None);
        }
        let map = rows_to_json(&result, 0);
        Ok(Some(MemoryItem::from_map(&map)?))
    }

    /// Get memories with optional filters.
    pub fn get_memories(
        &self,
        user_id: Option<&str>,
        memory_type: Option<MemoryType>,
        limit: usize,
    ) -> Result<Vec<MemoryItem>> {
        // v4: Use @ "NOW" time-travel instead of invalid_at == 0.0 filter
        let mut conditions = vec![format!(
            "*{}{{id, content, type, hash, user_id, agent_id, session_id, \
             speaker, document_date, \
             metadata_json, created_at, updated_at, \
             event_start: valid_at, event_end: invalid_at, @ \"NOW\"}}",
            MemorySchema::MEMORIES
        )];
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();

        if let Some(uid) = user_id {
            conditions.push("user_id == $uid".into());
            params.insert("uid".into(), DataValue::Str(uid.into()));
        }
        if let Some(mtype) = memory_type {
            conditions.push("type == $mtype".into());
            params.insert("mtype".into(), DataValue::Str(mtype.stored_name().into()));
        }

        let script = format!(
            "?[id, content, type, hash, user_id, agent_id, session_id, \
             speaker, document_date, \
             metadata_json, created_at, updated_at, valid_at, invalid_at] := \
             {}\n:order -updated_at\n:limit {}",
            conditions.join(", "),
            limit
        );

        let result = self.run_immutable(&script, params)?;
        rows_to_items(&result)
    }

    /// Get memories for a specific session, ordered by event time when available.
    pub fn get_memories_by_session(
        &self,
        session_id: &str,
        user_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<MemoryItem>> {
        let mut conditions = vec![format!(
            "*{}{{id, content, type, hash, user_id, agent_id, session_id, \
             speaker, document_date, \
             metadata_json, created_at, updated_at, \
             event_start: valid_at, event_end: invalid_at, @ \"NOW\"}}",
            MemorySchema::MEMORIES
        )];
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();

        conditions.push("session_id == $sid".into());
        params.insert("sid".into(), DataValue::Str(session_id.into()));

        if let Some(uid) = user_id {
            conditions.push("user_id == $uid".into());
            params.insert("uid".into(), DataValue::Str(uid.into()));
        }

        let script = format!(
            "?[id, content, type, hash, user_id, agent_id, session_id, \
             speaker, document_date, \
             metadata_json, created_at, updated_at, valid_at, invalid_at] := \
             {}\n:order valid_at, -updated_at\n:limit {}",
            conditions.join(", "),
            limit
        );

        let result = self.run_immutable(&script, params)?;
        rows_to_items(&result)
    }

    /// Invalidate (soft-delete) a memory using RETRACT validity.
    /// The memory's history is preserved — it just becomes invisible at "NOW".
    pub fn invalidate_memory(&self, id: &str) -> Result<()> {
        // RETRACT requires all value columns. Get current values, then retract.
        let script = format!(
            "?[id, vld, content, type, hash, user_id, agent_id, session_id, \
             speaker, document_date, \
             metadata_json, embedding, created_at, updated_at, event_start, event_end] := \
             *{rel}{{id, content, type, hash, user_id, agent_id, session_id, \
             speaker, document_date, \
             metadata_json, embedding, created_at, updated_at, event_start, event_end, @ \"NOW\"}}, \
             id == $id, vld = 'RETRACT'\n\
             :put {rel} {{id, vld => content, type, hash, user_id, agent_id, session_id, \
             speaker, document_date, \
             metadata_json, embedding, created_at, updated_at, event_start, event_end}}",
            rel = MemorySchema::MEMORIES
        );
        let params = BTreeMap::from([("id".into(), DataValue::Str(id.into()))]);
        self.run_mutable(&script, params)?;
        Ok(())
    }

    /// Hard-delete a memory by ID.
    /// Hard-delete uses RETRACT with Validity (preserves history but invisible at NOW).
    pub fn delete_memory(&self, id: &str) -> Result<()> {
        self.invalidate_memory(id)
    }

    /// Find existing memories that match by content hash (dedup check).
    pub fn find_by_hash(&self, hash: &str, user_id: Option<&str>) -> Result<Option<MemoryItem>> {
        let mut conditions = vec![format!(
            "*{}{{id, content, type, hash, user_id, agent_id, session_id, \
             speaker, document_date, \
             metadata_json, created_at, updated_at, \
             event_start: valid_at, event_end: invalid_at, @ \"NOW\"}}",
            MemorySchema::MEMORIES
        )];
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();

        conditions.push("hash == $h".into());
        params.insert("h".into(), DataValue::Str(hash.into()));

        if let Some(uid) = user_id {
            conditions.push("user_id == $uid".into());
            params.insert("uid".into(), DataValue::Str(uid.into()));
        }

        let script = format!(
            "?[id, content, type, hash, user_id, agent_id, session_id, \
             speaker, document_date, \
             metadata_json, created_at, updated_at, valid_at, invalid_at] := \
             {}\n:limit 1",
            conditions.join(", ")
        );

        let result = self.run_immutable(&script, params)?;
        if result.rows.is_empty() {
            return Ok(None);
        }
        let map = rows_to_json(&result, 0);
        Ok(Some(MemoryItem::from_map(&map)?))
    }

    /// Count total memories (current versions only via time-travel).
    pub fn memory_count(&self, memory_type: Option<MemoryType>) -> Result<usize> {
        let mut conditions = vec![format!(
            "*{}{{id, type, @ \"NOW\"}}",
            MemorySchema::MEMORIES
        )];
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();

        if let Some(mtype) = memory_type {
            conditions.push("type == $mtype".into());
            params.insert("mtype".into(), DataValue::Str(mtype.stored_name().into()));
        }

        let script = format!("?[count(id)] := {}", conditions.join(", "));
        let result = self.run_immutable(&script, params)?;
        if result.rows.is_empty() {
            return Ok(0);
        }
        Ok(dv_to_i64(&result.rows[0][0]) as usize)
    }

    // ──────────── Scoring helpers ────────────

    /// Apply confidence multiplier with sqrt dampening.
    fn apply_confidence(item: &MemoryItem, raw_score: f64) -> f64 {
        let confidence = item
            .metadata
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        raw_score * confidence.sqrt()
    }

    /// v5: Type-aware time decay factor.
    /// - Factual/semantic: low decay (lambda=0.002, half-life ~346 days) — facts persist
    /// - Episodic: medium decay (lambda=0.005, half-life ~138 days) — events age
    /// - Prospective: high decay (lambda=0.02, half-life ~35 days) — reminders expire fast
    fn time_decay_factor(&self, item: &MemoryItem) -> f64 {
        if !self.enable_time_decay {
            return 1.0;
        }
        let days_since = (chrono::Utc::now() - item.updated_at).num_days() as f64;
        let lambda = match item.memory_type {
            MemoryType::Factual | MemoryType::Semantic | MemoryType::Procedural => 0.002,
            MemoryType::Social => 0.003,
            MemoryType::Episodic | MemoryType::Spatial | MemoryType::Affective => 0.005,
            MemoryType::Prospective => 0.02,
            _ => 0.005,
        };
        (-lambda * days_since).exp()
    }

    /// Compute importance: 0.25*confidence + 0.25*reinforcement + 0.2*log_access + 0.3*recency.
    fn compute_importance(&self, item: &MemoryItem) -> f64 {
        let confidence = item
            .metadata
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.8);
        let access_count = item
            .metadata
            .get("access_count")
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as f64;
        let reinforcement = item.reinforcement_count() as f64;

        let reinforcement_factor = (1.0 + reinforcement).ln() / (1.0 + 10.0_f64).ln();
        let access_factor = (1.0 + access_count).ln() / (1.0 + 100.0_f64).ln();
        let recency_factor = self.time_decay_factor(item);

        0.25 * confidence + 0.25 * reinforcement_factor + 0.2 * access_factor + 0.3 * recency_factor
    }

    // ──────────── Search ────────────

    /// Semantic (vector) search.
    pub fn search_semantic(
        &self,
        query_embedding: &[f32],
        k: usize,
        user_id: Option<&str>,
        memory_type: Option<MemoryType>,
    ) -> Result<Vec<MemoryItem>> {
        let mut filter_parts = Vec::new();
        if let Some(uid) = user_id {
            filter_parts.push(format!("user_id == \"{}\"", escape_cozo_str(uid)));
        }
        if let Some(mtype) = memory_type {
            filter_parts.push(format!("type == \"{}\"", mtype.stored_name()));
        }
        // Note: with Validity schema, HNSW indexes all versions.
        // We still filter by user_id/type if specified.
        let filter = if filter_parts.is_empty() {
            String::new()
        } else {
            filter_parts.join(" && ")
        };

        let bind_fields = "id, content, type, hash, user_id, agent_id, session_id, \
                           speaker, document_date, \
                           metadata_json, created_at, updated_at, \
                           event_start: valid_at, event_end: invalid_at";
        let output_fields = "id, content, type, hash, user_id, agent_id, session_id, \
                             speaker, document_date, \
                             metadata_json, created_at, updated_at, valid_at, invalid_at";

        let filter_clause = if filter.is_empty() {
            String::new()
        } else {
            format!(", filter: {filter}")
        };
        let script = format!(
            "?[{output_fields}, distance] := ~{}:{}{{ {bind_fields} | \
             query: vec($q), k: {k}, ef: {ef}, bind_distance: distance{filter_clause} }}",
            MemorySchema::MEMORIES,
            MemorySchema::VECTOR_INDEX,
            ef = k * 2,
        );

        let embedding_dv: Vec<DataValue> = query_embedding
            .iter()
            .map(|&f| DataValue::from(f as f64))
            .collect();
        let params = BTreeMap::from([("q".into(), DataValue::List(embedding_dv))]);

        let result = self.run_immutable(&script, params)?;
        let mut items = Vec::new();
        for i in 0..result.rows.len() {
            let map = rows_to_json(&result, i);
            let item = MemoryItem::from_map(&map)?;
            let raw_score = 1.0 - map["distance"].as_f64().unwrap_or(0.0);
            let adjusted = Self::apply_confidence(&item, raw_score);
            items.push(item.with_score(adjusted));
        }
        Ok(items)
    }

    /// Full-text (BM25) search.
    pub fn search_text(&self, query: &str, k: usize) -> Result<Vec<MemoryItem>> {
        let sanitized = sanitize_fts_query(query);
        if sanitized.is_empty() {
            return Ok(vec![]);
        }

        let bind_fields = "id, content, type, hash, user_id, agent_id, session_id, \
                           speaker, document_date, \
                           metadata_json, created_at, updated_at, \
                           event_start: valid_at, event_end: invalid_at";
        let output_fields = "id, content, type, hash, user_id, agent_id, session_id, \
                             speaker, document_date, \
                             metadata_json, created_at, updated_at, valid_at, invalid_at";

        let script = format!(
            "?[{output_fields}, score] := ~{}:{}{{ {bind_fields} | \
             query: $q, k: {k}, bind_score: score }}",
            MemorySchema::MEMORIES,
            MemorySchema::FTS_INDEX,
        );

        let params = BTreeMap::from([("q".into(), DataValue::Str(sanitized.into()))]);
        let result = self.run_immutable(&script, params)?;

        let mut items = Vec::new();
        for i in 0..result.rows.len() {
            let map = rows_to_json(&result, i);
            let item = MemoryItem::from_map(&map)?;
            let raw_score = map["score"].as_f64().unwrap_or(0.0);
            items.push(item.with_score(raw_score));
        }
        Ok(items)
    }

    /// LSH (near-duplicate / Jaccard similarity) search.
    pub fn search_lsh(&self, query: &str, k: usize) -> Result<Vec<MemoryItem>> {
        let sanitized = sanitize_fts_query(query);
        if sanitized.is_empty() {
            return Ok(vec![]);
        }

        let bind_fields = "id, content, type, hash, user_id, agent_id, session_id, \
                           speaker, document_date, \
                           metadata_json, created_at, updated_at, \
                           event_start: valid_at, event_end: invalid_at";
        let output_fields = "id, content, type, hash, user_id, agent_id, session_id, \
                             speaker, document_date, \
                             metadata_json, created_at, updated_at, valid_at, invalid_at";

        let script = format!(
            "?[{output_fields}, score] := ~{}:{}{{ {bind_fields} | \
             query: $q, k: {k}, bind_score: score }}",
            MemorySchema::MEMORIES,
            MemorySchema::LSH_INDEX,
        );

        let params = BTreeMap::from([("q".into(), DataValue::Str(sanitized.into()))]);
        match self.run_immutable(&script, params) {
            Ok(result) => {
                let mut items = Vec::new();
                for i in 0..result.rows.len() {
                    let map = rows_to_json(&result, i);
                    let item = MemoryItem::from_map(&map)?;
                    let raw_score = map["score"].as_f64().unwrap_or(0.0);
                    items.push(item.with_score(raw_score));
                }
                Ok(items)
            }
            Err(_) => Ok(vec![]), // LSH may fail for short queries
        }
    }

    /// Hybrid search using Reciprocal Rank Fusion (RRF) across semantic, FTS, and LSH.
    pub fn search_hybrid(
        &self,
        query: &str,
        query_embedding: &[f32],
        k: usize,
        user_id: Option<&str>,
        memory_type: Option<MemoryType>,
    ) -> Result<Vec<MemoryItem>> {
        let semantic_list = self.search_semantic(query_embedding, k, user_id, memory_type)?;
        let text_list = self.search_text(query, k)?;
        let lsh_list = self.search_lsh(query, k)?;

        const K_RRF: f64 = 60.0;
        let mut rrf_scores: HashMap<String, f64> = HashMap::new();
        let mut item_map: HashMap<String, MemoryItem> = HashMap::new();

        let mut add_ranks = |list: &[MemoryItem]| {
            for (rank, item) in list.iter().enumerate() {
                *rrf_scores.entry(item.id.clone()).or_default() +=
                    1.0 / (K_RRF + rank as f64 + 1.0);
                item_map
                    .entry(item.id.clone())
                    .or_insert_with(|| item.clone());
            }
        };

        add_ranks(&semantic_list);
        // BM25 gets 2x weight in RRF — keyword precision is critical for factual recall
        add_ranks(&text_list);
        add_ranks(&text_list);
        add_ranks(&lsh_list);

        // Apply time decay + confidence boost to RRF scores
        let mut final_scores: Vec<(String, f64)> = rrf_scores
            .into_iter()
            .map(|(id, score)| {
                let item = &item_map[&id];
                let decayed = score * self.time_decay_factor(item);
                let with_confidence = Self::apply_confidence(item, decayed);
                (id, with_confidence)
            })
            .collect();

        final_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(final_scores
            .into_iter()
            .take(k)
            .filter_map(|(id, score)| item_map.remove(&id).map(|item| item.with_score(score)))
            .collect())
    }

    /// v5: Hybrid search with post-search dedup by recency.
    /// Fetches top-30, clusters by similarity, keeps newest per cluster, returns top-k.
    /// This prevents Claude from seeing conflicting versions of the same fact.
    pub fn search_hybrid_deduped(
        &self,
        query: &str,
        query_embedding: &[f32],
        k: usize,
        user_id: Option<&str>,
        memory_type: Option<MemoryType>,
    ) -> Result<Vec<MemoryItem>> {
        // Fetch more than needed for dedup headroom
        let raw = self.search_hybrid(query, query_embedding, k + 15, user_id, memory_type)?;

        if raw.len() <= 1 {
            return Ok(raw);
        }

        // Cluster by content similarity: if two results share >60% words, they're about the same topic
        let mut kept: Vec<MemoryItem> = Vec::new();
        for item in raw {
            let dominated = kept.iter().any(|existing| {
                word_overlap(&existing.content, &item.content) > 0.6
                    && existing.updated_at >= item.updated_at
            });
            if !dominated {
                // Remove any older item this one dominates
                kept.retain(|existing| {
                    !(word_overlap(&existing.content, &item.content) > 0.6
                        && item.updated_at > existing.updated_at)
                });
                kept.push(item);
            }
        }

        kept.truncate(k);
        Ok(kept)
    }

    /// Graph search: seeds from semantic, then traverses edges.
    pub fn search_graph(
        &self,
        query_embedding: &[f32],
        k: usize,
        user_id: Option<&str>,
        memory_type: Option<MemoryType>,
        hops: usize,
    ) -> Result<Vec<MemoryItem>> {
        let seed_count = k.clamp(1, 5);
        let seeds = self.search_semantic(query_embedding, seed_count, user_id, memory_type)?;

        if seeds.is_empty() {
            return Ok(vec![]);
        }

        let mut neighbor_scores: HashMap<String, f64> = HashMap::new();
        let mut neighbor_items: HashMap<String, MemoryItem> = HashMap::new();

        // Seed nodes get their semantic score
        for seed in &seeds {
            let score = seed.score.unwrap_or(0.0);
            neighbor_scores
                .entry(seed.id.clone())
                .and_modify(|s| *s = s.max(score))
                .or_insert(score);
            neighbor_items
                .entry(seed.id.clone())
                .or_insert_with(|| seed.clone());
        }

        for seed in &seeds {
            let seed_score = seed.score.unwrap_or(0.5);

            // Depth-1 neighbors
            let depth1 = self.get_neighbor_ids(&seed.id)?;
            for (to_id, weight) in &depth1 {
                let hop_score = seed_score * weight;
                neighbor_scores
                    .entry(to_id.clone())
                    .and_modify(|s| *s = s.max(hop_score))
                    .or_insert(hop_score);

                // Depth-2 neighbors
                if hops >= 2 {
                    let depth2 = self.get_neighbor_ids(to_id)?;
                    for (to_id2, weight2) in &depth2 {
                        if neighbor_items.contains_key(to_id2) {
                            continue;
                        }
                        let hop2_score = hop_score * weight2 * 0.5;
                        neighbor_scores
                            .entry(to_id2.clone())
                            .and_modify(|s| *s = s.max(hop2_score))
                            .or_insert(hop2_score);
                    }
                }
            }
        }

        // Fetch missing items
        let missing_ids: Vec<String> = neighbor_scores
            .keys()
            .filter(|id| !neighbor_items.contains_key(*id))
            .cloned()
            .collect();

        for id in missing_ids {
            if let Some(item) = self.get_memory(&id)? {
                if item.is_valid() {
                    neighbor_items.insert(id, item);
                } else {
                    neighbor_scores.remove(&id);
                }
            } else {
                neighbor_scores.remove(&id);
            }
        }

        let mut sorted: Vec<(String, f64)> = neighbor_scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(sorted
            .into_iter()
            .take(k)
            .filter_map(|(id, score)| {
                neighbor_items
                    .remove(&id)
                    .map(|item| item.with_score(score))
            })
            .collect())
    }

    /// Get direct neighbors (depth-1) of a memory node.
    fn get_neighbor_ids(&self, from_id: &str) -> Result<Vec<(String, f64)>> {
        let script = "?[to_id, weight] := *mem_edges{from_id, to_id, weight}, from_id == $id";
        let params = BTreeMap::from([("id".into(), DataValue::Str(from_id.into()))]);
        match self.run_immutable(script, params) {
            Ok(result) => {
                let mut neighbors = Vec::new();
                for row in &result.rows {
                    let to_id = dv_to_string(&row[0]);
                    let weight = dv_to_f64(&row[1]);
                    neighbors.push((to_id, weight));
                }
                Ok(neighbors)
            }
            Err(_) => Ok(vec![]),
        }
    }

    /// Search memories within a date range on the `event_start` column.
    /// Filters by ISO 8601 date strings converted to epoch timestamps.
    /// Optionally scoped to a user_id.
    pub fn search_temporal(
        &self,
        date_from: &str,
        date_to: &str,
        k: usize,
        user_id: Option<&str>,
    ) -> Result<Vec<MemoryItem>> {
        // Parse ISO 8601 strings to epoch timestamps (f64)
        let from_ts = chrono::DateTime::parse_from_rfc3339(date_from)
            .or_else(|_| {
                // Try parsing date-only format by appending time
                chrono::DateTime::parse_from_rfc3339(&format!("{date_from}T00:00:00Z"))
            })
            .map(|dt| dt.timestamp_millis() as f64 / 1000.0)
            .map_err(|e| {
                MemlocalError::InvalidArgument(format!("invalid date_from '{date_from}': {e}"))
            })?;

        let to_ts = chrono::DateTime::parse_from_rfc3339(date_to)
            .or_else(|_| {
                chrono::DateTime::parse_from_rfc3339(&format!("{date_to}T23:59:59Z"))
            })
            .map(|dt| dt.timestamp_millis() as f64 / 1000.0)
            .map_err(|e| {
                MemlocalError::InvalidArgument(format!("invalid date_to '{date_to}': {e}"))
            })?;

        let mut conditions = vec![format!(
            "*{}{{id, content, type, hash, user_id, agent_id, session_id, \
             speaker, document_date, \
             metadata_json, created_at, updated_at, \
             event_start: valid_at, event_end: invalid_at, @ \"NOW\"}}",
            MemorySchema::MEMORIES
        )];
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();

        // event_start must be within the date range (and non-zero, meaning it was set)
        conditions.push("valid_at >= $date_from".into());
        conditions.push("valid_at <= $date_to".into());
        conditions.push("valid_at > 0.0".into());
        params.insert("date_from".into(), DataValue::from(from_ts));
        params.insert("date_to".into(), DataValue::from(to_ts));

        if let Some(uid) = user_id {
            conditions.push("user_id == $uid".into());
            params.insert("uid".into(), DataValue::Str(uid.into()));
        }

        let script = format!(
            "?[id, content, type, hash, user_id, agent_id, session_id, \
             speaker, document_date, \
             metadata_json, created_at, updated_at, valid_at, invalid_at] := \
             {}\n:order -valid_at\n:limit {}",
            conditions.join(", "),
            k
        );

        let result = self.run_immutable(&script, params)?;
        rows_to_items(&result)
    }

    /// Unified search dispatching to the right mode.
    pub fn search(
        &self,
        query: &str,
        query_embedding: Option<&[f32]>,
        mode: SearchMode,
        k: usize,
        user_id: Option<&str>,
        memory_type: Option<MemoryType>,
    ) -> Result<Vec<MemoryItem>> {
        match mode {
            SearchMode::Semantic => {
                let emb = query_embedding.ok_or_else(|| {
                    MemlocalError::InvalidArgument(
                        "queryEmbedding required for semantic search".into(),
                    )
                })?;
                self.search_semantic(emb, k, user_id, memory_type)
            }
            SearchMode::Text => self.search_text(query, k),
            SearchMode::Graph => match query_embedding {
                Some(emb) => self.search_graph(emb, k, user_id, memory_type, 2),
                None => self.search_text(query, k),
            },
            SearchMode::Hybrid => match query_embedding {
                Some(emb) => self.search_hybrid(query, emb, k, user_id, memory_type),
                None => self.search_text(query, k),
            },
        }
    }

    /// Get important memories ranked by composite score.
    pub fn get_important_memories(
        &self,
        user_id: Option<&str>,
        limit: usize,
        min_importance: f64,
    ) -> Result<Vec<MemoryItem>> {
        let all = self.get_memories(user_id, None, 500)?;
        let mut scored: Vec<MemoryItem> = all
            .into_iter()
            .map(|item| {
                let importance = self.compute_importance(&item);
                item.with_score(importance)
            })
            .filter(|item| item.score.unwrap_or(0.0) >= min_importance)
            .collect();
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(limit);
        Ok(scored)
    }

    // ──────────── Edges ────────────

    /// Insert or update a knowledge-graph edge.
    pub fn put_edge(&self, edge: &MemoryEdge) -> Result<()> {
        let m = edge.to_map();
        let script = format!(
            "?[from_id, to_id, relation, weight, created_at] <- \
             [[$from_id, $to_id, $relation, $weight, $created_at]]\n\
             :put {} {{from_id, to_id, relation, weight, created_at}}",
            MemorySchema::EDGES
        );
        let params = BTreeMap::from([
            (
                "from_id".into(),
                DataValue::Str(json_str(&m["from_id"]).into()),
            ),
            ("to_id".into(), DataValue::Str(json_str(&m["to_id"]).into())),
            (
                "relation".into(),
                DataValue::Str(json_str(&m["relation"]).into()),
            ),
            (
                "weight".into(),
                DataValue::from(m["weight"].as_f64().unwrap_or(1.0)),
            ),
            (
                "created_at".into(),
                DataValue::from(m["created_at"].as_f64().unwrap_or(0.0)),
            ),
        ]);
        self.run_mutable(&script, params)?;
        Ok(())
    }

    /// Get all edges from a memory node.
    pub fn get_edges_from(&self, memory_id: &str) -> Result<Vec<MemoryEdge>> {
        let script = "?[from_id, to_id, relation, weight, created_at] := \
             *mem_edges{from_id, to_id, relation, weight, created_at}, from_id == $id";
        let params = BTreeMap::from([("id".into(), DataValue::Str(memory_id.into()))]);
        let result = self.run_immutable(script, params)?;
        rows_to_edges(&result)
    }

    /// Get all edges to a memory node.
    pub fn get_edges_to(&self, memory_id: &str) -> Result<Vec<MemoryEdge>> {
        let script = "?[from_id, to_id, relation, weight, created_at] := \
             *mem_edges{from_id, to_id, relation, weight, created_at}, to_id == $id";
        let params = BTreeMap::from([("id".into(), DataValue::Str(memory_id.into()))]);
        let result = self.run_immutable(script, params)?;
        rows_to_edges(&result)
    }

    /// Remove an edge.
    pub fn remove_edge(&self, from_id: &str, to_id: &str, relation: &str) -> Result<()> {
        let script = format!(
            "?[from_id, to_id, relation] <- [[$from_id, $to_id, $relation]]\n\
             :rm {} {{from_id, to_id, relation}}",
            MemorySchema::EDGES
        );
        let params = BTreeMap::from([
            ("from_id".into(), DataValue::Str(from_id.into())),
            ("to_id".into(), DataValue::Str(to_id.into())),
            ("relation".into(), DataValue::Str(relation.into())),
        ]);
        self.run_mutable(&script, params)?;
        Ok(())
    }

    /// Run PageRank on the memory graph.
    pub fn page_rank(&self, iterations: usize) -> Result<HashMap<String, f64>> {
        let script = format!(
            "?[node, score] <~ PageRank(*{}{{from_id: from, to_id: to}}, \
             iters: {})",
            MemorySchema::EDGES,
            iterations
        );
        let result = self.run_immutable(&script, BTreeMap::new())?;
        let mut ranks = HashMap::new();
        for row in &result.rows {
            ranks.insert(dv_to_string(&row[0]), dv_to_f64(&row[1]));
        }
        Ok(ranks)
    }

    /// Run community detection on the memory graph.
    pub fn community_detection(&self) -> Result<HashMap<String, i64>> {
        let script = format!(
            "?[node, community] <~ CommunityDetectionLouvain(*{}{{from_id: from, to_id: to}})",
            MemorySchema::EDGES
        );
        let result = self.run_immutable(&script, BTreeMap::new())?;
        let mut communities = HashMap::new();
        for row in &result.rows {
            communities.insert(dv_to_string(&row[0]), dv_to_i64(&row[1]));
        }
        Ok(communities)
    }

    /// Find shortest path between two memory nodes.
    pub fn shortest_path(&self, from_id: &str, to_id: &str) -> Result<Option<Vec<String>>> {
        let script = format!(
            "?[path] <~ ShortestPathBFS(*{}{{from_id: from, to_id: to}}, \
             starting: [$from], goals: [$to])",
            MemorySchema::EDGES
        );
        let params = BTreeMap::from([
            ("from".into(), DataValue::Str(from_id.into())),
            ("to".into(), DataValue::Str(to_id.into())),
        ]);
        let result = self.run_immutable(&script, params)?;
        if result.rows.is_empty() {
            return Ok(None);
        }
        if let DataValue::List(path) = &result.rows[0][0] {
            let strings: Vec<String> = path.iter().map(dv_to_string).collect();
            Ok(Some(strings))
        } else {
            Ok(None)
        }
    }

    // ──────────── Conversations ────────────

    /// Append a message to a conversation session.
    pub fn put_message(&self, message: &Message, seq: i64) -> Result<()> {
        let script = format!(
            "?[session_id, seq, role, content, timestamp, metadata_json] <- \
             [[$session_id, $seq, $role, $content, $timestamp, $metadata_json]]\n\
             :put {} {{session_id, seq, role, content, timestamp, metadata_json}}",
            MemorySchema::CONVERSATIONS
        );
        let params = BTreeMap::from([
            (
                "session_id".into(),
                DataValue::Str(message.session_id.as_deref().unwrap_or("").into()),
            ),
            ("seq".into(), DataValue::from(seq)),
            ("role".into(), DataValue::Str(message.role.clone().into())),
            (
                "content".into(),
                DataValue::Str(message.content.clone().into()),
            ),
            (
                "timestamp".into(),
                DataValue::from(message.timestamp.timestamp_millis() as f64 / 1000.0),
            ),
            ("metadata_json".into(), DataValue::Str("{}".into())),
        ]);
        self.run_mutable(&script, params)?;
        Ok(())
    }

    /// Get messages for a session, ordered by sequence.
    pub fn get_messages(&self, session_id: &str, limit: Option<usize>) -> Result<Vec<Message>> {
        let mut script = format!(
            "?[session_id, seq, role, content, timestamp, metadata_json] := \
             *{}{{session_id, seq, role, content, timestamp, metadata_json}}, \
             session_id == $sid\n:order seq",
            MemorySchema::CONVERSATIONS
        );
        if let Some(lim) = limit {
            script.push_str(&format!("\n:limit {lim}"));
        }

        let params = BTreeMap::from([("sid".into(), DataValue::Str(session_id.into()))]);
        let result = self.run_immutable(&script, params)?;

        let mut messages = Vec::new();
        for row in &result.rows {
            let role = dv_to_string(&row[2]);
            let content = dv_to_string(&row[3]);
            let ts = dv_to_f64(&row[4]);
            let millis = (ts * 1000.0).round() as i64;
            let timestamp = chrono::TimeZone::timestamp_millis_opt(&chrono::Utc, millis)
                .single()
                .unwrap_or_default();
            let sid = dv_to_string(&row[0]);
            messages.push(Message {
                role,
                content,
                timestamp,
                session_id: if sid.is_empty() { None } else { Some(sid) },
                metadata: None,
            });
        }
        Ok(messages)
    }

    /// Count messages in a session.
    pub fn message_count(&self, session_id: &str) -> Result<usize> {
        let script = "?[count(seq)] := *mem_conversations{session_id, seq}, session_id == $sid";
        let params = BTreeMap::from([("sid".into(), DataValue::Str(session_id.into()))]);
        let result = self.run_immutable(script, params)?;
        if result.rows.is_empty() {
            return Ok(0);
        }
        Ok(dv_to_i64(&result.rows[0][0]) as usize)
    }

    // ──────────── User Profile ────────────

    /// Upsert a user profile.
    pub fn put_profile(&self, profile: &UserProfile) -> Result<()> {
        let m = profile.to_map();
        let script = format!(
            "?[user_id, static_facts_json, dynamic_context_json, updated_at] <- \
             [[$user_id, $static_facts_json, $dynamic_context_json, $updated_at]]\n\
             :put {} {{user_id, static_facts_json, dynamic_context_json, updated_at}}",
            MemorySchema::PROFILES
        );
        let params = BTreeMap::from([
            (
                "user_id".into(),
                DataValue::Str(json_str(&m["user_id"]).into()),
            ),
            (
                "static_facts_json".into(),
                DataValue::Str(json_str(&m["static_facts_json"]).into()),
            ),
            (
                "dynamic_context_json".into(),
                DataValue::Str(json_str(&m["dynamic_context_json"]).into()),
            ),
            (
                "updated_at".into(),
                DataValue::from(m["updated_at"].as_f64().unwrap_or(0.0)),
            ),
        ]);
        self.run_mutable(&script, params)?;
        Ok(())
    }

    /// Get a user profile.
    pub fn get_profile(&self, user_id: &str) -> Result<Option<UserProfile>> {
        let script = format!(
            "?[user_id, static_facts_json, dynamic_context_json, updated_at] := \
             *{}{{user_id, static_facts_json, dynamic_context_json, updated_at}}, \
             user_id == $uid",
            MemorySchema::PROFILES
        );
        let params = BTreeMap::from([("uid".into(), DataValue::Str(user_id.into()))]);
        let result = self.run_immutable(&script, params)?;
        if result.rows.is_empty() {
            return Ok(None);
        }
        let map = rows_to_json(&result, 0);
        Ok(Some(UserProfile::from_map(&map)?))
    }

    // ──────────── Prospective ────────────

    /// Insert or update a prospective memory item.
    pub fn put_prospective(&self, item: &ProspectiveItem) -> Result<()> {
        let m = item.to_map();
        let script = format!(
            "?[id, content, trigger_type, trigger_condition, user_id, completed, \
             created_at, completed_at] <- [[$id, $content, $trigger_type, \
             $trigger_condition, $user_id, $completed, $created_at, $completed_at]]\n\
             :put {} {{id, content, trigger_type, trigger_condition, user_id, \
             completed, created_at, completed_at}}",
            MemorySchema::PROSPECTIVE
        );
        let params = BTreeMap::from([
            ("id".into(), DataValue::Str(json_str(&m["id"]).into())),
            (
                "content".into(),
                DataValue::Str(json_str(&m["content"]).into()),
            ),
            (
                "trigger_type".into(),
                DataValue::Str(json_str(&m["trigger_type"]).into()),
            ),
            (
                "trigger_condition".into(),
                DataValue::Str(json_str(&m["trigger_condition"]).into()),
            ),
            (
                "user_id".into(),
                DataValue::Str(json_str(&m["user_id"]).into()),
            ),
            (
                "completed".into(),
                DataValue::from(m["completed"].as_i64().unwrap_or(0)),
            ),
            (
                "created_at".into(),
                DataValue::from(m["created_at"].as_f64().unwrap_or(0.0)),
            ),
            (
                "completed_at".into(),
                DataValue::from(m["completed_at"].as_f64().unwrap_or(0.0)),
            ),
        ]);
        self.run_mutable(&script, params)?;
        Ok(())
    }

    /// Get pending (uncompleted) prospective items.
    pub fn get_pending_prospective(&self, user_id: Option<&str>) -> Result<Vec<ProspectiveItem>> {
        let mut conditions = vec![format!(
            "*{}{{id, content, trigger_type, trigger_condition, user_id, \
             completed, created_at, completed_at}}",
            MemorySchema::PROSPECTIVE
        )];
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();

        conditions.push("completed == 0".into());
        if let Some(uid) = user_id {
            conditions.push("user_id == $uid".into());
            params.insert("uid".into(), DataValue::Str(uid.into()));
        }

        let script = format!(
            "?[id, content, trigger_type, trigger_condition, user_id, \
             completed, created_at, completed_at] := {}",
            conditions.join(", ")
        );

        let result = self.run_immutable(&script, params)?;
        let mut items = Vec::new();
        for i in 0..result.rows.len() {
            let map = rows_to_json(&result, i);
            items.push(ProspectiveItem::from_map(&map)?);
        }
        Ok(items)
    }

    /// Mark a prospective item as completed.
    pub fn complete_prospective(&self, id: &str) -> Result<()> {
        let now = chrono::Utc::now().timestamp_millis() as f64 / 1000.0;
        let script = format!(
            "?[id, completed, completed_at] <- [[$id, 1, {now}]]\n\
             :update {} {{id => completed, completed_at}}",
            MemorySchema::PROSPECTIVE
        );
        let params = BTreeMap::from([("id".into(), DataValue::Str(id.into()))]);
        self.run_mutable(&script, params)?;
        Ok(())
    }

    // ──────────── Export/Import ────────────

    /// Export all relations as JSON.
    pub fn export_relations(&self) -> Result<serde_json::Value> {
        let result = self
            .db
            .export_relations(
                [
                    "mem_items",
                    "mem_edges",
                    "mem_conversations",
                    "mem_profiles",
                    "mem_prospective",
                    "mem_triples",
                    "mem_summaries",
                ]
                .iter()
                .copied(),
            )
            .map_err(|e| MemlocalError::Database(e.to_string()))?;

        // Convert NamedRows map to JSON
        let mut export = serde_json::Map::new();
        for (name, rows) in result {
            let headers = &rows.headers;
            let mut json_rows = Vec::new();
            for row in &rows.rows {
                let mut obj = serde_json::Map::new();
                for (i, header) in headers.iter().enumerate() {
                    obj.insert(header.clone(), dv_to_json(&row[i]));
                }
                json_rows.push(serde_json::Value::Object(obj));
            }
            export.insert(
                name.to_string(),
                serde_json::json!({
                    "headers": headers,
                    "rows": json_rows
                }),
            );
        }
        Ok(serde_json::Value::Object(export))
    }

    // ──────────── Triples ────────────

    /// Insert or update a semantic triple.
    pub fn put_triple(&self, triple: &Triple) -> Result<()> {
        let script = format!(
            "?[subject, predicate, object, memory_id, speaker, mention_count, \
             last_mentioned, session_id, confidence] <- \
             [[$subject, $predicate, $object, $memory_id, $speaker, $mention_count, \
             $last_mentioned, $session_id, $confidence]]\n\
             :put {} {{subject, predicate, object => memory_id, speaker, mention_count, \
             last_mentioned, session_id, confidence}}",
            MemorySchema::TRIPLES
        );
        let params = BTreeMap::from([
            ("subject".into(), DataValue::Str(triple.subject.clone().into())),
            ("predicate".into(), DataValue::Str(triple.predicate.clone().into())),
            ("object".into(), DataValue::Str(triple.object.clone().into())),
            ("memory_id".into(), DataValue::Str(triple.memory_id.clone().into())),
            ("speaker".into(), DataValue::Str(triple.speaker.clone().into())),
            ("mention_count".into(), DataValue::from(triple.mention_count as i64)),
            ("last_mentioned".into(), DataValue::from(triple.last_mentioned)),
            ("session_id".into(), DataValue::Str(triple.session_id.clone().into())),
            ("confidence".into(), DataValue::from(triple.confidence)),
        ]);
        self.run_mutable(&script, params)?;
        Ok(())
    }

    /// Search triples by subject and/or predicate.
    pub fn search_triples(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Result<Vec<Triple>> {
        let mut conditions = vec![format!(
            "*{}{{subject, predicate, object, memory_id, speaker, mention_count, \
             last_mentioned, session_id, confidence}}",
            MemorySchema::TRIPLES
        )];
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();

        if let Some(s) = subject {
            conditions.push("subject == $s".into());
            params.insert("s".into(), DataValue::Str(s.into()));
        }
        if let Some(p) = predicate {
            conditions.push("predicate == $p".into());
            params.insert("p".into(), DataValue::Str(p.into()));
        }
        if let Some(o) = object {
            conditions.push("object == $o".into());
            params.insert("o".into(), DataValue::Str(o.into()));
        }

        let script = format!(
            "?[subject, predicate, object, memory_id, speaker, mention_count, \
             last_mentioned, session_id, confidence] := {}",
            conditions.join(", ")
        );

        let result = self.run_immutable(&script, params)?;
        let mut triples = Vec::new();
        for row in &result.rows {
            triples.push(Triple {
                subject: dv_to_string(&row[0]),
                predicate: dv_to_string(&row[1]),
                object: dv_to_string(&row[2]),
                memory_id: dv_to_string(&row[3]),
                speaker: dv_to_string(&row[4]),
                mention_count: dv_to_i64(&row[5]) as u64,
                last_mentioned: dv_to_f64(&row[6]),
                session_id: dv_to_string(&row[7]),
                confidence: dv_to_f64(&row[8]),
            });
        }
        Ok(triples)
    }

    /// Full-text search on triples.
    pub fn search_triples_fts(&self, query: &str, k: usize) -> Result<Vec<Triple>> {
        let sanitized = sanitize_fts_query(query);
        if sanitized.is_empty() {
            return Ok(vec![]);
        }
        let script = format!(
            "?[subject, predicate, object, memory_id, speaker, mention_count, \
             last_mentioned, session_id, confidence, score] := \
             ~{}:{}{{subject, predicate, object, memory_id, speaker, mention_count, \
             last_mentioned, session_id, confidence | query: $q, k: {}, bind_score: score}}",
            MemorySchema::TRIPLES,
            MemorySchema::TRIPLES_FTS_INDEX,
            k
        );
        let params = BTreeMap::from([("q".into(), DataValue::Str(sanitized.into()))]);
        let result = self.run_immutable(&script, params)?;
        let mut triples = Vec::new();
        for row in &result.rows {
            triples.push(Triple {
                subject: dv_to_string(&row[0]),
                predicate: dv_to_string(&row[1]),
                object: dv_to_string(&row[2]),
                memory_id: dv_to_string(&row[3]),
                speaker: dv_to_string(&row[4]),
                mention_count: dv_to_i64(&row[5]) as u64,
                last_mentioned: dv_to_f64(&row[6]),
                session_id: dv_to_string(&row[7]),
                confidence: dv_to_f64(&row[8]),
            });
        }
        Ok(triples)
    }

    /// Increment the mention count for an existing triple.
    pub fn increment_triple_mention(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        last_mentioned: f64,
    ) -> Result<()> {
        // First get the current mention_count
        let existing = self.search_triples(Some(subject), Some(predicate), Some(object))?;
        if let Some(triple) = existing.first() {
            let new_count = triple.mention_count + 1;
            let script = format!(
                "?[subject, predicate, object, mention_count, last_mentioned] <- \
                 [[$s, $p, $o, {}, {}]]\n\
                 :update {} {{subject, predicate, object => mention_count, last_mentioned}}",
                new_count, last_mentioned, MemorySchema::TRIPLES
            );
            let params = BTreeMap::from([
                ("s".into(), DataValue::Str(subject.into())),
                ("p".into(), DataValue::Str(predicate.into())),
                ("o".into(), DataValue::Str(object.into())),
            ]);
            self.run_mutable(&script, params)?;
        }
        Ok(())
    }

    /// Speaker-filtered semantic search: filters by speaker column BEFORE vector similarity.
    pub fn search_by_speaker(
        &self,
        query_embedding: &[f32],
        speaker: &str,
        k: usize,
        user_id: Option<&str>,
    ) -> Result<Vec<MemoryItem>> {
        let mut filter_parts = vec![format!("speaker == \"{}\"", escape_cozo_str(speaker))];
        if let Some(uid) = user_id {
            filter_parts.push(format!("user_id == \"{}\"", escape_cozo_str(uid)));
        }
        let filter = filter_parts.join(" && ");

        let bind_fields = "id, content, type, hash, user_id, agent_id, session_id, \
                           speaker, document_date, metadata_json, created_at, updated_at, \
                           event_start: valid_at, event_end: invalid_at";
        let output_fields = "id, content, type, hash, user_id, agent_id, session_id, \
                             speaker, document_date, metadata_json, created_at, updated_at, valid_at, invalid_at";

        let script = format!(
            "?[{output_fields}, distance] := ~{}:{}{{ {bind_fields} | \
             query: vec($q), k: {k}, ef: {ef}, bind_distance: distance, filter: {filter} }}",
            MemorySchema::MEMORIES,
            MemorySchema::VECTOR_INDEX,
            ef = k * 2,
        );

        let embedding_dv: Vec<DataValue> = query_embedding
            .iter()
            .map(|&f| DataValue::from(f as f64))
            .collect();
        let params = BTreeMap::from([("q".into(), DataValue::List(embedding_dv))]);

        let result = self.run_immutable(&script, params)?;
        let mut items = Vec::new();
        for i in 0..result.rows.len() {
            let map = rows_to_json(&result, i);
            let item = MemoryItem::from_map(&map)?;
            let raw_score = 1.0 - map["distance"].as_f64().unwrap_or(0.0);
            let adjusted = Self::apply_confidence(&item, raw_score);
            items.push(item.with_score(adjusted));
        }
        Ok(items)
    }

    /// Search triples filtered by speaker.
    pub fn search_triples_by_speaker(
        &self,
        query: &str,
        speaker: &str,
        k: usize,
    ) -> Result<Vec<Triple>> {
        let sanitized = sanitize_fts_query(query);
        if sanitized.is_empty() {
            return Ok(vec![]);
        }
        let script = format!(
            "?[subject, predicate, object, memory_id, speaker, mention_count, \
             last_mentioned, session_id, confidence] := \
             ~{}:{}{{subject, predicate, object, memory_id, speaker, mention_count, \
             last_mentioned, session_id, confidence | query: $q, k: {} }}, speaker == $speaker",
            MemorySchema::TRIPLES,
            MemorySchema::TRIPLES_FTS_INDEX,
            k
        );
        let params = BTreeMap::from([
            ("q".into(), DataValue::Str(sanitized.into())),
            ("speaker".into(), DataValue::Str(speaker.into())),
        ]);
        // Note: CozoDB FTS may not support post-filter via condition. If it errors, fall back to
        // search_triples_fts and filter in Rust.
        match self.run_immutable(&script, params) {
            Ok(result) => {
                let mut triples = Vec::new();
                for row in &result.rows {
                    triples.push(Triple {
                        subject: dv_to_string(&row[0]),
                        predicate: dv_to_string(&row[1]),
                        object: dv_to_string(&row[2]),
                        memory_id: dv_to_string(&row[3]),
                        speaker: dv_to_string(&row[4]),
                        mention_count: dv_to_i64(&row[5]) as u64,
                        last_mentioned: dv_to_f64(&row[6]),
                        session_id: dv_to_string(&row[7]),
                        confidence: dv_to_f64(&row[8]),
                    });
                }
                Ok(triples)
            }
            Err(_) => {
                // Fallback: search all then filter by speaker
                let all = self.search_triples_fts(query, k * 2)?;
                Ok(all.into_iter().filter(|t| t.speaker == speaker).take(k).collect())
            }
        }
    }

    /// Get all known speakers (distinct subjects from triples that are people).
    pub fn get_known_speakers(&self, _user_id: Option<&str>) -> Result<Vec<String>> {
        let script = format!(
            "?[speaker] := *{}{{speaker}}, speaker != ''",
            MemorySchema::TRIPLES
        );
        let result = self.run_immutable(&script, BTreeMap::new())?;
        let mut speakers = Vec::new();
        for row in &result.rows {
            speakers.push(dv_to_string(&row[0]));
        }
        Ok(speakers)
    }

    // ──────────── Summaries ────────────

    /// Store a session summary with embedding.
    pub fn put_summary(
        &self,
        session_id: &str,
        summary: &str,
        embedding: &[f32],
        speakers: &[String],
        topics: &[String],
        doc_date: f64,
    ) -> Result<()> {
        let embedding_dv: Vec<DataValue> = embedding
            .iter()
            .map(|&f| DataValue::from(f as f64))
            .collect();
        let speakers_json = serde_json::to_string(speakers).unwrap_or_else(|_| "[]".to_string());
        let topics_json = serde_json::to_string(topics).unwrap_or_else(|_| "[]".to_string());

        let script = format!(
            "?[session_id, summary, speakers_json, key_topics_json, document_date, embedding] <- \
             [[$session_id, $summary, $speakers_json, $key_topics_json, $doc_date, vec($embedding)]]\n\
             :put {} {{session_id => summary, speakers_json, key_topics_json, document_date, embedding}}",
            MemorySchema::SUMMARIES
        );
        let params = BTreeMap::from([
            ("session_id".into(), DataValue::Str(session_id.into())),
            ("summary".into(), DataValue::Str(summary.into())),
            ("speakers_json".into(), DataValue::Str(speakers_json.into())),
            ("key_topics_json".into(), DataValue::Str(topics_json.into())),
            ("doc_date".into(), DataValue::from(doc_date)),
            ("embedding".into(), DataValue::List(embedding_dv)),
        ]);
        self.run_mutable(&script, params)?;
        Ok(())
    }

    /// Search summaries by vector similarity.
    pub fn search_summaries(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<SessionSummary>> {
        let embedding_dv: Vec<DataValue> = query_embedding
            .iter()
            .map(|&f| DataValue::from(f as f64))
            .collect();
        let script = format!(
            "?[session_id, summary, speakers_json, key_topics_json, document_date, distance] := \
             ~{}:{}{{session_id, summary, speakers_json, key_topics_json, document_date | \
             query: vec($q), k: {}, bind_distance: distance, ef: {}}}",
            MemorySchema::SUMMARIES,
            MemorySchema::SUMMARIES_VECTOR_INDEX,
            k,
            k * 2
        );
        let params = BTreeMap::from([("q".into(), DataValue::List(embedding_dv))]);
        let result = self.run_immutable(&script, params)?;
        let mut summaries = Vec::new();
        for row in &result.rows {
            let speakers_str = dv_to_string(&row[2]);
            let topics_str = dv_to_string(&row[3]);
            let speakers: Vec<String> = serde_json::from_str(&speakers_str).unwrap_or_default();
            let key_topics: Vec<String> = serde_json::from_str(&topics_str).unwrap_or_default();
            let distance = dv_to_f64(&row[5]);
            summaries.push(SessionSummary {
                session_id: dv_to_string(&row[0]),
                summary: dv_to_string(&row[1]),
                speakers,
                key_topics,
                document_date: dv_to_f64(&row[4]),
                score: Some(1.0 - distance),
            });
        }
        Ok(summaries)
    }

    // ──────────── Export/Import + Compact ────────────

    /// Compact the database for optimal query performance.
    /// Call after bulk operations (storing many memories).
    pub fn compact(&self) -> Result<()> {
        self.try_run(MemorySchema::compact_statement());
        Ok(())
    }

    // ──────────── CozoDB-Specific Optimizations ────────────

    /// Recursive Datalog graph search: finds all memories reachable within N hops.
    /// Replaces imperative 2-hop traversal with a single CozoDB query.
    pub fn search_graph_recursive(
        &self,
        seed_ids: &[String],
        max_hops: usize,
        user_id: Option<&str>,
    ) -> Result<Vec<MemoryItem>> {
        if seed_ids.is_empty() {
            return Ok(vec![]);
        }

        // Build seed list for Datalog
        let seeds_list = seed_ids
            .iter()
            .map(|id| format!("[\"{}\"]", id.replace('"', "\\\"")))
            .collect::<Vec<_>>()
            .join(", ");

        let mut filter_clause = String::new();
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();
        if let Some(uid) = user_id {
            filter_clause = ", user_id == $uid".to_string();
            params.insert("uid".into(), DataValue::Str(uid.into()));
        }

        // Recursive Datalog: find all memories reachable within N hops
        let script = format!(
            "seed[id] <- [{seeds}] \n\
             reachable[id, 0] := seed[id] \n\
             reachable[to_id, d + 1] := reachable[from_id, d], \
                 *{edges}{{from_id, to_id}}, d < {hops} \n\
             ?[id, content, type, hash, user_id, agent_id, session_id, \
               speaker, document_date, metadata_json, created_at, updated_at, \
               valid_at, invalid_at] := \
                 reachable[id, _], \
                 *{mem}{{id, content, type, hash, user_id, agent_id, session_id, \
                 speaker, document_date, metadata_json, created_at, updated_at, \
                 event_start: valid_at, event_end: invalid_at, @ \"NOW\"}}{filter}",
            seeds = seeds_list,
            edges = MemorySchema::EDGES,
            hops = max_hops,
            mem = MemorySchema::MEMORIES,
            filter = filter_clause,
        );

        let result = self.run_immutable(&script, params)?;
        rows_to_items(&result)
    }

    /// Time-travel query: retrieves memories as they existed at a specific point in time.
    /// Uses CozoDB's HNSW vector search and filters by created_at timestamp in Rust.
    pub fn search_at_time(
        &self,
        query_embedding: &[f32],
        k: usize,
        at_time: f64, // UTC epoch seconds
        user_id: Option<&str>,
    ) -> Result<Vec<MemoryItem>> {
        let mut filter_parts = Vec::new();
        if let Some(uid) = user_id {
            filter_parts.push(format!("user_id == \"{}\"", escape_cozo_str(uid)));
        }
        let filter = if filter_parts.is_empty() {
            String::new()
        } else {
            format!(", filter: {}", filter_parts.join(" && "))
        };

        let bind_fields = "id, content, type, hash, user_id, agent_id, session_id, \
                           speaker, document_date, metadata_json, created_at, updated_at, \
                           event_start: valid_at, event_end: invalid_at";
        let output_fields = "id, content, type, hash, user_id, agent_id, session_id, \
                             speaker, document_date, metadata_json, created_at, updated_at, \
                             valid_at, invalid_at";

        // Use CozoDB HNSW search and filter by created_at in Rust
        // Full CozoDB time-travel on HNSW requires passing a timestamp to the validity clause,
        // which may not work with HNSW indices. The created_at filter approach is the fallback.
        let script = format!(
            "?[{output_fields}, distance] := ~{}:{}{{ {bind_fields} | \
             query: vec($q), k: {k}, ef: {ef}, bind_distance: distance{filter} }}",
            MemorySchema::MEMORIES,
            MemorySchema::VECTOR_INDEX,
            ef = k * 2,
        );

        let embedding_dv: Vec<DataValue> = query_embedding
            .iter()
            .map(|&f| DataValue::from(f as f64))
            .collect();
        let params = BTreeMap::from([("q".into(), DataValue::List(embedding_dv))]);

        let result = self.run_immutable(&script, params)?;
        let mut items = Vec::new();
        for i in 0..result.rows.len() {
            let map = rows_to_json(&result, i);
            let item = MemoryItem::from_map(&map)?;
            let raw_score = 1.0 - map["distance"].as_f64().unwrap_or(0.0);
            // Filter out items that were created AFTER the at_time
            let created_ts = item.created_at.timestamp_millis() as f64 / 1000.0;
            if created_ts <= at_time {
                let adjusted = Self::apply_confidence(&item, raw_score);
                items.push(item.with_score(adjusted));
            }
        }
        Ok(items)
    }
}

// ──────────── Helpers ────────────

/// Jaccard-like word overlap ratio between two strings.
fn word_overlap(a: &str, b: &str) -> f64 {
    let words_a: std::collections::HashSet<&str> = a
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.len() > 2)
        .collect();
    let words_b: std::collections::HashSet<&str> = b
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.len() > 2)
        .collect();
    if words_a.is_empty() || words_b.is_empty() {
        return 0.0;
    }
    let intersection = words_a.intersection(&words_b).count();
    let smaller = words_a.len().min(words_b.len());
    intersection as f64 / smaller as f64
}

/// Escape double quotes in a string for safe interpolation into CozoScript filter clauses.
fn escape_cozo_str(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn sanitize_fts_query(query: &str) -> String {
    query
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn json_str(val: &serde_json::Value) -> String {
    match val {
        serde_json::Value::String(s) => s.clone(),
        _ => val.to_string(),
    }
}

fn dv_to_string(dv: &DataValue) -> String {
    match dv {
        DataValue::Str(s) => s.to_string(),
        DataValue::Num(n) => {
            if let cozo::Num::Int(i) = n {
                i.to_string()
            } else {
                format!("{n:?}")
            }
        }
        _ => format!("{dv:?}"),
    }
}

fn dv_to_f64(dv: &DataValue) -> f64 {
    match dv {
        DataValue::Num(n) => match n {
            cozo::Num::Float(f) => *f,
            cozo::Num::Int(i) => *i as f64,
        },
        _ => 0.0,
    }
}

fn dv_to_i64(dv: &DataValue) -> i64 {
    match dv {
        DataValue::Num(n) => match n {
            cozo::Num::Int(i) => *i,
            cozo::Num::Float(f) => *f as i64,
        },
        _ => 0,
    }
}

fn dv_to_json(dv: &DataValue) -> serde_json::Value {
    match dv {
        DataValue::Null => serde_json::Value::Null,
        DataValue::Bool(b) => serde_json::Value::Bool(*b),
        DataValue::Num(n) => match n {
            cozo::Num::Int(i) => serde_json::json!(*i),
            cozo::Num::Float(f) => serde_json::json!(*f),
        },
        DataValue::Str(s) => serde_json::Value::String(s.to_string()),
        DataValue::List(l) => serde_json::Value::Array(l.iter().map(dv_to_json).collect()),
        _ => serde_json::Value::String(format!("{dv:?}")),
    }
}

/// Convert row i of a NamedRows result to a JSON object.
fn rows_to_json(result: &NamedRows, row_idx: usize) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (col_idx, header) in result.headers.iter().enumerate() {
        map.insert(header.clone(), dv_to_json(&result.rows[row_idx][col_idx]));
    }
    serde_json::Value::Object(map)
}

fn rows_to_items(result: &NamedRows) -> Result<Vec<MemoryItem>> {
    let mut items = Vec::new();
    for i in 0..result.rows.len() {
        let map = rows_to_json(result, i);
        items.push(MemoryItem::from_map(&map)?);
    }
    Ok(items)
}

fn rows_to_edges(result: &NamedRows) -> Result<Vec<MemoryEdge>> {
    let mut edges = Vec::new();
    for i in 0..result.rows.len() {
        let map = rows_to_json(result, i);
        edges.push(MemoryEdge::from_map(&map)?);
    }
    Ok(edges)
}

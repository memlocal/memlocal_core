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

        let script = format!(
            "?[id, content, type, hash, user_id, agent_id, session_id, \
             metadata_json, embedding, created_at, updated_at, valid_at, invalid_at] <- \
             [[$id, $content, $type, $hash, $user_id, $agent_id, $session_id, \
             $metadata_json, vec($embedding), $created_at, $updated_at, $valid_at, $invalid_at]]\n\
             :put {} {{id, content, type, hash, user_id, agent_id, session_id, \
             metadata_json, embedding, created_at, updated_at, valid_at, invalid_at}}",
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
                "valid_at".into(),
                DataValue::from(m["valid_at"].as_f64().unwrap_or(0.0)),
            ),
            (
                "invalid_at".into(),
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

    /// Get a memory item by ID.
    pub fn get_memory(&self, id: &str) -> Result<Option<MemoryItem>> {
        let script = format!(
            "?[id, content, type, hash, user_id, agent_id, session_id, \
             metadata_json, created_at, updated_at, valid_at, invalid_at] := \
             *{}{{id, content, type, hash, user_id, agent_id, session_id, \
             metadata_json, created_at, updated_at, valid_at, invalid_at}}, id == $id",
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
        let mut conditions = vec![format!(
            "*{}{{id, content, type, hash, user_id, agent_id, session_id, \
             metadata_json, created_at, updated_at, valid_at, invalid_at}}",
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
        conditions.push("invalid_at == 0.0".into());

        let script = format!(
            "?[id, content, type, hash, user_id, agent_id, session_id, \
             metadata_json, created_at, updated_at, valid_at, invalid_at] := \
             {}\n:order -updated_at\n:limit {}",
            conditions.join(", "),
            limit
        );

        let result = self.run_immutable(&script, params)?;
        rows_to_items(&result)
    }

    /// Invalidate (soft-delete) a memory.
    pub fn invalidate_memory(&self, id: &str) -> Result<()> {
        let now = chrono::Utc::now().timestamp_millis() as f64 / 1000.0;
        let script = format!(
            "?[id, invalid_at] <- [[$id, {now}]]\n\
             :update {} {{id => invalid_at}}",
            MemorySchema::MEMORIES
        );
        let params = BTreeMap::from([("id".into(), DataValue::Str(id.into()))]);
        self.run_mutable(&script, params)?;
        Ok(())
    }

    /// Hard-delete a memory by ID.
    pub fn delete_memory(&self, id: &str) -> Result<()> {
        let script = format!("?[id] <- [[$id]]\n:rm {} {{id}}", MemorySchema::MEMORIES);
        let params = BTreeMap::from([("id".into(), DataValue::Str(id.into()))]);
        self.run_mutable(&script, params)?;
        Ok(())
    }

    /// Find existing memories that match by content hash (dedup check).
    pub fn find_by_hash(&self, hash: &str, user_id: Option<&str>) -> Result<Option<MemoryItem>> {
        let mut conditions = vec![format!(
            "*{}{{id, content, type, hash, user_id, agent_id, session_id, \
             metadata_json, created_at, updated_at, valid_at, invalid_at}}",
            MemorySchema::MEMORIES
        )];
        let mut params: BTreeMap<String, DataValue> = BTreeMap::new();

        conditions.push("hash == $h".into());
        params.insert("h".into(), DataValue::Str(hash.into()));
        conditions.push("invalid_at == 0.0".into());

        if let Some(uid) = user_id {
            conditions.push("user_id == $uid".into());
            params.insert("uid".into(), DataValue::Str(uid.into()));
        }

        let script = format!(
            "?[id, content, type, hash, user_id, agent_id, session_id, \
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

    /// Count total memories.
    pub fn memory_count(&self, memory_type: Option<MemoryType>) -> Result<usize> {
        let mut conditions = vec![format!("*{}{{id, type}}", MemorySchema::MEMORIES)];
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

    /// Compute exponential time-decay factor. lambda=0.005, half-life ~138 days.
    fn time_decay_factor(&self, item: &MemoryItem) -> f64 {
        if !self.enable_time_decay {
            return 1.0;
        }
        let days_since = (chrono::Utc::now() - item.updated_at).num_days() as f64;
        (-0.005 * days_since).exp()
    }

    /// Compute importance: 0.3*confidence + 0.3*log_access + 0.4*recency.
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
        let days_since = (chrono::Utc::now() - item.updated_at).num_days() as f64;

        let access_factor = (1.0 + access_count).ln() / (1.0 + 100.0_f64).ln();
        let recency_factor = (-0.0115 * days_since).exp();

        0.3 * confidence + 0.3 * access_factor + 0.4 * recency_factor
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
            filter_parts.push(format!("user_id == \"{uid}\""));
        }
        if let Some(mtype) = memory_type {
            filter_parts.push(format!("type == \"{}\"", mtype.stored_name()));
        }
        filter_parts.push("invalid_at == 0.0".into());
        let filter = filter_parts.join(" && ");

        let bind_fields = "id, content, type, hash, user_id, agent_id, session_id, \
                           metadata_json, created_at, updated_at, valid_at, invalid_at";

        let script = format!(
            "?[{bind_fields}, distance] := ~{}:{}{{ {bind_fields} | \
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

    /// Full-text (BM25) search.
    pub fn search_text(&self, query: &str, k: usize) -> Result<Vec<MemoryItem>> {
        let sanitized = sanitize_fts_query(query);
        if sanitized.is_empty() {
            return Ok(vec![]);
        }

        let bind_fields = "id, content, type, hash, user_id, agent_id, session_id, \
                           metadata_json, created_at, updated_at, valid_at, invalid_at";

        let script = format!(
            "?[{bind_fields}, score] := ~{}:{}{{ {bind_fields} | \
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
                           metadata_json, created_at, updated_at, valid_at, invalid_at";

        let script = format!(
            "?[{bind_fields}, score] := ~{}:{}{{ {bind_fields} | \
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
        add_ranks(&text_list);
        add_ranks(&lsh_list);

        // Apply time decay
        let mut final_scores: Vec<(String, f64)> = rrf_scores
            .into_iter()
            .map(|(id, score)| {
                let item = &item_map[&id];
                let decayed = score * self.time_decay_factor(item);
                (id, decayed)
            })
            .collect();

        final_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(final_scores
            .into_iter()
            .take(k)
            .filter_map(|(id, score)| item_map.remove(&id).map(|item| item.with_score(score)))
            .collect())
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
}

// ──────────── Helpers ────────────

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

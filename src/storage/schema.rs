/// CozoScript DDL for the memlocal schema.
///
/// All relations are prefixed with `mem_` to avoid namespace collisions.
/// Uses `:create` for each relation. Callers should catch "already exists"
/// errors for idempotency.
///
/// v4: mem_items uses `vld: Validity` for native CozoDB time-travel.
/// Old `valid_at`/`invalid_at` replaced by `event_start`/`event_end` + Validity.
pub struct MemorySchema;

impl MemorySchema {
    // ─────────── Relation names ───────────

    pub const MEMORIES: &'static str = "mem_items";
    pub const EDGES: &'static str = "mem_edges";
    pub const CONVERSATIONS: &'static str = "mem_conversations";
    pub const PROFILES: &'static str = "mem_profiles";
    pub const PROSPECTIVE: &'static str = "mem_prospective";

    // ─────────── Index names ───────────

    pub const VECTOR_INDEX: &'static str = "mem_vec_idx";
    pub const FTS_INDEX: &'static str = "mem_fts_idx";
    pub const LSH_INDEX: &'static str = "mem_lsh_idx";

    /// Generate the DDL for all relations.
    ///
    /// v4: mem_items has `vld: Validity` as a key column for native time-travel.
    /// `event_start`/`event_end` track when the described event occurs.
    pub fn create_statements(
        embedding_dim: u32,
        hnsw_m: u32,
        hnsw_ef_construction: u32,
    ) -> Vec<String> {
        let _ = (hnsw_m, hnsw_ef_construction);
        vec![
            // ── Core memory items (with time-travel via Validity) ──
            format!(
                ":create {} {{ id: String, vld: Validity => content: String, type: String, hash: String, \
                 user_id: String default '', agent_id: String default '', \
                 session_id: String default '', metadata_json: String default '{{}}', \
                 embedding: <F32; {}>, created_at: Float, updated_at: Float, \
                 event_start: Float default 0.0, event_end: Float default 0.0 }}",
                Self::MEMORIES,
                embedding_dim
            ),
            // ── Knowledge-graph edges ──
            format!(
                ":create {} {{ from_id: String, to_id: String, relation: String => \
                 weight: Float default 1.0, created_at: Float }}",
                Self::EDGES
            ),
            // ── Conversation messages ──
            format!(
                ":create {} {{ session_id: String, seq: Int => role: String, \
                 content: String, timestamp: Float, metadata_json: String default '{{}}' }}",
                Self::CONVERSATIONS
            ),
            // ── User profiles ──
            format!(
                ":create {} {{ user_id: String => static_facts_json: String default '{{}}', \
                 dynamic_context_json: String default '{{}}', updated_at: Float }}",
                Self::PROFILES
            ),
            // ── Prospective (future-oriented) items ──
            format!(
                ":create {} {{ id: String => content: String, trigger_type: String, \
                 trigger_condition: String, user_id: String default '', \
                 completed: Int default 0, created_at: Float, \
                 completed_at: Float default 0.0 }}",
                Self::PROSPECTIVE
            ),
        ]
    }

    /// HNSW vector index (works with Validity-keyed relations — tested).
    pub fn create_vector_index(
        embedding_dim: u32,
        hnsw_m: u32,
        hnsw_ef_construction: u32,
    ) -> String {
        format!(
            "::hnsw create {}:{} {{ dim: {}, m: {}, dtype: F32, fields: [embedding], \
             distance: Cosine, ef_construction: {} }}",
            Self::MEMORIES,
            Self::VECTOR_INDEX,
            embedding_dim,
            hnsw_m,
            hnsw_ef_construction
        )
    }

    /// FTS index with Stemmer for better recall ("running" matches "run").
    pub fn create_fts_index() -> String {
        format!(
            "::fts create {}:{} {{ extractor: content, tokenizer: Simple, \
             filters: [Lowercase, AlphaNumOnly, Stemmer('english')] }}",
            Self::MEMORIES,
            Self::FTS_INDEX
        )
    }

    /// LSH index for near-duplicate detection.
    pub fn create_lsh_index() -> String {
        format!(
            "::lsh create {}:{} {{ extractor: content, tokenizer: Simple, \
             filters: [Lowercase], n_perm: 200, target_threshold: 0.7, n_gram: 3 }}",
            Self::MEMORIES,
            Self::LSH_INDEX
        )
    }

    /// Compact the database after bulk operations.
    pub fn compact_statement() -> &'static str {
        "::compact"
    }
}

use serde::{Deserialize, Serialize};

/// Core engine configuration (storage + buffer settings).
/// LLM and embedding configs are NOT part of the core — they stay in the platform layer.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CoreConfig {
    pub storage: StorageConfig,
    /// Maximum messages in conversation buffer (default 20).
    #[serde(default = "default_conversation_buffer_size")]
    pub conversation_buffer_size: usize,
    /// Maximum items in sensory buffer (default 100).
    #[serde(default = "default_sensory_buffer_capacity")]
    pub sensory_buffer_capacity: usize,
    /// Sensory buffer TTL in milliseconds (default 5000).
    #[serde(default = "default_sensory_ttl_ms")]
    pub sensory_ttl_ms: u64,
}

/// Configuration for the CozoDB storage backend.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StorageConfig {
    /// Whether to use an in-memory database (non-persistent).
    #[serde(default)]
    pub in_memory: bool,
    /// Path to the SQLite database file. If None, auto-determined.
    pub db_path: Option<String>,
    /// HNSW index parameter: max edges per node (12–64).
    #[serde(default = "default_hnsw_m")]
    pub hnsw_m: u32,
    /// HNSW index parameter: candidate list size during construction (100–500).
    #[serde(default = "default_hnsw_ef_construction")]
    pub hnsw_ef_construction: u32,
    /// Embedding vector dimensions (default 1536 for OpenAI text-embedding-3-small).
    #[serde(default = "default_embedding_dimensions")]
    pub embedding_dimensions: u32,
    /// Minimum extraction confidence to store a memory (0.0–1.0).
    #[serde(default = "default_min_confidence")]
    pub min_confidence_to_store: f64,
    /// Whether to apply time-decay scoring in hybrid search.
    #[serde(default)]
    pub enable_time_decay: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            in_memory: false,
            db_path: None,
            hnsw_m: 16,
            hnsw_ef_construction: 100,
            embedding_dimensions: 1536,
            min_confidence_to_store: 0.3,
            enable_time_decay: true,
        }
    }
}

impl Default for CoreConfig {
    fn default() -> Self {
        Self {
            storage: StorageConfig::default(),
            conversation_buffer_size: 20,
            sensory_buffer_capacity: 100,
            sensory_ttl_ms: 5000,
        }
    }
}

fn default_conversation_buffer_size() -> usize {
    20
}
fn default_sensory_buffer_capacity() -> usize {
    100
}
fn default_sensory_ttl_ms() -> u64 {
    5000
}
fn default_hnsw_m() -> u32 {
    16
}
fn default_hnsw_ef_construction() -> u32 {
    100
}
fn default_embedding_dimensions() -> u32 {
    1536
}
fn default_min_confidence() -> f64 {
    0.3
}

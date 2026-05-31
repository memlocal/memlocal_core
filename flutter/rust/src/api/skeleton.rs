use std::sync::Arc;

use flutter_rust_bridge::DartFnFuture;
use memlocal_core::api::MemlocalEngine;
use memlocal_core::models::{CoreConfig, MemoryItem, MemoryType, StorageConfig};

/// Opaque handle held by Dart. (FRB treats unknown structs as opaque.)
pub struct Memlocal {
    pub(crate) engine: Arc<MemlocalEngine>,
}

impl Memlocal {
    /// Open an in-memory engine. Phase 0 smoke entry point.
    pub fn open_in_memory(dimensions: u32) -> Result<Memlocal, String> {
        let config = CoreConfig {
            storage: StorageConfig {
                in_memory: true,
                embedding_dimensions: dimensions,
                ..Default::default()
            },
            ..Default::default()
        };
        let engine = MemlocalEngine::open(config).map_err(|e| e.to_string())?;
        Ok(Memlocal { engine: Arc::new(engine) })
    }

    /// Total stored memories (None = all types).
    pub fn memory_count(&self) -> Result<u32, String> {
        self.engine
            .memory_count(None)
            .map(|c| c as u32)
            .map_err(|e| e.to_string())
    }

    /// Open a persistent engine backed by a SQLite file at `db_path`.
    pub fn open(db_path: String, dimensions: u32) -> Result<Memlocal, String> {
        let config = CoreConfig {
            storage: StorageConfig {
                in_memory: false,
                db_path: Some(db_path),
                embedding_dimensions: dimensions,
                ..Default::default()
            },
            ..Default::default()
        };
        let engine = MemlocalEngine::open(config).map_err(|e| e.to_string())?;
        Ok(Memlocal { engine: Arc::new(engine) })
    }

    /// Store `content` as a Factual memory using a caller-supplied embedding. Returns the new id.
    pub fn add_memory(&self, content: String, embedding: Vec<f32>) -> Result<String, String> {
        let item = MemoryItem::new(content, MemoryType::Factual);
        self.engine
            .put_memory(&item, &embedding)
            .map_err(|e| e.to_string())?;
        Ok(item.id)
    }

    /// Semantic (HNSW) search using a caller-supplied query embedding. Returns the top-k recalled memories.
    pub fn search_semantic(&self, embedding: Vec<f32>, k: u32) -> Result<Vec<RecalledMemory>, String> {
        let items = self
            .engine
            .search_semantic(&embedding, k as usize, None, None)
            .map_err(|e| e.to_string())?;
        Ok(items
            .into_iter()
            .map(|m| RecalledMemory {
                id: m.id,
                content: m.content,
                kind: m.memory_type.stored_name().to_string(),
                score: m.score,
            })
            .collect())
    }
}

/// A memory returned from a search, flattened for the FFI boundary.
pub struct RecalledMemory {
    pub id: String,
    pub content: String,
    /// The stored memory-type name (e.g. "factual").
    pub kind: String,
    /// Relevance score from the search, if available.
    pub score: Option<f64>,
}

/// Calls a Dart-provided async closure and returns its result.
/// Proves FRB can invoke Dart back from Rust (foundation for Dart-side providers).
pub async fn call_dart_closure(
    value: i32,
    callback: impl Fn(i32) -> DartFnFuture<i32>,
) -> i32 {
    callback(value).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opens_in_memory_and_counts_zero() {
        let m = Memlocal::open_in_memory(1536).expect("engine opens");
        assert_eq!(m.memory_count().expect("count works"), 0);
    }

    #[test]
    fn add_then_search_roundtrip() {
        let m = Memlocal::open_in_memory(8).expect("engine opens");
        let emb = vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let id = m.add_memory("hello world".to_string(), emb.clone()).expect("add");
        assert!(!id.is_empty());
        let hits = m.search_semantic(emb, 5).expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].content, "hello world");
    }
}

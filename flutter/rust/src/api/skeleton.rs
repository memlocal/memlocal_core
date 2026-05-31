use std::sync::Arc;

use memlocal_core::api::MemlocalEngine;
use memlocal_core::models::{CoreConfig, StorageConfig};

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opens_in_memory_and_counts_zero() {
        let m = Memlocal::open_in_memory(1536).expect("engine opens");
        assert_eq!(m.memory_count().expect("count works"), 0);
    }
}

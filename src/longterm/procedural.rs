use std::sync::Arc;

use crate::error::Result;
use crate::models::*;
use crate::storage::MemoryStore;

/// Procedural memory: skills, workflows, and routines.
pub struct ProceduralMemory {
    store: Arc<MemoryStore>,
}

impl ProceduralMemory {
    pub fn new(store: Arc<MemoryStore>) -> Self {
        Self { store }
    }

    pub fn record(&self, item: &MemoryItem, embedding: &[f32]) -> Result<()> {
        self.store.put_memory(item, embedding)
    }

    pub fn get_procedures(&self, user_id: Option<&str>, limit: usize) -> Result<Vec<MemoryItem>> {
        self.store
            .get_memories(user_id, Some(MemoryType::Procedural), limit)
    }

    pub fn search(
        &self,
        query_embedding: &[f32],
        k: usize,
        user_id: Option<&str>,
    ) -> Result<Vec<MemoryItem>> {
        self.store
            .search_semantic(query_embedding, k, user_id, Some(MemoryType::Procedural))
    }
}

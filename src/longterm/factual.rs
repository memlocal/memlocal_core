use std::sync::Arc;

use crate::error::Result;
use crate::models::*;
use crate::storage::MemoryStore;

/// Factual memory: stable personal facts and preferences.
pub struct FactualMemory {
    store: Arc<MemoryStore>,
}

impl FactualMemory {
    pub fn new(store: Arc<MemoryStore>) -> Self {
        Self { store }
    }

    pub fn record(&self, item: &MemoryItem, embedding: &[f32]) -> Result<()> {
        self.store.put_memory(item, embedding)
    }

    pub fn get_user_facts(&self, user_id: Option<&str>, limit: usize) -> Result<Vec<MemoryItem>> {
        self.store
            .get_memories(user_id, Some(MemoryType::Factual), limit)
    }

    pub fn search(
        &self,
        query_embedding: &[f32],
        k: usize,
        user_id: Option<&str>,
    ) -> Result<Vec<MemoryItem>> {
        self.store
            .search_semantic(query_embedding, k, user_id, Some(MemoryType::Factual))
    }
}

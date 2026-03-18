use std::collections::HashMap;
use std::sync::Arc;

use crate::error::Result;
use crate::models::*;
use crate::storage::MemoryStore;

/// Social memory: contact graph, relationships, interaction patterns.
pub struct SocialMemory {
    store: Arc<MemoryStore>,
}

impl SocialMemory {
    pub fn new(store: Arc<MemoryStore>) -> Self {
        Self { store }
    }

    pub fn record(&self, item: &MemoryItem, embedding: &[f32]) -> Result<()> {
        self.store.put_memory(item, embedding)
    }

    pub fn record_relationship(&self, edge: &MemoryEdge) -> Result<()> {
        self.store.put_edge(edge)
    }

    pub fn get_contacts(&self, user_id: Option<&str>, limit: usize) -> Result<Vec<MemoryItem>> {
        self.store
            .get_memories(user_id, Some(MemoryType::Social), limit)
    }

    pub fn get_relationships(&self, memory_id: &str) -> Result<Vec<MemoryEdge>> {
        let mut edges = self.store.get_edges_from(memory_id)?;
        edges.extend(self.store.get_edges_to(memory_id)?);
        Ok(edges)
    }

    pub fn detect_communities(&self) -> Result<HashMap<String, i64>> {
        self.store.community_detection()
    }

    pub fn page_rank(&self, iterations: usize) -> Result<HashMap<String, f64>> {
        self.store.page_rank(iterations)
    }

    pub fn shortest_path(&self, from_id: &str, to_id: &str) -> Result<Option<Vec<String>>> {
        self.store.shortest_path(from_id, to_id)
    }
}

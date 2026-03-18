use serde::{Deserialize, Serialize};

use super::memory_item::MemoryItem;
use super::search_mode::SearchMode;

/// Result container for memory search queries.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MemorySearchResult {
    /// The matched memory items (with scores populated).
    pub items: Vec<MemoryItem>,
    /// The original search query.
    pub query: String,
    /// The search mode that was used.
    pub mode: SearchMode,
    /// How long the search took in milliseconds.
    pub duration_ms: u64,
    /// Total count of matching items (may exceed items.len() when limited).
    pub total_count: Option<usize>,
}

impl MemorySearchResult {
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// An empty result.
    pub fn empty(query: String, mode: SearchMode) -> Self {
        Self {
            items: vec![],
            query,
            mode,
            duration_ms: 0,
            total_count: Some(0),
        }
    }
}

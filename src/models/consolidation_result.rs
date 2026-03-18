use serde::{Deserialize, Serialize};

use super::memory_item::MemoryItem;

/// Result of a memory consolidation pass.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ConsolidationResult {
    /// Total number of episodic memories that were consolidated.
    pub memories_consolidated: usize,
    /// Number of semantic summary memories created.
    pub summaries_created: usize,
    /// The newly created summary memory items.
    pub summaries: Vec<MemoryItem>,
    /// Wall-clock time taken for the consolidation pass.
    pub duration_ms: u64,
}

impl ConsolidationResult {
    /// Whether any consolidation occurred.
    pub fn has_changes(&self) -> bool {
        self.summaries_created > 0
    }
}

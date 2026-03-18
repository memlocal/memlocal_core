use serde::{Deserialize, Serialize};

use super::memory_item::MemoryItem;

/// Result of adding content to memory.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AddResult {
    /// The memory items that were created or updated.
    pub items: Vec<MemoryItem>,
    /// How long the operation took in milliseconds.
    pub duration_ms: u64,
}

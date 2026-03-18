use serde::{Deserialize, Serialize};

use super::memory_item::MemoryItem;

/// What the extraction pipeline decided to do with a piece of information.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryAction {
    /// Create a new memory.
    Add,
    /// Update an existing memory.
    Update,
    /// Delete / invalidate an existing memory.
    Delete,
    /// No action needed (duplicate or irrelevant).
    None,
}

/// A single change proposed or applied by the extraction pipeline.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MemoryDelta {
    pub action: MemoryAction,
    pub old_memory: Option<MemoryItem>,
    pub new_memory: Option<MemoryItem>,
    pub reason: Option<String>,
    pub confidence: f64,
}

impl MemoryDelta {
    /// Factory for an add delta.
    pub fn add(item: MemoryItem, reason: Option<String>, confidence: f64) -> Self {
        Self {
            action: MemoryAction::Add,
            new_memory: Some(item),
            old_memory: None,
            reason,
            confidence,
        }
    }

    /// Factory for an update delta.
    pub fn update(
        old: MemoryItem,
        updated: MemoryItem,
        reason: Option<String>,
        confidence: f64,
    ) -> Self {
        Self {
            action: MemoryAction::Update,
            old_memory: Some(old),
            new_memory: Some(updated),
            reason,
            confidence,
        }
    }

    /// Factory for a delete delta.
    pub fn delete(item: MemoryItem, reason: Option<String>) -> Self {
        Self {
            action: MemoryAction::Delete,
            old_memory: Some(item),
            new_memory: None,
            reason,
            confidence: 1.0,
        }
    }

    /// Factory for a no-op delta.
    pub fn none(reason: Option<String>) -> Self {
        Self {
            action: MemoryAction::None,
            old_memory: None,
            new_memory: None,
            reason,
            confidence: 1.0,
        }
    }
}

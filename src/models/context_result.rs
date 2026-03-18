use serde::{Deserialize, Serialize};

use super::memory_item::MemoryItem;
use super::prospective_item::ProspectiveItem;
use super::user_profile::UserProfile;

/// Result of assembling a context block for LLM injection.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ContextResult {
    /// The assembled context block text.
    pub context_block: String,
    /// Memories relevant to the current query.
    pub relevant_memories: Vec<MemoryItem>,
    /// High-importance memories regardless of query.
    pub important_memories: Vec<MemoryItem>,
    /// Prospective triggers that fired.
    pub triggered_reminders: Vec<ProspectiveItem>,
    /// The user profile (if loaded).
    pub profile: Option<UserProfile>,
    /// How long the context assembly took in milliseconds.
    pub duration_ms: u64,
}

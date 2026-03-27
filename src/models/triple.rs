use serde::{Deserialize, Serialize};

/// A semantic triple (subject, predicate, object) extracted from conversation.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub memory_id: String,
    pub speaker: String,
    pub mention_count: u64,
    pub last_mentioned: f64,
    pub session_id: String,
    pub confidence: f64,
}

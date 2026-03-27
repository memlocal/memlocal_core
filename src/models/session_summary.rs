use serde::{Deserialize, Serialize};

/// A summary of a conversation session.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SessionSummary {
    pub session_id: String,
    pub summary: String,
    pub speakers: Vec<String>,
    pub key_topics: Vec<String>,
    pub document_date: f64,
    /// Relevance score from search (transient).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
}

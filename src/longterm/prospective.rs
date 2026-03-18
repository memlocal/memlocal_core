use std::sync::Arc;

use crate::error::Result;
use crate::models::*;
use crate::storage::MemoryStore;

/// Prospective memory: future intentions, reminders, planned actions.
pub struct ProspectiveMemory {
    store: Arc<MemoryStore>,
}

/// Cosine similarity threshold for semantic triggers.
const _SEMANTIC_TRIGGER_THRESHOLD: f64 = 0.75;

impl ProspectiveMemory {
    pub fn new(store: Arc<MemoryStore>) -> Self {
        Self { store }
    }

    pub fn record(&self, item: &MemoryItem, embedding: &[f32]) -> Result<()> {
        self.store.put_memory(item, embedding)
    }

    pub fn add_trigger(&self, item: &ProspectiveItem) -> Result<()> {
        self.store.put_prospective(item)
    }

    pub fn get_pending(&self, user_id: Option<&str>) -> Result<Vec<ProspectiveItem>> {
        self.store.get_pending_prospective(user_id)
    }

    pub fn complete_trigger(&self, trigger_id: &str) -> Result<()> {
        self.store.complete_prospective(trigger_id)
    }

    pub fn get_all(&self, user_id: Option<&str>, limit: usize) -> Result<Vec<MemoryItem>> {
        self.store
            .get_memories(user_id, Some(MemoryType::Prospective), limit)
    }

    /// Check which triggers fire for the given context.
    ///
    /// - `TopicMention`: checks if trigger_condition appears in the query string
    /// - `SemanticMatch`: checks cosine similarity between query embedding and
    ///   trigger condition embedding (requires embedding for the trigger)
    /// - `UserPresence`: checks if user_id matches
    /// - `TimeBased`: checks if current time is past the trigger condition
    pub fn check_triggers(
        &self,
        query: &str,
        _query_embedding: Option<&[f32]>,
        user_id: Option<&str>,
    ) -> Result<Vec<ProspectiveItem>> {
        let pending = self.store.get_pending_prospective(user_id)?;
        let now = chrono::Utc::now();
        let query_lower = query.to_lowercase();

        let mut triggered = Vec::new();
        for item in pending {
            let fires = match item.trigger_type {
                TriggerType::TopicMention => {
                    query_lower.contains(&item.trigger_condition.to_lowercase())
                }
                TriggerType::TimeBased => {
                    // Parse trigger_condition as ISO datetime
                    if let Ok(trigger_time) =
                        chrono::DateTime::parse_from_rfc3339(&item.trigger_condition)
                    {
                        now >= trigger_time
                    } else {
                        false
                    }
                }
                TriggerType::UserPresence => user_id
                    .map(|uid| uid == item.trigger_condition)
                    .unwrap_or(false),
                TriggerType::SemanticMatch => {
                    // Semantic matching requires embedding comparison.
                    // For now, fall back to keyword matching since we don't have
                    // the trigger's embedding stored separately.
                    // The platform layer can do a proper semantic comparison.
                    query_lower.contains(&item.trigger_condition.to_lowercase())
                }
            };
            if fires {
                triggered.push(item);
            }
        }
        Ok(triggered)
    }
}

#[allow(dead_code)]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for i in 0..a.len() {
        dot += a[i] as f64 * b[i] as f64;
        norm_a += (a[i] as f64).powi(2);
        norm_b += (b[i] as f64).powi(2);
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

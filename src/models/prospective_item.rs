use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{MemlocalError, Result};

/// How a prospective memory item should be triggered.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TriggerType {
    /// Trigger when a specific topic is mentioned.
    TopicMention,
    /// Trigger at a specific time.
    TimeBased,
    /// Trigger when talking to a specific user.
    UserPresence,
    /// Trigger when a semantic condition matches.
    SemanticMatch,
}

impl TriggerType {
    /// The string stored in CozoDB.
    pub fn stored_name(&self) -> &'static str {
        match self {
            Self::TopicMention => "topic_mention",
            Self::TimeBased => "time_based",
            Self::UserPresence => "user_presence",
            Self::SemanticMatch => "semantic_match",
        }
    }

    /// Look up by stored name. Defaults to `TopicMention`.
    pub fn from_stored_name(name: &str) -> Self {
        match name {
            "topic_mention" => Self::TopicMention,
            "time_based" => Self::TimeBased,
            "user_presence" => Self::UserPresence,
            "semantic_match" => Self::SemanticMatch,
            _ => Self::TopicMention,
        }
    }
}

/// A "remember to do X" item — future-oriented memory that fires when
/// conditions are met.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ProspectiveItem {
    pub id: String,
    pub content: String,
    pub trigger_type: TriggerType,
    pub trigger_condition: String,
    pub user_id: Option<String>,
    pub completed: bool,
    pub created_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

impl ProspectiveItem {
    /// Convert to a map for CozoDB storage.
    /// Note: `completed` is stored as Int (0/1) in CozoDB.
    pub fn to_map(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id,
            "content": self.content,
            "trigger_type": self.trigger_type.stored_name(),
            "trigger_condition": self.trigger_condition,
            "user_id": self.user_id.as_deref().unwrap_or(""),
            "completed": if self.completed { 1 } else { 0 },
            "created_at": self.created_at
                .map(|dt| dt.timestamp_millis() as f64 / 1000.0)
                .unwrap_or_else(|| Utc::now().timestamp_millis() as f64 / 1000.0),
            "completed_at": self.completed_at
                .map(|dt| dt.timestamp_millis() as f64 / 1000.0)
                .unwrap_or(0.0),
        })
    }

    /// Create from a storage map.
    pub fn from_map(map: &serde_json::Value) -> Result<Self> {
        let id = map["id"]
            .as_str()
            .ok_or_else(|| MemlocalError::Query("missing 'id'".into()))?
            .to_string();
        let content = map["content"]
            .as_str()
            .ok_or_else(|| MemlocalError::Query("missing 'content'".into()))?
            .to_string();
        let trigger_type =
            TriggerType::from_stored_name(map["trigger_type"].as_str().unwrap_or("topic_mention"));
        let trigger_condition = map["trigger_condition"].as_str().unwrap_or("").to_string();
        let user_id = map["user_id"].as_str().and_then(|s| {
            if s.is_empty() {
                None
            } else {
                Some(s.to_string())
            }
        });
        let completed = map["completed"].as_i64().unwrap_or(0) == 1;

        let created_at_val = map["created_at"].as_f64().unwrap_or(0.0);
        let completed_at_val = map["completed_at"].as_f64().unwrap_or(0.0);

        let created_at = if created_at_val > 0.0 {
            let millis = (created_at_val * 1000.0).round() as i64;
            Utc.timestamp_millis_opt(millis).single()
        } else {
            None
        };
        let completed_at = if completed_at_val > 0.0 {
            let millis = (completed_at_val * 1000.0).round() as i64;
            Utc.timestamp_millis_opt(millis).single()
        } else {
            None
        };

        Ok(Self {
            id,
            content,
            trigger_type,
            trigger_condition,
            user_id,
            completed,
            created_at,
            completed_at,
        })
    }
}

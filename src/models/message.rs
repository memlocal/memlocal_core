use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{MemlocalError, Result};

/// A single message in a conversation.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub session_id: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

impl Message {
    /// Convenience constructor for a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
            timestamp: Utc::now(),
            session_id: None,
            metadata: None,
        }
    }

    /// Convenience constructor for an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
            timestamp: Utc::now(),
            session_id: None,
            metadata: None,
        }
    }

    /// Convenience constructor for a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
            timestamp: Utc::now(),
            session_id: None,
            metadata: None,
        }
    }

    /// Convert to a map for storage.
    pub fn to_map(&self) -> serde_json::Value {
        let mut map = serde_json::json!({
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.timestamp_millis() as f64 / 1000.0,
        });
        if let Some(sid) = &self.session_id {
            map["session_id"] = serde_json::Value::String(sid.clone());
        }
        if let Some(meta) = &self.metadata {
            map["metadata"] = meta.clone();
        }
        map
    }

    /// Create from a storage map.
    pub fn from_map(map: &serde_json::Value) -> Result<Self> {
        let role = map["role"]
            .as_str()
            .ok_or_else(|| MemlocalError::Query("missing 'role'".into()))?
            .to_string();
        let content = map["content"]
            .as_str()
            .ok_or_else(|| MemlocalError::Query("missing 'content'".into()))?
            .to_string();
        let ts_val = map["timestamp"]
            .as_f64()
            .ok_or_else(|| MemlocalError::Query("missing 'timestamp'".into()))?;
        let millis = (ts_val * 1000.0).round() as i64;
        let timestamp = Utc
            .timestamp_millis_opt(millis)
            .single()
            .unwrap_or_default();
        let session_id = map
            .get("session_id")
            .and_then(|v| v.as_str())
            .map(String::from);
        let metadata = map.get("metadata").cloned();

        Ok(Self {
            role,
            content,
            timestamp,
            session_id,
            metadata,
        })
    }
}

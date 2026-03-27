use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::memory_type::MemoryType;
use crate::error::{MemlocalError, Result};

/// A single memory item stored in the memory layer.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MemoryItem {
    pub id: String,
    pub content: String,
    #[serde(rename = "type")]
    pub memory_type: MemoryType,
    pub hash: String,
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub session_id: Option<String>,
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub valid_at: Option<DateTime<Utc>>,
    pub invalid_at: Option<DateTime<Utc>>,
    /// Relevance score from a search query (transient, not stored).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
}

impl MemoryItem {
    /// Compute a SHA-256 hash of the content for deduplication.
    pub fn compute_hash(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Whether this memory is currently valid (not invalidated).
    pub fn is_valid(&self) -> bool {
        self.invalid_at.is_none()
    }

    /// Convert to a map suitable for CozoDB storage.
    pub fn to_map(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id,
            "content": self.content,
            "type": self.memory_type.stored_name(),
            "hash": self.hash,
            "user_id": self.user_id.as_deref().unwrap_or(""),
            "agent_id": self.agent_id.as_deref().unwrap_or(""),
            "session_id": self.session_id.as_deref().unwrap_or(""),
            "metadata_json": serde_json::to_string(&self.metadata).unwrap_or_else(|_| "{}".to_string()),
            "created_at": self.created_at.timestamp_millis() as f64 / 1000.0,
            "updated_at": self.updated_at.timestamp_millis() as f64 / 1000.0,
            "valid_at": self.valid_at.map(|dt| dt.timestamp_millis() as f64 / 1000.0).unwrap_or(0.0),
            "invalid_at": self.invalid_at.map(|dt| dt.timestamp_millis() as f64 / 1000.0).unwrap_or(0.0),
        })
    }

    /// Create from a storage map (CozoDB row as JSON).
    pub fn from_map(map: &serde_json::Value) -> Result<Self> {
        let id = map["id"]
            .as_str()
            .ok_or_else(|| MemlocalError::Query("missing 'id'".into()))?
            .to_string();
        let content = map["content"]
            .as_str()
            .ok_or_else(|| MemlocalError::Query("missing 'content'".into()))?
            .to_string();
        let type_str = map["type"]
            .as_str()
            .ok_or_else(|| MemlocalError::Query("missing 'type'".into()))?;
        let hash = map["hash"]
            .as_str()
            .ok_or_else(|| MemlocalError::Query("missing 'hash'".into()))?
            .to_string();

        let user_id = empty_to_none(map["user_id"].as_str());
        let agent_id = empty_to_none(map["agent_id"].as_str());
        let session_id = empty_to_none(map["session_id"].as_str());

        let metadata: serde_json::Value = match map["metadata_json"].as_str() {
            Some(s) if !s.is_empty() => {
                serde_json::from_str(s).unwrap_or(serde_json::Value::Object(Default::default()))
            }
            _ => serde_json::Value::Object(Default::default()),
        };

        let created_at = epoch_to_datetime(map["created_at"].as_f64().unwrap_or(0.0));
        let updated_at = epoch_to_datetime(map["updated_at"].as_f64().unwrap_or(0.0));

        let valid_at_val = map["valid_at"].as_f64().unwrap_or(0.0);
        let invalid_at_val = map["invalid_at"].as_f64().unwrap_or(0.0);

        let valid_at = if valid_at_val > 0.0 {
            Some(epoch_to_datetime(valid_at_val))
        } else {
            None
        };
        let invalid_at = if invalid_at_val > 0.0 {
            Some(epoch_to_datetime(invalid_at_val))
        } else {
            None
        };

        let score = map.get("score").and_then(|v| v.as_f64());

        Ok(Self {
            id,
            content,
            memory_type: MemoryType::from_stored_name(type_str),
            hash,
            user_id,
            agent_id,
            session_id,
            metadata,
            created_at,
            updated_at,
            valid_at,
            invalid_at,
            score,
        })
    }

    /// Creates a copy with an overridden score.
    pub fn with_score(mut self, score: f64) -> Self {
        self.score = Some(score);
        self
    }

    pub fn reinforcement_count(&self) -> u64 {
        self.metadata.get("reinforcement_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(1)
    }

    pub fn is_latest(&self) -> bool {
        self.metadata.get("is_latest")
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
    }

    pub fn speaker(&self) -> &str {
        self.metadata.get("speaker")
            .and_then(|v| v.as_str())
            .unwrap_or("")
    }
}

fn empty_to_none(s: Option<&str>) -> Option<String> {
    match s {
        Some(s) if !s.is_empty() => Some(s.to_string()),
        _ => None,
    }
}

fn epoch_to_datetime(epoch_secs: f64) -> DateTime<Utc> {
    let millis = (epoch_secs * 1000.0).round() as i64;
    Utc.timestamp_millis_opt(millis)
        .single()
        .unwrap_or_default()
}

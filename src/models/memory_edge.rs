use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};

use super::memory_relation::MemoryRelation;
use crate::error::{MemlocalError, Result};

/// A directed edge in the memory knowledge graph.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MemoryEdge {
    pub from_id: String,
    pub to_id: String,
    pub relation: MemoryRelation,
    pub weight: f64,
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
}

impl MemoryEdge {
    /// Create a new edge with defaults.
    pub fn new(from_id: String, to_id: String, relation: MemoryRelation) -> Self {
        Self {
            from_id,
            to_id,
            relation,
            weight: 1.0,
            metadata: serde_json::Value::Object(Default::default()),
            created_at: Utc::now(),
        }
    }

    /// Convert to a map for CozoDB storage.
    pub fn to_map(&self) -> serde_json::Value {
        serde_json::json!({
            "from_id": self.from_id,
            "to_id": self.to_id,
            "relation": self.relation.stored_name(),
            "weight": self.weight,
            "created_at": self.created_at.timestamp_millis() as f64 / 1000.0,
        })
    }

    /// Create from a storage map.
    pub fn from_map(map: &serde_json::Value) -> Result<Self> {
        let from_id = map["from_id"]
            .as_str()
            .ok_or_else(|| MemlocalError::Query("missing 'from_id'".into()))?
            .to_string();
        let to_id = map["to_id"]
            .as_str()
            .ok_or_else(|| MemlocalError::Query("missing 'to_id'".into()))?
            .to_string();
        let relation_str = map["relation"]
            .as_str()
            .ok_or_else(|| MemlocalError::Query("missing 'relation'".into()))?;
        let weight = map["weight"].as_f64().unwrap_or(1.0);
        let created_at_val = map["created_at"].as_f64().unwrap_or(0.0);
        let created_at = if created_at_val > 0.0 {
            let millis = (created_at_val * 1000.0).round() as i64;
            Utc.timestamp_millis_opt(millis)
                .single()
                .unwrap_or_default()
        } else {
            Utc::now()
        };

        Ok(Self {
            from_id,
            to_id,
            relation: MemoryRelation::from_stored_name(relation_str),
            weight,
            metadata: serde_json::Value::Object(Default::default()),
            created_at,
        })
    }
}

use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::error::{MemlocalError, Result};

/// An automatically maintained user profile built from extracted memories.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct UserProfile {
    pub user_id: String,
    /// Long-lived facts about the user (name, occupation, preferences, etc.).
    pub static_facts: BTreeMap<String, String>,
    /// Dynamic context that changes over time (current mood, recent topics, etc.).
    pub dynamic_context: BTreeMap<String, String>,
    pub updated_at: Option<DateTime<Utc>>,
}

impl UserProfile {
    /// Whether the profile has any content.
    pub fn is_empty(&self) -> bool {
        self.static_facts.is_empty() && self.dynamic_context.is_empty()
    }

    pub fn is_not_empty(&self) -> bool {
        !self.is_empty()
    }

    /// Convert to a map for CozoDB storage.
    pub fn to_map(&self) -> serde_json::Value {
        serde_json::json!({
            "user_id": self.user_id,
            "static_facts_json": serde_json::to_string(&self.static_facts).unwrap_or_else(|_| "{}".to_string()),
            "dynamic_context_json": serde_json::to_string(&self.dynamic_context).unwrap_or_else(|_| "{}".to_string()),
            "updated_at": self.updated_at
                .map(|dt| dt.timestamp_millis() as f64 / 1000.0)
                .unwrap_or_else(|| Utc::now().timestamp_millis() as f64 / 1000.0),
        })
    }

    /// Create from a storage map.
    pub fn from_map(map: &serde_json::Value) -> Result<Self> {
        let user_id = map["user_id"]
            .as_str()
            .ok_or_else(|| MemlocalError::Query("missing 'user_id'".into()))?
            .to_string();

        let static_facts = decode_string_map(map["static_facts_json"].as_str());
        let dynamic_context = decode_string_map(map["dynamic_context_json"].as_str());

        let updated_at_val = map["updated_at"].as_f64().unwrap_or(0.0);
        let updated_at = if updated_at_val > 0.0 {
            let millis = (updated_at_val * 1000.0).round() as i64;
            Utc.timestamp_millis_opt(millis).single()
        } else {
            None
        };

        Ok(Self {
            user_id,
            static_facts,
            dynamic_context,
            updated_at,
        })
    }

    /// Produce a textual summary for LLM context injection.
    pub fn to_summary(&self) -> String {
        let mut buf = String::new();
        if !self.static_facts.is_empty() {
            buf.push_str("User Facts:\n");
            for (k, v) in &self.static_facts {
                buf.push_str(&format!("  - {k}: {v}\n"));
            }
        }
        if !self.dynamic_context.is_empty() {
            buf.push_str("Current Context:\n");
            for (k, v) in &self.dynamic_context {
                buf.push_str(&format!("  - {k}: {v}\n"));
            }
        }
        buf
    }
}

fn decode_string_map(json: Option<&str>) -> BTreeMap<String, String> {
    match json {
        Some(s) if !s.is_empty() => serde_json::from_str::<BTreeMap<String, serde_json::Value>>(s)
            .map(|m| {
                m.into_iter()
                    .map(|(k, v)| (k, v.to_string().trim_matches('"').to_string()))
                    .collect()
            })
            .unwrap_or_default(),
        _ => BTreeMap::new(),
    }
}

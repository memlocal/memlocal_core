use std::sync::Arc;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::tool_definitions::tool_names;
use crate::error::{MemlocalError, Result};
use crate::models::*;
use crate::storage::MemoryStore;

/// Trait for embedding generation — implemented by the platform layer.
pub trait EmbeddingProvider: Send + Sync {
    fn embed_one(&self, text: &str) -> Result<Vec<f32>>;
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub tool_name: String,
    pub content: String,
    pub duration_ms: u64,
    pub success: bool,
}

pub struct ToolExecutor {
    store: Arc<MemoryStore>,
}

impl ToolExecutor {
    pub fn new(store: Arc<MemoryStore>) -> Self {
        Self { store }
    }

    /// Execute a single tool call.
    pub fn execute(
        &self,
        tool_call: &ToolCall,
        embedding_provider: &dyn EmbeddingProvider,
    ) -> ToolResult {
        let start = std::time::Instant::now();
        let result = self.dispatch(&tool_call.name, &tool_call.arguments, embedding_provider);
        let duration_ms = start.elapsed().as_millis() as u64;

        match result {
            Ok(value) => ToolResult {
                tool_call_id: tool_call.id.clone(),
                tool_name: tool_call.name.clone(),
                content: serde_json::to_string(&value).unwrap_or_default(),
                duration_ms,
                success: true,
            },
            Err(e) => ToolResult {
                tool_call_id: tool_call.id.clone(),
                tool_name: tool_call.name.clone(),
                content: serde_json::json!({"error": e.to_string()}).to_string(),
                duration_ms,
                success: false,
            },
        }
    }

    /// Execute multiple tool calls in sequence.
    pub fn execute_all(
        &self,
        tool_calls: &[ToolCall],
        embedding_provider: &dyn EmbeddingProvider,
    ) -> Vec<ToolResult> {
        tool_calls
            .iter()
            .map(|tc| self.execute(tc, embedding_provider))
            .collect()
    }

    fn dispatch(
        &self,
        name: &str,
        args: &serde_json::Value,
        embedding_provider: &dyn EmbeddingProvider,
    ) -> Result<serde_json::Value> {
        match name {
            tool_names::ADD_MEMORY => self.add_memory(args, embedding_provider),
            tool_names::SEARCH_MEMORY => self.search_memory(args, embedding_provider),
            tool_names::GET_MEMORIES => self.get_memories(args),
            tool_names::DELETE_MEMORY => self.delete_memory(args),
            tool_names::GET_PROFILE => self.get_profile(args),
            tool_names::ADD_RELATIONSHIP => self.add_relationship(args),
            tool_names::GET_RELATIONSHIPS => self.get_relationships(args),
            tool_names::ADD_REMINDER => self.add_reminder(args, embedding_provider),
            tool_names::GET_CONTEXT => self.get_context(args, embedding_provider),
            _ => Err(MemlocalError::InvalidArgument(format!(
                "Unknown tool: {name}"
            ))),
        }
    }

    fn add_memory(
        &self,
        args: &serde_json::Value,
        embedding_provider: &dyn EmbeddingProvider,
    ) -> Result<serde_json::Value> {
        let content = args["content"]
            .as_str()
            .ok_or_else(|| MemlocalError::InvalidArgument("missing 'content'".into()))?;
        let type_str = args["memory_type"].as_str().unwrap_or("factual");
        let user_id = args["user_id"].as_str().map(String::from);

        let memory_type = MemoryType::from_stored_name(type_str);
        let id = Uuid::new_v4().to_string();
        let now = Utc::now();

        let item = MemoryItem {
            id: id.clone(),
            content: content.to_string(),
            memory_type,
            hash: MemoryItem::compute_hash(content),
            user_id,
            agent_id: None,
            session_id: None,
            metadata: serde_json::Value::Object(Default::default()),
            created_at: now,
            updated_at: now,
            valid_at: None,
            invalid_at: None,
            score: None,
        };

        let embedding = embedding_provider.embed_one(content)?;
        self.store.put_memory(&item, &embedding)?;

        Ok(serde_json::json!({
            "status": "stored",
            "memory_id": id,
            "type": memory_type.stored_name(),
        }))
    }

    fn search_memory(
        &self,
        args: &serde_json::Value,
        embedding_provider: &dyn EmbeddingProvider,
    ) -> Result<serde_json::Value> {
        let query = args["query"]
            .as_str()
            .ok_or_else(|| MemlocalError::InvalidArgument("missing 'query'".into()))?;
        let mode_str = args["mode"].as_str().unwrap_or("hybrid");
        let limit = args["limit"].as_u64().unwrap_or(10) as usize;
        let user_id = args["user_id"].as_str();
        let type_str = args["memory_type"].as_str();

        let mode = SearchMode::from_str_lossy(mode_str);
        let memory_type = type_str.map(MemoryType::from_stored_name);

        let start = std::time::Instant::now();

        let needs_embedding = matches!(
            mode,
            SearchMode::Semantic | SearchMode::Hybrid | SearchMode::Graph
        );
        let embedding = if needs_embedding {
            Some(embedding_provider.embed_one(query)?)
        } else {
            None
        };

        let items = self.store.search(
            query,
            embedding.as_deref(),
            mode,
            limit,
            user_id,
            memory_type,
        )?;

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(serde_json::json!({
            "results": items.iter().map(|m| serde_json::json!({
                "id": m.id,
                "content": m.content,
                "type": m.memory_type.stored_name(),
                "score": m.score,
                "created_at": m.created_at.to_rfc3339(),
            })).collect::<Vec<_>>(),
            "total": items.len(),
            "search_time_ms": duration_ms,
            "mode": mode_str,
        }))
    }

    fn get_memories(&self, args: &serde_json::Value) -> Result<serde_json::Value> {
        let user_id = args["user_id"].as_str();
        let type_str = args["memory_type"].as_str();
        let limit = args["limit"].as_u64().unwrap_or(20) as usize;

        let memory_type = type_str.map(MemoryType::from_stored_name);
        let items = self.store.get_memories(user_id, memory_type, limit)?;

        Ok(serde_json::json!({
            "memories": items.iter().map(|m| serde_json::json!({
                "id": m.id,
                "content": m.content,
                "type": m.memory_type.stored_name(),
                "created_at": m.created_at.to_rfc3339(),
            })).collect::<Vec<_>>(),
            "total": items.len(),
        }))
    }

    fn delete_memory(&self, args: &serde_json::Value) -> Result<serde_json::Value> {
        let memory_id = args["memory_id"]
            .as_str()
            .ok_or_else(|| MemlocalError::InvalidArgument("missing 'memory_id'".into()))?;
        self.store.delete_memory(memory_id)?;
        Ok(serde_json::json!({
            "status": "deleted",
            "memory_id": memory_id,
        }))
    }

    fn get_profile(&self, args: &serde_json::Value) -> Result<serde_json::Value> {
        let user_id = args["user_id"]
            .as_str()
            .ok_or_else(|| MemlocalError::InvalidArgument("missing 'user_id'".into()))?;
        let profile = self.store.get_profile(user_id)?;
        match profile {
            Some(p) => Ok(serde_json::json!({
                "user_id": p.user_id,
                "summary": p.to_summary(),
                "static_facts": p.static_facts,
                "dynamic_context": p.dynamic_context,
            })),
            None => Ok(serde_json::json!({
                "status": "not_found",
                "user_id": user_id,
            })),
        }
    }

    fn add_relationship(&self, args: &serde_json::Value) -> Result<serde_json::Value> {
        let from_id = args["from_id"]
            .as_str()
            .ok_or_else(|| MemlocalError::InvalidArgument("missing 'from_id'".into()))?;
        let to_id = args["to_id"]
            .as_str()
            .ok_or_else(|| MemlocalError::InvalidArgument("missing 'to_id'".into()))?;
        let relation_str = args["relation"]
            .as_str()
            .ok_or_else(|| MemlocalError::InvalidArgument("missing 'relation'".into()))?;
        let weight = args["weight"].as_f64().unwrap_or(1.0);

        let relation = MemoryRelation::from_stored_name(relation_str);
        let edge = MemoryEdge {
            from_id: from_id.to_string(),
            to_id: to_id.to_string(),
            relation,
            weight,
            metadata: serde_json::Value::Object(Default::default()),
            created_at: Utc::now(),
        };

        self.store.put_edge(&edge)?;

        Ok(serde_json::json!({
            "status": "created",
            "from_id": from_id,
            "to_id": to_id,
            "relation": relation.stored_name(),
        }))
    }

    fn get_relationships(&self, args: &serde_json::Value) -> Result<serde_json::Value> {
        let memory_id = args["memory_id"]
            .as_str()
            .ok_or_else(|| MemlocalError::InvalidArgument("missing 'memory_id'".into()))?;

        let from = self.store.get_edges_from(memory_id)?;
        let to = self.store.get_edges_to(memory_id)?;

        Ok(serde_json::json!({
            "memory_id": memory_id,
            "outgoing": from.iter().map(|e| serde_json::json!({
                "to": e.to_id,
                "relation": e.relation.stored_name(),
                "weight": e.weight,
            })).collect::<Vec<_>>(),
            "incoming": to.iter().map(|e| serde_json::json!({
                "from": e.from_id,
                "relation": e.relation.stored_name(),
                "weight": e.weight,
            })).collect::<Vec<_>>(),
        }))
    }

    fn add_reminder(
        &self,
        args: &serde_json::Value,
        embedding_provider: &dyn EmbeddingProvider,
    ) -> Result<serde_json::Value> {
        let content = args["content"]
            .as_str()
            .ok_or_else(|| MemlocalError::InvalidArgument("missing 'content'".into()))?;
        let trigger_type_str = args["trigger_type"]
            .as_str()
            .ok_or_else(|| MemlocalError::InvalidArgument("missing 'trigger_type'".into()))?;
        let trigger_condition = args["trigger_condition"]
            .as_str()
            .ok_or_else(|| MemlocalError::InvalidArgument("missing 'trigger_condition'".into()))?;
        let user_id = args["user_id"].as_str().map(String::from);

        let trigger_type = TriggerType::from_stored_name(trigger_type_str);
        let id = Uuid::new_v4().to_string();

        let item = ProspectiveItem {
            id: id.clone(),
            content: content.to_string(),
            trigger_type,
            trigger_condition: trigger_condition.to_string(),
            user_id: user_id.clone(),
            completed: false,
            created_at: Some(Utc::now()),
            completed_at: None,
        };

        self.store.put_prospective(&item)?;

        // Also store as a regular memory for semantic search
        let memory_text = if trigger_type == TriggerType::SemanticMatch {
            trigger_condition.to_string()
        } else {
            format!("Reminder: {content} (trigger: {trigger_condition})")
        };

        let memory_item = MemoryItem {
            id: Uuid::new_v4().to_string(),
            content: format!("Reminder: {content} (trigger: {trigger_condition})"),
            memory_type: MemoryType::Prospective,
            hash: MemoryItem::compute_hash(content),
            user_id,
            agent_id: None,
            session_id: None,
            metadata: serde_json::json!({
                "reminder_id": id,
                "trigger_type": trigger_type_str,
                "trigger_condition": trigger_condition,
            }),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            valid_at: None,
            invalid_at: None,
            score: None,
        };

        let embedding = embedding_provider.embed_one(&memory_text)?;
        self.store.put_memory(&memory_item, &embedding)?;

        Ok(serde_json::json!({
            "status": "created",
            "reminder_id": id,
            "trigger_type": trigger_type_str,
        }))
    }

    fn get_context(
        &self,
        args: &serde_json::Value,
        embedding_provider: &dyn EmbeddingProvider,
    ) -> Result<serde_json::Value> {
        let query = args["query"]
            .as_str()
            .ok_or_else(|| MemlocalError::InvalidArgument("missing 'query'".into()))?;
        let user_id = args["user_id"].as_str();

        let start = std::time::Instant::now();

        let embedding = embedding_provider.embed_one(query)?;
        let memories = self
            .store
            .search_hybrid(query, &embedding, 10, user_id, None)?;

        let profile = if let Some(uid) = user_id {
            self.store.get_profile(uid)?
        } else {
            None
        };

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(serde_json::json!({
            "relevant_memories": memories.iter().map(|m| serde_json::json!({
                "id": m.id,
                "content": m.content,
                "type": m.memory_type.stored_name(),
                "score": m.score,
            })).collect::<Vec<_>>(),
            "user_profile": profile.as_ref().map(|p| p.to_summary()),
            "retrieval_time_ms": duration_ms,
        }))
    }
}

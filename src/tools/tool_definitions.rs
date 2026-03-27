use serde::{Deserialize, Serialize};

/// Tool name constants.
pub mod tool_names {
    pub const ADD_MEMORY: &str = "add_memory";
    pub const SEARCH_MEMORY: &str = "search_memory";
    pub const GET_MEMORIES: &str = "get_memories";
    pub const DELETE_MEMORY: &str = "delete_memory";
    pub const GET_PROFILE: &str = "get_user_profile";
    pub const ADD_RELATIONSHIP: &str = "add_relationship";
    pub const GET_RELATIONSHIPS: &str = "get_relationships";
    pub const ADD_REMINDER: &str = "add_reminder";
    pub const GET_CONTEXT: &str = "get_context";
    pub const ADD_MEMORIES: &str = "add_memories";
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Returns all 9 memory tool definitions in provider-agnostic format.
pub fn all_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: tool_names::ADD_MEMORY.into(),
            description: "Store a memory. Use when the user shares a personal fact, preference, \
                experience, or any information worth remembering for future conversations."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The memory content to store."
                    },
                    "memory_type": {
                        "type": "string",
                        "description": "The type of memory.",
                        "enum": ["episodic", "semantic", "factual", "procedural", "social", "spatial", "prospective", "affective"]
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Optional user identifier."
                    }
                },
                "required": ["content"]
            }),
        },
        ToolDefinition {
            name: tool_names::SEARCH_MEMORY.into(),
            description: "Search stored memories by semantic similarity, text match, or hybrid. \
                Use when you need to recall something the user told you before."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    },
                    "mode": {
                        "type": "string",
                        "description": "Search mode: semantic, text, hybrid, or graph.",
                        "enum": ["semantic", "text", "hybrid", "graph"]
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default 10)."
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Optional user identifier to scope search."
                    },
                    "memory_type": {
                        "type": "string",
                        "description": "Optional memory type filter.",
                        "enum": ["episodic", "semantic", "factual", "procedural", "social", "spatial", "prospective", "affective"]
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Optional ISO 8601 date to filter results from (e.g. '2023-05-01'). Only returns memories with events on or after this date."
                    },
                    "date_to": {
                        "type": "string",
                        "description": "Optional ISO 8601 date to filter results until (e.g. '2023-05-31'). Only returns memories with events on or before this date."
                    }
                },
                "required": ["query"]
            }),
        },
        ToolDefinition {
            name: tool_names::GET_MEMORIES.into(),
            description: "Get a list of stored memories, optionally filtered by type or user."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "Optional user identifier."
                    },
                    "memory_type": {
                        "type": "string",
                        "description": "Optional memory type filter.",
                        "enum": ["episodic", "semantic", "factual", "procedural", "social", "spatial", "prospective", "affective"]
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default 20)."
                    }
                },
                "required": []
            }),
        },
        ToolDefinition {
            name: tool_names::DELETE_MEMORY.into(),
            description: "Delete a specific memory by its ID.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The ID of the memory to delete."
                    }
                },
                "required": ["memory_id"]
            }),
        },
        ToolDefinition {
            name: tool_names::GET_PROFILE.into(),
            description: "Get the user profile summary built from accumulated memories.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user identifier."
                    }
                },
                "required": ["user_id"]
            }),
        },
        ToolDefinition {
            name: tool_names::ADD_RELATIONSHIP.into(),
            description:
                "Create a relationship (edge) between two memories in the knowledge graph.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "from_id": {
                        "type": "string",
                        "description": "The source memory ID."
                    },
                    "to_id": {
                        "type": "string",
                        "description": "The target memory ID."
                    },
                    "relation": {
                        "type": "string",
                        "description": "The type of relationship.",
                        "enum": ["relates_to", "contradicts", "supersedes", "caused_by", "part_of", "prefers_over", "follows", "instance_of", "belongs_to", "similar_to"]
                    },
                    "weight": {
                        "type": "number",
                        "description": "Relationship strength from 0.0 to 1.0 (default 1.0)."
                    }
                },
                "required": ["from_id", "to_id", "relation"]
            }),
        },
        ToolDefinition {
            name: tool_names::GET_RELATIONSHIPS.into(),
            description: "Get all relationships connected to a specific memory.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The memory ID to get relationships for."
                    }
                },
                "required": ["memory_id"]
            }),
        },
        ToolDefinition {
            name: tool_names::ADD_REMINDER.into(),
            description: "Add a prospective memory / reminder that triggers in a future context."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "What to remember to do."
                    },
                    "trigger_type": {
                        "type": "string",
                        "description": "How the reminder should be triggered.",
                        "enum": ["topic_mention", "time_based", "user_presence", "semantic_match"]
                    },
                    "trigger_condition": {
                        "type": "string",
                        "description": "The condition that triggers the reminder (e.g., a topic keyword)."
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Optional user identifier."
                    }
                },
                "required": ["content", "trigger_type", "trigger_condition"]
            }),
        },
        ToolDefinition {
            name: tool_names::GET_CONTEXT.into(),
            description: "Get assembled context (relevant memories, profile, attention items) \
                for the current conversation."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The current query or topic to retrieve relevant context for."
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Optional user identifier."
                    }
                },
                "required": ["query"]
            }),
        },
        ToolDefinition {
            name: tool_names::ADD_MEMORIES.into(),
            description: "Extract and store memories from raw conversation text. \
                Automatically classifies memory types, assigns confidence scores, \
                and resolves temporal references to absolute dates. \
                Use this instead of add_memory when processing natural conversation."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Raw conversation text to extract memories from."
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Optional user identifier."
                    },
                    "preserve_source": {
                        "type": "boolean",
                        "description": "If true, also store raw text segments as Episodic memories alongside extracted facts. Recommended for conversation ingestion."
                    }
                },
                "required": ["text"]
            }),
        },
    ]
}

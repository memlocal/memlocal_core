use serde::{Deserialize, Serialize};
use std::fmt;

use super::memory_category::MemoryCategory;

/// Fine-grained memory type within the taxonomy.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    // ── Sensory ──
    SensoryBuffer,
    // ── Short-Term ──
    WorkingMemory,
    AttentionContext,
    ConversationBuffer,
    // ── Long-Term ──
    Episodic,
    Semantic,
    Factual,
    Procedural,
    Social,
    Spatial,
    Prospective,
    Affective,
}

impl MemoryType {
    /// The high-level category this type belongs to.
    pub fn category(&self) -> MemoryCategory {
        match self {
            Self::SensoryBuffer => MemoryCategory::Sensory,
            Self::WorkingMemory | Self::AttentionContext | Self::ConversationBuffer => {
                MemoryCategory::ShortTerm
            }
            _ => MemoryCategory::LongTerm,
        }
    }

    /// Human-readable display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::SensoryBuffer => "Sensory Buffer",
            Self::WorkingMemory => "Working Memory",
            Self::AttentionContext => "Attention Context",
            Self::ConversationBuffer => "Conversation Buffer",
            Self::Episodic => "Episodic",
            Self::Semantic => "Semantic",
            Self::Factual => "Factual / Profile",
            Self::Procedural => "Procedural",
            Self::Social => "Social",
            Self::Spatial => "Spatial",
            Self::Prospective => "Prospective",
            Self::Affective => "Affective",
        }
    }

    /// Short description.
    pub fn description(&self) -> &'static str {
        match self {
            Self::SensoryBuffer => "Ultra-short buffer for raw perception before interpretation.",
            Self::WorkingMemory => "Temporary scratchpad for calculations and partial plans.",
            Self::AttentionContext => "What the model is actively reasoning about right now.",
            Self::ConversationBuffer => "Recent messages needed to stay coherent.",
            Self::Episodic => "Past conversations, completed tasks, user journeys.",
            Self::Semantic => "General knowledge facts and relationships.",
            Self::Factual => "User preferences, account details, constraints.",
            Self::Procedural => "Agent skills library, tool-use policies, SOPs.",
            Self::Social => "Contact graph, team relationships, interaction history.",
            Self::Spatial => "Location history, route memory, spatial relationships.",
            Self::Prospective => "Reminders, follow-ups, scheduled actions.",
            Self::Affective => "User sentiment, tone preferences, emotional salience.",
        }
    }

    /// Default time-to-live in milliseconds, or `None` for permanent.
    pub fn default_ttl_ms(&self) -> Option<u64> {
        match self {
            Self::SensoryBuffer => Some(5000),
            Self::WorkingMemory => Some(300_000),
            Self::AttentionContext => Some(60_000),
            Self::ConversationBuffer => Some(300_000),
            _ => None,
        }
    }

    /// Whether this type is persistent (no TTL).
    pub fn is_persistent(&self) -> bool {
        self.default_ttl_ms().is_none()
    }

    /// The string stored in the database to identify this type.
    pub fn stored_name(&self) -> &'static str {
        match self {
            Self::SensoryBuffer => "sensory_buffer",
            Self::WorkingMemory => "working_memory",
            Self::AttentionContext => "attention_context",
            Self::ConversationBuffer => "conversation_buffer",
            Self::Episodic => "episodic",
            Self::Semantic => "semantic",
            Self::Factual => "factual",
            Self::Procedural => "procedural",
            Self::Social => "social",
            Self::Spatial => "spatial",
            Self::Prospective => "prospective",
            Self::Affective => "affective",
        }
    }

    /// Look up a `MemoryType` from its stored name string.
    /// Defaults to `Semantic` if not found (matching Dart implementation).
    pub fn from_stored_name(name: &str) -> Self {
        match name {
            "sensory_buffer" => Self::SensoryBuffer,
            "working_memory" => Self::WorkingMemory,
            "attention_context" => Self::AttentionContext,
            "conversation_buffer" => Self::ConversationBuffer,
            "episodic" => Self::Episodic,
            "semantic" => Self::Semantic,
            "factual" => Self::Factual,
            "procedural" => Self::Procedural,
            "social" => Self::Social,
            "spatial" => Self::Spatial,
            "prospective" => Self::Prospective,
            "affective" => Self::Affective,
            _ => Self::Semantic,
        }
    }
}

impl fmt::Display for MemoryType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

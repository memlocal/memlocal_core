use serde::{Deserialize, Serialize};
use std::fmt;

/// High-level memory category mirroring human cognitive tiers.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum MemoryCategory {
    /// Ultra-short buffers for raw perception (milliseconds).
    Sensory,
    /// Temporary context supporting the current task (seconds to minutes).
    ShortTerm,
    /// Persistent knowledge surviving across sessions (days to years).
    LongTerm,
}

impl fmt::Display for MemoryCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sensory => write!(f, "Sensory"),
            Self::ShortTerm => write!(f, "Short-Term"),
            Self::LongTerm => write!(f, "Long-Term"),
        }
    }
}

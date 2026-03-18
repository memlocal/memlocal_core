use serde::{Deserialize, Serialize};
use std::fmt;

/// Strategy for searching memories.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    /// Vector similarity search using embeddings.
    Semantic,
    /// BM25 full-text search.
    Text,
    /// Graph traversal from closest vector match.
    Graph,
    /// All three modes merged and re-ranked.
    Hybrid,
}

impl SearchMode {
    /// Human-readable display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Semantic => "Semantic (Vector)",
            Self::Text => "Full-Text (BM25)",
            Self::Graph => "Graph Traversal",
            Self::Hybrid => "Hybrid",
        }
    }

    /// Parse from string. Defaults to Hybrid.
    pub fn from_str_lossy(s: &str) -> Self {
        match s {
            "semantic" => Self::Semantic,
            "text" => Self::Text,
            "graph" => Self::Graph,
            "hybrid" => Self::Hybrid,
            _ => Self::Hybrid,
        }
    }
}

impl fmt::Display for SearchMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

use serde::{Deserialize, Serialize};
use std::fmt;

/// The relationship type between two memory nodes in the knowledge graph.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum MemoryRelation {
    /// A relates to B.
    RelatesTo,
    /// A contradicts B (marks B as potentially invalid).
    Contradicts,
    /// A supersedes B (a newer version of the same fact).
    Supersedes,
    /// A is caused by B.
    CausedBy,
    /// A is a part of B.
    PartOf,
    /// A is associated with user preference B.
    PrefersOver,
    /// A temporally follows B (sequence / episode chain).
    Follows,
    /// A occurred during the same conversation session as B.
    DuringSession,
    /// A is an instance of category B.
    InstanceOf,
    /// A is owned by / belongs to entity B.
    BelongsTo,
    /// A is similar to B (auto-discovered via embedding proximity).
    SimilarTo,
}

impl MemoryRelation {
    /// The string stored in CozoDB.
    pub fn stored_name(&self) -> &'static str {
        match self {
            Self::RelatesTo => "relates_to",
            Self::Contradicts => "contradicts",
            Self::Supersedes => "supersedes",
            Self::CausedBy => "caused_by",
            Self::PartOf => "part_of",
            Self::PrefersOver => "prefers_over",
            Self::Follows => "follows",
            Self::DuringSession => "during_session",
            Self::InstanceOf => "instance_of",
            Self::BelongsTo => "belongs_to",
            Self::SimilarTo => "similar_to",
        }
    }

    /// Look up by stored name. Defaults to `RelatesTo` if not found.
    pub fn from_stored_name(name: &str) -> Self {
        match name {
            "relates_to" => Self::RelatesTo,
            "contradicts" => Self::Contradicts,
            "supersedes" => Self::Supersedes,
            "caused_by" => Self::CausedBy,
            "part_of" => Self::PartOf,
            "prefers_over" => Self::PrefersOver,
            "follows" => Self::Follows,
            "during_session" => Self::DuringSession,
            "instance_of" => Self::InstanceOf,
            "belongs_to" => Self::BelongsTo,
            "similar_to" => Self::SimilarTo,
            _ => Self::RelatesTo,
        }
    }
}

impl fmt::Display for MemoryRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.stored_name())
    }
}

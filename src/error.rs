use thiserror::Error;

/// All errors that can originate from memlocal_core.
#[derive(Error, Debug)]
pub enum MemlocalError {
    #[error("Database error: {0}")]
    Database(String),

    #[error("Schema error: {0}")]
    Schema(String),

    #[error("Query error: {0}")]
    Query(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Convenience type alias.
pub type Result<T> = std::result::Result<T, MemlocalError>;

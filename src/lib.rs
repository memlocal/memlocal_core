//! memlocal_core — the core memory engine for the memlocal ecosystem.
//!
//! This crate provides a local-first AI memory layer modelled on human
//! cognitive architecture. It manages sensory, short-term, and long-term
//! memory backed by an embedded CozoDB database with vector search,
//! full-text search, and graph algorithms.
//!
//! Platform-specific concerns (LLM API calls, embedding generation,
//! HTTP networking) live behind the optional `http` feature. Platform SDKs
//! (memlocal_flutter, memlocal-android, etc.) can instead implement the
//! `EmbeddingProvider` trait using their own HTTP stack.

pub mod api;
pub mod consolidation;
pub mod error;
pub mod longterm;
pub mod models;
pub mod shortterm;
pub mod storage;
pub mod tools;

#[cfg(feature = "http")]
pub mod http;

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use crate::error::Result;
use crate::models::*;
use crate::storage::MemoryStore;

/// Cosine similarity threshold for grouping memories into a cluster.
const CLUSTER_THRESHOLD: f64 = 0.65;

/// Consolidates related episodic memories into higher-level semantic summaries.
///
/// The clustering logic stays in Rust. LLM summarization is exposed as a trait
/// for the platform layer to implement.
pub struct MemoryConsolidator {
    store: Arc<MemoryStore>,
}

/// Trait for LLM-based summarization — implemented by the platform layer.
pub trait Summarizer: Send + Sync {
    fn summarize(&self, contents: &[String]) -> Result<String>;
}

/// Input data for session summarization (the LLM call is done by the platform layer).
#[derive(Clone, Debug)]
pub struct SessionSummaryInput {
    pub session_id: String,
    pub text: String,
    pub speakers: Vec<String>,
    pub key_topics: Vec<String>,
    pub memory_count: usize,
}

impl MemoryConsolidator {
    pub fn new(store: Arc<MemoryStore>) -> Self {
        Self { store }
    }

    /// Find contradicting triples (same subject + predicate, different object).
    /// Returns pairs of (old_triple, new_triple) where old should be marked not-latest.
    pub fn find_contradicting_triples(&self) -> Result<Vec<(Triple, Triple)>> {
        // Get all triples (None filters = no filter)
        let all_triples = self.store.search_triples(None, None, None)?;

        let mut by_sp: HashMap<(String, String), Vec<Triple>> = HashMap::new();
        for triple in all_triples {
            let key = (triple.subject.clone(), triple.predicate.clone());
            by_sp.entry(key).or_default().push(triple);
        }

        let mut contradictions = Vec::new();
        for (_key, mut triples) in by_sp {
            if triples.len() > 1 {
                // Sort by last_mentioned descending -- newest is the "latest"
                triples.sort_by(|a, b| {
                    b.last_mentioned
                        .partial_cmp(&a.last_mentioned)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let newest = &triples[0];
                for old in &triples[1..] {
                    if old.object != newest.object {
                        contradictions.push((old.clone(), newest.clone()));
                    }
                }
            }
        }
        Ok(contradictions)
    }

    /// Summarize a session's memories into a SessionSummaryInput.
    /// The actual LLM call is done by the caller -- this method gathers the memories
    /// and returns the text to be summarized, plus speaker/topic metadata.
    pub fn prepare_session_for_summarization(
        &self,
        session_id: &str,
        user_id: Option<&str>,
    ) -> Result<Option<SessionSummaryInput>> {
        let memories = self.store.get_memories_by_session(session_id, user_id, 50)?;
        if memories.is_empty() {
            return Ok(None);
        }

        // Collect speakers from memory metadata
        let mut speakers = BTreeSet::new();
        let mut topics = BTreeSet::new();
        let mut text_parts = Vec::new();

        for m in &memories {
            let speaker = m.speaker();
            if !speaker.is_empty() {
                speakers.insert(speaker.to_string());
            }
            // Extract key topics from the content (simple heuristic: capitalized words > 3 chars)
            for word in m.content.split_whitespace() {
                let trimmed = word.trim_matches(|c: char| !c.is_alphanumeric());
                if trimmed.len() > 3
                    && trimmed
                        .chars()
                        .next()
                        .map(|c| c.is_uppercase())
                        .unwrap_or(false)
                {
                    topics.insert(trimmed.to_string());
                }
            }
            text_parts.push(m.content.clone());
        }

        Ok(Some(SessionSummaryInput {
            session_id: session_id.to_string(),
            text: text_parts.join("\n"),
            speakers: speakers.into_iter().collect(),
            key_topics: topics.into_iter().take(10).collect(),
            memory_count: memories.len(),
        }))
    }

    /// Find clusters of episodic memories eligible for consolidation.
    /// Returns groups of memory items that should be consolidated together.
    ///
    /// Note: This only does clustering. The actual summarization (LLM call)
    /// and storage of summaries happens in the platform layer.
    pub fn find_clusters(
        &self,
        user_id: Option<&str>,
        embeddings: &[(String, Vec<f32>)], // (memory_id, embedding)
        min_cluster_size: usize,
        min_episodic_age_secs: u64,
    ) -> Result<Vec<Vec<MemoryItem>>> {
        // 1. Fetch episodic memories
        let all_episodic = self
            .store
            .get_memories(user_id, Some(MemoryType::Episodic), 200)?;

        // 2. Filter to old, unconsolidated items
        let cutoff = chrono::Utc::now() - chrono::Duration::seconds(min_episodic_age_secs as i64);
        let eligible: Vec<MemoryItem> = all_episodic
            .into_iter()
            .filter(|m| {
                m.created_at < cutoff
                    && m.metadata.get("consolidated").and_then(|v| v.as_bool()) != Some(true)
            })
            .collect();

        if eligible.len() < min_cluster_size {
            return Ok(vec![]);
        }

        // 3. Match embeddings to eligible items
        let emb_map: std::collections::HashMap<&str, &[f32]> = embeddings
            .iter()
            .map(|(id, emb)| (id.as_str(), emb.as_slice()))
            .collect();

        let mut items_with_emb: Vec<(&MemoryItem, &[f32])> = Vec::new();
        for item in &eligible {
            if let Some(emb) = emb_map.get(item.id.as_str()) {
                items_with_emb.push((item, emb));
            }
        }

        // 4. Greedy clustering
        let clusters = greedy_cluster(&items_with_emb, min_cluster_size);
        Ok(clusters)
    }

    /// Compute cosine similarity between two vectors.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let mut dot = 0.0_f64;
        let mut norm_a = 0.0_f64;
        let mut norm_b = 0.0_f64;
        for i in 0..a.len() {
            dot += a[i] as f64 * b[i] as f64;
            norm_a += (a[i] as f64).powi(2);
            norm_b += (b[i] as f64).powi(2);
        }
        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom == 0.0 {
            0.0
        } else {
            dot / denom
        }
    }
}

fn greedy_cluster(
    items_with_emb: &[(&MemoryItem, &[f32])],
    min_size: usize,
) -> Vec<Vec<MemoryItem>> {
    let n = items_with_emb.len();
    let mut visited = vec![false; n];
    let mut clusters = Vec::new();

    for i in 0..n {
        if visited[i] {
            continue;
        }
        let mut cluster = vec![items_with_emb[i].0.clone()];
        visited[i] = true;

        for j in (i + 1)..n {
            if visited[j] {
                continue;
            }
            if MemoryConsolidator::cosine_similarity(items_with_emb[i].1, items_with_emb[j].1)
                >= CLUSTER_THRESHOLD
            {
                cluster.push(items_with_emb[j].0.clone());
                visited[j] = true;
            }
        }

        if cluster.len() >= min_size {
            clusters.push(cluster);
        }
    }

    clusters
}

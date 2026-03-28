use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

use crate::error::{MemlocalError, Result};
use crate::tools::RerankerProvider;

/// Jina AI reranker client using the public rerank API.
pub struct JinaReranker {
    api_key: String,
    model: String,
    base_url: String,
    client: Client,
}

impl JinaReranker {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: "jina-reranker-v2-base-multilingual".to_string(),
            base_url: "https://api.jina.ai".to_string(),
            client: Client::new(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}

impl RerankerProvider for JinaReranker {
    fn rerank(&self, query: &str, documents: &[String], top_k: usize) -> Result<Vec<(usize, f64)>> {
        if documents.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

        let url = format!("{}/v1/rerank", self.base_url.trim_end_matches('/'));
        let body = RerankRequest {
            model: self.model.clone(),
            query: query.to_string(),
            documents: documents.to_vec(),
            top_n: top_k.min(documents.len()),
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| MemlocalError::Internal(format!("HTTP request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response
                .text()
                .unwrap_or_else(|_| "could not read body".to_string());
            return Err(MemlocalError::Internal(format!(
                "Jina API error ({status}): {body_text}"
            )));
        }

        let mut parsed: RerankResponse = response
            .json()
            .map_err(|e| MemlocalError::Internal(format!("Failed to parse response: {e}")))?;
        parsed
            .results
            .sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal));

        Ok(parsed
            .results
            .into_iter()
            .map(|result| (result.index, result.relevance_score))
            .collect())
    }
}

#[derive(Serialize)]
struct RerankRequest {
    model: String,
    query: String,
    documents: Vec<String>,
    top_n: usize,
}

#[derive(Deserialize)]
struct RerankResponse {
    results: Vec<RerankResult>,
}

#[derive(Deserialize)]
struct RerankResult {
    index: usize,
    relevance_score: f64,
}
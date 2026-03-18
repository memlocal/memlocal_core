use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

use crate::error::{MemlocalError, Result};
use crate::tools::EmbeddingProvider;

/// OpenAI-compatible embedding provider using `reqwest` (blocking).
///
/// Works with OpenAI's `/v1/embeddings` endpoint and any API-compatible
/// service (e.g., Azure OpenAI, local servers with OpenAI-compatible API).
///
/// ```rust,no_run
/// use memlocal_core::http::OpenAiEmbeddingProvider;
/// use memlocal_core::tools::EmbeddingProvider;
///
/// let provider = OpenAiEmbeddingProvider::new("sk-...");
/// let embedding = provider.embed_one("Hello world").unwrap();
/// assert_eq!(embedding.len(), 1536);
/// ```
pub struct OpenAiEmbeddingProvider {
    api_key: String,
    model: String,
    dimensions: u32,
    base_url: String,
    client: Client,
}

impl OpenAiEmbeddingProvider {
    /// Create a new provider with the default model (`text-embedding-3-small`, 1536 dims).
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: "text-embedding-3-small".to_string(),
            dimensions: 1536,
            base_url: "https://api.openai.com".to_string(),
            client: Client::new(),
        }
    }

    /// Override the model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the embedding dimensions.
    pub fn with_dimensions(mut self, dimensions: u32) -> Self {
        self.dimensions = dimensions;
        self
    }

    /// Override the base URL (for Azure, local servers, etc.).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Embed multiple texts in a single API call.
    pub fn embed_many(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/v1/embeddings", self.base_url);

        let body = EmbeddingRequest {
            model: &self.model,
            input: texts.to_vec(),
            dimensions: Some(self.dimensions),
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
                "OpenAI API error ({status}): {body_text}"
            )));
        }

        let resp: EmbeddingResponse = response
            .json()
            .map_err(|e| MemlocalError::Internal(format!("Failed to parse response: {e}")))?;

        // Sort by index to ensure correct order
        let mut data = resp.data;
        data.sort_by_key(|d| d.index);

        Ok(data.into_iter().map(|d| d.embedding).collect())
    }
}

impl EmbeddingProvider for OpenAiEmbeddingProvider {
    fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_many(&[text])?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| MemlocalError::Internal("Empty embedding response".into()))
    }
}

// ── OpenAI API types ──

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

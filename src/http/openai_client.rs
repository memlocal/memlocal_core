use reqwest::blocking::Client;

use crate::error::{MemlocalError, Result};
use crate::tools::{LlmProvider, ToolCall, ToolDefinition};

use super::anthropic_client::{LlmMessage, LlmResponse, ToolCallingLlm};

/// OpenAI-compatible chat completion client with tool-calling support.
///
/// Works with OpenAI's `/v1/chat/completions` endpoint and any API-compatible
/// service. Supports both simple completions (`LlmProvider`) and multi-turn
/// tool calling (`ToolCallingLlm`).
pub struct OpenAiClient {
    api_key: String,
    model: String,
    base_url: String,
    max_tokens: u32,
    temperature: Option<f64>,
    client: Client,
}

impl OpenAiClient {
    /// Create a new client defaulting to GPT-5 Mini.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: "gpt-5-mini".into(),
            base_url: "https://api.openai.com".into(),
            max_tokens: 1024,
            temperature: None,
            client: Client::new(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Returns true if the model is in the GPT-5 or "o" reasoning family,
    /// which uses `max_completion_tokens` instead of `max_tokens` and does
    /// not support `temperature`.
    fn is_reasoning_model(&self) -> bool {
        self.model.starts_with("gpt-5") || self.model.starts_with("o")
    }
}

impl LlmProvider for OpenAiClient {
    fn complete(&self, system: &str, user: &str) -> Result<String> {
        let url = format!("{}/v1/chat/completions", self.base_url);

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        });

        if self.is_reasoning_model() {
            body["max_completion_tokens"] = serde_json::json!(self.max_tokens);
        } else {
            body["max_tokens"] = serde_json::json!(self.max_tokens);
            if let Some(temp) = self.temperature {
                body["temperature"] = serde_json::json!(temp);
            }
        }

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

        let data: serde_json::Value = response
            .json()
            .map_err(|e| MemlocalError::Internal(format!("Failed to parse response: {e}")))?;

        let content = data["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        Ok(content)
    }
}

impl ToolCallingLlm for OpenAiClient {
    fn complete_with_tools(
        &self,
        messages: &[LlmMessage],
        tools: &[ToolDefinition],
    ) -> Result<LlmResponse> {
        let url = format!("{}/v1/chat/completions", self.base_url);

        // Convert LlmMessage to OpenAI message format
        let api_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|msg| match msg.role.as_str() {
                "system" => serde_json::json!({
                    "role": "system",
                    "content": msg.content,
                }),
                "user" => serde_json::json!({
                    "role": "user",
                    "content": msg.content,
                }),
                "assistant" if !msg.tool_calls.is_empty() => {
                    let tool_calls: Vec<serde_json::Value> = msg
                        .tool_calls
                        .iter()
                        .map(|tc| {
                            serde_json::json!({
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": tc.arguments.to_string(),
                                }
                            })
                        })
                        .collect();
                    let mut obj = serde_json::json!({
                        "role": "assistant",
                        "tool_calls": tool_calls,
                    });
                    if !msg.content.is_empty() {
                        obj["content"] = serde_json::json!(msg.content);
                    }
                    obj
                }
                "assistant" => serde_json::json!({
                    "role": "assistant",
                    "content": msg.content,
                }),
                "tool_result" => {
                    let tool_call_id = msg.tool_call_id.as_deref().unwrap_or("");
                    serde_json::json!({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": msg.content,
                    })
                }
                _ => serde_json::json!({
                    "role": msg.role,
                    "content": msg.content,
                }),
            })
            .collect();

        // Build request body
        let mut body = serde_json::json!({
            "model": self.model,
            "messages": api_messages,
        });

        if self.is_reasoning_model() {
            body["max_completion_tokens"] = serde_json::json!(self.max_tokens);
        } else {
            body["max_tokens"] = serde_json::json!(self.max_tokens);
            if let Some(temp) = self.temperature {
                body["temperature"] = serde_json::json!(temp);
            }
        }

        // Convert tool definitions to OpenAI format
        if !tools.is_empty() {
            let api_tools: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                    })
                })
                .collect();
            body["tools"] = serde_json::Value::Array(api_tools);
        }

        // Send request
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

        let data: serde_json::Value = response
            .json()
            .map_err(|e| MemlocalError::Internal(format!("Failed to parse response: {e}")))?;

        // Parse response
        let message = &data["choices"][0]["message"];

        let content = message["content"].as_str().unwrap_or("").to_string();

        let mut tool_calls = Vec::new();
        if let Some(tc_array) = message["tool_calls"].as_array() {
            for tc in tc_array {
                let id = tc["id"].as_str().unwrap_or("").to_string();
                let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
                // OpenAI returns arguments as a JSON string, not an object
                let args_str = tc["function"]["arguments"].as_str().unwrap_or("{}");
                let arguments: serde_json::Value = serde_json::from_str(args_str)
                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                tool_calls.push(ToolCall {
                    id,
                    name,
                    arguments,
                });
            }
        }

        let input_tokens = data["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
        let output_tokens = data["usage"]["completion_tokens"].as_u64().unwrap_or(0);

        Ok(LlmResponse {
            content,
            tool_calls,
            input_tokens,
            output_tokens,
        })
    }
}

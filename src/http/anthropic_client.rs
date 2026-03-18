use reqwest::blocking::Client;

use crate::error::{MemlocalError, Result};
use crate::tools::{ToolCall, ToolDefinition};

/// A message in a conversation with an LLM.
#[derive(Clone, Debug)]
pub struct LlmMessage {
    pub role: String,
    pub content: String,
    /// For tool_result messages: the tool_use_id this result corresponds to.
    pub tool_call_id: Option<String>,
    /// For assistant messages: tool calls the model wants to make.
    pub tool_calls: Vec<ToolCall>,
}

impl LlmMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
            tool_call_id: None,
            tool_calls: vec![],
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
            tool_call_id: None,
            tool_calls: vec![],
        }
    }

    pub fn assistant(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
            tool_call_id: None,
            tool_calls,
        }
    }

    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool_result".into(),
            content: content.into(),
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: vec![],
        }
    }
}

/// Response from the LLM.
#[derive(Clone, Debug)]
pub struct LlmResponse {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl LlmResponse {
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }
}

/// Anthropic Claude API client with tool-calling support.
pub struct AnthropicClient {
    api_key: String,
    model: String,
    base_url: String,
    max_tokens: u32,
    temperature: Option<f64>,
    client: Client,
}

impl AnthropicClient {
    /// Create a new client defaulting to Claude Haiku 4.5.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: "claude-haiku-4-5-20251001".into(),
            base_url: "https://api.anthropic.com".into(),
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

    /// Send a completion request with optional tools.
    pub fn complete(
        &self,
        messages: &[LlmMessage],
        tools: &[ToolDefinition],
    ) -> Result<LlmResponse> {
        let url = format!("{}/v1/messages", self.base_url);

        // Separate system messages from conversation messages
        let mut system_text = String::new();
        let mut api_messages: Vec<serde_json::Value> = Vec::new();

        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    if !system_text.is_empty() {
                        system_text.push('\n');
                    }
                    system_text.push_str(&msg.content);
                }
                "tool_result" => {
                    // Tool results are sent as user messages with content blocks
                    let tool_call_id = msg.tool_call_id.as_deref().unwrap_or("");
                    api_messages.push(serde_json::json!({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": msg.content,
                        }]
                    }));
                }
                "assistant" if !msg.tool_calls.is_empty() => {
                    // Assistant message with tool calls
                    let mut content_blocks: Vec<serde_json::Value> = Vec::new();
                    if !msg.content.is_empty() {
                        content_blocks.push(serde_json::json!({
                            "type": "text",
                            "text": msg.content,
                        }));
                    }
                    for tc in &msg.tool_calls {
                        content_blocks.push(serde_json::json!({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }));
                    }
                    api_messages.push(serde_json::json!({
                        "role": "assistant",
                        "content": content_blocks,
                    }));
                }
                _ => {
                    // Regular user/assistant message
                    api_messages.push(serde_json::json!({
                        "role": msg.role,
                        "content": msg.content,
                    }));
                }
            }
        }

        // Build request body
        let mut body = serde_json::json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": api_messages,
        });

        if !system_text.is_empty() {
            body["system"] = serde_json::Value::String(system_text);
        }

        if let Some(temp) = self.temperature {
            body["temperature"] = serde_json::json!(temp);
        }

        if !tools.is_empty() {
            let api_tools: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.parameters,
                    })
                })
                .collect();
            body["tools"] = serde_json::Value::Array(api_tools);
        }

        // Send request
        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
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
                "Anthropic API error ({status}): {body_text}"
            )));
        }

        let data: serde_json::Value = response
            .json()
            .map_err(|e| MemlocalError::Internal(format!("Failed to parse response: {e}")))?;

        // Parse response content blocks
        let mut text_content = String::new();
        let mut tool_calls = Vec::new();

        if let Some(content_blocks) = data["content"].as_array() {
            for block in content_blocks {
                match block["type"].as_str() {
                    Some("text") => {
                        if let Some(text) = block["text"].as_str() {
                            if !text_content.is_empty() {
                                text_content.push('\n');
                            }
                            text_content.push_str(text);
                        }
                    }
                    Some("tool_use") => {
                        let id = block["id"].as_str().unwrap_or("").to_string();
                        let name = block["name"].as_str().unwrap_or("").to_string();
                        let input = block["input"].clone();
                        tool_calls.push(ToolCall {
                            id,
                            name,
                            arguments: input,
                        });
                    }
                    _ => {}
                }
            }
        }

        let input_tokens = data["usage"]["input_tokens"].as_u64().unwrap_or(0);
        let output_tokens = data["usage"]["output_tokens"].as_u64().unwrap_or(0);

        Ok(LlmResponse {
            content: text_content,
            tool_calls,
            input_tokens,
            output_tokens,
        })
    }
}

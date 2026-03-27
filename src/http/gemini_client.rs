use reqwest::blocking::Client;

use crate::error::{MemlocalError, Result};
use crate::tools::{LlmProvider, ToolCall, ToolDefinition};

use super::anthropic_client::{LlmMessage, LlmResponse, ToolCallingLlm};

/// Google Gemini API client with tool-calling support.
///
/// Uses the `generativelanguage.googleapis.com` REST API.
/// Supports both simple completions (`LlmProvider`) and multi-turn
/// tool calling (`ToolCallingLlm`).
pub struct GeminiClient {
    api_key: String,
    model: String,
    client: Client,
}

impl GeminiClient {
    /// Create a new client defaulting to Gemini 2.0 Flash Lite.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: "gemini-2.0-flash-lite".into(),
            client: Client::new(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    fn endpoint_url(&self) -> String {
        format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            self.model, self.api_key
        )
    }
}

impl LlmProvider for GeminiClient {
    fn complete(&self, system: &str, user: &str) -> Result<String> {
        let url = self.endpoint_url();

        let body = serde_json::json!({
            "system_instruction": {
                "parts": [{"text": system}]
            },
            "contents": [{
                "role": "user",
                "parts": [{"text": user}]
            }]
        });

        let response = self
            .client
            .post(&url)
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
                "Gemini API error ({status}): {body_text}"
            )));
        }

        let data: serde_json::Value = response
            .json()
            .map_err(|e| MemlocalError::Internal(format!("Failed to parse response: {e}")))?;

        let content = data["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .unwrap_or("")
            .to_string();

        Ok(content)
    }
}

impl ToolCallingLlm for GeminiClient {
    fn complete_with_tools(
        &self,
        messages: &[LlmMessage],
        tools: &[ToolDefinition],
    ) -> Result<LlmResponse> {
        let url = self.endpoint_url();

        // Separate system instruction from conversation contents
        let mut system_text = String::new();
        let mut contents: Vec<serde_json::Value> = Vec::new();

        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    if !system_text.is_empty() {
                        system_text.push('\n');
                    }
                    system_text.push_str(&msg.content);
                }
                "user" => {
                    contents.push(serde_json::json!({
                        "role": "user",
                        "parts": [{"text": msg.content}]
                    }));
                }
                "assistant" if !msg.tool_calls.is_empty() => {
                    // Assistant message with function calls
                    let parts: Vec<serde_json::Value> = msg
                        .tool_calls
                        .iter()
                        .map(|tc| {
                            serde_json::json!({
                                "functionCall": {
                                    "name": tc.name,
                                    "args": tc.arguments,
                                }
                            })
                        })
                        .collect();
                    contents.push(serde_json::json!({
                        "role": "model",
                        "parts": parts,
                    }));
                }
                "assistant" => {
                    // Gemini uses "model" instead of "assistant"
                    contents.push(serde_json::json!({
                        "role": "model",
                        "parts": [{"text": msg.content}]
                    }));
                }
                "tool_result" => {
                    // Gemini expects function responses as user-role parts
                    let name = msg
                        .tool_call_id
                        .as_deref()
                        .unwrap_or("unknown");
                    contents.push(serde_json::json!({
                        "role": "user",
                        "parts": [{
                            "functionResponse": {
                                "name": name,
                                "response": {
                                    "content": msg.content,
                                }
                            }
                        }]
                    }));
                }
                _ => {
                    contents.push(serde_json::json!({
                        "role": "user",
                        "parts": [{"text": msg.content}]
                    }));
                }
            }
        }

        // Build request body
        let mut body = serde_json::json!({
            "contents": contents,
        });

        if !system_text.is_empty() {
            body["system_instruction"] = serde_json::json!({
                "parts": [{"text": system_text}]
            });
        }

        // Convert tool definitions to Gemini format
        if !tools.is_empty() {
            let function_declarations: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    })
                })
                .collect();
            body["tools"] = serde_json::json!([{
                "function_declarations": function_declarations,
            }]);
        }

        // Send request
        let response = self
            .client
            .post(&url)
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
                "Gemini API error ({status}): {body_text}"
            )));
        }

        let data: serde_json::Value = response
            .json()
            .map_err(|e| MemlocalError::Internal(format!("Failed to parse response: {e}")))?;

        // Parse response parts
        let mut text_content = String::new();
        let mut tool_calls = Vec::new();

        if let Some(parts) = data["candidates"][0]["content"]["parts"].as_array() {
            for part in parts {
                if let Some(text) = part["text"].as_str() {
                    if !text_content.is_empty() {
                        text_content.push('\n');
                    }
                    text_content.push_str(text);
                }
                if let Some(fc) = part.get("functionCall") {
                    let name = fc["name"].as_str().unwrap_or("").to_string();
                    let args = fc["args"].clone();
                    // Gemini does not return tool call IDs; generate one
                    let id = format!("gemini_{}", uuid::Uuid::new_v4());
                    tool_calls.push(ToolCall {
                        id,
                        name,
                        arguments: args,
                    });
                }
            }
        }

        // Gemini usage metadata
        let input_tokens = data["usageMetadata"]["promptTokenCount"]
            .as_u64()
            .unwrap_or(0);
        let output_tokens = data["usageMetadata"]["candidatesTokenCount"]
            .as_u64()
            .unwrap_or(0);

        Ok(LlmResponse {
            content: text_content,
            tool_calls,
            input_tokens,
            output_tokens,
        })
    }
}

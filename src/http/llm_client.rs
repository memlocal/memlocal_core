use crate::error::Result;
use crate::tools::{EmbeddingProvider, ToolDefinition, ToolExecutor};

use super::anthropic_client::{LlmMessage, ToolCallingLlm};

/// Run a full agentic tool-calling loop.
///
/// Sends the conversation to Claude, and if Claude responds with tool calls,
/// executes them via the `ToolExecutor`, appends the results as messages,
/// and re-sends — looping until Claude gives a final text answer or
/// `max_rounds` is reached.
///
/// Returns the final text response from Claude.
///
/// The `messages` vec is mutated in place — after this returns, it contains
/// the full conversation trace including all tool calls and results.
pub fn run_with_tools(
    client: &dyn ToolCallingLlm,
    messages: &mut Vec<LlmMessage>,
    tools: &[ToolDefinition],
    tool_executor: &ToolExecutor,
    embedding_provider: &dyn EmbeddingProvider,
    max_rounds: usize,
) -> Result<String> {
    for round in 0..max_rounds {
        let response = client.complete_with_tools(messages, tools)?;

        if response.has_tool_calls() {
            // Append assistant message with tool calls to the conversation
            messages.push(LlmMessage::assistant(
                response.content.clone(),
                response.tool_calls.clone(),
            ));

            // Execute each tool call and append results
            for tc in &response.tool_calls {
                let result = tool_executor.execute(tc, embedding_provider);
                messages.push(LlmMessage::tool_result(&tc.id, &result.content));
            }

            log::debug!(
                "[run_with_tools] Round {}: {} tool call(s) executed",
                round + 1,
                response.tool_calls.len()
            );
        } else {
            // No tool calls — Claude gave a final answer
            messages.push(LlmMessage::assistant(response.content.clone(), vec![]));
            return Ok(response.content);
        }
    }

    // Max rounds exceeded — return whatever we have
    let last_response = client.complete_with_tools(messages, &[])?;
    messages.push(LlmMessage::assistant(last_response.content.clone(), vec![]));
    Ok(last_response.content)
}

use std::sync::Arc;

use crate::error::Result;
use crate::models::Message;
use crate::storage::MemoryStore;

/// Conversation buffer backed by CozoDB with a sliding window.
pub struct ConversationBuffer {
    store: Arc<MemoryStore>,
    session_id: String,
    max_messages: usize,
    messages: Vec<Message>,
    seq: i64,
    loaded: bool,
}

impl ConversationBuffer {
    pub fn new(store: Arc<MemoryStore>, session_id: String, max_messages: usize) -> Self {
        Self {
            store,
            session_id,
            max_messages,
            messages: Vec::new(),
            seq: 0,
            loaded: false,
        }
    }

    /// Load messages from CozoDB.
    pub fn load(&mut self) -> Result<()> {
        if self.loaded {
            return Ok(());
        }
        self.messages = self
            .store
            .get_messages(&self.session_id, Some(self.max_messages))?;
        self.seq = self.store.message_count(&self.session_id)? as i64;
        self.loaded = true;
        Ok(())
    }

    /// Append a message and persist it.
    pub fn append(&mut self, message: Message) -> Result<()> {
        if !self.loaded {
            self.load()?;
        }
        self.seq += 1;
        let mut msg = message;
        msg.session_id = Some(self.session_id.clone());
        self.store.put_message(&msg, self.seq)?;
        self.messages.push(msg);

        // Trim to sliding window
        if self.messages.len() > self.max_messages {
            let excess = self.messages.len() - self.max_messages;
            self.messages.drain(0..excess);
        }
        Ok(())
    }

    /// Get all buffered messages.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Get the most recent n messages.
    pub fn recent(&self, n: usize) -> &[Message] {
        let start = self.messages.len().saturating_sub(n);
        &self.messages[start..]
    }

    /// Clear the in-memory buffer (does NOT delete from DB).
    pub fn clear(&mut self) {
        self.messages.clear();
    }
}

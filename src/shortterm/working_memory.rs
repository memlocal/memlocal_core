use std::collections::HashSet;

use crate::models::*;

/// Working memory: the active context assembled for each LLM call.
pub struct WorkingMemory {
    relevant_memories: Vec<MemoryItem>,
    important_memories: Vec<MemoryItem>,
    triggered_reminders: Vec<ProspectiveItem>,
    user_profile: Option<UserProfile>,
    attention_items: Vec<MemoryItem>,
}

impl WorkingMemory {
    pub fn new() -> Self {
        Self {
            relevant_memories: Vec::new(),
            important_memories: Vec::new(),
            triggered_reminders: Vec::new(),
            user_profile: None,
            attention_items: Vec::new(),
        }
    }

    pub fn set_relevant(&mut self, items: Vec<MemoryItem>) {
        self.relevant_memories = items;
    }

    pub fn set_important(&mut self, items: Vec<MemoryItem>) {
        self.important_memories = items;
    }

    pub fn set_triggered_reminders(&mut self, reminders: Vec<ProspectiveItem>) {
        self.triggered_reminders = reminders;
    }

    pub fn set_profile(&mut self, profile: Option<UserProfile>) {
        self.user_profile = profile;
    }

    pub fn focus(&mut self, item: MemoryItem) {
        self.attention_items.retain(|i| i.id != item.id);
        self.attention_items.push(item);
    }

    pub fn unfocus(&mut self, item_id: &str) {
        self.attention_items.retain(|i| i.id != item_id);
    }

    pub fn clear(&mut self) {
        self.relevant_memories.clear();
        self.important_memories.clear();
        self.triggered_reminders.clear();
        self.user_profile = None;
        self.attention_items.clear();
    }

    pub fn relevant_memories(&self) -> &[MemoryItem] {
        &self.relevant_memories
    }

    pub fn important_memories(&self) -> &[MemoryItem] {
        &self.important_memories
    }

    pub fn triggered_reminders(&self) -> &[ProspectiveItem] {
        &self.triggered_reminders
    }

    pub fn user_profile(&self) -> Option<&UserProfile> {
        self.user_profile.as_ref()
    }

    pub fn attention_items(&self) -> &[MemoryItem] {
        &self.attention_items
    }

    /// Whether there is any context to inject.
    pub fn has_context(&self) -> bool {
        !self.relevant_memories.is_empty()
            || !self.important_memories.is_empty()
            || !self.triggered_reminders.is_empty()
            || self
                .user_profile
                .as_ref()
                .map(|p| p.is_not_empty())
                .unwrap_or(false)
            || !self.attention_items.is_empty()
    }

    /// Build a text context block for injection into an LLM system prompt.
    ///
    /// Tiered structure:
    /// 1. Triggered Reminders (highest priority)
    /// 2. User Profile
    /// 3. Important Memories (deduplicated against relevant set)
    /// 4. Relevant Memories (grouped by type)
    /// 5. Focused Items (attention context)
    pub fn to_context_block(&self) -> String {
        let mut buf = String::new();

        // 1. Triggered reminders
        if !self.triggered_reminders.is_empty() {
            buf.push_str("=== Triggered Reminders ===\n");
            for reminder in &self.triggered_reminders {
                buf.push_str(&format!("! {}\n", reminder.content));
                buf.push_str(&format!(
                    "  (trigger: {} — {})\n",
                    reminder.trigger_type.stored_name(),
                    reminder.trigger_condition
                ));
            }
            buf.push('\n');
        }

        // 2. User profile
        if let Some(profile) = &self.user_profile {
            if profile.is_not_empty() {
                buf.push_str("=== User Profile ===\n");
                buf.push_str(&profile.to_summary());
                buf.push('\n');
            }
        }

        // 3. Important memories (deduplicated against relevant set)
        let relevant_ids: HashSet<&str> = self
            .relevant_memories
            .iter()
            .map(|m| m.id.as_str())
            .collect();
        let unique_important: Vec<&MemoryItem> = self
            .important_memories
            .iter()
            .filter(|m| !relevant_ids.contains(m.id.as_str()))
            .collect();
        if !unique_important.is_empty() {
            buf.push_str("=== Important Memories ===\n");
            for item in unique_important {
                buf.push_str(&format!(
                    "- [{}] {}\n",
                    item.memory_type.display_name(),
                    item.content
                ));
            }
            buf.push('\n');
        }

        // 4. Relevant memories grouped by type
        if !self.relevant_memories.is_empty() {
            buf.push_str("=== Relevant Memories ===\n");
            let mut grouped: std::collections::BTreeMap<&str, Vec<&MemoryItem>> =
                std::collections::BTreeMap::new();
            for item in &self.relevant_memories {
                grouped
                    .entry(item.memory_type.display_name())
                    .or_default()
                    .push(item);
            }
            for (type_name, items) in &grouped {
                buf.push_str(&format!("{type_name}:\n"));
                for item in items {
                    let score_str = match item.score {
                        Some(s) => format!(" (relevance: {s:.2})"),
                        None => String::new(),
                    };
                    // v5: Add recency annotation so Claude prefers recent info
                    let age = format_age(item.updated_at);
                    buf.push_str(&format!("  - [{age}] {}{}\n", item.content, score_str));
                }
            }
            buf.push('\n');
        }

        // 5. Focused items
        if !self.attention_items.is_empty() {
            buf.push_str("=== Focused Items ===\n");
            for item in &self.attention_items {
                buf.push_str(&format!(
                    "- [{}] {}\n",
                    item.memory_type.display_name(),
                    item.content
                ));
            }
        }

        buf.trim_end().to_string()
    }
}

impl Default for WorkingMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// Format how old a memory is in human-readable form.
fn format_age(updated_at: chrono::DateTime<chrono::Utc>) -> String {
    let days = (chrono::Utc::now() - updated_at).num_days();
    if days == 0 {
        "today".to_string()
    } else if days == 1 {
        "1 day ago".to_string()
    } else if days < 7 {
        format!("{days} days ago")
    } else if days < 30 {
        format!("{} weeks ago", days / 7)
    } else if days < 365 {
        format!("{} months ago", days / 30)
    } else {
        format!("{} years ago", days / 365)
    }
}

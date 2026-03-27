use std::cmp::Ordering;
use std::collections::HashSet;

use crate::models::*;

/// Working memory: the active context assembled for each LLM call.
pub struct WorkingMemory {
    relevant_memories: Vec<MemoryItem>,
    important_memories: Vec<MemoryItem>,
    keyword_matches: Vec<MemoryItem>,
    key_triples: Vec<Triple>,
    session_summaries: Vec<SessionSummary>,
    triggered_reminders: Vec<ProspectiveItem>,
    user_profile: Option<UserProfile>,
    attention_items: Vec<MemoryItem>,
}

impl WorkingMemory {
    pub fn new() -> Self {
        Self {
            relevant_memories: Vec::new(),
            important_memories: Vec::new(),
            keyword_matches: Vec::new(),
            key_triples: Vec::new(),
            session_summaries: Vec::new(),
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

    pub fn set_keyword_matches(&mut self, items: Vec<MemoryItem>) {
        self.keyword_matches = items;
    }

    pub fn set_triggered_reminders(&mut self, reminders: Vec<ProspectiveItem>) {
        self.triggered_reminders = reminders;
    }

    pub fn set_profile(&mut self, profile: Option<UserProfile>) {
        self.user_profile = profile;
    }

    pub fn set_key_triples(&mut self, triples: Vec<Triple>) {
        self.key_triples = triples;
    }

    pub fn set_session_summaries(&mut self, summaries: Vec<SessionSummary>) {
        self.session_summaries = summaries;
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
        self.keyword_matches.clear();
        self.key_triples.clear();
        self.session_summaries.clear();
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

    pub fn key_triples(&self) -> &[Triple] {
        &self.key_triples
    }

    pub fn session_summaries(&self) -> &[SessionSummary] {
        &self.session_summaries
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
            || !self.keyword_matches.is_empty()
            || !self.key_triples.is_empty()
            || !self.session_summaries.is_empty()
            || !self.attention_items.is_empty()
    }

    /// Build a text context block for injection into an LLM system prompt.
    ///
    /// Tiered structure:
    /// 1. Triggered Reminders (highest priority)
    /// 2. User Profile
    /// 3. Top Evidence (highest-ranked query matches)
    /// 4. Raw Conversation Excerpts (limited, supporting evidence)
    /// 5. Important Memories (deduplicated against relevant set)
    /// 6. Supporting Memories (grouped by type)
    /// 7. Focused Items (attention context)
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
                buf.push_str("=== USER PROFILE ===\n");
                buf.push_str(&profile.to_summary());
                buf.push('\n');
            }
        }

        // KEY FACTS: BM25 keyword matches displayed prominently at the top.
        if !self.keyword_matches.is_empty() {
            buf.push_str("=== KEY FACTS (exact keyword matches — read these first) ===\n");
            let mut km_seen = HashSet::new();
            let mut keyword_matches = self.keyword_matches.clone();
            sort_by_score_desc(&mut keyword_matches);
            for item in &keyword_matches {
                if km_seen.insert(item.id.as_str()) {
                    buf.push_str(&format!(
                        "- [{}{}] {}\n",
                        item.memory_type.display_name(),
                        format_temporal_tag(item),
                        item.content
                    ));
                }
            }
            buf.push('\n');
        }

        // KEY FACTS from triples (structured knowledge — highest signal, lowest noise)
        if !self.key_triples.is_empty() {
            buf.push_str("=== KEY FACTS (structured knowledge) ===\n");
            // Group triples by speaker
            let mut by_speaker: std::collections::BTreeMap<String, Vec<&Triple>> =
                std::collections::BTreeMap::new();
            for triple in &self.key_triples {
                let key = if triple.speaker.is_empty() {
                    "General".to_string()
                } else {
                    triple.speaker.clone()
                };
                by_speaker.entry(key).or_default().push(triple);
            }
            for (speaker, triples) in &by_speaker {
                buf.push_str(&format!("  About {}:\n", speaker));
                for t in triples {
                    buf.push_str(&format!(
                        "  - {} {} {} (mentioned {}x, confidence: {:.0}%)\n",
                        t.subject, t.predicate, t.object, t.mention_count,
                        t.confidence * 100.0
                    ));
                }
            }
            buf.push('\n');
        }

        // Separate raw conversation excerpts from extracted memories
        let is_raw_conversation = |item: &MemoryItem| -> bool {
            item.metadata
                .get("source")
                .and_then(|v| v.as_str())
                .map(|s| s == "raw_conversation")
                .unwrap_or(false)
        };

        let mut top_evidence: Vec<&MemoryItem> = self
            .relevant_memories
            .iter()
            .filter(|m| !is_raw_conversation(m))
            .collect();
        sort_refs_by_score_desc(&mut top_evidence);
        top_evidence.truncate(8);

        if !top_evidence.is_empty() {
            buf.push_str("=== TOP EVIDENCE (highest-ranked answer candidates) ===\n");
            for item in &top_evidence {
                let score_str = item
                    .score
                    .map(|score| format!(" relevance: {score:.2}"))
                    .unwrap_or_default();
                let temporal_tag = format_temporal_tag(item);
                buf.push_str(&format!(
                    "- [{}{}{}] {}\n",
                    item.memory_type.display_name(),
                    temporal_tag,
                    score_str,
                    item.content
                ));
            }
            buf.push('\n');
        }

        // Collect raw conversation excerpts from all memory pools.
        let mut raw_conversations: Vec<&MemoryItem> = self
            .relevant_memories
            .iter()
            .chain(self.important_memories.iter())
            .filter(|m| is_raw_conversation(m))
            .collect();
        sort_refs_by_temporal_then_score(&mut raw_conversations);

        // Deduplicate raw conversations by ID
        let mut raw_seen = HashSet::new();
        let raw_conversations: Vec<&MemoryItem> = raw_conversations
            .into_iter()
            .filter(|m| raw_seen.insert(m.id.as_str()))
            .take(5)
            .collect();

        if !raw_conversations.is_empty() {
            buf.push_str("=== RAW CONVERSATION EXCERPTS ===\n");
            for item in &raw_conversations {
                let session_tag = item
                    .session_id
                    .as_deref()
                    .map(|s| format!("Session {s}"))
                    .unwrap_or_default();
                let date_tag = item
                    .valid_at
                    .map(|dt| dt.format("%-d %b %Y").to_string())
                    .unwrap_or_default();
                let prefix = match (session_tag.is_empty(), date_tag.is_empty()) {
                    (false, false) => format!("[{session_tag}, {date_tag}] "),
                    (false, true) => format!("[{session_tag}] "),
                    (true, false) => format!("[{date_tag}] "),
                    (true, true) => String::new(),
                };
                buf.push_str(&format!("{prefix}{}\n", item.content));
            }
            buf.push('\n');
        }

        // Session summaries (narrative context for multi-hop questions)
        if !self.session_summaries.is_empty() {
            buf.push_str("=== SESSION CONTEXT ===\n");
            for summary in &self.session_summaries {
                let speakers = if summary.speakers.is_empty() {
                    String::new()
                } else {
                    format!(" [{}]", summary.speakers.join(", "))
                };
                buf.push_str(&format!(
                    "- Session {}{}: {}\n",
                    summary.session_id, speakers, summary.summary
                ));
            }
            buf.push('\n');
        }

        // 5. Important memories (deduplicated against relevant set and raw conversations)
        let relevant_ids: HashSet<&str> = self
            .relevant_memories
            .iter()
            .map(|m| m.id.as_str())
            .collect();
        let unique_important: Vec<&MemoryItem> = self
            .important_memories
            .iter()
            .filter(|m| !relevant_ids.contains(m.id.as_str()))
            .filter(|m| !is_raw_conversation(m))
            .collect();
        if !unique_important.is_empty() {
            buf.push_str("=== Important Memories ===\n");
            for item in unique_important {
                let temporal_tag = format_temporal_tag(item);
                buf.push_str(&format!(
                    "- [{}{}] {}\n",
                    item.memory_type.display_name(),
                    temporal_tag,
                    item.content
                ));
            }
            buf.push('\n');
        }

        // 6. Supporting memories grouped by type (excluding raw conversations and top evidence)
        let top_evidence_ids: HashSet<&str> = top_evidence.iter().map(|m| m.id.as_str()).collect();
        let extracted_relevant: Vec<&MemoryItem> = self
            .relevant_memories
            .iter()
            .filter(|m| !is_raw_conversation(m))
            .filter(|m| !top_evidence_ids.contains(m.id.as_str()))
            .collect();

        if !extracted_relevant.is_empty() {
            buf.push_str("=== SUPPORTING MEMORIES ===\n");
            // Group by speaker (from metadata), then by type within speaker
            let mut by_speaker: std::collections::BTreeMap<String, Vec<&MemoryItem>> =
                std::collections::BTreeMap::new();
            for item in &extracted_relevant {
                let speaker_key = item.speaker().to_string();
                let key = if speaker_key.is_empty() {
                    "General".to_string()
                } else {
                    speaker_key
                };
                by_speaker.entry(key).or_default().push(item);
            }
            for (speaker, items) in &by_speaker {
                buf.push_str(&format!("  {}:\n", speaker));
                for item in items.iter().take(8) {
                    let score_str = match item.score {
                        Some(s) => format!(" (relevance: {s:.2})"),
                        None => String::new(),
                    };
                    let temporal_tag = format_temporal_tag(item);
                    let age = format_age(item.updated_at);
                    buf.push_str(&format!(
                        "    - [{}{}{}] {}{}\n",
                        item.memory_type.display_name(),
                        if age.is_empty() {
                            String::new()
                        } else {
                            format!(", {}", age)
                        },
                        temporal_tag,
                        item.content,
                        score_str,
                    ));
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

fn sort_by_score_desc(items: &mut [MemoryItem]) {
    items.sort_by(|a, b| compare_scores_desc(a, b));
}

fn sort_refs_by_score_desc(items: &mut [&MemoryItem]) {
    items.sort_by(|a, b| compare_scores_desc(a, b));
}

fn sort_refs_by_temporal_then_score(items: &mut [&MemoryItem]) {
    items.sort_by(|a, b| compare_temporal_then_score(a, b));
}

fn compare_scores_desc(a: &MemoryItem, b: &MemoryItem) -> Ordering {
    b.score
        .unwrap_or(f64::MIN)
        .partial_cmp(&a.score.unwrap_or(f64::MIN))
        .unwrap_or(Ordering::Equal)
        .then_with(|| b.updated_at.cmp(&a.updated_at))
}

fn compare_temporal_then_score(a: &MemoryItem, b: &MemoryItem) -> Ordering {
    match (a.valid_at, b.valid_at) {
        (Some(a_dt), Some(b_dt)) => a_dt
            .cmp(&b_dt)
            .then_with(|| compare_scores_desc(a, b)),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => compare_scores_desc(a, b),
    }
}

impl Default for WorkingMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// Format a temporal annotation tag for a memory item.
/// If `valid_at` is set, returns something like ", event: May 2023".
/// If not set, returns an empty string.
fn format_temporal_tag(item: &MemoryItem) -> String {
    match item.valid_at {
        Some(dt) => {
            let formatted = dt.format("%b %Y").to_string();
            format!(", event: {formatted}")
        }
        None => String::new(),
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

use chrono::{DateTime, Datelike, Timelike, Utc};
use serde::{Deserialize, Serialize};

/// Temporal context passed to the extraction LLM so it can resolve
/// relative dates ("tomorrow", "next Saturday", "9pm today") to UTC.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TemporalContext {
    /// Current time in UTC.
    pub now_utc: DateTime<Utc>,
    /// Offset from UTC in minutes. E.g., +330 for IST (UTC+05:30).
    pub timezone_offset_minutes: i32,
    /// Human-readable timezone name, e.g., "Asia/Kolkata".
    pub timezone_name: String,
}

impl TemporalContext {
    /// Create a context for Indian Standard Time.
    pub fn ist() -> Self {
        Self {
            now_utc: Utc::now(),
            timezone_offset_minutes: 330,
            timezone_name: "Asia/Kolkata".to_string(),
        }
    }

    /// Create a context for a given offset.
    pub fn new(timezone_offset_minutes: i32, timezone_name: impl Into<String>) -> Self {
        Self {
            now_utc: Utc::now(),
            timezone_offset_minutes,
            timezone_name: timezone_name.into(),
        }
    }

    /// Create a context with a specific datetime (for historical conversations).
    pub fn with_datetime(
        now_utc: DateTime<Utc>,
        timezone_offset_minutes: i32,
        timezone_name: impl Into<String>,
    ) -> Self {
        Self {
            now_utc,
            timezone_offset_minutes,
            timezone_name: timezone_name.into(),
        }
    }

    /// Format for the LLM prompt, e.g.:
    /// `Wednesday, 19 March 2026 16:00 +05:30 (Asia/Kolkata)`
    ///
    /// Includes day-of-week (critical for "next Saturday" resolution),
    /// the UTC offset, and the named timezone.
    pub fn format_for_prompt(&self) -> String {
        let local = self.now_utc + chrono::Duration::minutes(self.timezone_offset_minutes as i64);

        const DAYS: &[&str] = &[
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ];
        const MONTHS: &[&str] = &[
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ];

        let day_name = DAYS[local.weekday().num_days_from_monday() as usize];
        let month_name = MONTHS[(local.month0()) as usize];
        let sign = if self.timezone_offset_minutes >= 0 {
            '+'
        } else {
            '-'
        };
        let abs_offset = self.timezone_offset_minutes.unsigned_abs();
        let oh = abs_offset / 60;
        let om = abs_offset % 60;

        format!(
            "{}, {} {} {} {:02}:{:02} {}{:02}:{:02} ({})",
            day_name,
            local.day(),
            month_name,
            local.year(),
            local.hour(),
            local.minute(),
            sign,
            oh,
            om,
            self.timezone_name,
        )
    }

    /// The UTC offset as a string like "+05:30" or "-08:00".
    pub fn offset_string(&self) -> String {
        let sign = if self.timezone_offset_minutes >= 0 {
            '+'
        } else {
            '-'
        };
        let abs = self.timezone_offset_minutes.unsigned_abs();
        format!("{}{:02}:{:02}", sign, abs / 60, abs % 60)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Extraction prompt — tells the LLM how to extract, classify, and temporally
// anchor memories from raw conversation text.
// ═══════════════════════════════════════════════════════════════════════════════

pub const EXTRACTION_SYSTEM: &str = r#"You are a memory extraction engine. Analyze the text and extract
distinct facts, preferences, events, skills, relationships, emotions,
locations, and intentions.

For each extracted memory, output a JSON array of objects. Each object must
have these fields:
- "content": a single, self-contained factual statement
- "type": one of: episodic, semantic, factual, procedural, social, spatial, prospective, affective
- "confidence": a number from 0.0 to 1.0 indicating certainty

Optional temporal fields (include ONLY for time-anchored items):
- "valid_at": UTC ISO 8601 datetime when this event/fact starts (e.g., "2026-03-20T15:30:00Z")
- "invalid_at": UTC ISO 8601 datetime when this memory expires/becomes irrelevant
  For single-day events, set to end-of-day UTC. Omit for persistent facts.

Type classification rules:
- episodic: events, experiences, meetings, trips (things that happened or will happen)
- factual: preferences, personal details, stable facts ("likes sushi", "works at Google")
- semantic: general knowledge, insights, learned concepts
- procedural: workflows, skills, routines, how-to knowledge
- social: relationships, people, team dynamics ("manager is Bob")
- spatial: locations, places, addresses
- prospective: reminders, future intentions ("need to follow up with X")
- affective: emotions, moods, feelings, stress levels

Confidence guidelines:
- >0.8: Explicit, unambiguous statements
- 0.5-0.8: Reasonable inferences
- <0.5: Speculative; prefer not to extract

CRITICAL — SPEAKER IDENTITY:
The text may be a conversation between two or more named speakers.
Do NOT flatten all speakers to "The user". Use the ACTUAL SPEAKER NAME:
- If "Caroline: I love abstract art" → "Caroline loves abstract art"
- If "Melanie: My kids really like dinosaurs" → "Melanie's kids like dinosaurs"
- If a single user is speaking (no named speakers), use "The user ..."
Always attribute facts, opinions, and experiences to the correct named speaker.

CRITICAL — Temporal resolution rules:
You will be given CURRENT DATE AND TIME with timezone. Use it to resolve ALL relative references:
- "tomorrow" = today + 1 day, preserve timezone. Convert to UTC for valid_at.
- "today at 9pm" = today's date + 21:00 in the user's local timezone → convert to UTC.
- "next Saturday" = the NEXT Saturday on or after today. Check the day-of-week carefully!
- "in 2 hours" = current time + 2 hours.
- "this evening" = today at 18:00 local time → UTC.
- "last week" = 7 days ago from today.
ALL valid_at/invalid_at MUST be in UTC with Z suffix. Do the timezone math carefully.
When content contains relative dates, REWRITE the content with the resolved absolute date:
  "Meeting with X tomorrow at 9pm" → "Caroline has a meeting with X on Thursday, 20 March 2026 at 21:00 IST (15:30 UTC)"
When the text includes session timestamps like [Session N, datetime], use that datetime
as the anchor for the event. Set valid_at to the session datetime converted to UTC.

CRITICAL — PRESERVE EXACT WORDING:
When extracting, you MUST keep the exact original words for:
- Proper nouns: "Devaraj Market" NOT "market", "Koramangala" NOT "neighborhood"
- People's names: "Arjun" NOT "colleague", "Priya" NOT "wife"
- Numbers and amounts: "$45,000" NOT "around 45K", "3:30pm" NOT "afternoon"
- Specific terms: "fusion curry" NOT "experimental cooking", "half-marathon" NOT "running goal"
- Place names, company names, product names: keep exactly as stated
The extracted content MUST contain the SAME specific words the speaker used.

Rules:
1. Each memory must be atomic — one fact per item.
2. Use the speaker's actual name (not "The user") when names are present in the text.
3. If you see temporal references, ALWAYS resolve them and include valid_at/invalid_at.
4. For preferences, use factual type.
5. For emotions/moods, use affective type.
6. For "remember to..." or future intentions, use prospective type.
7. Ignore greetings, filler, and meta-conversation.

Only output the JSON array. No explanation, no markdown fencing."#;

/// Build the user message for memory extraction.
pub fn build_extraction_user(text: &str, temporal: &TemporalContext) -> String {
    format!(
        "CURRENT DATE AND TIME: {}\n\nTEXT TO EXTRACT MEMORIES FROM:\n{}",
        temporal.format_for_prompt(),
        text
    )
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dedup/update prompt — decides ADD/UPDATE/DELETE/NONE for each candidate
// ═══════════════════════════════════════════════════════════════════════════════

pub const DEDUP_SYSTEM: &str = r#"You are a memory deduplication engine. Given existing memories
and a new candidate memory, decide what to do.

Output a single JSON object with:
- "action": "ADD" (new info), "UPDATE" (replaces existing), "SKIP" (duplicate), "CONTRADICTION" (conflicts)
- "existing_id": ID of the existing memory being affected (for UPDATE/CONTRADICTION). null for ADD/SKIP.
- "reason": brief one-line explanation

Rules:
1. If the new fact is essentially the same as an existing one → SKIP
2. If the new fact is a more recent/updated version of an existing fact → UPDATE
3. If the new fact directly contradicts an existing fact → CONTRADICTION
4. If the new fact is genuinely new information → ADD

Only output the JSON object. No explanation, no markdown."#;

/// Build the dedup user message.
pub fn build_dedup_user(
    existing_memories: &[(String, String)], // (id, content)
    new_content: &str,
) -> String {
    let mut buf = String::from("EXISTING MEMORIES:\n");
    for (id, content) in existing_memories {
        buf.push_str(&format!("  [{id}] {content}\n"));
    }
    buf.push_str(&format!("\nNEW CANDIDATE:\n  {new_content}"));
    buf
}

// ═══════════════════════════════════════════════════════════════════════════════
// Consolidation prompt (same as Dart)
// ═══════════════════════════════════════════════════════════════════════════════

pub const CONSOLIDATION_SYSTEM: &str = r#"You are a memory consolidation engine. Given a list of related episodic
memories about the same person, produce a single concise semantic memory
that captures the key insight, pattern, or relationship.

Rules:
1. Output only the summary text — one sentence, third-person factual.
2. Start with "The user ..." when referring to the person.
3. Capture the essence, not every detail.
4. Do not start with "Based on" or "These memories show".

Only output the single summary sentence. No explanation, no JSON, no markdown."#;

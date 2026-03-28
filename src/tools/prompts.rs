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

pub const EXTRACTION_SYSTEM: &str = r#"You are a memory extraction engine (v2). Analyze the text and extract
distinct facts, preferences, events, skills, relationships, emotions,
locations, and intentions.

OUTPUT FORMAT — structured JSON object (NOT an array):
{
  "memories": [ ... ],
    "observations": [ ... ],
  "session_summary": "...",
  "speakers_detected": ["..."]
}

Each object in the "memories" array MUST have ALL of these fields:
- "content": a single, self-contained factual statement
- "type": one of: episodic, semantic, factual, procedural, social, spatial, prospective, affective
- "confidence": a number from 0.0 to 1.0 indicating certainty
- "speaker": the name of the person this fact is about or who stated it (see SPEAKER IDENTITY below)
- "triple": {"subject": "...", "predicate": "...", "object": "..."} — a semantic triple (see TRIPLE RULES below)
- "contradicts_pattern": a search pattern string, or null (see CONTRADICTION DETECTION below)

Optional temporal fields (include ONLY for time-anchored items):
- "valid_at": UTC ISO 8601 datetime when this event/fact starts (e.g., "2026-03-20T15:30:00Z")
- "invalid_at": UTC ISO 8601 datetime when this memory expires/becomes irrelevant
  For single-day events, set to end-of-day UTC. Omit for persistent facts.

Top-level fields:
- "observations": array of ultra-literal factual statements that should be preserved exactly as
    stated, even if they feel too specific or narrow for the main memory list. Use this for facts
    such as relationship status, exact counts, named books or artists, gifts, reminders, and other
    precise details that are easy to lose during abstraction. Each observation item MUST have:
    - "content": a single factual statement using the original wording as closely as possible
    - "speaker": the person the fact is about or who stated it
    - "confidence": a number from 0.0 to 1.0
    Optional temporal fields: "valid_at", "invalid_at"
- "session_summary": 2-3 sentences summarizing the conversation's narrative arc. Mention speakers
  by name and what they discussed. Include specific dates and key facts mentioned.
- "speakers_detected": array of ALL speaker names detected in the text. Use actual names.
  If a single unnamed user is speaking, use ["user"].

TRIPLE RULES:
Every memory MUST include a "triple" field with:
- "subject": a named entity (person name, place, thing). NEVER a pronoun like "she" or "they".
- "predicate": a short verb phrase (e.g., "loves", "works at", "is allergic to", "lives in").
- "object": a specific value (e.g., "abstract art", "Google", "peanuts", "Koramangala").
Examples:
  Content: "Caroline loves abstract art" → {"subject": "Caroline", "predicate": "loves", "object": "abstract art"}
  Content: "Melanie's kids like dinosaurs" → {"subject": "Melanie's kids", "predicate": "like", "object": "dinosaurs"}
  Content: "Arjun works at Google" → {"subject": "Arjun", "predicate": "works at", "object": "Google"}

CONTRADICTION DETECTION:
When extracting a memory that might update or contradict something previously stated in the
conversation, set "contradicts_pattern" to a search string of the form "subject predicate_keyword".
Examples:
- Someone changes their favorite color → "contradicts_pattern": "Caroline favorite color"
- Someone moves to a new city → "contradicts_pattern": "Caroline lives in"
- Someone changes jobs → "contradicts_pattern": "Arjun works at"
Set to null when no contradiction is suspected. Most memories will have null.

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
- If "Caroline: I love abstract art" → content: "Caroline loves abstract art", speaker: "Caroline"
- If "Melanie: My kids really like dinosaurs" → content: "Melanie's kids like dinosaurs", speaker: "Melanie"
- If a single user is speaking (no named speakers), set speaker to "user"
Always attribute facts, opinions, and experiences to the correct named speaker.
The "speaker" field must match the person the fact is ABOUT or who stated it.

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

SESSION TIMESTAMP ANCHORING:
When the text includes session timestamps like [Session N, datetime], use that datetime
as the anchor for ALL events in that session. Set valid_at to the session datetime converted to UTC.
Events within a session that reference "today", "this morning", etc. should resolve relative to
the session datetime, not the current time.

CRITICAL — PRESERVE EXACT WORDING:
When extracting, you MUST keep the exact original words for:
- Proper nouns: "Devaraj Market" NOT "market", "Koramangala" NOT "neighborhood"
- People's names: "Arjun" NOT "colleague", "Priya" NOT "wife"
- Numbers and amounts: "$45,000" NOT "around 45K", "3:30pm" NOT "afternoon"
- Specific terms: "fusion curry" NOT "experimental cooking", "half-marathon" NOT "running goal"
- Place names, company names, product names: keep exactly as stated
The extracted content MUST contain the SAME specific words the speaker used.

WRONG: "Melanie's kids enjoy creative activities" (too vague, not what was said)
RIGHT: "Melanie's kids like dinosaurs and nature" (exact words from text)

WRONG: "Caroline participated in community events" (vague generalization)
RIGHT: "Caroline attended a pride parade" (specific event mentioned)

WRONG: "They discussed weekend plans" (vague, uses pronoun)
RIGHT: "Arjun plans to visit Devaraj Market on Saturday" (specific, named, exact)

OBSERVATION EXAMPLES:
- If the speaker says "I'm dating Stefan", include an observation like "Caroline is dating Stefan"
- If the speaker says "I bought three beach towels", include an observation like
    "Caroline bought three beach towels"
- If the speaker says "The bowl was hand-painted", include an observation like
    "The bowl was hand-painted"
- If the speaker says "My kids like dinosaurs and nature", include an observation like
    "Melanie's kids like dinosaurs and nature"

Rules:
1. Each memory must be atomic — one fact per item.
2. Use the speaker's actual name (not "The user") when names are present in the text.
3. If you see temporal references, ALWAYS resolve them and include valid_at/invalid_at.
4. For preferences, use factual type.
5. For emotions/moods, use affective type.
6. For "remember to..." or future intentions, use prospective type.
7. Ignore greetings, filler, and meta-conversation.
8. Every memory MUST have a triple and a speaker — no exceptions.
9. Observations should stay literal and specific. Do not generalize or paraphrase them.

Only output the JSON object. No explanation, no markdown fencing."#;

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

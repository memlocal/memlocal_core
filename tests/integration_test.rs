use std::sync::Arc;

use chrono::Utc;
use uuid::Uuid;

use memlocal_core::error::Result;
use memlocal_core::models::*;
use memlocal_core::storage::MemoryStore;
use memlocal_core::tools::{EmbeddingProvider, ToolCall, ToolExecutor};

// ═══════════════════════════════════════════════════════════════════════════════
// Test helpers
// ═══════════════════════════════════════════════════════════════════════════════

const DIM: u32 = 64; // small dims for fast tests

/// Deterministic mock embedding — each seed gives a distinct vector.
fn mock_embedding(seed: u32) -> Vec<f32> {
    (0..DIM)
        .map(|i| (((seed.wrapping_mul(31).wrapping_add(i)) % 100) as f32) / 100.0)
        .collect()
}

/// Mock embedding provider that hashes the text to pick a seed.
struct MockEmbedding;

impl EmbeddingProvider for MockEmbedding {
    fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let seed: u32 = text
            .bytes()
            .fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
        Ok(mock_embedding(seed))
    }
}

fn open_store() -> Arc<MemoryStore> {
    let config = StorageConfig {
        in_memory: true,
        db_path: None,
        hnsw_m: 16,
        hnsw_ef_construction: 100,
        embedding_dimensions: DIM,
        min_confidence_to_store: 0.3,
        enable_time_decay: false,
    };
    Arc::new(MemoryStore::open(&config, DIM).expect("failed to open in-memory store"))
}

fn make_item(content: &str, user_id: Option<&str>, memory_type: MemoryType) -> MemoryItem {
    MemoryItem {
        id: Uuid::new_v4().to_string(),
        content: content.to_string(),
        memory_type,
        hash: MemoryItem::compute_hash(content),
        user_id: user_id.map(String::from),
        agent_id: None,
        session_id: None,
        metadata: serde_json::json!({}),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        valid_at: None,
        invalid_at: None,
        score: None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group A: Storage CRUD
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_put_and_get_memory() {
    let store = open_store();
    let item = make_item(
        "I love hiking in the mountains",
        Some("alice"),
        MemoryType::Factual,
    );
    let emb = mock_embedding(1);

    store.put_memory(&item, &emb).unwrap();

    let retrieved = store.get_memory(&item.id).unwrap();
    assert!(retrieved.is_some());
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.id, item.id);
    assert_eq!(retrieved.content, "I love hiking in the mountains");
    assert_eq!(retrieved.memory_type, MemoryType::Factual);
    assert_eq!(retrieved.user_id.as_deref(), Some("alice"));
}

#[test]
fn test_get_memories_with_filters() {
    let store = open_store();

    let items = vec![
        (
            make_item("Fact about Alice", Some("alice"), MemoryType::Factual),
            mock_embedding(10),
        ),
        (
            make_item("Episode for Alice", Some("alice"), MemoryType::Episodic),
            mock_embedding(11),
        ),
        (
            make_item("Fact about Bob", Some("bob"), MemoryType::Factual),
            mock_embedding(12),
        ),
    ];
    for (item, emb) in &items {
        store.put_memory(item, emb).unwrap();
    }

    // Filter by user
    let alice_memories = store.get_memories(Some("alice"), None, 100).unwrap();
    assert_eq!(alice_memories.len(), 2);

    // Filter by type
    let factual = store
        .get_memories(None, Some(MemoryType::Factual), 100)
        .unwrap();
    assert_eq!(factual.len(), 2);

    // Filter by user + type
    let alice_factual = store
        .get_memories(Some("alice"), Some(MemoryType::Factual), 100)
        .unwrap();
    assert_eq!(alice_factual.len(), 1);
    assert_eq!(alice_factual[0].content, "Fact about Alice");
}

#[test]
fn test_delete_memory() {
    let store = open_store();
    let item = make_item("Temporary memory", Some("alice"), MemoryType::Episodic);
    store.put_memory(&item, &mock_embedding(20)).unwrap();

    assert!(store.get_memory(&item.id).unwrap().is_some());

    store.delete_memory(&item.id).unwrap();
    assert!(store.get_memory(&item.id).unwrap().is_none());
}

#[test]
fn test_invalidate_memory() {
    let store = open_store();
    let item = make_item("Will be invalidated", Some("alice"), MemoryType::Factual);
    store.put_memory(&item, &mock_embedding(30)).unwrap();

    store.invalidate_memory(&item.id).unwrap();

    // Item still exists but is_valid should be false
    let retrieved = store.get_memory(&item.id).unwrap().unwrap();
    assert!(!retrieved.is_valid());

    // Should not appear in filtered queries (which exclude invalid_at > 0)
    let valid = store.get_memories(Some("alice"), None, 100).unwrap();
    assert!(valid.is_empty());
}

#[test]
fn test_find_by_hash() {
    let store = open_store();
    let content = "Unique content for hash test";
    let item = make_item(content, Some("alice"), MemoryType::Factual);
    store.put_memory(&item, &mock_embedding(40)).unwrap();

    let hash = MemoryItem::compute_hash(content);
    let found = store.find_by_hash(&hash, Some("alice")).unwrap();
    assert!(found.is_some());
    assert_eq!(found.unwrap().id, item.id);

    // Non-existent hash
    let not_found = store.find_by_hash("0000", None).unwrap();
    assert!(not_found.is_none());
}

#[test]
fn test_memory_count() {
    let store = open_store();

    store
        .put_memory(
            &make_item("a", None, MemoryType::Factual),
            &mock_embedding(50),
        )
        .unwrap();
    store
        .put_memory(
            &make_item("b", None, MemoryType::Factual),
            &mock_embedding(51),
        )
        .unwrap();
    store
        .put_memory(
            &make_item("c", None, MemoryType::Episodic),
            &mock_embedding(52),
        )
        .unwrap();

    assert_eq!(store.memory_count(None).unwrap(), 3);
    assert_eq!(store.memory_count(Some(MemoryType::Factual)).unwrap(), 2);
    assert_eq!(store.memory_count(Some(MemoryType::Episodic)).unwrap(), 1);
    assert_eq!(store.memory_count(Some(MemoryType::Social)).unwrap(), 0);
}

#[test]
fn test_put_and_get_edges() {
    let store = open_store();

    let a = make_item("Item A", None, MemoryType::Factual);
    let b = make_item("Item B", None, MemoryType::Factual);
    store.put_memory(&a, &mock_embedding(60)).unwrap();
    store.put_memory(&b, &mock_embedding(61)).unwrap();

    let edge = MemoryEdge::new(a.id.clone(), b.id.clone(), MemoryRelation::RelatesTo);
    store.put_edge(&edge).unwrap();

    let from_edges = store.get_edges_from(&a.id).unwrap();
    assert_eq!(from_edges.len(), 1);
    assert_eq!(from_edges[0].to_id, b.id);
    assert_eq!(from_edges[0].relation, MemoryRelation::RelatesTo);

    let to_edges = store.get_edges_to(&b.id).unwrap();
    assert_eq!(to_edges.len(), 1);
    assert_eq!(to_edges[0].from_id, a.id);

    // Remove edge
    store.remove_edge(&a.id, &b.id, "relates_to").unwrap();
    let after = store.get_edges_from(&a.id).unwrap();
    assert!(after.is_empty());
}

#[test]
fn test_put_and_get_profile() {
    let store = open_store();

    let mut profile = UserProfile::default();
    profile.user_id = "alice".to_string();
    profile.static_facts.insert("name".into(), "Alice".into());
    profile
        .static_facts
        .insert("occupation".into(), "Engineer".into());
    profile
        .dynamic_context
        .insert("mood".into(), "happy".into());
    profile.updated_at = Some(Utc::now());

    store.put_profile(&profile).unwrap();

    let retrieved = store.get_profile("alice").unwrap();
    assert!(retrieved.is_some());
    let p = retrieved.unwrap();
    assert_eq!(p.user_id, "alice");
    assert_eq!(p.static_facts.get("name").unwrap(), "Alice");
    assert_eq!(p.dynamic_context.get("mood").unwrap(), "happy");
}

#[test]
fn test_put_and_get_messages() {
    let store = open_store();

    let m1 = Message {
        role: "user".into(),
        content: "Hello!".into(),
        timestamp: Utc::now(),
        session_id: Some("sess1".into()),
        metadata: None,
    };
    let m2 = Message {
        role: "assistant".into(),
        content: "Hi there!".into(),
        timestamp: Utc::now(),
        session_id: Some("sess1".into()),
        metadata: None,
    };

    store.put_message(&m1, 1).unwrap();
    store.put_message(&m2, 2).unwrap();

    let messages = store.get_messages("sess1", None).unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, "user");
    assert_eq!(messages[1].role, "assistant");

    assert_eq!(store.message_count("sess1").unwrap(), 2);
    assert_eq!(store.message_count("other").unwrap(), 0);
}

#[test]
fn test_prospective_lifecycle() {
    let store = open_store();

    let item = ProspectiveItem {
        id: Uuid::new_v4().to_string(),
        content: "Remind Alice about the meeting".to_string(),
        trigger_type: TriggerType::TopicMention,
        trigger_condition: "meeting".to_string(),
        user_id: Some("alice".to_string()),
        completed: false,
        created_at: Some(Utc::now()),
        completed_at: None,
    };

    store.put_prospective(&item).unwrap();

    let pending = store.get_pending_prospective(Some("alice")).unwrap();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].content, "Remind Alice about the meeting");
    assert!(!pending[0].completed);

    // Complete it
    store.complete_prospective(&item.id).unwrap();

    let after = store.get_pending_prospective(Some("alice")).unwrap();
    assert!(after.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group B: Search
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_search_text() {
    let store = open_store();

    store
        .put_memory(
            &make_item("I love programming in Rust", None, MemoryType::Factual),
            &mock_embedding(100),
        )
        .unwrap();
    store
        .put_memory(
            &make_item(
                "Python is great for data science",
                None,
                MemoryType::Factual,
            ),
            &mock_embedding(101),
        )
        .unwrap();
    store
        .put_memory(
            &make_item(
                "Hiking in the Rocky Mountains was amazing",
                None,
                MemoryType::Episodic,
            ),
            &mock_embedding(102),
        )
        .unwrap();

    let results = store.search_text("Rust programming", 10).unwrap();
    assert!(!results.is_empty());
    // The top result should contain "Rust"
    assert!(results[0].content.contains("Rust"));
}

#[test]
fn test_search_semantic() {
    let store = open_store();
    let embed = MockEmbedding;

    let items = vec![
        "I enjoy cooking Italian food",
        "Machine learning is fascinating",
        "The weather is beautiful today",
    ];

    for content in &items {
        let item = make_item(content, None, MemoryType::Factual);
        let emb = embed.embed_one(content).unwrap();
        store.put_memory(&item, &emb).unwrap();
    }

    // Search with an embedding close to "cooking" content
    let query_emb = embed.embed_one("cooking Italian pasta").unwrap();
    let results = store.search_semantic(&query_emb, 3, None, None).unwrap();
    assert!(!results.is_empty());
    // All items should be returned (we have 3 items, asking for 3)
    assert!(results.len() <= 3);
}

#[test]
fn test_search_hybrid() {
    let store = open_store();
    let embed = MockEmbedding;

    let contents = vec![
        "Rust is a systems programming language",
        "Python is used for machine learning",
        "JavaScript runs in the browser",
        "Go is great for concurrent servers",
    ];

    for content in &contents {
        let item = make_item(content, None, MemoryType::Semantic);
        let emb = embed.embed_one(content).unwrap();
        store.put_memory(&item, &emb).unwrap();
    }

    let query_emb = embed.embed_one("Rust programming language").unwrap();
    let results = store
        .search_hybrid("Rust programming", &query_emb, 4, None, None)
        .unwrap();
    assert!(!results.is_empty());
    // Hybrid merges FTS + semantic + LSH via RRF, top result should mention Rust
    assert!(results[0].content.contains("Rust"));
}

#[test]
fn test_search_graph() {
    let store = open_store();
    let embed = MockEmbedding;

    // Create 3 connected items: A -> B -> C
    let a = make_item("Rust ownership system", None, MemoryType::Semantic);
    let b = make_item("Borrow checker in Rust", None, MemoryType::Semantic);
    let c = make_item("Lifetimes and references", None, MemoryType::Semantic);

    let emb_a = embed.embed_one(&a.content).unwrap();
    let emb_b = embed.embed_one(&b.content).unwrap();
    let emb_c = embed.embed_one(&c.content).unwrap();

    store.put_memory(&a, &emb_a).unwrap();
    store.put_memory(&b, &emb_b).unwrap();
    store.put_memory(&c, &emb_c).unwrap();

    store
        .put_edge(&MemoryEdge::new(
            a.id.clone(),
            b.id.clone(),
            MemoryRelation::RelatesTo,
        ))
        .unwrap();
    store
        .put_edge(&MemoryEdge::new(
            b.id.clone(),
            c.id.clone(),
            MemoryRelation::RelatesTo,
        ))
        .unwrap();

    // Graph search should find seeds + neighbors
    let query_emb = embed.embed_one("Rust ownership").unwrap();
    let results = store.search_graph(&query_emb, 5, None, None, 2).unwrap();
    // Should find at least the seed items
    assert!(!results.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group C: Tool Executor (simulates LLM tool calls)
// ═══════════════════════════════════════════════════════════════════════════════

fn make_tool_call(name: &str, args: serde_json::Value) -> ToolCall {
    ToolCall {
        id: Uuid::new_v4().to_string(),
        name: name.to_string(),
        arguments: args,
    }
}

#[test]
fn test_tool_add_memory() {
    let store = open_store();
    let executor = ToolExecutor::new(Arc::clone(&store));
    let embed = MockEmbedding;

    let call = make_tool_call(
        "add_memory",
        serde_json::json!({
            "content": "Alice's favorite color is blue",
            "memory_type": "factual",
            "user_id": "alice"
        }),
    );

    let result = executor.execute(&call, &embed);
    assert!(result.success, "Tool failed: {}", result.content);

    let parsed: serde_json::Value = serde_json::from_str(&result.content).unwrap();
    assert_eq!(parsed["status"], "stored");
    assert!(parsed["memory_id"].as_str().is_some());

    // Verify it's actually in the store
    let count = store.memory_count(None).unwrap();
    assert_eq!(count, 1);
}

#[test]
fn test_tool_search_memory() {
    let store = open_store();
    let executor = ToolExecutor::new(Arc::clone(&store));
    let embed = MockEmbedding;

    // Add some memories first
    for content in &["Rust is fast", "Python is easy", "Go is concurrent"] {
        let call = make_tool_call(
            "add_memory",
            serde_json::json!({
                "content": content,
                "memory_type": "semantic"
            }),
        );
        let r = executor.execute(&call, &embed);
        assert!(r.success, "add failed: {}", r.content);
    }

    // Search
    let search_call = make_tool_call(
        "search_memory",
        serde_json::json!({
            "query": "Rust performance",
            "mode": "hybrid",
            "limit": 3
        }),
    );

    let result = executor.execute(&search_call, &embed);
    assert!(result.success, "Search failed: {}", result.content);

    let parsed: serde_json::Value = serde_json::from_str(&result.content).unwrap();
    let results = parsed["results"].as_array().unwrap();
    assert!(!results.is_empty());
    assert!(parsed["total"].as_u64().unwrap() > 0);
}

#[test]
fn test_tool_get_memories() {
    let store = open_store();
    let executor = ToolExecutor::new(Arc::clone(&store));
    let embed = MockEmbedding;

    // Add 2 memories
    for content in &["Fact one", "Fact two"] {
        let call = make_tool_call(
            "add_memory",
            serde_json::json!({
                "content": content,
                "memory_type": "factual",
                "user_id": "bob"
            }),
        );
        executor.execute(&call, &embed);
    }

    let call = make_tool_call(
        "get_memories",
        serde_json::json!({
            "user_id": "bob",
            "memory_type": "factual"
        }),
    );

    let result = executor.execute(&call, &embed);
    assert!(result.success);

    let parsed: serde_json::Value = serde_json::from_str(&result.content).unwrap();
    assert_eq!(parsed["total"].as_u64().unwrap(), 2);
}

#[test]
fn test_tool_delete_memory() {
    let store = open_store();
    let executor = ToolExecutor::new(Arc::clone(&store));
    let embed = MockEmbedding;

    // Add
    let add_call = make_tool_call(
        "add_memory",
        serde_json::json!({
            "content": "To be deleted"
        }),
    );
    let add_result = executor.execute(&add_call, &embed);
    let parsed: serde_json::Value = serde_json::from_str(&add_result.content).unwrap();
    let memory_id = parsed["memory_id"].as_str().unwrap().to_string();

    // Delete
    let del_call = make_tool_call(
        "delete_memory",
        serde_json::json!({
            "memory_id": memory_id
        }),
    );
    let del_result = executor.execute(&del_call, &embed);
    assert!(del_result.success);

    assert_eq!(store.memory_count(None).unwrap(), 0);
}

#[test]
fn test_tool_add_and_get_relationship() {
    let store = open_store();
    let executor = ToolExecutor::new(Arc::clone(&store));
    let embed = MockEmbedding;

    // Add two memories
    let add1 = executor.execute(
        &make_tool_call(
            "add_memory",
            serde_json::json!({
                "content": "Memory A"
            }),
        ),
        &embed,
    );
    let add2 = executor.execute(
        &make_tool_call(
            "add_memory",
            serde_json::json!({
                "content": "Memory B"
            }),
        ),
        &embed,
    );

    let id_a: serde_json::Value = serde_json::from_str(&add1.content).unwrap();
    let id_b: serde_json::Value = serde_json::from_str(&add2.content).unwrap();
    let from_id = id_a["memory_id"].as_str().unwrap();
    let to_id = id_b["memory_id"].as_str().unwrap();

    // Add relationship
    let rel_call = make_tool_call(
        "add_relationship",
        serde_json::json!({
            "from_id": from_id,
            "to_id": to_id,
            "relation": "relates_to",
            "weight": 0.9
        }),
    );
    let rel_result = executor.execute(&rel_call, &embed);
    assert!(rel_result.success);

    // Get relationships
    let get_call = make_tool_call(
        "get_relationships",
        serde_json::json!({
            "memory_id": from_id
        }),
    );
    let get_result = executor.execute(&get_call, &embed);
    assert!(get_result.success);

    let parsed: serde_json::Value = serde_json::from_str(&get_result.content).unwrap();
    let outgoing = parsed["outgoing"].as_array().unwrap();
    assert_eq!(outgoing.len(), 1);
    assert_eq!(outgoing[0]["relation"], "relates_to");
}

#[test]
fn test_tool_add_reminder() {
    let store = open_store();
    let executor = ToolExecutor::new(Arc::clone(&store));
    let embed = MockEmbedding;

    let call = make_tool_call(
        "add_reminder",
        serde_json::json!({
            "content": "Remind me to buy groceries",
            "trigger_type": "topic_mention",
            "trigger_condition": "groceries",
            "user_id": "alice"
        }),
    );

    let result = executor.execute(&call, &embed);
    assert!(result.success, "Failed: {}", result.content);

    let parsed: serde_json::Value = serde_json::from_str(&result.content).unwrap();
    assert_eq!(parsed["status"], "created");

    // Should also have stored a prospective memory in the DB
    let pending = store.get_pending_prospective(Some("alice")).unwrap();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].trigger_condition, "groceries");
}

#[test]
fn test_tool_get_context() {
    let store = open_store();
    let executor = ToolExecutor::new(Arc::clone(&store));
    let embed = MockEmbedding;

    // Seed some memories
    for content in &[
        "Alice works at Acme Corp",
        "Alice likes sushi",
        "The project deadline is Friday",
    ] {
        executor.execute(
            &make_tool_call(
                "add_memory",
                serde_json::json!({
                    "content": content,
                    "user_id": "alice"
                }),
            ),
            &embed,
        );
    }

    // Add a profile
    let mut profile = UserProfile::default();
    profile.user_id = "alice".to_string();
    profile.static_facts.insert("name".into(), "Alice".into());
    store.put_profile(&profile).unwrap();

    // Get context
    let ctx_call = make_tool_call(
        "get_context",
        serde_json::json!({
            "query": "What food does Alice like?",
            "user_id": "alice"
        }),
    );
    let result = executor.execute(&ctx_call, &embed);
    assert!(result.success, "Failed: {}", result.content);

    let parsed: serde_json::Value = serde_json::from_str(&result.content).unwrap();
    let memories = parsed["relevant_memories"].as_array().unwrap();
    assert!(!memories.is_empty());
    // Profile should be included
    assert!(parsed["user_profile"].as_str().is_some());
}

#[test]
fn test_tool_get_profile() {
    let store = open_store();
    let executor = ToolExecutor::new(Arc::clone(&store));
    let embed = MockEmbedding;

    // Create profile directly
    let mut profile = UserProfile::default();
    profile.user_id = "bob".to_string();
    profile
        .static_facts
        .insert("language".into(), "English".into());
    store.put_profile(&profile).unwrap();

    let call = make_tool_call(
        "get_user_profile",
        serde_json::json!({
            "user_id": "bob"
        }),
    );
    let result = executor.execute(&call, &embed);
    assert!(result.success);

    let parsed: serde_json::Value = serde_json::from_str(&result.content).unwrap();
    assert_eq!(parsed["user_id"], "bob");
    assert!(parsed["static_facts"]["language"].as_str().is_some());
}

#[test]
fn test_tool_full_workflow() {
    let store = open_store();
    let executor = ToolExecutor::new(Arc::clone(&store));
    let embed = MockEmbedding;

    // 1. Add several memories via tool calls (simulating LLM tool use)
    let memories = vec![
        ("Alice is a software engineer at TechCorp", "factual"),
        (
            "Alice had a great meeting with the design team yesterday",
            "episodic",
        ),
        ("Alice prefers dark mode in all her apps", "factual"),
        ("The team uses Rust for the backend", "semantic"),
        ("Alice's manager is Bob", "social"),
    ];

    let mut memory_ids = Vec::new();
    for (content, mtype) in &memories {
        let call = make_tool_call(
            "add_memory",
            serde_json::json!({
                "content": content,
                "memory_type": mtype,
                "user_id": "alice"
            }),
        );
        let result = executor.execute(&call, &embed);
        assert!(
            result.success,
            "Failed to add '{content}': {}",
            result.content
        );
        let parsed: serde_json::Value = serde_json::from_str(&result.content).unwrap();
        memory_ids.push(parsed["memory_id"].as_str().unwrap().to_string());
    }

    // 2. Add relationships
    let rel_call = make_tool_call(
        "add_relationship",
        serde_json::json!({
            "from_id": &memory_ids[0],  // Alice is engineer
            "to_id": &memory_ids[3],    // Team uses Rust
            "relation": "relates_to"
        }),
    );
    assert!(executor.execute(&rel_call, &embed).success);

    let rel_call2 = make_tool_call(
        "add_relationship",
        serde_json::json!({
            "from_id": &memory_ids[4],  // Alice's manager is Bob
            "to_id": &memory_ids[0],    // Alice is engineer
            "relation": "belongs_to"
        }),
    );
    assert!(executor.execute(&rel_call2, &embed).success);

    // 3. Verify count
    assert_eq!(store.memory_count(None).unwrap(), 5);

    // 4. Search via tool
    let search = make_tool_call(
        "search_memory",
        serde_json::json!({
            "query": "What does Alice do for work?",
            "mode": "hybrid",
            "user_id": "alice",
            "limit": 3
        }),
    );
    let search_result = executor.execute(&search, &embed);
    assert!(search_result.success);
    let parsed: serde_json::Value = serde_json::from_str(&search_result.content).unwrap();
    assert!(!parsed["results"].as_array().unwrap().is_empty());

    // 5. Get context
    let ctx = make_tool_call(
        "get_context",
        serde_json::json!({
            "query": "Tell me about Alice's work",
            "user_id": "alice"
        }),
    );
    let ctx_result = executor.execute(&ctx, &embed);
    assert!(ctx_result.success);
    let ctx_parsed: serde_json::Value = serde_json::from_str(&ctx_result.content).unwrap();
    assert!(!ctx_parsed["relevant_memories"]
        .as_array()
        .unwrap()
        .is_empty());

    // 6. Get memories by type
    let get_factual = make_tool_call(
        "get_memories",
        serde_json::json!({
            "user_id": "alice",
            "memory_type": "factual"
        }),
    );
    let factual_result = executor.execute(&get_factual, &embed);
    assert!(factual_result.success);
    let factual_parsed: serde_json::Value = serde_json::from_str(&factual_result.content).unwrap();
    assert_eq!(factual_parsed["total"].as_u64().unwrap(), 2);

    // 7. Delete one memory
    let del = make_tool_call(
        "delete_memory",
        serde_json::json!({
            "memory_id": &memory_ids[1]  // Delete the episodic one
        }),
    );
    assert!(executor.execute(&del, &embed).success);
    assert_eq!(store.memory_count(None).unwrap(), 4);

    println!("✓ Full workflow test passed: add → relate → search → context → filter → delete");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group D: Live OpenAI API tests (only run with OPENAI_API_KEY env var)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "http")]
mod live_api {
    use super::*;
    use memlocal_core::http::OpenAiEmbeddingProvider;

    fn get_api_key() -> Option<String> {
        // Load .env file from the crate root (silently ignore if missing)
        let _ = dotenvy::dotenv();
        std::env::var("OPENAI_API_KEY").ok()
    }

    /// Open a store with 1536 dims (OpenAI default).
    fn open_store_1536() -> Arc<MemoryStore> {
        let config = StorageConfig {
            in_memory: true,
            embedding_dimensions: 1536,
            ..StorageConfig::default()
        };
        Arc::new(MemoryStore::open(&config, 1536).unwrap())
    }

    #[test]
    fn test_live_openai_embedding() {
        let Some(api_key) = get_api_key() else {
            eprintln!("⏭ Skipping live API test — set OPENAI_API_KEY to run");
            return;
        };

        let provider = OpenAiEmbeddingProvider::new(api_key);
        let embedding = provider.embed_one("Hello, world!").unwrap();

        assert_eq!(embedding.len(), 1536, "Expected 1536 dimensions");
        // Verify it's not all zeros
        assert!(
            embedding.iter().any(|&v| v != 0.0),
            "Embedding should not be all zeros"
        );

        println!(
            "✓ Live embedding: {} dimensions, first 5 values: {:?}",
            embedding.len(),
            &embedding[..5]
        );
    }

    #[test]
    fn test_live_embed_many() {
        let Some(api_key) = get_api_key() else {
            eprintln!("⏭ Skipping live API test — set OPENAI_API_KEY to run");
            return;
        };

        let provider = OpenAiEmbeddingProvider::new(api_key);
        let embeddings = provider
            .embed_many(&["Rust programming", "Python scripting", "Go concurrency"])
            .unwrap();

        assert_eq!(embeddings.len(), 3);
        for emb in &embeddings {
            assert_eq!(emb.len(), 1536);
        }

        println!(
            "✓ Live batch embedding: {} vectors of {} dims",
            embeddings.len(),
            embeddings[0].len()
        );
    }

    #[test]
    fn test_live_add_and_search() {
        let Some(api_key) = get_api_key() else {
            eprintln!("⏭ Skipping live API test — set OPENAI_API_KEY to run");
            return;
        };

        let store = open_store_1536();
        let provider = OpenAiEmbeddingProvider::new(&api_key);

        // Add memories with real embeddings
        let contents = vec![
            "I love cooking Italian pasta with fresh basil and tomatoes",
            "Machine learning algorithms can predict stock market trends",
            "The weather in San Francisco is foggy and cool in summer",
            "Rust's borrow checker ensures memory safety at compile time",
        ];

        for content in &contents {
            let item = make_item(content, Some("test_user"), MemoryType::Factual);
            let emb = provider.embed_one(content).unwrap();
            store.put_memory(&item, &emb).unwrap();
        }

        assert_eq!(store.memory_count(None).unwrap(), 4);

        // Semantic search — "cooking recipes" should find the pasta item
        let query_emb = provider
            .embed_one("cooking recipes and food preparation")
            .unwrap();
        let results = store.search_semantic(&query_emb, 4, None, None).unwrap();
        assert!(!results.is_empty());
        println!("✓ Semantic search results:");
        for (i, item) in results.iter().enumerate() {
            println!(
                "  {i}. [score={:.4}] {}",
                item.score.unwrap_or(0.0),
                item.content
            );
        }
        // Verify the cooking item appears somewhere in results
        assert!(
            results
                .iter()
                .any(|r| r.content.contains("cooking") || r.content.contains("pasta")),
            "Results should contain the cooking item"
        );

        // Hybrid search
        let hybrid_emb = provider.embed_one("Rust memory safety").unwrap();
        let hybrid = store
            .search_hybrid("Rust memory safety", &hybrid_emb, 4, None, None)
            .unwrap();
        assert!(!hybrid.is_empty());
        println!("✓ Hybrid search results:");
        for (i, item) in hybrid.iter().enumerate() {
            println!(
                "  {i}. [score={:.4}] {}",
                item.score.unwrap_or(0.0),
                item.content
            );
        }
        // Verify the Rust item appears somewhere in results
        assert!(
            hybrid.iter().any(|r| r.content.contains("Rust")),
            "Results should contain the Rust item"
        );
    }

    #[test]
    fn test_live_tool_executor_with_real_embeddings() {
        let Some(api_key) = get_api_key() else {
            eprintln!("⏭ Skipping live API test — set OPENAI_API_KEY to run");
            return;
        };

        let store = open_store_1536();
        let executor = ToolExecutor::new(Arc::clone(&store));
        let provider = OpenAiEmbeddingProvider::new(&api_key);

        // Add memories via tool calls (as an LLM would)
        let tool_calls = vec![
            serde_json::json!({"content": "The user's name is Alice and she works at Google", "memory_type": "factual", "user_id": "alice"}),
            serde_json::json!({"content": "Alice prefers dark mode in all applications", "memory_type": "factual", "user_id": "alice"}),
            serde_json::json!({"content": "Had a productive standup meeting this morning", "memory_type": "episodic", "user_id": "alice"}),
        ];

        for args in &tool_calls {
            let call = make_tool_call("add_memory", args.clone());
            let result = executor.execute(&call, &provider);
            assert!(result.success, "Failed: {}", result.content);
        }

        // Search as an LLM would
        let search_call = make_tool_call(
            "search_memory",
            serde_json::json!({
                "query": "What does Alice do for a living?",
                "mode": "hybrid",
                "user_id": "alice",
                "limit": 3
            }),
        );

        let result = executor.execute(&search_call, &provider);
        assert!(result.success, "Search failed: {}", result.content);

        let parsed: serde_json::Value = serde_json::from_str(&result.content).unwrap();
        let results = parsed["results"].as_array().unwrap();
        assert!(!results.is_empty());
        println!("✓ Live tool search results:");
        for r in results {
            println!(
                "  - [score={:.4}] {}",
                r["score"].as_f64().unwrap_or(0.0),
                r["content"].as_str().unwrap_or("")
            );
        }
        // Results should contain the Google/work item somewhere
        assert!(
            results
                .iter()
                .any(|r| r["content"].as_str().unwrap_or("").contains("Google")),
            "Results should mention Google somewhere"
        );
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Live Claude Tool-Calling Test
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_live_claude_tool_calling() {
        let _ = dotenvy::dotenv();
        let Some(llm_key) = std::env::var("LLM_API_KEY").or_else(|_| std::env::var("ANTHROPIC_API_KEY")).ok() else {
            eprintln!("⏭ Skipping Claude test — set LLM_API_KEY to run");
            return;
        };
        let Some(openai_key) = std::env::var("OPENAI_API_KEY").ok() else {
            eprintln!("⏭ Skipping Claude test — set OPENAI_API_KEY to run");
            return;
        };

        use memlocal_core::http::{run_with_tools, AnthropicClient, LlmMessage};
        use memlocal_core::tools::all_tool_definitions;

        let store = open_store_1536();
        let executor = ToolExecutor::new(Arc::clone(&store));
        let embedder = OpenAiEmbeddingProvider::new(&openai_key);

        // 1. Pre-seed memories with real embeddings
        println!("\n══════════════════════════════════════════");
        println!("  LIVE CLAUDE TOOL-CALLING TEST");
        println!("══════════════════════════════════════════\n");

        let memories = vec![
            (
                "Alice loves eating sushi, especially salmon nigiri",
                "factual",
            ),
            ("Alice is a software engineer working at Google", "factual"),
            (
                "Alice had a great hiking trip in Yosemite last weekend",
                "episodic",
            ),
            ("Alice prefers dark mode and uses VS Code", "factual"),
            ("The team standup is every day at 10am", "procedural"),
            (
                "Alice's manager is Bob, who leads the backend team",
                "social",
            ),
        ];

        println!("📝 Seeding {} memories...", memories.len());
        for (content, mtype) in &memories {
            let call = make_tool_call(
                "add_memory",
                serde_json::json!({
                    "content": content,
                    "memory_type": mtype,
                    "user_id": "alice"
                }),
            );
            let result = executor.execute(&call, &embedder);
            assert!(result.success, "Failed to seed memory: {}", result.content);
            println!("  ✓ Stored: {content}");
        }

        // 2. Create Anthropic client
        let claude = AnthropicClient::new(&llm_key).with_temperature(0.0);

        let tools = all_tool_definitions();

        // 3. Ask a question — Claude should use tools to find the answer
        let question = "What kind of food does Alice enjoy? And where does she work?";
        println!("\n🧑 User: {question}\n");

        let mut messages = vec![
            LlmMessage::system(
                "You are a helpful assistant with access to a memory system. \
                 Use the memory tools to look up information about the user before answering. \
                 Always search memory before responding to questions about the user.",
            ),
            LlmMessage::user(question),
        ];

        let answer = run_with_tools(&claude, &mut messages, &tools, &executor, &embedder, 5)
            .expect("run_with_tools failed");

        // 4. Print the full conversation trace
        println!("── Conversation Trace ──");
        for msg in &messages {
            match msg.role.as_str() {
                "system" => println!("  [SYSTEM] {}", &msg.content[..msg.content.len().min(80)]),
                "user" => println!("  [USER] {}", msg.content),
                "assistant" if !msg.tool_calls.is_empty() => {
                    if !msg.content.is_empty() {
                        println!("  [ASSISTANT] {}", msg.content);
                    }
                    for tc in &msg.tool_calls {
                        println!("  [TOOL CALL] {}({})", tc.name, tc.arguments);
                    }
                }
                "assistant" => println!("  [ASSISTANT] {}", msg.content),
                "tool_result" => {
                    let preview = if msg.content.len() > 120 {
                        format!("{}...", &msg.content[..120])
                    } else {
                        msg.content.clone()
                    };
                    println!("  [TOOL RESULT] {preview}");
                }
                _ => {}
            }
        }

        // 5. Verify the answer
        println!("\n🤖 Claude's answer:\n{answer}\n");

        let answer_lower = answer.to_lowercase();
        assert!(
            answer_lower.contains("sushi") || answer_lower.contains("food"),
            "Answer should mention sushi or food"
        );
        assert!(
            answer_lower.contains("google") || answer_lower.contains("software"),
            "Answer should mention Google or software engineer"
        );

        println!("══════════════════════════════════════════");
        println!("  ✓ Claude successfully used memory tools!");
        println!("══════════════════════════════════════════\n");
    }

    // ═════════════════════════════════════════════════════════════════════════
    // 30-Day Tech Founder Simulation
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_live_30_day_founder_simulation() {
        let _ = dotenvy::dotenv();
        let Some(llm_key) = std::env::var("LLM_API_KEY").or_else(|_| std::env::var("ANTHROPIC_API_KEY")).ok() else {
            eprintln!("⏭ Skipping founder sim — set LLM_API_KEY");
            return;
        };
        let Some(openai_key) = std::env::var("OPENAI_API_KEY").ok() else {
            eprintln!("⏭ Skipping founder sim — set OPENAI_API_KEY");
            return;
        };

        use memlocal_core::http::{run_with_tools, AnthropicClient, LlmMessage};
        use memlocal_core::tools::all_tool_definitions;

        let store = open_store_1536();
        let executor = ToolExecutor::new(Arc::clone(&store));
        let embedder = OpenAiEmbeddingProvider::new(&openai_key);
        let claude = AnthropicClient::new(&llm_key).with_temperature(0.0);
        let tools = all_tool_definitions();

        println!("\n{}", "═".repeat(60));
        println!("  30-DAY TECH FOUNDER MEMORY SIMULATION");
        println!("{}\n", "═".repeat(60));

        // ── Phase 1: Seed 35 memories across 30 days ──

        let founder_thoughts: Vec<(&str, &str, i32)> = vec![
            // (content, memory_type, day_offset from 30 days ago)
            // Week 1: Early ideation
            ("Had the initial idea for memlocal — a local-first AI memory layer that gives LLMs persistent memory without cloud dependency. Privacy-first approach.", "episodic", 30),
            ("Key insight: human memory isn't a flat database. It has tiers — sensory, short-term, long-term — and we should model AI memory the same way.", "semantic", 29),
            ("Tech stack decision: CozoDB for the core database. It supports vector search (HNSW), full-text search, AND graph queries in one engine. Perfect for our use case.", "procedural", 28),
            ("Competitor analysis: Mem0 is cloud-only, Zep requires server infrastructure. Our edge is local-first with zero infrastructure requirement.", "semantic", 27),
            ("Met with Sarah Chen, angel investor. She's interested but wants to see a working demo with real LLM integration before committing.", "episodic", 26),

            // Week 2: Building and hiring
            ("Decided to write the core in Rust for performance and cross-platform FFI. Flutter wrapper for mobile, can later add Python/Swift/Kotlin bindings.", "procedural", 23),
            ("Hired Raj as the first engineer. He has deep experience with embedded databases and Rust. Starting Monday.", "social", 22),
            ("Raj suggested using flutter_rust_bridge for the Dart FFI layer. Good call — it handles all the codegen automatically.", "procedural", 21),
            ("Architecture decision: embeddings stay in the platform layer (Dart/Python), not in the Rust core. This keeps the core dependency-free from HTTP clients.", "procedural", 20),
            ("First customer call with DataFlow Inc. They want memory for their customer support chatbot. Need: per-user isolation, semantic search, conversation history.", "episodic", 19),
            ("Customer feedback from DataFlow: they need minimum 50ms search latency. Our in-memory CozoDB benchmarks at 2ms for 10K memories — way under budget.", "factual", 19),

            // Week 3: Product development
            ("Implemented the 12-type memory taxonomy: sensory buffer, working memory, attention context, conversation buffer, episodic, semantic, factual, procedural, social, spatial, prospective, affective.", "procedural", 16),
            ("Knowledge graph working — can now traverse relationships between memories. PageRank identifies the most connected/important nodes.", "semantic", 15),
            ("Hybrid search with RRF (Reciprocal Rank Fusion) merging semantic + BM25 + LSH gives much better results than any single mode alone.", "semantic", 14),
            ("Bug fix: the HNSW filter parameter was 'bind_filter' but CozoDB actually uses 'filter'. Cost us 2 hours of debugging.", "episodic", 13),
            ("Second call with Sarah Chen. Showed her the demo — she was impressed by the tool-calling loop. She's in for $150K at $2M pre-money.", "episodic", 12),
            ("Sarah introduced us to Marcus Rivera at Sequoia. Scheduling a call for next week.", "social", 12),

            // Week 4: Scaling and partnerships
            ("Marcus Rivera call went well. He wants to see our MRR trajectory. We don't have revenue yet — need to launch the developer SDK first.", "episodic", 8),
            ("Pricing decision: free tier up to 10K memories per user, $29/mo for 100K, $99/mo for unlimited. Enterprise custom pricing.", "factual", 7),
            ("Raj finished the consolidation engine. It clusters old episodic memories by cosine similarity and summarizes them via LLM — like sleep consolidation in the brain.", "procedural", 7),
            ("Partnership discussion with Vercel. They want to integrate memlocal into their AI SDK. Would give us massive distribution.", "episodic", 6),
            ("Technical milestone: the full pipeline works end-to-end. Store → search → retrieve → inject into LLM context. Sub-100ms total latency.", "factual", 5),
            ("DataFlow signed a pilot agreement. $5K/mo for 3 months. First revenue!", "episodic", 5),

            // Week 5: Launch prep
            ("Wrote the SDK documentation. Developer experience is critical — if integration takes more than 5 lines of code, we've failed.", "procedural", 4),
            ("Landing page copy: 'Give your AI a memory. Local-first. Privacy-native. Works with any LLM.' Need to A/B test this.", "semantic", 3),
            ("Raj found a memory leak in the sensory buffer — TTL eviction wasn't triggering when the buffer was exactly at capacity. Fixed and added regression test.", "episodic", 3),
            ("Board update email drafted. Key metrics: 1 paying customer ($5K MRR), 1 committed angel ($150K), 1 VC interested (Sequoia), 2 engineers, core product working.", "factual", 2),
            ("Prospective meeting with Google Developer Relations about featuring memlocal in their AI toolkit showcase at I/O.", "social", 2),
            ("Launch plan: open-source the Rust core on GitHub, paid hosted version for teams, SDK for Flutter/Python/TypeScript.", "procedural", 1),

            // Current state / misc
            ("Our burn rate is $18K/month. With Sarah's $150K we have ~8 months of runway. Need to hit $20K MRR before Series A.", "factual", 0),
            ("Team morale is high. Raj is shipping fast. Need to hire a DevRel person and a designer in the next 30 days.", "social", 0),
            ("Note to self: the prospective memory system (reminders/triggers) could be a killer feature for AI assistants. Nobody else has this.", "semantic", 0),
            ("Affective note: feeling optimistic but stretched thin. Need to delegate more and focus on fundraising + partnerships.", "affective", 0),
        ];

        println!("Phase 1: Seeding {} memories across 30 days...\n", founder_thoughts.len());

        let mut memory_ids: Vec<String> = Vec::new();
        let base_time = Utc::now() - chrono::Duration::days(30);

        for (content, mtype, day_offset) in &founder_thoughts {
            let timestamp = base_time + chrono::Duration::days((30 - day_offset) as i64);
            let id = Uuid::new_v4().to_string();
            let item = MemoryItem {
                id: id.clone(),
                content: content.to_string(),
                memory_type: MemoryType::from_stored_name(mtype),
                hash: MemoryItem::compute_hash(content),
                user_id: Some("founder".into()),
                agent_id: None,
                session_id: None,
                metadata: serde_json::json!({"confidence": 0.95}),
                created_at: timestamp,
                updated_at: timestamp,
                valid_at: None,
                invalid_at: None,
                score: None,
            };
            let emb = embedder.embed_one(content).unwrap();
            store.put_memory(&item, &emb).unwrap();
            memory_ids.push(id);
        }
        println!("  ✓ {} memories stored\n", memory_ids.len());

        // ── Phase 2: Create relationships ──

        println!("Phase 2: Creating knowledge graph edges...\n");

        let relationships = vec![
            (5, 2, "relates_to"),   // Rust decision → CozoDB decision
            (15, 12, "caused_by"),  // Sarah invests → after seeing demo
            (16, 15, "follows"),    // Marcus intro → Sarah investment
            (17, 16, "follows"),    // Marcus call → Marcus intro
            (21, 10, "caused_by"), // DataFlow signs → after customer feedback
            (19, 11, "relates_to"), // Consolidation → memory taxonomy
            (6, 7, "relates_to"),  // Raj hired → Raj suggestion
            (29, 14, "relates_to"), // Burn rate → pricing
        ];

        for (from_idx, to_idx, relation) in &relationships {
            let edge = MemoryEdge::new(
                memory_ids[*from_idx].clone(),
                memory_ids[*to_idx].clone(),
                MemoryRelation::from_stored_name(relation),
            );
            store.put_edge(&edge).unwrap();
        }
        println!("  ✓ {} relationships created\n", relationships.len());

        // ── Phase 3: User profile ──

        println!("Phase 3: Setting up founder profile...\n");

        let mut profile = UserProfile::default();
        profile.user_id = "founder".to_string();
        profile.static_facts.insert("name".into(), "Alex".into());
        profile.static_facts.insert("role".into(), "CEO & Co-founder".into());
        profile.static_facts.insert("company".into(), "memlocal".into());
        profile.static_facts.insert("stage".into(), "Pre-seed".into());
        profile.dynamic_context.insert("focus".into(), "Launch prep and fundraising".into());
        profile.dynamic_context.insert("mood".into(), "Optimistic but stretched thin".into());
        profile.updated_at = Some(Utc::now());
        store.put_profile(&profile).unwrap();
        println!("  ✓ Profile created for Alex (CEO, memlocal)\n");

        // ── Phase 4: Prospective reminders ──

        println!("Phase 4: Adding prospective reminders...\n");

        let reminders = vec![
            ("Follow up with Marcus Rivera at Sequoia about the MRR question", "topic_mention", "Sequoia"),
            ("Send board update email with key metrics", "topic_mention", "board"),
            ("Schedule interview for DevRel hire", "topic_mention", "hiring"),
            ("Prepare demo for Google I/O showcase", "topic_mention", "Google"),
        ];

        for (content, trigger_type, trigger_condition) in &reminders {
            let item = ProspectiveItem {
                id: Uuid::new_v4().to_string(),
                content: content.to_string(),
                trigger_type: TriggerType::from_stored_name(trigger_type),
                trigger_condition: trigger_condition.to_string(),
                user_id: Some("founder".into()),
                completed: false,
                created_at: Some(Utc::now()),
                completed_at: None,
            };
            store.put_prospective(&item).unwrap();
        }
        println!("  ✓ {} reminders set\n", reminders.len());

        // ── Phase 5: Ask questions via Claude ──

        let questions: Vec<(&str, Vec<&str>)> = vec![
            (
                "What were the key technical architecture decisions we made, and why?",
                vec!["Rust", "CozoDB", "flutter_rust_bridge", "embeddings", "platform layer"],
            ),
            (
                "Give me a summary of our investor relations. Who have we talked to and what's the status?",
                vec!["Sarah", "150K", "Marcus", "Sequoia"],
            ),
            (
                "What's the current state of our team and who do we need to hire?",
                vec!["Raj", "DevRel", "designer"],
            ),
            (
                "What customer feedback have we received and how should it influence the product?",
                vec!["DataFlow", "50ms", "latency", "per-user"],
            ),
            (
                "What's our financial situation — burn rate, runway, and revenue?",
                vec!["18K", "runway", "5K", "MRR"],
            ),
        ];

        println!("Phase 5: Asking {} questions via Claude Haiku...\n", questions.len());
        println!("{}", "─".repeat(60));

        let system_prompt = "You are a personal AI assistant for Alex, a tech startup founder. \
            You have access to a memory system containing Alex's notes, thoughts, and records from the past 30 days. \
            When asked a question, ALWAYS use the memory tools to search for relevant information first. \
            Use search_memory with mode 'hybrid' for best results. You can also use get_context for broader retrieval. \
            Be specific — cite facts, names, numbers, and dates from the memories. \
            Keep answers concise but thorough.";

        let mut total_tool_calls = 0;
        let mut validations_passed = 0;
        let mut validations_total = 0;

        for (i, (question, expected_keywords)) in questions.iter().enumerate() {
            println!("\nQ{}: {question}\n", i + 1);

            let mut messages = vec![
                LlmMessage::system(system_prompt),
                LlmMessage::user(*question),
            ];

            let answer = run_with_tools(&claude, &mut messages, &tools, &executor, &embedder, 5)
                .expect("run_with_tools failed");

            // Count tool calls in this conversation
            let round_tool_calls: usize = messages
                .iter()
                .map(|m| m.tool_calls.len())
                .sum();
            total_tool_calls += round_tool_calls;

            // Print tool usage
            for msg in &messages {
                if !msg.tool_calls.is_empty() {
                    for tc in &msg.tool_calls {
                        let args_preview = tc.arguments.to_string();
                        let args_short = if args_preview.len() > 80 {
                            format!("{}...", &args_preview[..80])
                        } else {
                            args_preview
                        };
                        println!("  🔧 {}({})", tc.name, args_short);
                    }
                }
            }

            // Print answer (truncated if very long)
            let answer_display = if answer.len() > 500 {
                format!("{}...", &answer[..500])
            } else {
                answer.clone()
            };
            println!("\n  🤖 {answer_display}\n");

            // Validate keywords
            let answer_lower = answer.to_lowercase();
            for kw in expected_keywords {
                validations_total += 1;
                if answer_lower.contains(&kw.to_lowercase()) {
                    validations_passed += 1;
                    println!("  ✅ Contains '{kw}'");
                } else {
                    println!("  ❌ Missing '{kw}'");
                }
            }

            println!("{}", "─".repeat(60));
        }

        // ── Phase 6: Summary report ──

        println!("\n{}", "═".repeat(60));
        println!("  SIMULATION REPORT");
        println!("{}", "═".repeat(60));
        println!("  Memories stored:      {}", founder_thoughts.len());
        println!("  Relationships:        {}", relationships.len());
        println!("  Reminders:            {}", reminders.len());
        println!("  Questions asked:      {}", questions.len());
        println!("  Total tool calls:     {total_tool_calls}");
        println!("  Keyword validations:  {validations_passed}/{validations_total} passed");
        println!("{}", "═".repeat(60));

        // At least 60% of keyword validations should pass
        let pass_rate = validations_passed as f64 / validations_total as f64;
        assert!(
            pass_rate >= 0.6,
            "Keyword validation pass rate too low: {:.0}% ({validations_passed}/{validations_total})",
            pass_rate * 100.0
        );

        println!(
            "\n  ✓ Simulation passed! ({:.0}% keyword accuracy)\n",
            pass_rate * 100.0
        );
    }
}

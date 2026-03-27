//! FFI bridge API for flutter_rust_bridge.
//!
//! All complex types cross the FFI boundary as JSON strings.
//! Vec<f32> is passed as native float arrays (FRB handles these efficiently).

use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::consolidation::MemoryConsolidator;
use crate::error::Result;
use crate::longterm;
use crate::models::*;
use crate::shortterm::{ConversationBuffer, SensoryBuffer, WorkingMemory};
use crate::storage::MemoryStore;
use crate::tools::{EmbeddingProvider, LlmProvider, ToolCall, ToolExecutor};

/// The main engine — held as an opaque pointer by the platform layer.
pub struct MemlocalEngine {
    store: Arc<MemoryStore>,
    sensory_buffer: Mutex<SensoryBuffer>,
    conversation_buffers: Mutex<std::collections::HashMap<String, ConversationBuffer>>,
    working_memory: Mutex<WorkingMemory>,
    tool_executor: ToolExecutor,
    consolidator: MemoryConsolidator,
    config: CoreConfig,
    // Long-term subtypes
    pub episodic: longterm::EpisodicMemory,
    pub semantic: longterm::SemanticMemory,
    pub factual: longterm::FactualMemory,
    pub procedural: longterm::ProceduralMemory,
    pub social: longterm::SocialMemory,
    pub spatial: longterm::SpatialMemory,
    pub prospective: longterm::ProspectiveMemory,
    pub affective: longterm::AffectiveMemory,
}

impl MemlocalEngine {
    /// Create and initialize the engine.
    pub fn open(config: CoreConfig) -> Result<Self> {
        let store = Arc::new(MemoryStore::open(
            &config.storage,
            config.storage.embedding_dimensions,
        )?);

        let sensory_buffer = SensoryBuffer::new(
            config.sensory_buffer_capacity,
            Duration::from_millis(config.sensory_ttl_ms),
        );

        Ok(Self {
            sensory_buffer: Mutex::new(sensory_buffer),
            conversation_buffers: Mutex::new(std::collections::HashMap::new()),
            working_memory: Mutex::new(WorkingMemory::new()),
            tool_executor: ToolExecutor::new(Arc::clone(&store)),
            consolidator: MemoryConsolidator::new(Arc::clone(&store)),
            episodic: longterm::EpisodicMemory::new(Arc::clone(&store)),
            semantic: longterm::SemanticMemory::new(Arc::clone(&store)),
            factual: longterm::FactualMemory::new(Arc::clone(&store)),
            procedural: longterm::ProceduralMemory::new(Arc::clone(&store)),
            social: longterm::SocialMemory::new(Arc::clone(&store)),
            spatial: longterm::SpatialMemory::new(Arc::clone(&store)),
            prospective: longterm::ProspectiveMemory::new(Arc::clone(&store)),
            affective: longterm::AffectiveMemory::new(Arc::clone(&store)),
            config,
            store,
        })
    }

    pub fn close(&self) -> Result<()> {
        self.store.close()
    }

    // --- Memory CRUD ---

    pub fn put_memory(&self, item: &MemoryItem, embedding: &[f32]) -> Result<()> {
        self.store.put_memory(item, embedding)
    }

    pub fn get_memory(&self, id: &str) -> Result<Option<MemoryItem>> {
        self.store.get_memory(id)
    }

    pub fn get_memories(
        &self,
        user_id: Option<&str>,
        memory_type: Option<MemoryType>,
        limit: usize,
    ) -> Result<Vec<MemoryItem>> {
        self.store.get_memories(user_id, memory_type, limit)
    }

    pub fn delete_memory(&self, id: &str) -> Result<()> {
        self.store.delete_memory(id)
    }

    pub fn invalidate_memory(&self, id: &str) -> Result<()> {
        self.store.invalidate_memory(id)
    }

    pub fn memory_count(&self, memory_type: Option<MemoryType>) -> Result<usize> {
        self.store.memory_count(memory_type)
    }

    // --- Search ---

    pub fn search_semantic(
        &self,
        embedding: &[f32],
        k: usize,
        user_id: Option<&str>,
        memory_type: Option<MemoryType>,
    ) -> Result<Vec<MemoryItem>> {
        self.store
            .search_semantic(embedding, k, user_id, memory_type)
    }

    pub fn search_text(&self, query: &str, k: usize) -> Result<Vec<MemoryItem>> {
        self.store.search_text(query, k)
    }

    pub fn search_hybrid(
        &self,
        query: &str,
        embedding: &[f32],
        k: usize,
        user_id: Option<&str>,
        memory_type: Option<MemoryType>,
    ) -> Result<Vec<MemoryItem>> {
        self.store
            .search_hybrid(query, embedding, k, user_id, memory_type)
    }

    pub fn search_graph(
        &self,
        embedding: &[f32],
        k: usize,
        user_id: Option<&str>,
        memory_type: Option<MemoryType>,
    ) -> Result<Vec<MemoryItem>> {
        self.store
            .search_graph(embedding, k, user_id, memory_type, 2)
    }

    // --- Edges ---

    pub fn put_edge(&self, edge: &MemoryEdge) -> Result<()> {
        self.store.put_edge(edge)
    }

    pub fn get_edges(&self, memory_id: &str) -> Result<Vec<MemoryEdge>> {
        let mut edges = self.store.get_edges_from(memory_id)?;
        edges.extend(self.store.get_edges_to(memory_id)?);
        Ok(edges)
    }

    // --- Conversation Buffer ---

    pub fn append_message(&self, message: Message, session_id: &str) -> Result<()> {
        let mut buffers = self.conversation_buffers.lock().unwrap();
        let buffer = buffers.entry(session_id.to_string()).or_insert_with(|| {
            ConversationBuffer::new(
                Arc::clone(&self.store),
                session_id.to_string(),
                self.config.conversation_buffer_size,
            )
        });
        buffer.append(message)
    }

    pub fn get_messages(&self, session_id: &str, limit: Option<usize>) -> Result<Vec<Message>> {
        self.store.get_messages(session_id, limit)
    }

    // --- Sensory Buffer ---

    pub fn sensory_add(&self, message: Message) {
        let mut buf = self.sensory_buffer.lock().unwrap();
        buf.add(message);
    }

    pub fn sensory_items(&self) -> Vec<Message> {
        let mut buf = self.sensory_buffer.lock().unwrap();
        buf.items().into_iter().cloned().collect()
    }

    pub fn sensory_clear(&self) {
        let mut buf = self.sensory_buffer.lock().unwrap();
        buf.clear();
    }

    // --- Profile ---

    pub fn put_profile(&self, profile: &UserProfile) -> Result<()> {
        self.store.put_profile(profile)
    }

    pub fn get_profile(&self, user_id: &str) -> Result<Option<UserProfile>> {
        self.store.get_profile(user_id)
    }

    // --- Prospective ---

    pub fn put_prospective(&self, item: &ProspectiveItem) -> Result<()> {
        self.store.put_prospective(item)
    }

    pub fn get_pending_prospective(&self, user_id: Option<&str>) -> Result<Vec<ProspectiveItem>> {
        self.store.get_pending_prospective(user_id)
    }

    pub fn complete_prospective(&self, id: &str) -> Result<()> {
        self.store.complete_prospective(id)
    }

    // --- Working Memory ---

    pub fn working_memory_set_relevant(&self, items: Vec<MemoryItem>) {
        let mut wm = self.working_memory.lock().unwrap();
        wm.set_relevant(items);
    }

    pub fn working_memory_set_important(&self, items: Vec<MemoryItem>) {
        let mut wm = self.working_memory.lock().unwrap();
        wm.set_important(items);
    }

    pub fn working_memory_set_profile(&self, profile: Option<UserProfile>) {
        let mut wm = self.working_memory.lock().unwrap();
        wm.set_profile(profile);
    }

    pub fn working_memory_set_reminders(&self, reminders: Vec<ProspectiveItem>) {
        let mut wm = self.working_memory.lock().unwrap();
        wm.set_triggered_reminders(reminders);
    }

    pub fn working_memory_context_block(&self) -> String {
        let wm = self.working_memory.lock().unwrap();
        wm.to_context_block()
    }

    pub fn working_memory_clear(&self) {
        let mut wm = self.working_memory.lock().unwrap();
        wm.clear();
    }

    // --- Tools ---

    pub fn get_tool_definitions() -> Vec<crate::tools::ToolDefinition> {
        crate::tools::all_tool_definitions()
    }

    pub fn execute_tool(
        &self,
        tool_call: &ToolCall,
        embedding_provider: &dyn EmbeddingProvider,
    ) -> crate::tools::ToolResult {
        self.tool_executor.execute(tool_call, embedding_provider)
    }

    /// Prepare context with iterative retrieval (agent mode).
    /// Uses an LLM to assess context sufficiency and refine queries.
    pub fn prepare_context_iterative(
        &self,
        query: &str,
        embedding_provider: &dyn EmbeddingProvider,
        llm_provider: &dyn LlmProvider,
        user_id: Option<&str>,
        max_results: Option<usize>,
    ) -> Result<String> {
        self.tool_executor.prepare_context_iterative(
            query, embedding_provider, llm_provider, user_id, max_results
        ).map(|pc| pc.context_block)
    }

    // --- Consolidation ---

    pub fn find_consolidation_clusters(
        &self,
        user_id: Option<&str>,
        embeddings: &[(String, Vec<f32>)],
        min_age_secs: u64,
        min_cluster_size: usize,
    ) -> Result<Vec<Vec<MemoryItem>>> {
        self.consolidator
            .find_clusters(user_id, embeddings, min_cluster_size, min_age_secs)
    }

    // --- Important memories ---

    pub fn get_important_memories(
        &self,
        user_id: Option<&str>,
        limit: usize,
        min_importance: f64,
    ) -> Result<Vec<MemoryItem>> {
        self.store
            .get_important_memories(user_id, limit, min_importance)
    }

    // --- Export/Import ---

    pub fn export_relations(&self) -> Result<serde_json::Value> {
        self.store.export_relations()
    }
}

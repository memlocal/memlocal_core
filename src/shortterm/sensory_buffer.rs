use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::models::Message;

struct TimedItem {
    message: Message,
    added_at: Instant,
}

/// Ultra-short in-memory buffer with capacity limit and TTL eviction.
pub struct SensoryBuffer {
    capacity: usize,
    ttl: Duration,
    buffer: VecDeque<TimedItem>,
}

impl SensoryBuffer {
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            capacity,
            ttl,
            buffer: VecDeque::with_capacity(capacity),
        }
    }

    /// Add a message to the buffer.
    pub fn add(&mut self, message: Message) {
        self.evict_expired();
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(TimedItem {
            message,
            added_at: Instant::now(),
        });
    }

    /// Get all non-expired items.
    pub fn items(&mut self) -> Vec<&Message> {
        self.evict_expired();
        self.buffer.iter().map(|ti| &ti.message).collect()
    }

    /// Get the most recent n items.
    pub fn recent(&mut self, n: usize) -> Vec<&Message> {
        self.evict_expired();
        self.buffer
            .iter()
            .rev()
            .take(n)
            .map(|ti| &ti.message)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    pub fn len(&mut self) -> usize {
        self.evict_expired();
        self.buffer.len()
    }

    pub fn is_empty(&mut self) -> bool {
        self.evict_expired();
        self.buffer.is_empty()
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    fn evict_expired(&mut self) {
        let now = Instant::now();
        while let Some(front) = self.buffer.front() {
            if now.duration_since(front.added_at) > self.ttl {
                self.buffer.pop_front();
            } else {
                break;
            }
        }
    }
}

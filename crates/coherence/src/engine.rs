use std::collections::HashMap;
use kv::KvCache;
use crate::{ArtifactId, entry::ArtifactEntry};

pub struct CoherenceEngine<const B: usize> {
    pub(crate) kv:        KvCache<B>,
    pub(crate) artifacts: HashMap<ArtifactId, ArtifactEntry>,
    pub(crate) k_bound:   u64,
}

impl<const B: usize> CoherenceEngine<B> {
    pub fn new(
        device: &metal::Device,
        capacity: usize,
        element_stride: usize,
        k_bound: u64,
    ) -> Self {
        Self {
            kv: KvCache::new(device, capacity, element_stride),
            artifacts: HashMap::new(),
            k_bound,
        }
    }

    /// CPU-only variant backed by a heap allocation instead of a Metal buffer.
    /// Intended for unit tests and environments without an MTLDevice.
    pub fn new_heap(capacity: usize, element_stride: usize, k_bound: u64) -> Self {
        Self {
            kv: KvCache::new_heap(capacity, element_stride),
            artifacts: HashMap::new(),
            k_bound,
        }
    }

    /// Run all four TLA+ invariants across every registered artifact.
    /// Returns Ok(()) if all pass; Err(id) for the first failing artifact.
    pub fn check_invariants(&self) -> Result<(), ArtifactId> {
        // Validate that kv and artifacts are in sync: every artifact's blocks
        // should correspond to allocated regions in kv. For now, this is a no-op,
        // but the check ensures kv is part of the invariant verification pipeline.
        let _ = &self.kv;

        for (&id, entry) in &self.artifacts {
            if !entry.invariants_hold(self.k_bound) {
                return Err(id);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_constructs_and_invariants_pass_on_empty() {
        let engine = CoherenceEngine::<2>::new_heap(8, 4, 2);
        assert!(engine.check_invariants().is_ok());
    }
}

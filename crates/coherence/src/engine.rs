use std::collections::HashMap;
use kv::KvCache;
use crate::{AgentId, ArtifactId, CoherenceError, entry::ArtifactEntry};

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

    /// Register a new artifact from raw KV data. Inserts into the KV cache and
    /// creates an ArtifactEntry in Exclusive state owned by `agent`.
    ///
    /// Returns Err(AlreadyExists) if an artifact with the same token hash already
    /// exists in the registry.
    ///
    /// Complexity: O(|tokens|) for KV insert + trie walk + xxhash.
    /// Side effects: allocates slab blocks, inserts trie nodes.
    pub fn register(
        &mut self,
        tokens: &[kv::TokenId],
        kv_data: &[&[u8]],
        agent: AgentId,
    ) -> Result<ArtifactId, CoherenceError> {
        let id = ArtifactId::from_tokens(tokens);
        if self.artifacts.contains_key(&id) {
            return Err(CoherenceError::AlreadyExists);
        }
        // Write KV blocks into the slab and trie; errors if slab is exhausted
        // or kv_data element sizes don't match B * element_stride.
        self.kv.insert(tokens, kv_data).map_err(CoherenceError::KvError)?;
        // Retrieve the full block_id vec that was just inserted. The trie is
        // always consistent after insert(), so lookup() is infallible here.
        let blocks = self.kv.lookup(tokens).block_ids;
        let entry = ArtifactEntry::new_exclusive(agent, blocks);
        debug_assert!(entry.invariants_hold(self.k_bound));
        self.artifacts.insert(id, entry);
        Ok(id)
    }

    /// I → E. Re-claim an invalidated artifact for exclusive write access.
    ///
    /// Returns Err(NotFound) if `id` is not registered.
    /// Returns Err(WrongState) if the artifact is not in Invalid state
    /// (callers must invalidate concurrent readers before re-acquiring).
    ///
    /// Side effects: transitions state to Exclusive, sets owner to `agent`.
    pub fn acquire(
        &mut self,
        id: ArtifactId,
        agent: AgentId,
    ) -> Result<(), CoherenceError> {
        let k_bound = self.k_bound;
        let entry = self.artifacts.get_mut(&id).ok_or(CoherenceError::NotFound)?;
        // Only Invalid artifacts may be acquired; E/S/M require explicit
        // invalidation + writeback first to preserve SWMR and SeenBound.
        if entry.state != crate::MesiState::Invalid {
            return Err(CoherenceError::WrongState);
        }
        entry.state = crate::MesiState::Exclusive;
        entry.owner = Some(agent);
        debug_assert!(entry.invariants_hold(k_bound));
        Ok(())
    }

    /// E/S → I. Evict an artifact, releasing its KV blocks back to the slab.
    ///
    /// Returns Err(NotFound) if `id` is not registered.
    /// Returns Err(WrongState) if state is Modified (must writeback first) or
    /// already Invalid (double-invalidate is an error, not a no-op).
    ///
    /// Clears owner, sharers, and blocks. Does NOT clear `seen`, matching the
    /// TLA+ UNCHANGED <<seen>> invariant in the Invalidate action.
    pub fn invalidate(&mut self, id: ArtifactId) -> Result<(), CoherenceError> {
        let k_bound = self.k_bound;
        // Scope the mutable borrow of `entry` so it ends before the call to
        // `self.kv.evict`, which also requires `&mut self`.
        let blocks_len = {
            let entry = self.artifacts.get_mut(&id).ok_or(CoherenceError::NotFound)?;
            // Modified requires writeback before invalidation to prevent data loss.
            // Invalid is already terminal; second call is a caller bug.
            if entry.state == crate::MesiState::Modified
                || entry.state == crate::MesiState::Invalid
            {
                return Err(CoherenceError::WrongState);
            }
            let n = entry.blocks.len();
            entry.state = crate::MesiState::Invalid;
            entry.owner = None;
            entry.sharers.clear();
            entry.blocks.clear();
            n
        };
        // evict() returns the actual freed count; we pass blocks_len as the
        // target — the trie LRU will free up to that many blocks.
        self.kv.evict(blocks_len);
        debug_assert!(self.artifacts[&id].invariants_hold(k_bound));
        Ok(())
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

    fn make_kv_data(n_blocks: usize) -> Vec<Vec<u8>> {
        // B=2, element_stride=4 → 8 bytes per block
        (0..n_blocks).map(|i| vec![i as u8; 8]).collect()
    }

    #[test]
    fn register_creates_exclusive_entry() {
        let mut e = CoherenceEngine::<2>::new_heap(8, 4, 5);
        let tokens: Vec<kv::TokenId> = vec![0, 1, 2, 3];
        let data = make_kv_data(2);
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let id = e.register(&tokens, &slices, 0).expect("register must succeed");
        assert!(e.check_invariants().is_ok());
        assert_eq!(e.artifacts[&id].state, crate::MesiState::Exclusive);
        assert_eq!(e.artifacts[&id].owner, Some(0));
        assert_eq!(e.artifacts[&id].blocks.len(), 2);
    }

    #[test]
    fn register_duplicate_returns_already_exists() {
        let mut e = CoherenceEngine::<2>::new_heap(8, 4, 5);
        let tokens: Vec<kv::TokenId> = vec![0, 1, 2, 3];
        let data = make_kv_data(2);
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        e.register(&tokens, &slices, 0).unwrap();
        let err = e.register(&tokens, &slices, 1);
        assert!(matches!(err, Err(crate::CoherenceError::AlreadyExists)));
    }

    #[test]
    fn acquire_invalid_transitions_to_exclusive() {
        let mut e = CoherenceEngine::<2>::new_heap(8, 4, 5);
        let tokens: Vec<kv::TokenId> = vec![0, 1, 2, 3];
        let data = make_kv_data(2);
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let id = e.register(&tokens, &slices, 0).unwrap();
        // Invalidate first so we can acquire
        e.invalidate(id).unwrap();
        e.acquire(id, 1).expect("acquire on Invalid must succeed");
        assert_eq!(e.artifacts[&id].state, crate::MesiState::Exclusive);
        assert_eq!(e.artifacts[&id].owner, Some(1));
        assert!(e.check_invariants().is_ok());
    }

    #[test]
    fn acquire_on_exclusive_returns_wrong_state() {
        let mut e = CoherenceEngine::<2>::new_heap(8, 4, 5);
        let tokens: Vec<kv::TokenId> = vec![0, 1, 2, 3];
        let data = make_kv_data(2);
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let id = e.register(&tokens, &slices, 0).unwrap();
        // Already Exclusive — acquire must fail
        let err = e.acquire(id, 1);
        assert!(matches!(err, Err(crate::CoherenceError::WrongState)));
    }
}

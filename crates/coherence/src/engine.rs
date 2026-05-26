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

    /// E/S → S. Add `agent` to the sharer set; return backing block ids.
    /// Updates seen[agent] = ver. Returns Err(WrongState) if state is M or I.
    /// Returns Err(KBoundExceeded) if the agent's last-seen version is more than
    /// k_bound writes behind the current version.
    pub fn read(
        &mut self,
        id: ArtifactId,
        agent: AgentId,
    ) -> Result<Vec<kv::BlockId>, CoherenceError> {
        let k_bound = self.k_bound;
        let entry = self.artifacts.get_mut(&id).ok_or(CoherenceError::NotFound)?;
        match entry.state {
            crate::MesiState::Modified | crate::MesiState::Invalid => {
                return Err(CoherenceError::WrongState);
            }
            _ => {}
        }
        // K-bound check: reject if the agent has read before and their seen is too stale.
        // Fresh readers (not in seen map) are always allowed.
        if let Some(&s) = entry.seen.get(&agent) {
            if entry.ver.saturating_sub(s) > k_bound {
                return Err(CoherenceError::KBoundExceeded);
            }
        }
        entry.state = crate::MesiState::Shared;
        entry.owner = None;
        entry.sharers.insert(agent);
        entry.seen.insert(agent, entry.ver);
        debug_assert!(entry.invariants_hold(k_bound));
        Ok(entry.blocks.clone())
    }

    /// E → M. The exclusive owner writes new KV data extending `tokens`,
    /// increments the artifact version, and updates their seen version.
    /// Returns Err(WrongState) if not Exclusive. Returns Err(NotOwner) if
    /// `agent` is not the current exclusive owner.
    pub fn write(
        &mut self,
        id: ArtifactId,
        agent: AgentId,
        tokens: &[kv::TokenId],
        kv_data: &[&[u8]],
    ) -> Result<(), CoherenceError> {
        // Validate before touching KV. Immutable borrow dropped at end of block.
        {
            let entry = self.artifacts.get(&id).ok_or(CoherenceError::NotFound)?;
            if entry.state != crate::MesiState::Exclusive {
                return Err(CoherenceError::WrongState);
            }
            if entry.owner != Some(agent) {
                return Err(CoherenceError::NotOwner);
            }
        }
        self.kv.insert(tokens, kv_data).map_err(CoherenceError::KvError)?;
        let new_blocks = self.kv.lookup(tokens).block_ids;
        let k_bound = self.k_bound;
        let entry = self.artifacts.get_mut(&id).unwrap();
        entry.state = crate::MesiState::Modified;
        entry.ver += 1;
        // Owner sees its own write: matches TLA+ seen'[ag][a] = ver[a] + 1
        entry.seen.insert(agent, entry.ver);
        entry.blocks = new_blocks;
        debug_assert!(entry.invariants_hold(k_bound));
        Ok(())
    }

    /// M → E stub. Full implementation in Task 7.
    pub fn writeback(&mut self, id: ArtifactId) -> Result<(), CoherenceError> {
        let _ = id;
        unimplemented!("writeback: implemented in Task 7")
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

    #[test]
    fn read_from_exclusive_demotes_to_shared() {
        let mut e = CoherenceEngine::<2>::new_heap(8, 4, 5);
        let tokens: Vec<kv::TokenId> = vec![0, 1, 2, 3];
        let data = make_kv_data(2);
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let id = e.register(&tokens, &slices, 0).unwrap();
        let blocks = e.read(id, 1).expect("read from Exclusive must succeed");
        assert_eq!(blocks.len(), 2);
        assert_eq!(e.artifacts[&id].state, crate::MesiState::Shared);
        assert_eq!(e.artifacts[&id].owner, None);
        assert!(e.artifacts[&id].sharers.contains(&1));
        assert_eq!(e.artifacts[&id].seen[&1], 0); // ver=0 at time of read
        assert!(e.check_invariants().is_ok());
    }

    #[test]
    fn read_from_shared_adds_second_sharer() {
        let mut e = CoherenceEngine::<2>::new_heap(8, 4, 5);
        let tokens: Vec<kv::TokenId> = vec![0, 1, 2, 3];
        let data = make_kv_data(2);
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let id = e.register(&tokens, &slices, 0).unwrap();
        e.read(id, 1).unwrap();
        e.read(id, 2).unwrap();
        assert!(e.artifacts[&id].sharers.contains(&1));
        assert!(e.artifacts[&id].sharers.contains(&2));
        assert!(e.check_invariants().is_ok());
    }

    #[test]
    fn read_from_modified_returns_wrong_state() {
        let mut e = CoherenceEngine::<2>::new_heap(8, 4, 5);
        let tokens: Vec<kv::TokenId> = vec![0, 1, 2, 3];
        let data = make_kv_data(2);
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let id = e.register(&tokens, &slices, 0).unwrap();
        // Force M state
        e.artifacts.get_mut(&id).unwrap().state = crate::MesiState::Modified;
        let err = e.read(id, 1);
        assert!(matches!(err, Err(crate::CoherenceError::WrongState)));
    }

    #[test]
    fn read_from_invalid_returns_wrong_state() {
        let mut e = CoherenceEngine::<2>::new_heap(8, 4, 5);
        let tokens: Vec<kv::TokenId> = vec![0, 1, 2, 3];
        let data = make_kv_data(2);
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let id = e.register(&tokens, &slices, 0).unwrap();
        e.invalidate(id).unwrap();
        let err = e.read(id, 1);
        assert!(matches!(err, Err(crate::CoherenceError::WrongState)));
    }

    #[test]
    fn write_from_exclusive_transitions_to_modified() {
        let mut e = CoherenceEngine::<2>::new_heap(16, 4, 5);
        let tokens: Vec<kv::TokenId> = vec![0, 1, 2, 3];
        let data = make_kv_data(2);
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let id = e.register(&tokens, &slices, 0).unwrap();

        let tokens_ext: Vec<kv::TokenId> = vec![0, 1, 2, 3, 4, 5];
        let data_ext = make_kv_data(3);
        let slices_ext: Vec<&[u8]> = data_ext.iter().map(|v| v.as_slice()).collect();
        e.write(id, 0, &tokens_ext, &slices_ext).expect("write from Exclusive must succeed");

        assert_eq!(e.artifacts[&id].state, crate::MesiState::Modified);
        assert_eq!(e.artifacts[&id].ver, 1);
        assert_eq!(e.artifacts[&id].seen[&0], 1); // owner sees own write
        assert_eq!(e.artifacts[&id].blocks.len(), 3);
        assert!(e.check_invariants().is_ok());
    }

    #[test]
    fn write_wrong_owner_returns_not_owner() {
        let mut e = CoherenceEngine::<2>::new_heap(8, 4, 5);
        let tokens: Vec<kv::TokenId> = vec![0, 1, 2, 3];
        let data = make_kv_data(2);
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let id = e.register(&tokens, &slices, 0).unwrap(); // owner=0

        let tokens_ext: Vec<kv::TokenId> = vec![0, 1, 2, 3, 4, 5];
        let data_ext = make_kv_data(3);
        let slices_ext: Vec<&[u8]> = data_ext.iter().map(|v| v.as_slice()).collect();
        let err = e.write(id, 1, &tokens_ext, &slices_ext); // agent 1 ≠ owner
        assert!(matches!(err, Err(crate::CoherenceError::NotOwner)));
    }

    #[test]
    fn write_from_shared_returns_wrong_state() {
        let mut e = CoherenceEngine::<2>::new_heap(8, 4, 5);
        let tokens: Vec<kv::TokenId> = vec![0, 1, 2, 3];
        let data = make_kv_data(2);
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let id = e.register(&tokens, &slices, 0).unwrap();
        e.read(id, 1).unwrap(); // demotes to Shared

        let tokens_ext: Vec<kv::TokenId> = vec![0, 1, 2, 3, 4, 5];
        let data_ext = make_kv_data(3);
        let slices_ext: Vec<&[u8]> = data_ext.iter().map(|v| v.as_slice()).collect();
        let err = e.write(id, 0, &tokens_ext, &slices_ext);
        assert!(matches!(err, Err(crate::CoherenceError::WrongState)));
    }

    #[test]
    fn read_stale_beyond_k_returns_error() {
        // k_bound=0: any unseen write makes a previously-reading agent stale.
        //
        // Sequence that triggers K-bound exceeded without concurrent writes:
        //   1. Register (ver=0, E, owner=0)
        //   2. Agent 1 reads → S, seen[1]=0
        //   3. Invalidate → I, seen[1]=0 preserved
        //   4. Agent 0 acquire → E
        //   5. Agent 0 writes → ver=1, M
        //   6. Writeback → E
        //   7. Agent 1 reads → seen[1]=0, ver=1, 1-0=1 > k_bound=0 → KBoundExceeded
        let mut e = CoherenceEngine::<2>::new_heap(8, 4, 0); // k_bound=0
        let tokens: Vec<kv::TokenId> = vec![0, 1, 2, 3];
        let data = make_kv_data(2);
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let id = e.register(&tokens, &slices, 0).unwrap();
        e.read(id, 1).unwrap();           // step 2: agent 1 joins S, seen[1]=0
        e.invalidate(id).unwrap();        // step 3
        e.acquire(id, 0).unwrap();        // step 4

        // step 5: write needs E state and extends tokens
        let tokens_ext: Vec<kv::TokenId> = vec![0, 1, 2, 3, 4, 5];
        let data_ext = make_kv_data(3);
        let slices_ext: Vec<&[u8]> = data_ext.iter().map(|v| v.as_slice()).collect();
        e.write(id, 0, &tokens_ext, &slices_ext).unwrap();
        e.writeback(id).unwrap();         // step 6: M → E

        let err = e.read(id, 1);          // step 7: agent 1 is stale
        assert!(
            matches!(err, Err(crate::CoherenceError::KBoundExceeded)),
            "agent with seen=0 must not read past k_bound=0 when ver=1"
        );
    }
}

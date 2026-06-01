use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use kv::KvCache;
use crate::{AgentId, ArtifactId, CoherenceError, entry::ArtifactEntry, routing::RoutingIndex};

pub struct CoherenceEngine<const B: usize> {
    pub(crate) routing:   Mutex<RoutingIndex<B>>,
    pub(crate) artifacts: HashMap<ArtifactId, Arc<Mutex<ArtifactEntry>>>,
    pub(crate) kv:        Mutex<KvCache<B>>,
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
            routing:   Mutex::new(RoutingIndex::new()),
            artifacts: HashMap::new(),
            kv:        Mutex::new(KvCache::new(device, capacity, element_stride)),
            k_bound,
        }
    }

    /// CPU-only variant backed by a heap allocation instead of a Metal buffer.
    /// Intended for unit tests and environments without an MTLDevice.
    pub fn new_heap(capacity: usize, element_stride: usize, k_bound: u64) -> Self {
        Self {
            routing:   Mutex::new(RoutingIndex::new()),
            artifacts: HashMap::new(),
            kv:        Mutex::new(KvCache::new_heap(capacity, element_stride)),
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
        let blocks = {
            let kv = self.kv.get_mut().unwrap();
            kv.insert(tokens, kv_data).map_err(CoherenceError::KvError)?;
            kv.lookup(tokens).block_ids
        };
        let entry = ArtifactEntry::new_exclusive(agent, blocks, tokens.to_vec());
        debug_assert!(entry.invariants_hold(self.k_bound));
        self.routing.get_mut().unwrap().insert(tokens, id);
        self.artifacts.insert(id, Arc::new(Mutex::new(entry)));
        Ok(id)
    }

    /// I → E. Re-claim an invalidated artifact for exclusive write access.
    ///
    /// Returns Err(NotFound) if `id` is not registered.
    /// Returns Err(WrongState) if the artifact is not in Invalid state
    /// (callers must invalidate concurrent readers before re-acquiring).
    ///
    /// Side effects: transitions state to Exclusive, sets owner to `agent`.
    pub fn acquire(&self, id: ArtifactId, agent: AgentId) -> Result<(), CoherenceError> {
        let k_bound = self.k_bound;
        let entry_arc = self.artifacts.get(&id).ok_or(CoherenceError::NotFound)?;
        let mut entry = entry_arc.lock().unwrap();
        // Only Invalid artifacts may be acquired; E/S/M require explicit
        // invalidation + writeback first to preserve SWMR and SeenBound.
        if entry.state != crate::MesiState::Invalid {
            return Err(CoherenceError::WrongState);
        }
        entry.state = crate::MesiState::Exclusive;
        entry.owner = Some(agent);
        // blocks was cleared by invalidate(); tokens must match (blocks.len() * B == 0).
        // Clear the stale token sequence so the tokens/blocks size invariant holds.
        entry.tokens.clear();
        debug_assert!(entry.invariants_hold(k_bound));
        Ok(())
    }

    /// E/S → I. Invalidate an artifact, releasing its KV blocks back to the slab.
    ///
    /// Returns Err(NotFound) if `id` is not registered.
    /// Returns Err(WrongState) if state is Modified (must writeback first) or
    /// already Invalid (double-invalidate is an error, not a no-op).
    ///
    /// Clears owner, sharers, and blocks. Does NOT clear `seen`, matching the
    /// TLA+ UNCHANGED <<seen>> invariant in the Invalidate action. The `tokens`
    /// field is also left stale after invalidation — it retains the token
    /// sequence from the last valid state. Callers should not read `tokens`
    /// from an Invalid entry. `acquire()` clears `tokens` as part of the
    /// Invalid → Exclusive transition, restoring the `tokens.len() == blocks.len() * B`
    /// invariant before any new write.
    ///
    /// Uses `kv.release()` for targeted per-artifact block release rather than
    /// the global LRU sweep used by the deprecated `kv.evict()` path.
    pub fn invalidate(&self, id: ArtifactId) -> Result<(), CoherenceError> {
        let k_bound = self.k_bound;
        let entry_arc = self.artifacts.get(&id).ok_or(CoherenceError::NotFound)?;
        // Step 1: lock artifact, transition to Invalid, extract blocks + tokens.
        let (tokens, blocks) = {
            let mut entry = entry_arc.lock().unwrap();
            // Modified requires writeback before invalidation to prevent data loss.
            // Invalid is already terminal; second call is a caller bug.
            if entry.state == crate::MesiState::Modified
                || entry.state == crate::MesiState::Invalid
            {
                return Err(CoherenceError::WrongState);
            }
            let tokens = entry.tokens.clone();
            let blocks = std::mem::take(&mut entry.blocks);
            entry.state  = crate::MesiState::Invalid;
            entry.owner  = None;
            entry.sharers.clear();
            (tokens, blocks)
        }; // artifact lock released
        // Step 2: free slab blocks (targeted per-artifact release — not a global LRU sweep).
        self.kv.lock().unwrap().release(&blocks);
        // Step 3: remove from routing (artifact no longer routable).
        self.routing.lock().unwrap().remove(&tokens);
        debug_assert!(entry_arc.lock().unwrap().invariants_hold(k_bound));
        Ok(())
    }

    /// E/S → S. Add `agent` to the sharer set; return backing block ids.
    /// Updates seen[agent] = ver. Returns Err(WrongState) if state is M or I.
    /// Returns Err(KBoundExceeded) if the agent's last-seen version is more than
    /// k_bound writes behind the current version.
    pub fn read(&self, id: ArtifactId, agent: AgentId) -> Result<Vec<kv::BlockId>, CoherenceError> {
        let k_bound = self.k_bound;
        let entry_arc = self.artifacts.get(&id).ok_or(CoherenceError::NotFound)?;
        let mut entry = entry_arc.lock().unwrap();
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
        let ver = entry.ver;
        entry.state = crate::MesiState::Shared;
        entry.owner = None;
        entry.sharers.insert(agent);
        entry.seen.insert(agent, ver);
        debug_assert!(entry.invariants_hold(k_bound));
        Ok(entry.blocks.clone())
    }

    /// E → M. The exclusive owner writes new KV data extending `tokens`,
    /// increments the artifact version, and updates their seen version.
    /// Returns Err(WrongState) if not Exclusive. Returns Err(NotOwner) if
    /// `agent` is not the current exclusive owner.
    pub fn write(
        &self,
        id: ArtifactId,
        agent: AgentId,
        tokens: &[kv::TokenId],
        kv_data: &[&[u8]],
    ) -> Result<(), CoherenceError> {
        let k_bound = self.k_bound;
        let entry_arc = self.artifacts.get(&id).ok_or(CoherenceError::NotFound)?;
        let mut entry = entry_arc.lock().unwrap();
        if entry.state != crate::MesiState::Exclusive {
            return Err(CoherenceError::WrongState);
        }
        if entry.owner != Some(agent) {
            return Err(CoherenceError::NotOwner);
        }
        let old_tokens = entry.tokens.clone();
        // Lock ordering: artifact (already held) → routing → kv.
        self.routing.lock().unwrap().remove(&old_tokens);
        let new_blocks = {
            let mut kv = self.kv.lock().unwrap();
            kv.insert(tokens, kv_data).map_err(CoherenceError::KvError)?;
            kv.lookup(tokens).block_ids
        };
        entry.state = crate::MesiState::Modified;
        entry.ver  += 1;
        // Owner sees its own write: matches TLA+ seen'[ag][a] = ver[a] + 1
        let new_ver = entry.ver;
        entry.seen.insert(agent, new_ver);
        entry.blocks = new_blocks;
        entry.tokens = tokens.to_vec();
        debug_assert!(entry.invariants_hold(k_bound));
        Ok(())
    }

    /// M → E. Stabilise a modified artifact without evicting it.
    /// Retains the current owner in Exclusive state for continued use.
    /// Returns Err(WrongState) if state is not Modified.
    pub fn writeback(&self, id: ArtifactId) -> Result<(), CoherenceError> {
        let k_bound = self.k_bound;
        let entry_arc = self.artifacts.get(&id).ok_or(CoherenceError::NotFound)?;
        let mut entry = entry_arc.lock().unwrap();
        if entry.state != crate::MesiState::Modified {
            return Err(CoherenceError::WrongState);
        }
        entry.state = crate::MesiState::Exclusive;
        let tokens = entry.tokens.clone();
        // Insert into routing while holding artifact lock (artifact → routing ordering).
        self.routing.lock().unwrap().insert(&tokens, id);
        debug_assert!(entry.invariants_hold(k_bound));
        Ok(())
    }

    /// Create a new artifact by CoW-forking an existing one.
    /// `source` must be in Exclusive or Shared state.
    /// The new artifact starts in Exclusive state owned by `agent`.
    /// Calls `kv.fork(tokens)` for zero-copy prefix sharing.
    ///
    /// Returns Err(NotFound) if source is not registered.
    /// Returns Err(WrongState) if source is M or I.
    /// Returns Err(AlreadyExists) if tokens hash to an already-registered ArtifactId.
    pub fn register_fork(
        &mut self,
        tokens: &[kv::TokenId],
        source: ArtifactId,
        agent: AgentId,
    ) -> Result<ArtifactId, CoherenceError> {
        // Validate source state (lock dropped before kv access).
        {
            let src = self.artifacts.get(&source).ok_or(CoherenceError::NotFound)?;
            match src.lock().unwrap().state {
                crate::MesiState::Exclusive | crate::MesiState::Shared => {}
                _ => return Err(CoherenceError::WrongState),
            }
        }
        let new_id = ArtifactId::from_tokens(tokens);
        if self.artifacts.contains_key(&new_id) {
            return Err(CoherenceError::AlreadyExists);
        }
        // kv.fork does a longest-prefix match on tokens — zero memcpy for shared prefix.
        let blocks = self.kv.get_mut().unwrap().fork(tokens);
        // tokens.to_vec() is the caller's intended sequence; blocks.len() * B is
        // what's actually cached. Store only the cached portion in entry.tokens.
        debug_assert!(
            blocks.len() * B <= tokens.len(),
            "kv.fork() returned more blocks ({}) than tokens ({}) / B ({}) allows",
            blocks.len(), tokens.len(), B
        );
        let n = blocks.len() * B;
        let entry = ArtifactEntry::new_exclusive(agent, blocks, tokens[..n].to_vec());
        debug_assert!(entry.invariants_hold(self.k_bound));
        self.routing.get_mut().unwrap().insert(&tokens[..n], new_id);
        self.artifacts.insert(new_id, Arc::new(Mutex::new(entry)));
        Ok(new_id)
    }

    /// Find the artifact in E or S state with the longest token prefix match
    /// against `tokens`. Returns None if no readable artifact covers any prefix.
    /// M and I artifacts are skipped.
    ///
    /// Delegates to `RoutingIndex::longest_prefix`. O(|tokens| / B).
    #[must_use]
    pub fn lookup(&self, tokens: &[kv::TokenId]) -> Option<crate::RouteResult> {
        // Step 1: O(k) routing lookup — routing Mutex acquired then released.
        let (artifact_id, matched_blocks) = {
            let routing = self.routing.lock().unwrap();
            routing.longest_prefix(tokens)
        }?;
        // routing Mutex released before acquiring the per-artifact Mutex (lock-ordering: artifact first).
        // Step 2: read blocks from the per-artifact entry.
        let entry_arc = self.artifacts.get(&artifact_id)?;
        let entry = entry_arc.lock().unwrap();
        if !matches!(entry.state, crate::MesiState::Exclusive | crate::MesiState::Shared) {
            return None;
        }
        let blocks = entry.blocks[..matched_blocks].to_vec();
        Some(crate::RouteResult {
            artifact_id,
            matched_tokens: matched_blocks * B,
            blocks,
        })
    }

    /// Returns `(used_blocks, total_blocks)` for Prometheus metrics.
    pub fn stats(&self) -> (usize, usize) {
        let kv    = self.kv.lock().unwrap();
        let total = kv.capacity();
        let used  = total - kv.free_count();
        (used, total)
    }

    /// Run all four TLA+ invariants across every registered artifact.
    /// Returns Ok(()) if all pass; Err(id) for the first failing artifact.
    pub fn check_invariants(&self) -> Result<(), ArtifactId> {
        for (&id, entry_arc) in &self.artifacts {
            let entry = entry_arc.lock().unwrap();
            if !entry.invariants_hold(self.k_bound) {
                return Err(id);
            }
            // tokens.len() must equal blocks.len() * B for live entries.
            // Invalid entries have blocks.clear() called but tokens is left
            // stale intentionally (mirrors the UNCHANGED <<seen>> rationale),
            // so skip this check for Invalid state.
            if entry.state != crate::MesiState::Invalid
                && entry.tokens.len() != entry.blocks.len() * B
            {
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
        assert_eq!(e.artifacts[&id].lock().unwrap().state, crate::MesiState::Exclusive);
        assert_eq!(e.artifacts[&id].lock().unwrap().owner, Some(0));
        assert_eq!(e.artifacts[&id].lock().unwrap().blocks.len(), 2);
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
        assert_eq!(e.artifacts[&id].lock().unwrap().state, crate::MesiState::Exclusive);
        assert_eq!(e.artifacts[&id].lock().unwrap().owner, Some(1));
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
        assert_eq!(e.artifacts[&id].lock().unwrap().state, crate::MesiState::Shared);
        assert_eq!(e.artifacts[&id].lock().unwrap().owner, None);
        assert!(e.artifacts[&id].lock().unwrap().sharers.contains(&1));
        assert_eq!(e.artifacts[&id].lock().unwrap().seen[&1], 0); // ver=0 at time of read
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
        assert!(e.artifacts[&id].lock().unwrap().sharers.contains(&1));
        assert!(e.artifacts[&id].lock().unwrap().sharers.contains(&2));
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
        e.artifacts[&id].lock().unwrap().state = crate::MesiState::Modified;
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

        assert_eq!(e.artifacts[&id].lock().unwrap().state, crate::MesiState::Modified);
        assert_eq!(e.artifacts[&id].lock().unwrap().ver, 1);
        assert_eq!(e.artifacts[&id].lock().unwrap().seen[&0], 1); // owner sees own write
        assert_eq!(e.artifacts[&id].lock().unwrap().blocks.len(), 3);
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

    #[test]
    fn writeback_modified_to_exclusive() {
        let mut e = CoherenceEngine::<2>::new_heap(16, 4, 5);
        let tokens: Vec<kv::TokenId> = vec![0, 1, 2, 3];
        let data = make_kv_data(2);
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let id = e.register(&tokens, &slices, 0).unwrap();

        let tokens_ext: Vec<kv::TokenId> = vec![0, 1, 2, 3, 4, 5];
        let data_ext = make_kv_data(3);
        let slices_ext: Vec<&[u8]> = data_ext.iter().map(|v| v.as_slice()).collect();
        e.write(id, 0, &tokens_ext, &slices_ext).unwrap();
        assert_eq!(e.artifacts[&id].lock().unwrap().state, crate::MesiState::Modified);

        e.writeback(id).expect("writeback must succeed from Modified");
        assert_eq!(e.artifacts[&id].lock().unwrap().state, crate::MesiState::Exclusive);
        assert_eq!(e.artifacts[&id].lock().unwrap().owner, Some(0)); // owner retained
        assert!(e.check_invariants().is_ok());
    }

    #[test]
    fn writeback_from_exclusive_returns_wrong_state() {
        let mut e = CoherenceEngine::<2>::new_heap(8, 4, 5);
        let tokens: Vec<kv::TokenId> = vec![0, 1, 2, 3];
        let data = make_kv_data(2);
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let id = e.register(&tokens, &slices, 0).unwrap();
        let err = e.writeback(id); // already E, not M
        assert!(matches!(err, Err(crate::CoherenceError::WrongState)));
    }

    #[test]
    fn invalidate_exclusive_transitions_to_invalid() {
        let mut e = CoherenceEngine::<2>::new_heap(8, 4, 5);
        let tokens: Vec<kv::TokenId> = vec![0, 1, 2, 3];
        let data = make_kv_data(2);
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let id = e.register(&tokens, &slices, 0).unwrap();
        e.invalidate(id).expect("invalidate from E must succeed");
        assert_eq!(e.artifacts[&id].lock().unwrap().state, crate::MesiState::Invalid);
        assert_eq!(e.artifacts[&id].lock().unwrap().owner, None);
        assert!(e.artifacts[&id].lock().unwrap().blocks.is_empty());
        assert!(e.check_invariants().is_ok());
    }

    #[test]
    fn invalidate_shared_clears_all_sharers() {
        let mut e = CoherenceEngine::<2>::new_heap(8, 4, 5);
        let tokens: Vec<kv::TokenId> = vec![0, 1, 2, 3];
        let data = make_kv_data(2);
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let id = e.register(&tokens, &slices, 0).unwrap();
        e.read(id, 1).unwrap();
        e.read(id, 2).unwrap();
        assert_eq!(e.artifacts[&id].lock().unwrap().sharers.len(), 2);
        e.invalidate(id).expect("invalidate from S must succeed");
        assert!(e.artifacts[&id].lock().unwrap().sharers.is_empty());
        assert!(e.check_invariants().is_ok());
    }

    #[test]
    fn invalidate_modified_returns_wrong_state() {
        let mut e = CoherenceEngine::<2>::new_heap(16, 4, 5);
        let tokens: Vec<kv::TokenId> = vec![0, 1, 2, 3];
        let data = make_kv_data(2);
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let id = e.register(&tokens, &slices, 0).unwrap();
        let tokens_ext: Vec<kv::TokenId> = vec![0, 1, 2, 3, 4, 5];
        let data_ext = make_kv_data(3);
        let slices_ext: Vec<&[u8]> = data_ext.iter().map(|v| v.as_slice()).collect();
        e.write(id, 0, &tokens_ext, &slices_ext).unwrap();
        let err = e.invalidate(id); // must writeback first
        assert!(matches!(err, Err(crate::CoherenceError::WrongState)));
    }

    #[test]
    fn seen_preserved_across_invalidate() {
        // TLA+ UNCHANGED <<seen>> in Invalidate: seen values survive invalidation.
        let mut e = CoherenceEngine::<2>::new_heap(8, 4, 0); // k_bound=0
        let tokens: Vec<kv::TokenId> = vec![0, 1, 2, 3];
        let data = make_kv_data(2);
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let id = e.register(&tokens, &slices, 0).unwrap();
        e.read(id, 1).unwrap();           // seen[1] = 0
        e.invalidate(id).unwrap();
        assert_eq!(e.artifacts[&id].lock().unwrap().seen.get(&1), Some(&0),
            "seen must not be cleared by invalidate");
    }

    #[test]
    fn register_fork_creates_exclusive_artifact_with_shared_prefix() {
        let mut e = CoherenceEngine::<2>::new_heap(16, 4, 5);
        let tokens: Vec<kv::TokenId> = vec![0, 1, 2, 3];
        let data = make_kv_data(2);
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let base_id = e.register(&tokens, &slices, 0).unwrap();

        // Fork: agent 1 starts a divergent sequence from the shared prefix
        let fork_tokens: Vec<kv::TokenId> = vec![0, 1, 2, 3, 4, 5];
        let fork_id = e.register_fork(&fork_tokens, base_id, 1)
            .expect("register_fork must succeed");

        assert_ne!(fork_id, base_id, "fork must get a new ArtifactId");
        assert_eq!(e.artifacts[&fork_id].lock().unwrap().state, crate::MesiState::Exclusive);
        assert_eq!(e.artifacts[&fork_id].lock().unwrap().owner, Some(1));
        // Fork prefix blocks are shared with base (CoW)
        let fork_blocks = e.artifacts[&fork_id].lock().unwrap().blocks.clone();
        let base_blocks = e.artifacts[&base_id].lock().unwrap().blocks.clone();
        assert_eq!(
            fork_blocks[..2],
            base_blocks[..2],
            "first 2 blocks must be shared with the base artifact"
        );
        assert!(e.check_invariants().is_ok());
    }

    #[test]
    fn register_fork_from_invalid_returns_wrong_state() {
        let mut e = CoherenceEngine::<2>::new_heap(8, 4, 5);
        let tokens: Vec<kv::TokenId> = vec![0, 1, 2, 3];
        let data = make_kv_data(2);
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let base_id = e.register(&tokens, &slices, 0).unwrap();
        e.invalidate(base_id).unwrap();

        let fork_tokens: Vec<kv::TokenId> = vec![0, 1, 2, 3, 4, 5];
        let err = e.register_fork(&fork_tokens, base_id, 1);
        assert!(matches!(err, Err(crate::CoherenceError::WrongState)));
    }

    #[test]
    fn register_fork_from_modified_returns_wrong_state() {
        let mut e = CoherenceEngine::<2>::new_heap(16, 4, 5);
        let tokens: Vec<kv::TokenId> = vec![0, 1, 2, 3];
        let data = make_kv_data(2);
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let base_id = e.register(&tokens, &slices, 0).unwrap();
        let tokens_ext: Vec<kv::TokenId> = vec![0, 1, 2, 3, 4, 5];
        let data_ext = make_kv_data(3);
        let slices_ext: Vec<&[u8]> = data_ext.iter().map(|v| v.as_slice()).collect();
        e.write(base_id, 0, &tokens_ext, &slices_ext).unwrap();
        // base is now Modified

        let fork_tokens: Vec<kv::TokenId> = vec![0, 1, 2, 3, 6, 7];
        let err = e.register_fork(&fork_tokens, base_id, 1);
        assert!(matches!(err, Err(crate::CoherenceError::WrongState)));
    }

    #[test]
    fn register_fork_duplicate_token_hash_returns_already_exists() {
        let mut e = CoherenceEngine::<2>::new_heap(16, 4, 5);
        let tokens: Vec<kv::TokenId> = vec![0, 1, 2, 3];
        let data = make_kv_data(2);
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let base_id = e.register(&tokens, &slices, 0).unwrap();

        let fork_tokens: Vec<kv::TokenId> = vec![0, 1, 2, 3, 4, 5];
        e.register_fork(&fork_tokens, base_id, 1).unwrap();
        // Same tokens → same ArtifactId → AlreadyExists
        let err = e.register_fork(&fork_tokens, base_id, 2);
        assert!(matches!(err, Err(crate::CoherenceError::AlreadyExists)));
    }

    #[test]
    fn fork_artifact_stays_readable_after_source_invalidated() {
        // Regression: invalidating the source artifact must not corrupt the fork's slab
        // references. The fork holds its own incref on the shared prefix blocks.
        let mut e = CoherenceEngine::<2>::new_heap(16, 4, 5);
        let base_tokens: Vec<kv::TokenId> = vec![0, 1, 2, 3];
        let d = make_kv_data(2);
        let s: Vec<&[u8]> = d.iter().map(|v| v.as_slice()).collect();
        let base_id = e.register(&base_tokens, &s, 0).unwrap();

        let fork_tokens: Vec<kv::TokenId> = vec![0, 1, 2, 3, 4, 5];
        let fork_id = e.register_fork(&fork_tokens, base_id, 1).unwrap();
        let data_fork = make_kv_data(3);
        let slices_fork: Vec<&[u8]> = data_fork.iter().map(|v| v.as_slice()).collect();
        e.write(fork_id, 1, &fork_tokens, &slices_fork).unwrap();
        e.writeback(fork_id).unwrap();

        // Invalidate the source artifact; the fork must still be readable.
        e.invalidate(base_id).unwrap();
        e.check_invariants().unwrap();

        let blocks = e.read(fork_id, 2)
            .expect("fork must remain readable after source is invalidated");
        assert_eq!(blocks.len(), 3, "fork must expose all 3 blocks");
    }

    #[test]
    fn invalidate_releases_exact_blocks_not_global_lru() {
        // Two separate artifacts; invalidating one must not free the other's blocks.
        let mut e = CoherenceEngine::<2>::new_heap(8, 4, 5);
        let t1: Vec<kv::TokenId> = vec![0, 1, 2, 3];
        let t2: Vec<kv::TokenId> = vec![4, 5, 6, 7];
        let d = make_kv_data(2);
        let s: Vec<&[u8]> = d.iter().map(|v| v.as_slice()).collect();
        let id1 = e.register(&t1, &s, 0).unwrap();
        e.register(&t2, &s, 1).unwrap();
        // Slab: 4 blocks used, 4 free
        e.invalidate(id1).unwrap();
        // After invalidating id1 (2 blocks), slab should have exactly 2 more free.
        assert_eq!(e.kv.lock().unwrap().free_count(), 6, "invalidate must free exactly the artifact's 2 blocks");
    }
}

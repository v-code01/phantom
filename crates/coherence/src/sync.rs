use std::sync::{Arc, Mutex};
use kv::TokenId;
use crate::{AgentId, ArtifactId, CoherenceError, RouteResult, engine::CoherenceEngine};

/// Thread-safe wrapper over [`CoherenceEngine`]. Backed by an `Arc<Mutex<...>>`
/// so clones are cheap (reference-count bump) and all share the same engine state.
///
/// All methods acquire the inner `Mutex` for the duration of the call — callers
/// should not hold results across calls that themselves lock (e.g., do not call
/// `lookup` and then `read` while holding the lookup result under a separate lock).
///
/// # Invariants
/// - `Clone` is an `Arc` clone: O(1), no deep copy.
/// - Mutex poisoning propagates as `unwrap()` panics — acceptable because poison
///   implies an engine panic in a prior call, which is already a fatal state.
/// - This type is the **only** route into `CoherenceEngine` from Router and
///   Scheduler. Neither component ever holds or names the inner `Mutex`.
#[derive(Clone)]
pub struct SyncEngine<const B: usize>(Arc<Mutex<CoherenceEngine<B>>>);

impl<const B: usize> SyncEngine<B> {
    /// Construct a `SyncEngine` backed by a GPU-resident Metal buffer.
    ///
    /// # Arguments
    /// * `device`         — Metal device that owns the KV buffer allocation.
    /// * `capacity`       — maximum number of KV blocks in the slab.
    /// * `element_stride` — byte size of a single KV element within a block.
    /// * `k_bound`        — maximum staleness (in writer versions) tolerated per agent.
    pub fn new(
        device: &metal::Device,
        capacity: usize,
        element_stride: usize,
        k_bound: u64,
    ) -> Self {
        Self(Arc::new(Mutex::new(
            CoherenceEngine::new(device, capacity, element_stride, k_bound),
        )))
    }

    /// Construct a `SyncEngine` backed by a heap allocation.
    ///
    /// Intended for unit tests and environments without an `MTLDevice`. Semantics
    /// are identical to `new()` — only the backing storage differs.
    ///
    /// # Arguments
    /// * `capacity`       — maximum number of KV blocks in the slab.
    /// * `element_stride` — byte size of a single KV element within a block.
    /// * `k_bound`        — maximum staleness (in writer versions) tolerated per agent.
    pub fn new_heap(capacity: usize, element_stride: usize, k_bound: u64) -> Self {
        Self(Arc::new(Mutex::new(
            CoherenceEngine::new_heap(capacity, element_stride, k_bound),
        )))
    }

    /// Find the artifact in E or S state with the longest token prefix matching
    /// `tokens`. Returns `None` if no readable artifact covers any prefix.
    ///
    /// Complexity: O(n * |tokens|) over registered artifacts — same as the
    /// underlying `CoherenceEngine::lookup`.
    #[must_use]
    pub fn lookup(&self, tokens: &[TokenId]) -> Option<RouteResult> {
        self.0.lock().unwrap().lookup(tokens)
    }

    /// Register a new artifact from raw KV data.
    ///
    /// Returns `Err(AlreadyExists)` if an artifact with the same token hash is
    /// already registered. Propagates KV slab errors as `Err(KvError(_))`.
    pub fn register(
        &self,
        tokens: &[TokenId],
        kv_data: &[&[u8]],
        agent: AgentId,
    ) -> Result<ArtifactId, CoherenceError> {
        self.0.lock().unwrap().register(tokens, kv_data, agent)
    }

    /// Create a new artifact by CoW-forking an existing one.
    ///
    /// `source` must be in `Exclusive` or `Shared` state. The new artifact
    /// starts in `Exclusive` state owned by `agent`. Prefix blocks are shared
    /// zero-copy via the underlying KV slab.
    ///
    /// Returns `Err(NotFound)` if `source` is not registered.
    /// Returns `Err(WrongState)` if `source` is `Modified` or `Invalid`.
    /// Returns `Err(AlreadyExists)` if `tokens` hash to an already-registered id.
    pub fn register_fork(
        &self,
        tokens: &[TokenId],
        source: ArtifactId,
        agent: AgentId,
    ) -> Result<ArtifactId, CoherenceError> {
        self.0.lock().unwrap().register_fork(tokens, source, agent)
    }

    /// I → E. Re-claim an invalidated artifact for exclusive write access.
    ///
    /// Returns `Err(NotFound)` if `id` is not registered.
    /// Returns `Err(WrongState)` if the artifact is not in `Invalid` state.
    pub fn acquire(&self, id: ArtifactId, agent: AgentId) -> Result<(), CoherenceError> {
        self.0.lock().unwrap().acquire(id, agent)
    }

    /// E/S → S. Add `agent` to the sharer set and return the backing block ids.
    ///
    /// Returns `Err(WrongState)` if state is `Modified` or `Invalid`.
    /// Returns `Err(KBoundExceeded)` if the agent's last-seen version is more
    /// than `k_bound` writes behind the current version.
    pub fn read(&self, id: ArtifactId, agent: AgentId) -> Result<Vec<kv::BlockId>, CoherenceError> {
        self.0.lock().unwrap().read(id, agent)
    }

    /// E → M. The exclusive owner writes new KV data extending `tokens`.
    ///
    /// Returns `Err(WrongState)` if not `Exclusive`.
    /// Returns `Err(NotOwner)` if `agent` is not the current exclusive owner.
    pub fn write(
        &self,
        id: ArtifactId,
        agent: AgentId,
        tokens: &[TokenId],
        kv_data: &[&[u8]],
    ) -> Result<(), CoherenceError> {
        self.0.lock().unwrap().write(id, agent, tokens, kv_data)
    }

    /// M → E. Stabilise a modified artifact, retaining the current owner.
    ///
    /// Returns `Err(WrongState)` if state is not `Modified`.
    pub fn writeback(&self, id: ArtifactId) -> Result<(), CoherenceError> {
        self.0.lock().unwrap().writeback(id)
    }

    /// E/S → I. Invalidate an artifact, releasing its KV blocks back to the slab.
    ///
    /// Returns `Err(WrongState)` if state is `Modified` (must writeback first) or
    /// already `Invalid` (double-invalidate is a caller bug).
    pub fn invalidate(&self, id: ArtifactId) -> Result<(), CoherenceError> {
        self.0.lock().unwrap().invalidate(id)
    }

    /// Run all four TLA+ invariants across every registered artifact.
    ///
    /// Returns `Ok(())` if all pass; `Err(id)` for the first failing artifact.
    /// Intended for use in tests and debug assertions.
    pub fn check_invariants(&self) -> Result<(), ArtifactId> {
        self.0.lock().unwrap().check_invariants()
    }

    /// Returns `(used_blocks, total_blocks)`. Intended for Prometheus metrics.
    pub fn stats(&self) -> (usize, usize) {
        self.0.lock().unwrap().stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kv::TokenId;

    fn make_data(n: usize) -> Vec<Vec<u8>> {
        (0..n).map(|i| vec![i as u8; 8]).collect()
    }

    #[test]
    fn sync_engine_constructs_heap() {
        let _e = SyncEngine::<2>::new_heap(8, 4, 5);
    }

    #[test]
    fn stats_reflects_allocation() {
        let engine = SyncEngine::<2>::new_heap(8, 4, 5);
        let (used0, total0) = engine.stats();
        assert_eq!(used0, 0, "fresh engine has 0 used blocks");
        assert_eq!(total0, 8, "total must equal capacity");

        let tokens: Vec<TokenId> = vec![0, 1, 2, 3]; // B=2 → 2 blocks
        let data: Vec<Vec<u8>> = (0..2).map(|i| vec![i as u8; 8]).collect();
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        engine.register(&tokens, &slices, 0).unwrap();

        let (used1, total1) = engine.stats();
        assert_eq!(used1, 2, "2-block artifact must consume 2 blocks");
        assert_eq!(total1, 8);
    }

    #[test]
    fn sync_engine_clone_shares_state() {
        let e1 = SyncEngine::<2>::new_heap(8, 4, 5);
        let e2 = e1.clone();
        let tokens: Vec<TokenId> = vec![0, 1, 2, 3];
        let data = make_data(2);
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let id = e1.register(&tokens, &slices, 0).unwrap();
        // e2 is a clone of the Arc — must see the same artifact
        assert!(e2.lookup(&tokens).is_some(), "clone must share engine state");
        let _ = e2.read(id, 1).unwrap();
        e2.check_invariants().unwrap();
    }

    #[test]
    fn sync_engine_concurrent_reads() {
        use std::thread;
        let engine = SyncEngine::<2>::new_heap(16, 4, 5);
        let tokens: Vec<TokenId> = vec![0, 1, 2, 3];
        let data = make_data(2);
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        engine.register(&tokens, &slices, 0).unwrap();

        let e1 = engine.clone();
        let e2 = engine.clone();
        let t1 = thread::spawn(move || {
            let id = e1.lookup(&[0u32, 1, 2, 3]).unwrap().artifact_id;
            e1.read(id, 1).unwrap()
        });
        let t2 = thread::spawn(move || {
            let id = e2.lookup(&[0u32, 1, 2, 3]).unwrap().artifact_id;
            e2.read(id, 2).unwrap()
        });
        let b1 = t1.join().unwrap();
        let b2 = t2.join().unwrap();
        assert_eq!(b1, b2, "concurrent readers must see same blocks");
        engine.check_invariants().unwrap();
    }

    #[test]
    fn sync_engine_register_fork_shares_prefix_blocks() {
        let engine = SyncEngine::<2>::new_heap(16, 4, 5);
        let base: Vec<TokenId> = vec![0, 1, 2, 3];
        let data = make_data(2);
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let base_id = engine.register(&base, &slices, 0).unwrap();

        // Fork extends the base with 2 more tokens
        let ext: Vec<TokenId> = vec![0, 1, 2, 3, 4, 5];
        let fork_id = engine.register_fork(&ext, base_id, 1).unwrap();

        // Read both — prefix blocks must be identical
        let base_blocks = engine.read(base_id, 0).unwrap();
        let fork_blocks = engine.read(fork_id, 1).unwrap();
        assert_eq!(base_blocks, fork_blocks, "fork must share prefix blocks with base");
        engine.check_invariants().unwrap();
    }
}

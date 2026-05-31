use coherence::{AgentId, ArtifactId, CoherenceError, SyncEngine};
use kv::{BlockId, TokenId};
use router::Router;

pub struct Request {
    pub tokens:  Vec<TokenId>,
    /// KV data for all blocks. Each element must be exactly B * element_stride bytes.
    pub kv_data: Vec<Vec<u8>>,
    pub agent:   AgentId,
}

pub struct Response {
    pub artifact_id: ArtifactId,
    pub blocks:      Vec<BlockId>,
}

#[derive(Debug)]
pub enum SchedulerError {
    Coherence(CoherenceError),
    /// K-bound recovery was attempted but the second read still failed.
    RecoveryFailed(CoherenceError),
}

pub struct Scheduler<const B: usize> {
    engine: SyncEngine<B>,
    router: Router<B>,
}

impl<const B: usize> Scheduler<B> {
    pub fn new(engine: SyncEngine<B>) -> Self {
        let router = Router::new(engine.clone());
        Self { engine, router }
    }

    /// Delegate to the underlying engine's invariant checker.
    /// Returns `Ok(())` if all artifact state is consistent, or
    /// `Err(artifact_id)` identifying the first offending artifact.
    pub fn check_invariants(&self) -> Result<(), coherence::ArtifactId> {
        self.engine.check_invariants()
    }

    pub fn handle(&self, req: &Request) -> Result<Response, SchedulerError> {
        match self.router.route(&req.tokens) {
            Some(hit) => {
                let blocks = self.read_with_recovery(hit.artifact_id, req.agent, req)?;
                Ok(Response { artifact_id: hit.artifact_id, blocks })
            }
            None => {
                let kv_slices: Vec<&[u8]> = req.kv_data.iter().map(|v| v.as_slice()).collect();
                let id = self.engine
                    .register(&req.tokens, &kv_slices, req.agent)
                    .map_err(SchedulerError::Coherence)?;
                let blocks = self.engine
                    .read(id, req.agent)
                    .map_err(SchedulerError::Coherence)?;
                Ok(Response { artifact_id: id, blocks })
            }
        }
    }

    fn read_with_recovery(
        &self,
        id: ArtifactId,
        agent: AgentId,
        req: &Request,
    ) -> Result<Vec<BlockId>, SchedulerError> {
        match self.engine.read(id, agent) {
            Ok(blocks) => Ok(blocks),
            Err(CoherenceError::KBoundExceeded) => self.recover(id, agent, req),
            Err(e) => Err(SchedulerError::Coherence(e)),
        }
    }

    /// Recovery path for KBoundExceeded.
    ///
    /// Sequence is strictly: invalidate → acquire → write → writeback → read.
    /// This re-anchors the artifact at the current version so `read` will accept
    /// the requesting agent's seen-version.
    fn recover(
        &self,
        id: ArtifactId,
        agent: AgentId,
        req: &Request,
    ) -> Result<Vec<BlockId>, SchedulerError> {
        // Allocate slices independently — separate borrow from `handle`'s local.
        let kv_slices: Vec<&[u8]> = req.kv_data.iter().map(|v| v.as_slice()).collect();
        self.engine.invalidate(id).map_err(SchedulerError::Coherence)?;
        self.engine.acquire(id, agent).map_err(SchedulerError::Coherence)?;
        self.engine.write(id, agent, &req.tokens, &kv_slices).map_err(SchedulerError::Coherence)?;
        self.engine.writeback(id).map_err(SchedulerError::Coherence)?;
        self.engine.read(id, agent).map_err(SchedulerError::RecoveryFailed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kv::TokenId;

    fn make_engine() -> coherence::SyncEngine<2> {
        coherence::SyncEngine::<2>::new_heap(32, 4, 5)
    }

    fn make_request(tokens: Vec<TokenId>, agent: coherence::AgentId) -> Request {
        let n_blocks = tokens.len() / 2; // B=2
        let kv_data: Vec<Vec<u8>> = (0..n_blocks).map(|i| vec![i as u8; 8]).collect();
        Request { tokens, kv_data, agent }
    }

    #[test]
    fn handle_cold_miss_registers_and_returns_blocks() {
        let engine = make_engine();
        let sched = Scheduler::new(engine);
        let req = make_request(vec![0, 1, 2, 3], 0);
        let resp = sched.handle(&req).expect("cold miss must succeed");
        assert_eq!(resp.blocks.len(), 2, "2-block artifact must return 2 blocks");
    }

    #[test]
    fn handle_cache_hit_reads_existing_artifact() {
        let engine = make_engine();
        let tokens: Vec<TokenId> = vec![0, 1, 2, 3];
        let data: Vec<Vec<u8>> = (0..2).map(|i| vec![i as u8; 8]).collect();
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let id = engine.register(&tokens, &slices, 0).unwrap();

        let sched = Scheduler::new(engine);
        let req = make_request(tokens, 1); // agent 1 requests same tokens
        let resp = sched.handle(&req).expect("cache hit must succeed");
        assert_eq!(resp.artifact_id, id, "hit must return the pre-registered artifact");
        assert_eq!(resp.blocks.len(), 2);
    }

    #[test]
    fn handle_kbound_recovery_succeeds() {
        // k_bound=0: any write makes prior readers stale.
        let engine = coherence::SyncEngine::<2>::new_heap(32, 4, 0);
        let tokens: Vec<TokenId> = vec![0, 1, 2, 3];
        let data: Vec<Vec<u8>> = (0..2).map(|i| vec![i as u8; 8]).collect();
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let id = engine.register(&tokens, &slices, 0).unwrap();

        // Agent 1 reads → seen[1]=0
        engine.read(id, 1).unwrap();
        // Invalidate, re-acquire, extend to ver=1
        engine.invalidate(id).unwrap();
        engine.acquire(id, 0).unwrap();
        let ext_tokens: Vec<TokenId> = vec![0, 1, 2, 3, 4, 5];
        let ext_data: Vec<Vec<u8>> = (0..3).map(|i| vec![i as u8; 8]).collect();
        let ext_slices: Vec<&[u8]> = ext_data.iter().map(|v| v.as_slice()).collect();
        engine.write(id, 0, &ext_tokens, &ext_slices).unwrap();
        engine.writeback(id).unwrap(); // ver=1, state=E

        // Agent 1's direct read would return KBoundExceeded (seen=0, ver=1, 1-0>0).
        // Scheduler must recover automatically via invalidate→acquire→write→writeback→read.
        let sched = Scheduler::new(engine);
        let req = make_request(ext_tokens, 1);
        let resp = sched.handle(&req).expect("K-bound recovery must succeed");
        assert!(!resp.blocks.is_empty(), "recovery must return valid blocks");
        sched.check_invariants().expect("engine invariants must hold after K-bound recovery");
    }
}

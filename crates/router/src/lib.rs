use coherence::{RouteResult, SyncEngine};
use kv::TokenId;

pub struct Router<const B: usize> {
    engine: SyncEngine<B>,
}

impl<const B: usize> Router<B> {
    pub fn new(engine: SyncEngine<B>) -> Self {
        Self { engine }
    }

    /// Find the best cached artifact for `tokens`. Returns None on a cold miss
    /// (no readable artifact covers any prefix of `tokens`).
    #[must_use]
    pub fn route(&self, tokens: &[TokenId]) -> Option<RouteResult> {
        self.engine.lookup(tokens)
    }
}

impl<const B: usize> Clone for Router<B> {
    fn clone(&self) -> Self {
        Self { engine: self.engine.clone() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_engine() -> coherence::SyncEngine<2> {
        coherence::SyncEngine::<2>::new_heap(16, 4, 5)
    }

    fn make_data(n: usize) -> Vec<Vec<u8>> {
        // 8 bytes = B * element_stride where B=2, element_stride=4
        (0..n).map(|i| vec![i as u8; 8]).collect()
    }

    #[test]
    fn route_cold_miss_returns_none() {
        let engine = make_engine();
        let router = Router::new(engine);
        assert!(router.route(&[0u32, 1, 2, 3]).is_none());
    }

    #[test]
    fn route_hit_returns_artifact() {
        let engine = make_engine();
        let tokens: Vec<TokenId> = vec![0, 1, 2, 3];
        let data = make_data(2);
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let id = engine.register(&tokens, &slices, 0).unwrap();
        let router = Router::new(engine);
        let result = router.route(&tokens).expect("must find registered artifact");
        assert_eq!(result.artifact_id, id);
        assert_eq!(result.matched_tokens, 4);
    }

    #[test]
    fn route_skips_invalidated_artifact() {
        let engine = make_engine();
        let tokens: Vec<TokenId> = vec![0, 1, 2, 3];
        let data = make_data(2);
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let id = engine.register(&tokens, &slices, 0).unwrap();
        engine.invalidate(id).unwrap();
        let router = Router::new(engine);
        assert!(router.route(&tokens).is_none(), "Invalid artifact must not be routed");
    }
}

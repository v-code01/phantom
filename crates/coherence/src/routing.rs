use std::collections::HashMap;
use xxhash_rust::xxh3::xxh3_64;
use kv::TokenId;
use crate::ArtifactId;

struct RoutingNode {
    tokens:      Box<[TokenId]>,
    artifact_id: Option<ArtifactId>,
    children:    HashMap<u64, RoutingNode>,
}

pub(crate) struct RoutingIndex<const B: usize> {
    root: HashMap<u64, RoutingNode>,
}

impl<const B: usize> RoutingIndex<B> {
    pub(crate) fn new() -> Self {
        Self { root: HashMap::new() }
    }

    fn hash_block(tokens: &[TokenId]) -> u64 {
        // SAFETY: TokenId is u32 — repr(transparent) u32, no interior padding.
        // Reinterpretation as &[u8] is valid in native-endian context (intra-process only).
        // This matches the same pattern used in ArtifactId::from_tokens (lib.rs).
        let bytes = unsafe {
            std::slice::from_raw_parts(
                tokens.as_ptr() as *const u8,
                std::mem::size_of_val(tokens),
            )
        };
        xxh3_64(bytes)
    }

    pub(crate) fn insert(&mut self, tokens: &[TokenId], id: ArtifactId) {
        assert_eq!(tokens.len() % B, 0, "tokens.len() must be multiple of B={B}");
        Self::insert_into(&mut self.root, tokens, id);
    }

    fn insert_into(map: &mut HashMap<u64, RoutingNode>, tokens: &[TokenId], id: ArtifactId) {
        let chunk = &tokens[..B];
        let rest  = &tokens[B..];
        let key   = Self::hash_block(chunk);
        let node  = map.entry(key).or_insert_with(|| RoutingNode {
            tokens:      chunk.to_vec().into_boxed_slice(),
            artifact_id: None,
            children:    HashMap::new(),
        });
        if rest.is_empty() {
            node.artifact_id = Some(id);
        } else {
            Self::insert_into(&mut node.children, rest, id);
        }
    }

    pub(crate) fn remove(&mut self, tokens: &[TokenId]) {
        if tokens.is_empty() { return; }
        debug_assert_eq!(tokens.len() % B, 0, "tokens.len() must be multiple of B={B}");
        if tokens.len() % B != 0 { return; }
        Self::remove_from(&mut self.root, tokens);
    }

    fn remove_from(map: &mut HashMap<u64, RoutingNode>, tokens: &[TokenId]) {
        let chunk = &tokens[..B];
        let rest  = &tokens[B..];
        let key   = Self::hash_block(chunk);
        if let Some(node) = map.get_mut(&key) {
            if rest.is_empty() {
                node.artifact_id = None;
                // Empty nodes are not pruned — they're harmless (longest_prefix skips them)
                // and avoiding recursion-on-return simplifies lock safety in Task 2+.
            } else {
                Self::remove_from(&mut node.children, rest);
            }
        }
    }

    /// Walk query tokens one B-block at a time. Returns the deepest node with
    /// `artifact_id = Some`. O(|tokens| / B).
    ///
    /// Returns `(artifact_id, matched_block_count)` or `None` on cold miss.
    pub(crate) fn longest_prefix(&self, tokens: &[TokenId]) -> Option<(ArtifactId, usize)> {
        let mut best:    Option<(ArtifactId, usize)> = None;
        let mut current: &HashMap<u64, RoutingNode>  = &self.root;
        let mut depth = 0usize;

        for chunk in tokens.chunks(B) {
            if chunk.len() < B { break; }
            let key = Self::hash_block(chunk);
            match current.get(&key) {
                Some(node) if node.tokens.as_ref() == chunk => {
                    depth += 1;
                    if let Some(id) = node.artifact_id {
                        best = Some((id, depth));
                    }
                    current = &node.children;
                }
                _ => break,
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(start: u32, n_blocks: usize) -> Vec<TokenId> {
        (start..start + (n_blocks * 2) as u32).collect() // B=2 in tests
    }

    #[test]
    fn cold_miss_returns_none() {
        let idx = RoutingIndex::<2>::new();
        assert!(idx.longest_prefix(&seq(0, 2)).is_none());
    }

    #[test]
    fn insert_then_longest_prefix_exact() {
        let mut idx = RoutingIndex::<2>::new();
        let id = ArtifactId(42);
        idx.insert(&seq(0, 3), id);
        let result = idx.longest_prefix(&seq(0, 3));
        assert_eq!(result, Some((id, 3)));
    }

    #[test]
    fn longest_prefix_partial_query() {
        let mut idx = RoutingIndex::<2>::new();
        let id = ArtifactId(7);
        idx.insert(&seq(0, 2), id);
        // Query is longer than artifact — matched_blocks = 2
        let result = idx.longest_prefix(&seq(0, 4));
        assert_eq!(result, Some((id, 2)));
    }

    #[test]
    fn longest_prefix_deeper_wins() {
        let mut idx = RoutingIndex::<2>::new();
        let id_a = ArtifactId(1);
        let id_b = ArtifactId(2);
        idx.insert(&seq(0, 2), id_a); // 2-block artifact
        idx.insert(&seq(0, 4), id_b); // 4-block artifact (extends same prefix)
        let result = idx.longest_prefix(&seq(0, 4));
        assert_eq!(result, Some((id_b, 4)), "deeper match must win");
    }

    #[test]
    fn remove_clears_artifact_id() {
        let mut idx = RoutingIndex::<2>::new();
        let id = ArtifactId(99);
        idx.insert(&seq(0, 2), id);
        idx.remove(&seq(0, 2));
        assert!(idx.longest_prefix(&seq(0, 2)).is_none());
    }

    #[test]
    fn remove_noop_for_unregistered_does_not_panic() {
        let mut idx = RoutingIndex::<2>::new();
        idx.remove(&seq(0, 2)); // must not panic
    }

    #[test]
    fn nested_prefixes_both_reachable() {
        let mut idx = RoutingIndex::<2>::new();
        let id_a = ArtifactId(10);
        let id_b = ArtifactId(20);
        idx.insert(&seq(0, 1), id_a); // 1 block
        idx.insert(&seq(0, 2), id_b); // 2 blocks (extends id_a)
        // Query matching 2 blocks → id_b
        assert_eq!(idx.longest_prefix(&seq(0, 2)), Some((id_b, 2)));
        // Query matching only 1 block → id_a
        let one_block_query: Vec<TokenId> = (0..2).collect(); // exactly B=2 tokens
        assert_eq!(idx.longest_prefix(&one_block_query), Some((id_a, 1)));
    }
}

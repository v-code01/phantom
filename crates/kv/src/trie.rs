use std::collections::HashMap;
use xxhash_rust::xxh3::xxh3_64;
use crate::{BlockId, LookupResult, TokenId};

struct TrieNode {
    block_id: BlockId,
    tokens: Box<[TokenId]>,
    children: HashMap<u64, usize>, // hash(child.tokens) → arena index
    // Fields below are written now; read in Tasks 6 (fork) and 7 (evict_lru).
    parent_idx: Option<usize>, // None if direct child of root
    parent_key: u64,           // key this node is stored under in parent
    rc: u32,   // active forks holding this block; incremented by fork().
               // Decremented by release(). A node is eligible for eviction only
               // when rc == 0 and it has no children.
    last_used: u64,            // monotonic clock for LRU
}

pub struct DualRadixTrie<const B: usize> {
    arena: Vec<Option<TrieNode>>,
    free_slots: Vec<usize>,
    root_children: HashMap<u64, usize>,
    clock: u64,
    /// Reverse map: BlockId → arena index. Populated on insert, removed on
    /// evict_lru or release. Enables O(1) targeted node lookup for release().
    block_to_node: HashMap<BlockId, usize>,
}

impl<const B: usize> DualRadixTrie<B> {
    pub fn new() -> Self {
        Self {
            arena: Vec::new(),
            free_slots: Vec::new(),
            root_children: HashMap::new(),
            clock: 0,
            block_to_node: HashMap::new(),
        }
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    fn hash_block(tokens: &[TokenId]) -> u64 {
        // SAFETY: TokenId is u32 — no interior padding; size_of::<u32>() bytes
        // are all value bytes. Viewing as &[u8] is well-defined.
        // NOTE: byte order is native-endian. Hash values are not portable across
        // hosts with different endianness; cached state must not be persisted
        // across heterogeneous architectures.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                tokens.as_ptr() as *const u8,
                std::mem::size_of_val(tokens),
            )
        };
        xxh3_64(bytes)
    }

    fn alloc_node(&mut self, node: TrieNode) -> usize {
        if let Some(idx) = self.free_slots.pop() {
            self.arena[idx] = Some(node);
            idx
        } else {
            let idx = self.arena.len();
            self.arena.push(Some(node));
            idx
        }
    }

    /// Walk the trie, returning all cached `BlockId`s that are a prefix of
    /// `tokens`.  Updates `last_used` on every matched node (two-pass borrow
    /// pattern: immutable lookup first, mutable update second).
    pub fn lookup(&mut self, tokens: &[TokenId]) -> LookupResult {
        let clock = self.tick();
        let mut matched_indices: Vec<usize> = Vec::new();
        let mut current_parent: Option<usize> = None;

        for chunk in tokens.chunks(B) {
            if chunk.len() < B {
                // Incomplete final block — cannot match a full edge.
                break;
            }
            let key = Self::hash_block(chunk);

            // Immutable lookup — borrow scope ends before the mutation below.
            let child_idx = {
                let map = match current_parent {
                    Some(p) => &self.arena[p].as_ref().unwrap().children,
                    None => &self.root_children,
                };
                map.get(&key).copied()
            };

            match child_idx {
                Some(idx)
                    if self.arena[idx].as_ref().unwrap().tokens.as_ref() == chunk =>
                {
                    matched_indices.push(idx);
                    current_parent = Some(idx);
                }
                _ => break,
            }
        }

        // Separate mutable pass — update last_used and collect block ids.
        let block_ids: Vec<BlockId> = matched_indices
            .iter()
            .map(|&idx| {
                let node = self.arena[idx].as_mut().unwrap();
                node.last_used = clock;
                node.block_id
            })
            .collect();

        LookupResult {
            matched_tokens: block_ids.len() * B,
            block_ids,
        }
    }

    /// Insert `blocks` keyed by `tokens` into the trie.  Blocks that are
    /// already present (matching token content) are silently skipped, so
    /// `insert` is idempotent for existing prefixes.
    pub fn insert(&mut self, tokens: &[TokenId], blocks: &[BlockId]) {
        assert_eq!(
            tokens.len() % B,
            0,
            "token count must be a multiple of B={B}"
        );
        assert_eq!(
            tokens.len() / B,
            blocks.len(),
            "blocks.len() must equal tokens.len() / B"
        );

        let clock = self.tick();
        let mut current_parent: Option<usize> = None;

        for (chunk, &block_id) in tokens.chunks(B).zip(blocks.iter()) {
            let key = Self::hash_block(chunk);

            // Immutable lookup — borrow scope ends before any mutation below.
            let existing = {
                let map = match current_parent {
                    Some(p) => &self.arena[p].as_ref().unwrap().children,
                    None => &self.root_children,
                };
                map.get(&key).copied()
            };

            if let Some(idx) = existing {
                if self.arena[idx].as_ref().unwrap().tokens.as_ref() == chunk {
                    // Edge already exists — descend without inserting.
                    current_parent = Some(idx);
                    continue;
                }
                // True xxh3 collision (different tokens, same 64-bit hash).
                // The existing node is orphaned in the arena — its parent_key
                // will still match `key`, but after the insert below the parent
                // maps `key` to the new node's index. Task 7's evict_lru must
                // guard against this: before removing from the parent's children
                // map, verify parent.children[node.parent_key] == node's own
                // arena index, and skip the remove if it doesn't match.
            }

            let new_node = TrieNode {
                block_id,
                tokens: chunk.to_vec().into_boxed_slice(),
                children: HashMap::new(),
                parent_idx: current_parent,
                parent_key: key,
                rc: 0,
                last_used: clock,
            };
            let new_idx = self.alloc_node(new_node);
            self.block_to_node.insert(block_id, new_idx);

            match current_parent {
                Some(p) => {
                    self.arena[p].as_mut().unwrap().children.insert(key, new_idx);
                }
                None => {
                    self.root_children.insert(key, new_idx);
                }
            }

            current_parent = Some(new_idx);
        }
    }

    /// Increment ref count on every matched node and return their BlockIds.
    /// Zero memcpy — caller is responsible for CoW: if writing to a returned
    /// block whose rc > 1, allocate a fresh block, copy, then call slab.decref.
    ///
    /// The caller must eventually decrement rc for each forked node (via a
    /// future `release` / `evict_lru` path). A node with rc > 0 is structurally
    /// immutable and must never be evicted.
    pub fn fork(&mut self, tokens: &[TokenId]) -> Vec<BlockId> {
        let clock = self.tick();
        let mut result: Vec<BlockId> = Vec::new();
        let mut current_parent: Option<usize> = None;

        for chunk in tokens.chunks(B) {
            if chunk.len() < B {
                break;
            }
            let key = Self::hash_block(chunk);

            let child_idx = {
                let map = match current_parent {
                    Some(p) => &self.arena[p].as_ref().unwrap().children,
                    None => &self.root_children,
                };
                map.get(&key).copied()
            };

            match child_idx {
                Some(idx)
                    if self.arena[idx].as_ref().unwrap().tokens.as_ref() == chunk =>
                {
                    let node = self.arena[idx].as_mut().unwrap();
                    node.rc = node.rc.checked_add(1).expect("TrieNode rc overflow");
                    node.last_used = clock;
                    result.push(node.block_id);
                    current_parent = Some(idx);
                }
                _ => break,
            }
        }

        result
    }

    /// Evict up to `target` leaf nodes (rc == 0, no children) sorted by
    /// last_used ascending (oldest first). Returns the freed BlockIds so the
    /// caller can decref the corresponding slab blocks.
    ///
    /// Uses a single-snapshot design: candidates are collected before any
    /// mutations. A node whose last child is evicted in this call may become a
    /// new leaf, but it will NOT be considered in the same call — a subsequent
    /// `evict_lru` call is required. Deep single-child chains require repeated
    /// calls to fully drain.
    pub fn evict_lru(&mut self, target: usize) -> Vec<BlockId> {
        // O(n) arena scan — acceptable at M1 working-set sizes (thousands of
        // blocks). Future: maintain a separate min-heap keyed by last_used for
        // O(log n) eviction.
        // Collect evictable: leaf nodes with rc == 0.
        let mut evictable: Vec<(u64, usize)> = self
            .arena
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| {
                slot.as_ref()
                    .filter(|n| n.rc == 0 && n.children.is_empty())
                    .map(|n| (n.last_used, i))
            })
            .collect();
        evictable.sort_unstable_by_key(|&(t, _)| t);

        let mut freed = Vec::new();
        for (_, idx) in evictable.into_iter().take(target) {
            let node = self.remove_node(idx);
            freed.push(node.block_id);
            self.block_to_node.remove(&node.block_id);
        }

        freed
    }

    /// Remove a node from the arena and unlink it from its parent. Returns the
    /// evicted `TrieNode`. Does NOT update `block_to_node` — the caller must do
    /// that after inspecting the returned node's `block_id`.
    ///
    /// Guards against the collision-orphan case: a true xxh3 collision can
    /// cause a live node to overwrite the parent's child pointer. Only removes
    /// from the parent map when parent.children[node.parent_key] still points
    /// to THIS node's index.
    fn remove_node(&mut self, idx: usize) -> TrieNode {
        let node = self.arena[idx].take().unwrap();
        match node.parent_idx {
            Some(parent_idx) => {
                if let Some(p) = self.arena[parent_idx].as_mut() {
                    if p.children.get(&node.parent_key) == Some(&idx) {
                        p.children.remove(&node.parent_key);
                    }
                }
            }
            None => {
                if self.root_children.get(&node.parent_key) == Some(&idx) {
                    self.root_children.remove(&node.parent_key);
                }
            }
        }
        self.free_slots.push(idx);
        node
    }

    /// Decrement rc for each block in `blocks`. Remove a node from the arena
    /// when rc reaches zero and the node has no children. Nodes with children
    /// are retained for routing even at rc=0. If `rc` is already zero the
    /// decrement is skipped; if the node is also a leaf it is removed
    /// unconditionally.
    ///
    /// Blocks are processed in **reverse** order (deepest / most-recent first)
    /// so that when a leaf node is removed and its parent's `children` map is
    /// updated, the parent is subsequently processed and may itself become
    /// removable.  Processing shallowest-first could leave an interior node
    /// alive (because it still had a child at the time of the check) even
    /// though all of its children are removed in the same call.
    pub fn release(&mut self, blocks: &[BlockId]) {
        for &bid in blocks.iter().rev() {
            let Some(&idx) = self.block_to_node.get(&bid) else { continue };
            let node = self.arena[idx].as_mut().unwrap();
            if node.rc > 0 {
                node.rc -= 1;
            }
            if node.rc == 0 && node.children.is_empty() {
                self.remove_node(idx);
                self.block_to_node.remove(&bid);
            }
        }
    }
}

impl<const B: usize> Default for DualRadixTrie<B> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a contiguous block of B tokens starting at `base`.
    fn block_tokens<const B: usize>(base: u32) -> Vec<TokenId> {
        (base * B as u32..(base + 1) * B as u32).collect()
    }

    /// Concatenate multiple blocks into a single token sequence.
    fn seq<const B: usize>(bases: &[u32]) -> Vec<TokenId> {
        bases.iter().flat_map(|&b| block_tokens::<B>(b)).collect()
    }

    #[test]
    fn lookup_empty_returns_zero() {
        let mut trie = DualRadixTrie::<4>::new();
        let result = trie.lookup(&seq::<4>(&[0]));
        assert_eq!(result.matched_tokens, 0);
        assert!(result.block_ids.is_empty());
    }

    #[test]
    fn insert_then_lookup_exact() {
        let mut trie = DualRadixTrie::<4>::new();
        let tokens = seq::<4>(&[0, 1, 2]);
        let blocks = vec![BlockId(0), BlockId(1), BlockId(2)];
        trie.insert(&tokens, &blocks);
        let result = trie.lookup(&tokens);
        assert_eq!(result.matched_tokens, 12); // 3 blocks × 4 tokens
        assert_eq!(result.block_ids, blocks);
    }

    #[test]
    fn lookup_partial_prefix() {
        let mut trie = DualRadixTrie::<4>::new();
        // Insert sequence A+B+C
        let abc = seq::<4>(&[0, 1, 2]);
        trie.insert(&abc, &[BlockId(0), BlockId(1), BlockId(2)]);
        // Query A+B+D (different third block)
        let abd: Vec<TokenId> = [seq::<4>(&[0, 1]), seq::<4>(&[3])].concat();
        let result = trie.lookup(&abd);
        assert_eq!(result.matched_tokens, 8, "first two blocks must match");
        assert_eq!(result.block_ids, vec![BlockId(0), BlockId(1)]);
    }

    #[test]
    fn fork_returns_existing_block_ids() {
        let mut trie = DualRadixTrie::<4>::new();
        let tokens = seq::<4>(&[0, 1]);
        let blocks = vec![BlockId(10), BlockId(11)];
        trie.insert(&tokens, &blocks);
        let forked = trie.fork(&tokens);
        assert_eq!(forked, blocks, "fork must return same block ids as insert");
    }

    #[test]
    fn fork_then_extend_independent() {
        let mut trie = DualRadixTrie::<4>::new();
        // Shared prefix: one block
        let prefix = seq::<4>(&[0]);
        trie.insert(&prefix, &[BlockId(0)]);

        // Agent A forks and extends
        trie.fork(&prefix);
        let a_ext = [seq::<4>(&[0]), seq::<4>(&[1])].concat();
        trie.insert(&a_ext, &[BlockId(0), BlockId(1)]);

        // Agent B forks and extends differently
        trie.fork(&prefix);
        let b_ext = [seq::<4>(&[0]), seq::<4>(&[2])].concat();
        trie.insert(&b_ext, &[BlockId(0), BlockId(2)]);

        let a_result = trie.lookup(&a_ext);
        assert_eq!(a_result.block_ids, vec![BlockId(0), BlockId(1)]);

        let b_result = trie.lookup(&b_ext);
        assert_eq!(b_result.block_ids, vec![BlockId(0), BlockId(2)]);
    }

    #[test]
    fn evict_lru_skips_forked_nodes() {
        let mut trie = DualRadixTrie::<4>::new();
        let tokens = seq::<4>(&[0]);
        trie.insert(&tokens, &[BlockId(0)]);
        trie.fork(&tokens); // rc becomes 1
        let freed = trie.evict_lru(10);
        assert!(freed.is_empty(), "forked node (rc=1) must not be evicted");
    }

    #[test]
    fn evict_lru_frees_oldest_first() {
        let mut trie = DualRadixTrie::<4>::new();
        // Three separate leaf nodes inserted in order; oldest has lowest clock.
        trie.insert(&seq::<4>(&[0]), &[BlockId(0)]);
        trie.insert(&seq::<4>(&[1]), &[BlockId(1)]);
        trie.insert(&seq::<4>(&[2]), &[BlockId(2)]);
        // Evict 1 → must free the oldest (BlockId(0), inserted first).
        let freed = trie.evict_lru(1);
        assert_eq!(freed, vec![BlockId(0)], "oldest leaf must be evicted first");
        // Verify the evicted node is unlinked — lookup must return zero matches.
        let result = trie.lookup(&seq::<4>(&[0]));
        assert_eq!(result.matched_tokens, 0, "evicted node must be unreachable via lookup");
        // Remaining two nodes must still be reachable.
        assert_eq!(trie.lookup(&seq::<4>(&[1])).matched_tokens, 4);
        assert_eq!(trie.lookup(&seq::<4>(&[2])).matched_tokens, 4);
    }

    // ── Property tests ────────────────────────────────────────────────────

    use proptest::prelude::*;

    proptest! {
        /// Any sequence of blocks inserted into the trie is always fully found
        /// on a subsequent lookup.
        #[test]
        fn prop_insert_lookup_roundtrip(n_blocks in 1usize..=8) {
            let mut trie = DualRadixTrie::<4>::new();
            // Use index-based tokens to guarantee unique blocks.
            let tokens: Vec<u32> = (0..(n_blocks * 4) as u32).collect();
            let block_ids: Vec<BlockId> = (0..n_blocks).map(BlockId).collect();
            trie.insert(&tokens, &block_ids);
            let result = trie.lookup(&tokens);
            prop_assert_eq!(result.matched_tokens, n_blocks * 4);
            prop_assert_eq!(result.block_ids, block_ids);
        }

        /// fork returns the same BlockIds that were provided at insert time.
        #[test]
        fn prop_fork_same_block_ids(n_blocks in 1usize..=6) {
            let mut trie = DualRadixTrie::<4>::new();
            let tokens: Vec<u32> = (0..(n_blocks * 4) as u32).collect();
            let block_ids: Vec<BlockId> = (0..n_blocks).map(BlockId).collect();
            trie.insert(&tokens, &block_ids);
            let forked = trie.fork(&tokens);
            prop_assert_eq!(forked, block_ids);
        }

        /// evict_lru never returns a BlockId whose node had rc > 0.
        #[test]
        fn prop_evict_never_touches_rc_nonzero(n_blocks in 1usize..=8) {
            let mut trie = DualRadixTrie::<4>::new();
            let mut forked_ids: Vec<BlockId> = Vec::new();
            for i in 0..n_blocks {
                // Each block gets 4 unique tokens based on its index.
                let tokens: Vec<u32> = (i as u32 * 4..i as u32 * 4 + 4).collect();
                trie.insert(&tokens, &[BlockId(i)]);
                if i % 2 == 0 {
                    trie.fork(&tokens);
                    forked_ids.push(BlockId(i));
                }
            }
            let freed = trie.evict_lru(n_blocks * 2);
            for fid in &forked_ids {
                prop_assert!(
                    !freed.contains(fid),
                    "forked block {:?} must not be evicted",
                    fid
                );
            }
        }
    }

    /// evict_lru never evicts an interior node that was forked (rc > 0, has children).
    /// This covers the case where `children.is_empty()` is the evictable filter —
    /// verifying that rc > 0 on interior nodes is independently respected.
    #[test]
    fn evict_lru_skips_interior_forked_node() {
        let mut trie = DualRadixTrie::<4>::new();
        // Build a 2-node chain: root → node A → node B
        let ab = seq::<4>(&[0, 1]);
        trie.insert(&ab, &[BlockId(10), BlockId(11)]);
        // Fork the prefix (node A becomes rc=1 and has node B as child)
        trie.fork(&seq::<4>(&[0]));
        // Attempt to evict everything
        let freed = trie.evict_lru(10);
        // Node B (leaf, rc=0) may be evicted; node A (interior, rc=1) must not be
        assert!(
            !freed.contains(&BlockId(10)),
            "interior forked node A must not be evicted even if rc guard were absent"
        );
    }

    #[test]
    fn release_leaf_with_rc_zero_removes_node() {
        let mut trie = DualRadixTrie::<4>::new();
        trie.insert(&seq::<4>(&[0]), &[BlockId(0)]);
        // rc=0, leaf — release must remove it
        trie.release(&[BlockId(0)]);
        let result = trie.lookup(&seq::<4>(&[0]));
        assert_eq!(result.matched_tokens, 0, "released node must be unreachable via lookup");
    }

    #[test]
    fn release_leaf_with_rc_one_decrements_and_removes() {
        let mut trie = DualRadixTrie::<4>::new();
        trie.insert(&seq::<4>(&[0]), &[BlockId(0)]);
        trie.fork(&seq::<4>(&[0])); // rc → 1
        // rc=1 → decrement → 0, still leaf → remove
        trie.release(&[BlockId(0)]);
        let result = trie.lookup(&seq::<4>(&[0]));
        assert_eq!(result.matched_tokens, 0, "forked node released to rc=0 must be unreachable");
    }

    #[test]
    fn release_interior_node_stays_when_has_children() {
        let mut trie = DualRadixTrie::<4>::new();
        // 2-block chain: [0] → [1]
        trie.insert(&seq::<4>(&[0, 1]), &[BlockId(0), BlockId(1)]);
        trie.fork(&seq::<4>(&[0])); // rc on [0] → 1
        // Release [0]: rc → 0 but has child [1] → must NOT be removed
        trie.release(&[BlockId(0)]);
        // Full 2-block chain still reachable
        let result = trie.lookup(&seq::<4>(&[0, 1]));
        assert_eq!(result.matched_tokens, 8, "interior node with children must survive release");
    }

    #[test]
    fn release_noop_for_unknown_blockid() {
        let mut trie = DualRadixTrie::<4>::new();
        // release on empty trie must not panic
        trie.release(&[BlockId(99)]);
    }
}

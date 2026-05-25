use std::collections::HashMap;
use xxhash_rust::xxh3::xxh3_64;
use crate::{BlockId, LookupResult, TokenId};

struct TrieNode {
    block_id: BlockId,
    tokens: Box<[TokenId]>,
    children: HashMap<u64, usize>, // hash(child.tokens) → arena index
    // Fields below are written now; read in Tasks 6 (fork) and 7 (evict_lru).
    #[allow(dead_code)]
    parent_idx: Option<usize>, // None if direct child of root
    #[allow(dead_code)]
    parent_key: u64,           // key this node is stored under in parent
    #[allow(dead_code)]
    rc: u32,                   // active forks holding this block
    last_used: u64,            // monotonic clock for LRU
}

pub struct DualRadixTrie<const B: usize> {
    arena: Vec<Option<TrieNode>>,
    free_slots: Vec<usize>,
    root_children: HashMap<u64, usize>,
    clock: u64,
}

impl<const B: usize> DualRadixTrie<B> {
    pub fn new() -> Self {
        Self {
            arena: Vec::new(),
            free_slots: Vec::new(),
            root_children: HashMap::new(),
            clock: 0,
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

    // fork and evict_lru are added in Tasks 6 and 7.
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
}

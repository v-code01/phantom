use crate::{
    block::BlockSlab,
    trie::DualRadixTrie,
    BlockId, LookupResult, TokenId,
};

#[derive(Debug)]
pub enum CacheError {
    OutOfBlocks,
    DataSizeMismatch,
}

pub struct KvCache<const B: usize> {
    slab: BlockSlab<B>,
    trie: DualRadixTrie<B>,
}

impl<const B: usize> KvCache<B> {
    pub fn new(device: &metal::Device, capacity: usize, element_stride: usize) -> Self {
        Self {
            slab: BlockSlab::new(device, capacity, element_stride),
            trie: DualRadixTrie::new(),
        }
    }

    /// CPU-only variant backed by a heap allocation instead of a Metal buffer.
    /// Intended for unit tests and environments without an MTLDevice.
    /// GPU visibility is not provided — do not use for production inference.
    pub fn new_heap(capacity: usize, element_stride: usize) -> Self {
        Self {
            slab: BlockSlab::new_heap(capacity, element_stride),
            trie: DualRadixTrie::new(),
        }
    }

    /// Find the longest cached prefix of `tokens`.
    pub fn lookup(&mut self, tokens: &[TokenId]) -> LookupResult {
        self.trie.lookup(tokens)
    }

    /// Write KV blocks for `tokens` into the slab and insert new nodes into
    /// the trie.  Existing prefix nodes are reused (no re-write).
    ///
    /// `kv_data[i]` must be exactly `B * element_stride` bytes.
    pub fn insert(&mut self, tokens: &[TokenId], kv_data: &[&[u8]]) -> Result<(), CacheError> {
        let n_blocks = tokens.len() / B;
        if tokens.len() % B != 0 || kv_data.len() != n_blocks {
            return Err(CacheError::DataSizeMismatch);
        }

        // Walk the trie to find the existing prefix length, then skip
        // re-allocating slab blocks for already-cached segments.
        let existing = self.trie.lookup(tokens);
        let skip_blocks = existing.matched_tokens / B;
        let mut allocated: Vec<BlockId> = existing.block_ids;

        let mut newly_allocated: Vec<BlockId> = Vec::new();
        for kv in &kv_data[skip_blocks..] {
            let id = match self.slab.alloc() {
                Some(id) => id,
                None => {
                    for &leaked in &newly_allocated {
                        self.slab.decref(leaked);
                    }
                    return Err(CacheError::OutOfBlocks);
                }
            };
            if kv.len() != B * self.slab.element_stride {
                self.slab.decref(id);
                for &leaked in &newly_allocated {
                    self.slab.decref(leaked);
                }
                return Err(CacheError::DataSizeMismatch);
            }
            // SAFETY: kv is a caller-provided slice of exactly B * element_stride
            // bytes (verified above). commit_block reads that many bytes from src.
            unsafe { self.slab.commit_block(id, kv.as_ptr()); }
            newly_allocated.push(id);
        }
        allocated.extend(newly_allocated);

        // Pass the full allocated vec (prefix ids + new ids) so the trie can
        // walk the existing prefix edges and insert only the new tail nodes.
        self.trie.insert(tokens, &allocated);
        Ok(())
    }

    /// CoW fork: increment ref counts on shared prefix trie nodes and return their
    /// block ids. Zero memcpy — the caller receives the same `BlockId`s as the
    /// inserted sequence.
    pub fn fork(&mut self, tokens: &[TokenId]) -> Vec<BlockId> {
        let ids = self.trie.fork(tokens);
        // Fix M1 slab ref-count leak: trie.fork() incremented trie rc but never
        // called slab.incref, leaving slab ref_count=1 for shared blocks. A decref
        // on either artifact path would free a block the other still holds.
        for &id in &ids {
            self.slab.incref(id);
        }
        ids
    }

    /// Evict up to `target` LRU blocks.  Returns the actual freed count.
    pub fn evict(&mut self, target: usize) -> usize {
        let freed_ids = self.trie.evict_lru(target);
        let n = freed_ids.len();
        for id in freed_ids {
            self.slab.decref(id);
        }
        n
    }

    /// Release blocks belonging to a specific artifact. Decrements trie rc for
    /// routing cleanup; calls slab.decref for every block unconditionally.
    /// Contrast with evict() which sweeps global LRU regardless of ownership.
    ///
    /// `blocks` must represent exactly one outstanding slab reference owned by
    /// this logical artifact — either from `alloc()` for a registered owner, or
    /// from `slab.incref()` for a fork holder. Passing the same block list twice
    /// (double-release) will undercount the reference and corrupt the slab.
    pub fn release(&mut self, blocks: &[BlockId]) {
        self.trie.release(blocks);
        for &id in blocks {
            self.slab.decref(id);
        }
    }

    pub fn free_count(&self) -> usize {
        self.slab.free_count()
    }

    /// Returns the total number of KV blocks in this cache.
    pub fn capacity(&self) -> usize {
        self.slab.capacity()
    }
}

// SAFETY: BlockSlab<B>: Sync (above). DualRadixTrie fields are Vec and HashMap
// which are Sync. All mutation goes through &mut self methods, serialized
// externally by the Mutex in SyncEngine.
unsafe impl<const B: usize> Sync for KvCache<B> {}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_kv_blocks(n: usize, stride: usize) -> Vec<Vec<u8>> {
        (0..n).map(|i| vec![i as u8; stride]).collect()
    }

    #[test]
    fn release_returns_slab_blocks_to_free_list() {
        // B=2, element_stride=4: 2 blocks × (2*4)=8 bytes each, capacity=4
        let mut kv = KvCache::<2>::new_heap(4, 4);
        let tokens: Vec<u32> = vec![0, 1, 2, 3];
        let data = make_kv_blocks(2, 8);
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        kv.insert(&tokens, &slices).unwrap();
        let blocks = kv.lookup(&tokens).block_ids;
        assert_eq!(kv.free_count(), 2, "2 blocks used, 2 free");
        kv.release(&blocks);
        assert_eq!(kv.free_count(), 4, "release must return all 2 blocks to slab");
    }

    #[test]
    fn fork_then_release_original_preserves_slab_refcount() {
        // After releasing the original's blocks, trie routing to those blocks is gone
        // (trie.release() removes the leaf nodes). But the SLAB blocks are still live
        // because the fork still holds a slab reference (slab rc went from 2 to 1).
        // This test verifies slab ref-count correctness; trie routing is separate.
        // B=2, capacity=4
        let mut kv = KvCache::<2>::new_heap(4, 4);
        let tokens: Vec<u32> = vec![0, 1, 2, 3];
        let data = make_kv_blocks(2, 8);
        let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        kv.insert(&tokens, &slices).unwrap();
        let orig_blocks = kv.lookup(&tokens).block_ids;
        let forked_blocks = kv.fork(&tokens); // incref — slab rc → 2
        assert_eq!(orig_blocks, forked_blocks);

        // Release original's blocks; slab rc → 1 — fork still holds them
        kv.release(&orig_blocks);
        // Slab should still have the blocks occupied (fork holds them)
        // free_count is 2 (the 2 slots are still in use by the fork)
        assert_eq!(kv.free_count(), 2, "fork still holds the blocks");

        // Releasing the fork's blocks brings free_count back to 4
        kv.release(&forked_blocks);
        assert_eq!(kv.free_count(), 4, "after releasing fork, all slots free");
    }

    #[test]
    fn kvcache_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<KvCache<4>>();
    }
}

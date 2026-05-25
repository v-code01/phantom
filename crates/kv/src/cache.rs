use crate::{
    block::BlockSlab,
    trie::DualRadixTrie,
    BlockId, TokenId,
};

#[derive(Debug)]
pub enum CacheError {
    OutOfBlocks,
    DataSizeMismatch,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LookupResult {
    pub matched_tokens: usize,
    pub block_ids: Vec<BlockId>,
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

        for kv in &kv_data[skip_blocks..] {
            let id = self.slab.alloc().ok_or(CacheError::OutOfBlocks)?;
            // Assert the caller-provided slice is exactly the right size before
            // committing. On mismatch, decref to avoid a slab leak then return.
            if kv.len() != B * self.slab.element_stride {
                self.slab.decref(id);
                return Err(CacheError::DataSizeMismatch);
            }
            // SAFETY: kv is a caller-provided slice of exactly B * element_stride
            // bytes (verified above). commit_block reads that many bytes from src.
            unsafe { self.slab.commit_block(id, kv.as_ptr()); }
            allocated.push(id);
        }

        // Pass the full allocated vec (prefix ids + new ids) so the trie can
        // walk the existing prefix edges and insert only the new tail nodes.
        self.trie.insert(tokens, &allocated);
        Ok(())
    }

    /// CoW fork: increment ref counts on shared prefix blocks, return their ids.
    pub fn fork(&mut self, tokens: &[TokenId]) -> Vec<BlockId> {
        self.trie.fork(tokens)
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

    pub fn free_count(&self) -> usize {
        self.slab.free_count()
    }
}

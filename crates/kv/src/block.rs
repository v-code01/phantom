use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use crate::BlockId;

#[derive(Debug)]
pub enum BlockError {
    NotCommitted,
}

enum BlockSlabBacking {
    #[allow(dead_code)] // Buffer held for its lifetime; only destructured on drop
    Metal(metal::Buffer),
    #[allow(dead_code)] // owned solely to keep the heap allocation alive
    Heap(Box<[u8]>),
}

pub struct BlockSlab<const B: usize> {
    data_ptr: *mut u8,
    _backing: BlockSlabBacking,
    pub element_stride: usize,
    pub capacity: usize,
    free_list: Vec<BlockId>,
    ref_counts: Vec<AtomicU32>,
    committed: Vec<AtomicBool>,
}

// SAFETY: data_ptr points into _backing which is owned and lives as long as
// BlockSlab. Single-threaded use in M1; Sync impl deferred to M3.
unsafe impl<const B: usize> Send for BlockSlab<B> {}

impl<const B: usize> BlockSlab<B> {
    pub fn new(device: &metal::Device, capacity: usize, element_stride: usize) -> Self {
        let size_bytes = capacity
            .checked_mul(B)
            .and_then(|x| x.checked_mul(element_stride))
            .expect("BlockSlab size overflow") as u64;
        let buffer = device.new_buffer(
            size_bytes,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let data_ptr = buffer.contents() as *mut u8;
        Self::from_raw(data_ptr, BlockSlabBacking::Metal(buffer), capacity, element_stride)
    }

    pub fn new_heap(capacity: usize, element_stride: usize) -> Self {
        let size_bytes = capacity
            .checked_mul(B)
            .and_then(|x| x.checked_mul(element_stride))
            .expect("BlockSlab size overflow");
        let mut heap = vec![0u8; size_bytes].into_boxed_slice();
        let data_ptr = heap.as_mut_ptr();
        Self::from_raw(data_ptr, BlockSlabBacking::Heap(heap), capacity, element_stride)
    }

    fn from_raw(
        data_ptr: *mut u8,
        backing: BlockSlabBacking,
        capacity: usize,
        element_stride: usize,
    ) -> Self {
        Self {
            data_ptr,
            _backing: backing,
            element_stride,
            capacity,
            free_list: (0..capacity).map(BlockId).collect(),
            ref_counts: (0..capacity).map(|_| AtomicU32::new(0)).collect(),
            committed: (0..capacity).map(|_| AtomicBool::new(false)).collect(),
        }
    }

    pub fn alloc(&mut self) -> Option<BlockId> {
        let id = self.free_list.pop()?;
        // Release ordering ensures the ref-count initialization is visible to any
        // thread that later acquires this BlockId. Makes the eventual M3 threading
        // transition safe without a latent data race.
        self.ref_counts[id.0].store(1, Ordering::Release);
        // committed is already false — decref resets it before returning the slot
        // to the free list, so a redundant store here is unnecessary.
        Some(id)
    }

    pub fn incref(&self, id: BlockId) {
        self.ref_counts[id.0].fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement the reference count for `id`. When the count reaches zero,
    /// the slot is reset and returned to the free list.
    ///
    /// Takes `&mut self` because returning a slot to the free list requires
    /// exclusive access to the `Vec`. At M1 (single-threaded), this is the
    /// correct contract: only the exclusive slab owner may release a block.
    /// M3 will revisit this with an interior-mutable free list for multi-writer
    /// support.
    pub fn decref(&mut self, id: BlockId) {
        debug_assert!(
            self.ref_counts[id.0].load(Ordering::Relaxed) > 0,
            "decref on block {} with zero ref count — double-free",
            id.0
        );
        let prev = self.ref_counts[id.0].fetch_sub(1, Ordering::AcqRel);
        if prev == 1 {
            self.committed[id.0].store(false, Ordering::Relaxed);
            self.free_list.push(id);
        }
    }

    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }

    /// Write `B * element_stride` bytes from `src` into block `id`, then set
    /// the I5 committed flag with a SeqCst store so any subsequent reader
    /// that loads `committed == true` is guaranteed to see all written bytes.
    ///
    /// # Safety
    /// - `src` must be valid for reads of `B * element_stride` bytes.
    /// - `id` must be an allocated block (ref count > 0). Calling this on a freed
    ///   slot is a logical error and produces stale data visible to future allocs.
    pub unsafe fn commit_block(&self, id: BlockId, src: *const u8) {
        debug_assert!(id.0 < self.capacity, "BlockId {} out of range (capacity {})", id.0, self.capacity);
        let dst = self.data_ptr.add(id.0 * B * self.element_stride);
        std::ptr::copy_nonoverlapping(src, dst, B * self.element_stride);
        // SeqCst store pairs with the SeqCst load in block_ptr. A Release/Acquire
        // pair would be sufficient for I5 correctness; SeqCst is used as a
        // conservative default. M3 can relax to Release/Acquire once the
        // multi-threaded protocol is fully specified.
        self.committed[id.0].store(true, Ordering::SeqCst);
    }

    /// Return a read-only pointer to the first byte of block `id`'s data.
    ///
    /// Returns `Err(BlockError::NotCommitted)` if the block has not yet had
    /// `commit_block` called on it — enforcing the I5 invariant that no
    /// reader sees a partially-written block.
    pub fn block_ptr(&self, id: BlockId) -> Result<*const u8, BlockError> {
        debug_assert!(id.0 < self.capacity, "BlockId {} out of range (capacity {})", id.0, self.capacity);
        // SeqCst load pairs with the SeqCst store in commit_block. Release/Acquire
        // is the minimal sufficient ordering for I5; SeqCst is used as a conservative
        // default pending the M3 threading specification.
        if !self.committed[id.0].load(Ordering::SeqCst) {
            return Err(BlockError::NotCommitted);
        }
        // SAFETY: id is within capacity; data_ptr is valid for the slab lifetime.
        Ok(unsafe { self.data_ptr.add(id.0 * B * self.element_stride) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_returns_valid_id() {
        let mut slab = BlockSlab::<16>::new_heap(4, 8);
        let id = slab.alloc().expect("fresh slab must yield a block");
        assert!(id.0 < 4, "block id must be within capacity");
    }

    #[test]
    fn decref_to_zero_frees_slot() {
        let mut slab = BlockSlab::<16>::new_heap(1, 8);
        let id = slab.alloc().unwrap();
        assert!(slab.alloc().is_none(), "slab must be exhausted after one alloc");
        slab.decref(id);
        assert!(slab.alloc().is_some(), "decreffed slot must be re-allocable");
    }

    #[test]
    fn alloc_exhaustion_returns_none() {
        let mut slab = BlockSlab::<16>::new_heap(2, 8);
        let _ = slab.alloc().unwrap();
        let _ = slab.alloc().unwrap();
        assert!(slab.alloc().is_none(), "third alloc from capacity-2 slab must return None");
    }

    #[test]
    fn uncommitted_block_ptr_returns_err() {
        let mut slab = BlockSlab::<4>::new_heap(1, 2);
        let id = slab.alloc().unwrap();
        assert!(
            matches!(slab.block_ptr(id), Err(BlockError::NotCommitted)),
            "block_ptr on uncommitted block must return Err(NotCommitted)"
        );
    }

    #[test]
    fn commit_write_read_roundtrip() {
        let mut slab = BlockSlab::<4>::new_heap(1, 2);
        let id = slab.alloc().unwrap();
        // B=4, element_stride=2 → block size = 4 * 2 = 8 bytes
        let data: Vec<u8> = (0u8..8).collect();
        unsafe { slab.commit_block(id, data.as_ptr()); }
        let ptr = slab.block_ptr(id).expect("committed block must be readable");
        let read_back = unsafe { std::slice::from_raw_parts(ptr, 8) };
        assert_eq!(read_back, data.as_slice(), "bytes written must be bytes read");
    }
}

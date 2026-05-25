use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use crate::BlockId;

#[derive(Debug)]
pub enum BlockError {
    NotCommitted,
}

enum BlockSlabBacking {
    #[cfg_attr(test, allow(dead_code))]
    Metal(metal::Buffer),
    #[allow(dead_code)] // owned solely to keep the heap allocation alive
    Heap(Box<[u8]>),
}

pub struct BlockSlab<const B: usize> {
    #[allow(dead_code)] // used in Task 3 commit_block / block_ptr
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
        let size_bytes = (capacity * B * element_stride) as u64;
        let buffer = device.new_buffer(
            size_bytes,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let data_ptr = buffer.contents() as *mut u8;
        Self::from_raw(data_ptr, BlockSlabBacking::Metal(buffer), capacity, element_stride)
    }

    #[cfg(test)]
    pub fn new_heap(capacity: usize, element_stride: usize) -> Self {
        let size_bytes = capacity * B * element_stride;
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
        self.ref_counts[id.0].store(1, Ordering::Relaxed);
        self.committed[id.0].store(false, Ordering::Relaxed);
        Some(id)
    }

    pub fn incref(&self, id: BlockId) {
        self.ref_counts[id.0].fetch_add(1, Ordering::Relaxed);
    }

    pub fn decref(&mut self, id: BlockId) {
        let prev = self.ref_counts[id.0].fetch_sub(1, Ordering::AcqRel);
        if prev == 1 {
            self.committed[id.0].store(false, Ordering::Relaxed);
            self.free_list.push(id);
        }
    }

    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }

    // commit_block and block_ptr are added in Task 3.
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
}

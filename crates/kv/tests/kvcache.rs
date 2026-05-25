use kv::{KvCache, TokenId};

/// Full workflow: shared prefix → CoW fork for two agents → independent extension
/// → evict a leaf block → verify slab free count.
///
/// Uses B=2, element_stride=4 → each block = 2 tokens × 4 bytes = 8 bytes.
/// Capacity = 8 blocks.
#[test]
fn full_workflow() {
    let mut cache = KvCache::<2>::new_heap(8, 4);

    // ── Insert 4-token shared prompt (2 blocks) ────────────────────────────
    let prompt: Vec<TokenId> = vec![0, 1, 2, 3];
    let kv0 = vec![0x00u8; 8]; // block 0: 2 tokens × 4 bytes
    let kv1 = vec![0x11u8; 8]; // block 1
    cache
        .insert(&prompt, &[kv0.as_slice(), kv1.as_slice()])
        .expect("insert shared prompt");

    // ── Fork for two agents ────────────────────────────────────────────────
    let blocks_a = cache.fork(&prompt);
    let blocks_b = cache.fork(&prompt);
    assert_eq!(blocks_a.len(), 2, "2 blocks for 4-token prompt with B=2");
    assert_eq!(blocks_a, blocks_b, "same prefix → same block ids");

    // ── Agent A extends with tokens [4,5] ─────────────────────────────────
    let a_full: Vec<TokenId> = vec![0, 1, 2, 3, 4, 5];
    let kv2a = vec![0xAAu8; 8];
    cache
        .insert(
            &a_full,
            &[kv0.as_slice(), kv1.as_slice(), kv2a.as_slice()],
        )
        .expect("insert agent A extension");

    // ── Agent B extends with tokens [6,7] ─────────────────────────────────
    let b_full: Vec<TokenId> = vec![0, 1, 2, 3, 6, 7];
    let kv2b = vec![0xBBu8; 8];
    cache
        .insert(
            &b_full,
            &[kv0.as_slice(), kv1.as_slice(), kv2b.as_slice()],
        )
        .expect("insert agent B extension");

    // ── Verify independent lookup ─────────────────────────────────────────
    let a_result = cache.lookup(&a_full);
    assert_eq!(a_result.matched_tokens, 6);
    assert_eq!(a_result.block_ids.len(), 3);

    let b_result = cache.lookup(&b_full);
    assert_eq!(b_result.matched_tokens, 6);
    assert_eq!(b_result.block_ids.len(), 3);

    // Shared prefix blocks must be identical.
    assert_eq!(
        a_result.block_ids[..2],
        b_result.block_ids[..2],
        "shared prefix must use same blocks"
    );
    // Extension blocks must differ (CoW branching).
    assert_ne!(
        a_result.block_ids[2], b_result.block_ids[2],
        "each agent's extension must use a distinct block"
    );

    // ── Evict 1 block (oldest rc=0 leaf) ─────────────────────────────────
    // Slab used: blocks 0,1 (shared, rc=2 each), block 2 (A ext, rc=0),
    //            block 3 (B ext, rc=0). free_count = 8 - 4 = 4.
    let free_before = cache.free_count();
    let freed = cache.evict(1);
    assert_eq!(freed, 1, "one block must be freed");
    assert_eq!(cache.free_count(), free_before + 1, "slab free list must grow by 1");
}

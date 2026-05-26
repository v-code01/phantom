use coherence::CoherenceEngine;
use kv::TokenId;

fn make_kv_data(n_blocks: usize) -> Vec<Vec<u8>> {
    // B=2, element_stride=4 → 8 bytes per block
    (0..n_blocks).map(|i| vec![i as u8; 8]).collect()
}

/// Three agents share a prompt artifact. Two agents read it, state machine
/// stays valid at every step per check_invariants().
#[test]
fn multi_agent_shared_artifact() {
    let mut e = CoherenceEngine::<2>::new_heap(16, 4, 5);

    // Agent 0 registers the shared prompt (4 tokens = 2 blocks).
    let prompt: Vec<TokenId> = vec![0, 1, 2, 3];
    let data = make_kv_data(2);
    let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
    let id = e.register(&prompt, &slices, 0).unwrap();
    e.check_invariants().unwrap();

    // Agents 1 and 2 read the prompt.
    let blocks1 = e.read(id, 1).expect("agent 1 read must succeed");
    e.check_invariants().unwrap();
    let blocks2 = e.read(id, 2).expect("agent 2 read must succeed");
    e.check_invariants().unwrap();
    assert_eq!(blocks1, blocks2, "both agents must see same blocks");

    // Invalidate clears the entry.
    e.invalidate(id).unwrap();
    e.check_invariants().unwrap();
}

/// Two agents fork from a shared base and extend independently.
/// Their shared prefix blocks must be identical; extension blocks must differ.
#[test]
fn fork_and_extend() {
    let mut e = CoherenceEngine::<2>::new_heap(16, 4, 5);

    // Agent 0 registers the shared base (4 tokens = 2 blocks).
    let base: Vec<TokenId> = vec![0, 1, 2, 3];
    let data = make_kv_data(2);
    let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
    let base_id = e.register(&base, &slices, 0).unwrap();
    e.check_invariants().unwrap();

    // Agent 1 forks and extends with [4, 5].
    let a_tokens: Vec<TokenId> = vec![0, 1, 2, 3, 4, 5];
    let a_id = e.register_fork(&a_tokens, base_id, 1).unwrap();
    let data_a = make_kv_data(3);
    let slices_a: Vec<&[u8]> = data_a.iter().map(|v| v.as_slice()).collect();
    e.write(a_id, 1, &a_tokens, &slices_a).unwrap();
    e.writeback(a_id).unwrap();
    e.check_invariants().unwrap();

    // Agent 2 forks and extends with [6, 7].
    let b_tokens: Vec<TokenId> = vec![0, 1, 2, 3, 6, 7];
    let b_id = e.register_fork(&b_tokens, base_id, 2).unwrap();
    let data_b = make_kv_data(3);
    let slices_b: Vec<&[u8]> = data_b.iter().map(|v| v.as_slice()).collect();
    e.write(b_id, 2, &b_tokens, &slices_b).unwrap();
    e.writeback(b_id).unwrap();
    e.check_invariants().unwrap();

    // Agent 3 reads both artifacts — extension blocks must differ.
    let ra = e.read(a_id, 3).expect("agent 3 reads A's artifact");
    let rb = e.read(b_id, 3).expect("agent 3 reads B's artifact");
    // Shared prefix (first 2 blocks) must be identical.
    assert_eq!(ra[..2], rb[..2], "shared prefix blocks must be identical");
    // Extension block must differ (CoW branching — distinct allocations).
    assert_ne!(ra[2], rb[2], "extension blocks must differ");
    e.check_invariants().unwrap();
}

use coherence::SyncEngine;
use kv::TokenId;
use scheduler::{Request, Scheduler};

fn make_request(tokens: Vec<TokenId>, agent: usize) -> Request {
    let n_blocks = tokens.len() / 2; // B=2
    let kv_data: Vec<Vec<u8>> = (0..n_blocks).map(|i| vec![i as u8; 8]).collect();
    Request { tokens, kv_data, agent }
}

/// Two agents share a prompt. Agent 0 registers it; agent 1 hits the cache.
/// check_invariants passes at every step.
#[test]
fn end_to_end_two_agents_shared_prompt() {
    let engine = SyncEngine::<2>::new_heap(32, 4, 5);
    let sched = Scheduler::new(engine.clone());

    // Agent 0: cold miss — registers the shared prompt (4 tokens = 2 blocks).
    let prompt: Vec<TokenId> = vec![0, 1, 2, 3];
    let resp0 = sched.handle(&make_request(prompt.clone(), 0))
        .expect("agent 0 cold miss must succeed");
    assert_eq!(resp0.blocks.len(), 2);
    engine.check_invariants().unwrap();

    // Agent 1: cache hit — reads the same prompt.
    let resp1 = sched.handle(&make_request(prompt.clone(), 1))
        .expect("agent 1 cache hit must succeed");
    assert_eq!(resp1.artifact_id, resp0.artifact_id, "both agents must reference same artifact");
    assert_eq!(resp1.blocks, resp0.blocks, "same prompt → same blocks");
    engine.check_invariants().unwrap();

    // Agent 2: longer query — prefix match, gets the 2 cached blocks.
    let long_query: Vec<TokenId> = vec![0, 1, 2, 3, 4, 5];
    // agent: 2 — AgentId is usize, no upper bound; B=2 is the block size const, not the agent limit
    let resp2 = sched.handle(&make_request(long_query, 2))
        .expect("agent 2 with extended query must succeed");
    // The prefix [0,1,2,3] is a cache hit for the 2-block artifact.
    // The extended tokens [4,5] cause a new cold-miss registration.
    // Either way, check_invariants must pass.
    assert_eq!(resp2.blocks.len(), 2, "prefix hit on 2-block base must return 2 blocks");
    engine.check_invariants().unwrap();
}

/// Prefix-hit scenario: two agents with different token extensions both route
/// to the same base artifact. M3 scheduler returns the cached prefix blocks;
/// full fork+extension is handled at the coherence layer by callers, not the scheduler.
#[test]
fn end_to_end_prefix_hit_multiple_agents() {
    let engine = SyncEngine::<2>::new_heap(32, 4, 5);
    let sched = Scheduler::new(engine.clone());

    // Agent 0 registers the base prompt.
    let base: Vec<TokenId> = vec![0, 1, 2, 3];
    let resp_base = sched.handle(&make_request(base.clone(), 0)).unwrap();
    engine.check_invariants().unwrap();

    // Agent 1 requests a longer sequence extending the base.
    let ext1: Vec<TokenId> = vec![0, 1, 2, 3, 4, 5];
    let resp1 = sched.handle(&make_request(ext1, 1)).unwrap();
    engine.check_invariants().unwrap();

    // Agent 2 requests a different extension of the base.
    let ext2: Vec<TokenId> = vec![0, 1, 2, 3, 6, 7];
    let resp2 = sched.handle(&make_request(ext2, 2)).unwrap();
    engine.check_invariants().unwrap();

    // Both extensions share the base prompt's blocks as prefix.
    // Both agents routed to the base artifact — they get the same 2 blocks.
    assert_eq!(resp1.blocks, resp_base.blocks, "agent 1 prefix hit must return base blocks");
    assert_eq!(resp2.blocks, resp_base.blocks, "agent 2 prefix hit must return base blocks");
    assert_eq!(resp1.artifact_id, resp_base.artifact_id, "agent 1 must route to base artifact");
    assert_eq!(resp2.artifact_id, resp_base.artifact_id, "agent 2 must route to base artifact");
}

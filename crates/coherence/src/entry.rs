use std::collections::{HashMap, HashSet};
use kv::BlockId;
use crate::{AgentId, MesiState};

pub(crate) struct ArtifactEntry {
    pub state:   MesiState,
    pub ver:     u64,
    pub owner:   Option<AgentId>,
    pub sharers: HashSet<AgentId>,
    /// Last version each agent observed via read(). Not cleared on Invalidate
    /// (matches TLA+ UNCHANGED <<seen>> in Invalidate action).
    pub seen:    HashMap<AgentId, u64>,
    pub blocks:  Vec<BlockId>,
}

impl ArtifactEntry {
    pub(crate) fn new_exclusive(owner: AgentId, blocks: Vec<BlockId>) -> Self {
        Self {
            state: MesiState::Exclusive,
            ver: 0,
            owner: Some(owner),
            sharers: HashSet::new(),
            seen: HashMap::new(),
            blocks,
        }
    }

    /// Returns true iff all four TLA+ invariants hold.
    ///
    /// SWMR: M/E state implies no sharers.
    /// SeenBound: every agent's seen version ≤ current ver.
    /// KBound: each sharer's last-seen version is within k_bound of ver.
    /// OwnerConsistency: M/E has an owner; I/S has no owner.
    pub(crate) fn invariants_hold(&self, k_bound: u64) -> bool {
        let swmr = match self.state {
            MesiState::Modified | MesiState::Exclusive => self.sharers.is_empty(),
            _ => true,
        };
        let seen_bound = self.seen.values().all(|&s| s <= self.ver);
        let k_ok = self.sharers.iter().all(|ag| {
            let s = self.seen.get(ag).copied().unwrap_or(0);
            self.ver.saturating_sub(s) <= k_bound
        });
        let owner_ok = match self.state {
            MesiState::Modified | MesiState::Exclusive => self.owner.is_some(),
            MesiState::Invalid | MesiState::Shared => self.owner.is_none(),
        };
        swmr && seen_bound && k_ok && owner_ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry_valid_exclusive(k: u64) -> bool {
        let e = ArtifactEntry::new_exclusive(0, vec![kv::BlockId(0)]);
        e.invariants_hold(k)
    }

    #[test]
    fn new_exclusive_satisfies_invariants() {
        assert!(entry_valid_exclusive(1));
    }

    #[test]
    fn modified_with_sharers_violates_swmr() {
        let mut e = ArtifactEntry::new_exclusive(0, vec![]);
        e.state = crate::MesiState::Modified;
        e.sharers.insert(1);
        assert!(!e.invariants_hold(5));
    }

    #[test]
    fn seen_past_ver_violates_seen_bound() {
        let mut e = ArtifactEntry::new_exclusive(0, vec![]);
        e.seen.insert(1, 99); // ver=0, seen=99: invalid
        assert!(!e.invariants_hold(5));
    }

    #[test]
    fn sharer_stale_beyond_k_violates_kbound() {
        let mut e = ArtifactEntry::new_exclusive(0, vec![]);
        e.state = crate::MesiState::Shared;
        e.owner = None;
        e.ver = 5;
        e.sharers.insert(1);
        e.seen.insert(1, 3); // 5 - 3 = 2 > k_bound=1
        assert!(!e.invariants_hold(1));
    }

    #[test]
    fn shared_state_with_owner_violates_owner_consistency() {
        let mut e = ArtifactEntry::new_exclusive(0, vec![]);
        e.state = crate::MesiState::Shared;
        // owner is still Some(0) — violates OwnerConsistency
        assert!(!e.invariants_hold(5));
    }
}

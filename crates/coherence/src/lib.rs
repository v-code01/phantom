pub(crate) mod engine;
pub(crate) mod entry;
pub mod sync;
pub use sync::SyncEngine;

pub use engine::CoherenceEngine;

use xxhash_rust::xxh3::xxh3_64;
use kv::{BlockId, TokenId};

pub type AgentId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArtifactId(pub u64);

impl ArtifactId {
    pub fn from_tokens(tokens: &[TokenId]) -> Self {
        // SAFETY: TokenId is u32 — no interior padding; all bytes are value bytes.
        // Native-endian, consistent within a single PHANTOM process.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                tokens.as_ptr() as *const u8,
                std::mem::size_of_val(tokens),
            )
        };
        ArtifactId(xxh3_64(bytes))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MesiState {
    Modified,
    Exclusive,
    Shared,
    Invalid,
}

#[derive(Debug)]
pub enum CoherenceError {
    NotFound,
    AlreadyExists,
    WrongState,
    NotOwner,
    KBoundExceeded,
    KvError(kv::CacheError),
}

#[derive(Debug, Clone)]
pub struct RouteResult {
    pub artifact_id: ArtifactId,
    pub matched_tokens: usize,
    pub blocks: Vec<BlockId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_id_from_tokens_is_deterministic() {
        let tokens: Vec<TokenId> = vec![0, 1, 2, 3];
        let a = ArtifactId::from_tokens(&tokens);
        let b = ArtifactId::from_tokens(&tokens);
        assert_eq!(a, b);
    }

    #[test]
    fn artifact_id_differs_for_different_tokens() {
        let a = ArtifactId::from_tokens(&[0u32, 1, 2, 3]);
        let b = ArtifactId::from_tokens(&[0u32, 1, 2, 4]);
        assert_ne!(a, b);
    }
}

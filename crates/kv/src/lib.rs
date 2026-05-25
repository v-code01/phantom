pub mod block;
pub mod cache;
pub mod trie;

pub use block::BlockError;
pub use cache::{CacheError, KvCache};
pub use trie::DualRadixTrie;

pub type TokenId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub usize);

#[derive(Debug, Clone, PartialEq)]
pub struct LookupResult {
    pub matched_tokens: usize,
    pub block_ids: Vec<BlockId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_compiles() {}

    #[test]
    fn types_exported() {
        let _id = BlockId(0);
        let _tok: TokenId = 42u32;
        let _ = format!("{:?}", BlockId(1));
    }
}

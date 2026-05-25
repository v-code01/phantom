use crate::BlockId;

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

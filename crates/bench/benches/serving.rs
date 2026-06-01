use std::time::{Duration, Instant};

use coherence::SyncEngine;
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use kv::BlockSlab;
use scheduler::{Request, Scheduler};

const B: usize = 16;
const STRIDE: usize = 64;

fn make_kv_data(n_blocks: usize) -> Vec<Vec<u8>> {
    (0..n_blocks).map(|_| vec![0u8; B * STRIDE]).collect()
}

fn bench_scheduler_cache_hit(_c: &mut Criterion) {}

fn bench_scheduler_cold_miss(_c: &mut Criterion) {}

fn bench_lookup_n_artifacts(_c: &mut Criterion) {}

fn bench_commit_block_throughput(_c: &mut Criterion) {}

fn bench_slab_alloc_decref(_c: &mut Criterion) {}

criterion_group!(
    benches,
    bench_scheduler_cache_hit,
    bench_scheduler_cold_miss,
    bench_lookup_n_artifacts,
    bench_commit_block_throughput,
    bench_slab_alloc_decref
);
criterion_main!(benches);

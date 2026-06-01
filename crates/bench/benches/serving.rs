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

fn bench_scheduler_cache_hit(c: &mut Criterion) {
    let tokens: Vec<u32> = (0u32..32).collect();
    let kv_data = make_kv_data(2);
    let kv_slices: Vec<&[u8]> = kv_data.iter().map(|v| v.as_slice()).collect();

    let mut group = c.benchmark_group("scheduler_cache_hit");

    // --- heap variant ---
    {
        let engine = SyncEngine::<B>::new_heap(256, STRIDE, 100);
        engine.register(&tokens, &kv_slices, 0).unwrap();
        let sched = Scheduler::new(engine);
        // Warm: agent 0 registers (E state); agent 1 reads → transitions to Shared
        let warm = Request { tokens: tokens.clone(), kv_data: kv_data.clone(), agent: 1 };
        sched.handle(&warm).unwrap();
        let req = Request { tokens: tokens.clone(), kv_data: kv_data.clone(), agent: 1 };
        group.bench_function("heap", |b| {
            b.iter(|| sched.handle(black_box(&req)))
        });
    }

    // --- metal variant — panics on non-Apple-Silicon per spec ---
    {
        let device = metal::Device::system_default()
            .expect("no Metal device — PHANTOM benchmarks require Apple Silicon");
        let engine = SyncEngine::<B>::new(&device, 256, STRIDE, 100);
        engine.register(&tokens, &kv_slices, 0).unwrap();
        let sched = Scheduler::new(engine);
        let warm = Request { tokens: tokens.clone(), kv_data: kv_data.clone(), agent: 1 };
        sched.handle(&warm).unwrap();
        let req = Request { tokens: tokens.clone(), kv_data: kv_data.clone(), agent: 1 };
        group.bench_function("metal", |b| {
            b.iter(|| sched.handle(black_box(&req)))
        });
    }

    group.finish();
}

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

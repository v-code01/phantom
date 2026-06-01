use std::cell::Cell;
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

fn bench_scheduler_cold_miss(c: &mut Criterion) {
    let kv_data = make_kv_data(2);

    let mut group = c.benchmark_group("scheduler_cold_miss");

    // --- heap variant ---
    // Capacity 16384 blocks = 8192 unique 2-block artifacts; sufficient for all
    // Criterion warm-up + measurement iterations combined.
    {
        let engine = SyncEngine::<B>::new_heap(16_384, STRIDE, 100);
        let sched = Scheduler::new(engine);
        // Cell<u64> for interior mutability across nested FnMut closures (no Sync needed).
        let counter = Cell::new(0u64);
        group.bench_function("heap", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let i = counter.get();
                    counter.set(i + 1);
                    // Allocate tokens + req outside the timed window
                    let tokens: Vec<u32> = (i * 32..(i + 1) * 32).map(|x| x as u32).collect();
                    let req = Request { tokens, kv_data: kv_data.clone(), agent: 0 };
                    let t0 = Instant::now();
                    black_box(sched.handle(&req)).unwrap();
                    total += t0.elapsed();
                }
                total
            });
        });
    }

    // --- metal variant ---
    {
        let device = metal::Device::system_default()
            .expect("no Metal device — PHANTOM benchmarks require Apple Silicon");
        let engine = SyncEngine::<B>::new(&device, 16_384, STRIDE, 100);
        let sched = Scheduler::new(engine);
        let counter = Cell::new(0u64);
        group.bench_function("metal", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let i = counter.get();
                    counter.set(i + 1);
                    let tokens: Vec<u32> = (i * 32..(i + 1) * 32).map(|x| x as u32).collect();
                    let req = Request { tokens, kv_data: kv_data.clone(), agent: 0 };
                    let t0 = Instant::now();
                    black_box(sched.handle(&req)).unwrap();
                    total += t0.elapsed();
                }
                total
            });
        });
    }

    group.finish();
}

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

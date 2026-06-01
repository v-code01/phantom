use std::cell::RefCell;
use std::time::{Duration, Instant};

use coherence::SyncEngine;
use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
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
    // A fresh scheduler is created per sample (outside the timed window), sized for
    // exactly `iters` cold misses. This avoids slab exhaustion across Criterion's
    // warm-up + measurement phases, which can collectively exceed a fixed capacity.
    group.bench_function("heap", |b| {
        b.iter_custom(|iters| {
            let engine = SyncEngine::<B>::new_heap(iters as usize * 2 + 64, STRIDE, 100);
            let sched = Scheduler::new(engine);
            let mut total = Duration::ZERO;
            for i in 0..iters {
                // tokens + req allocated before the clock starts — excluded from measurement
                let tokens: Vec<u32> = (i * 32..(i + 1) * 32).map(|x| x as u32).collect();
                let req = Request { tokens, kv_data: kv_data.clone(), agent: 0 };
                let t0 = Instant::now();
                black_box(sched.handle(&req)).unwrap();
                total += t0.elapsed();
            }
            total
        });
    });

    // --- metal variant ---
    {
        let device = metal::Device::system_default()
            .expect("no Metal device — PHANTOM benchmarks require Apple Silicon");
        group.bench_function("metal", |b| {
            b.iter_custom(|iters| {
                // Buffer size: iters × 2 blocks × 1024 B/block; e.g. ~2 MB at iters=1000.
                // Metal buffer creation is excluded from the timed window.
                let engine = SyncEngine::<B>::new(&device, iters as usize * 2 + 64, STRIDE, 100);
                let sched = Scheduler::new(engine);
                let mut total = Duration::ZERO;
                for i in 0..iters {
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

fn bench_lookup_n_artifacts(c: &mut Criterion) {
    // Heap only — isolates O(n) algorithmic cost from Metal throughput.
    //
    // Artifacts use disjoint token sequences (artifact i = i*32..(i+1)*32) per spec.
    // The query (0..32) matches only artifact 0; all others fail on their first token.
    // This measures HashMap iteration overhead (O(n) entries scanned), not prefix-scan
    // depth — which is intentional: the benchmark establishes when an O(1) routing index
    // (deferred to M5) becomes faster than the O(n) linear scan.
    let query_tokens: Vec<u32> = (0u32..32).collect();
    let kv_data = make_kv_data(2);
    let kv_slices: Vec<&[u8]> = kv_data.iter().map(|v| v.as_slice()).collect();

    let mut group = c.benchmark_group("lookup_n_artifacts");

    for &n in &[1usize, 10, 100, 1_000] {
        // capacity = n * 4 blocks: 2 blocks per artifact × n artifacts + 2n headroom
        let engine = SyncEngine::<B>::new_heap(n * 4, STRIDE, 100);
        for i in 0..n {
            let tokens: Vec<u32> = ((i as u32 * 32)..((i as u32 + 1) * 32)).collect();
            engine.register(&tokens, &kv_slices, 0).unwrap();
        }
        group.bench_with_input(BenchmarkId::new("lookup", n), &n, |b, _| {
            b.iter(|| engine.lookup(black_box(&query_tokens)))
        });
    }

    group.finish();
}

fn bench_commit_block_throughput(c: &mut Criterion) {
    // Non-zero source buffer: prevents compiler from emitting bzero instead of memcpy.
    let src: [u8; B * STRIDE] = std::array::from_fn(|i| i as u8);

    let mut group = c.benchmark_group("commit_block_throughput");
    // Report GB/s alongside ns/iter in Criterion output.
    group.throughput(Throughput::Bytes((B * STRIDE) as u64));

    // --- heap variant ---
    {
        let mut slab = BlockSlab::<B>::new_heap(1_024, STRIDE);
        let id = slab.alloc().expect("fresh slab must yield a block");
        group.bench_function("heap", |b| {
            b.iter(|| unsafe { slab.commit_block(black_box(id), src.as_ptr()) })
        });
    }

    // --- metal variant ---
    {
        let device = metal::Device::system_default()
            .expect("no Metal device — PHANTOM benchmarks require Apple Silicon");
        let mut slab = BlockSlab::<B>::new(&device, 1_024, STRIDE);
        let id = slab.alloc().expect("fresh slab must yield a block");
        group.bench_function("metal", |b| {
            b.iter(|| unsafe { slab.commit_block(black_box(id), src.as_ptr()) })
        });
    }

    group.finish();
}

fn bench_slab_alloc_decref(c: &mut Criterion) {
    let mut group = c.benchmark_group("slab_alloc_decref");

    // --- heap variant ---
    // RefCell provides interior mutability: `alloc` and `decref` both take &mut self,
    // so they cannot be called directly from a non-mut FnMut closure.
    {
        let slab = RefCell::new(BlockSlab::<B>::new_heap(1, STRIDE));
        group.bench_function("heap", |b| {
            b.iter_batched(
                || (),
                |_| {
                    // Single borrow_mut guard for both calls: one RefCell cycle,
                    // matching production semantics (one Mutex lock, alloc + decref, unlock).
                    let mut guard = slab.borrow_mut();
                    let id = black_box(guard.alloc()).unwrap();
                    guard.decref(black_box(id));
                },
                BatchSize::SmallInput,
            );
        });
    }

    // --- metal variant ---
    {
        let device = metal::Device::system_default()
            .expect("no Metal device — PHANTOM benchmarks require Apple Silicon");
        let slab = RefCell::new(BlockSlab::<B>::new(&device, 1, STRIDE));
        group.bench_function("metal", |b| {
            b.iter_batched(
                || (),
                |_| {
                    let mut guard = slab.borrow_mut();
                    let id = black_box(guard.alloc()).unwrap();
                    guard.decref(black_box(id));
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_scheduler_cache_hit,
    bench_scheduler_cold_miss,
    bench_lookup_n_artifacts,
    bench_commit_block_throughput,
    bench_slab_alloc_decref
);
criterion_main!(benches);

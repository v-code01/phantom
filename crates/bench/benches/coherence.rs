use coherence::SyncEngine;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, black_box};

const B:      usize = 16;
const STRIDE: usize = 64;

fn make_kv(n_blocks: usize) -> Vec<Vec<u8>> {
    (0..n_blocks).map(|_| vec![0u8; B * STRIDE]).collect()
}

/// Measures lookup time vs number of registered artifacts.
/// With the RoutingIndex, time should be flat (O(k)) not O(n).
fn bench_routing_lookup_vs_n_artifacts(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing_lookup_vs_n_artifacts");
    let kv = make_kv(2);
    let slices: Vec<&[u8]> = kv.iter().map(|v| v.as_slice()).collect();

    for &n in &[10usize, 100, 1000] {
        let engine = SyncEngine::<B>::new_heap(n * 4 + 16, STRIDE, 100);

        // Register n artifacts with distinct 2-block token sequences.
        for i in 0..n {
            let base: u32 = (i * 2 * B) as u32;
            let tokens: Vec<u32> = (base..base + 2 * B as u32).collect();
            let _ = engine.register(&tokens, &slices, 0);
        }

        // Query the last-registered artifact (worst case for the old O(n) scan).
        let last: u32 = ((n - 1) * 2 * B) as u32;
        let query: Vec<u32> = (last..last + 2 * B as u32).collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| engine.lookup(black_box(&query)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_routing_lookup_vs_n_artifacts);
criterion_main!(benches);

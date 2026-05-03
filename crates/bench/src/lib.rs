/// Statistical summary for a single latency metric (milliseconds).
/// Produced by the PHANTOM benchmark harness; compared against baseline_benchmark.py output.
pub struct Stats {
    pub p50:  f64,
    pub p95:  f64,
    pub p99:  f64,
    pub mean: f64,
}

/// Computes p50/p95/p99/mean from a slice of latency samples.
/// Samples need not be sorted on input. Panics on empty input.
pub fn compute_stats(samples: &[f64]) -> Stats {
    assert!(!samples.is_empty(), "cannot compute stats on empty sample set");
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let pct = |p: f64| -> f64 {
        let idx = ((p / 100.0) * (n - 1) as f64).round() as usize;
        sorted[idx.min(n - 1)]
    };
    Stats {
        p50:  pct(50.0),
        p95:  pct(95.0),
        p99:  pct(99.0),
        mean: sorted.iter().sum::<f64>() / n as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn crate_compiles() {}

    proptest! {
        #[test]
        fn stats_p50_between_min_and_max(
            samples in prop::collection::vec(0.0f64..1000.0, 2..100)
        ) {
            let stats = compute_stats(&samples);
            let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            prop_assert!(stats.p50 >= min);
            prop_assert!(stats.p50 <= max);
        }

        #[test]
        fn stats_p95_gte_p50(
            samples in prop::collection::vec(0.0f64..1000.0, 2..100)
        ) {
            let stats = compute_stats(&samples);
            prop_assert!(stats.p95 >= stats.p50);
        }

        #[test]
        fn stats_p99_gte_p95(
            samples in prop::collection::vec(0.0f64..1000.0, 2..100)
        ) {
            let stats = compute_stats(&samples);
            prop_assert!(stats.p99 >= stats.p95);
        }
    }
}

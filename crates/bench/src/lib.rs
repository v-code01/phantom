/// Statistical summary for a single latency metric (milliseconds).
/// Produced by the PHANTOM benchmark harness; compared against baseline_benchmark.py output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stats {
    pub p50:  f64,
    pub p95:  f64,
    pub p99:  f64,
    pub mean: f64,
}

/// Computes p50/p95/p99/mean from a slice of latency samples (milliseconds).
///
/// Uses the nearest-rank percentile method: `index = round(p/100 * (n-1))`.
/// Note: `baseline_benchmark.py` uses a different formula (`int(n * p/100)` for
/// percentiles, `statistics.median()` for p50). For n >= 20 the results converge;
/// for the 10-agent benchmark (n=10) values may differ by one rank position.
///
/// Panics on empty input.
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

    #[test]
    fn single_sample_all_percentiles_equal_value() {
        let stats = compute_stats(&[42.0]);
        assert_eq!(stats.p50, 42.0);
        assert_eq!(stats.p95, 42.0);
        assert_eq!(stats.p99, 42.0);
        assert_eq!(stats.mean, 42.0);
    }

    #[test]
    fn known_distribution_percentiles() {
        let samples: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let stats = compute_stats(&samples);
        // nearest-rank on sorted [1.0..100.0], n=100:
        // p50: round(0.5 * 99) = round(49.5) = 50 (half-away-from-zero), sorted[50] = 51.0
        // p95: round(0.95 * 99) = round(94.05) = 94, sorted[94] = 95.0
        // p99: round(0.99 * 99) = round(98.01) = 98, sorted[98] = 99.0
        // mean: sum(1..=100)/100 = 5050/100 = 50.5
        assert_eq!(stats.p50, 51.0);
        assert_eq!(stats.p95, 95.0);
        assert_eq!(stats.p99, 99.0);
        assert!((stats.mean - 50.5).abs() < 1e-10);
    }

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

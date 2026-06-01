.PHONY: test test-hardware clippy build verify-tla benchmark bench clean verify-dafny

test:
	cargo test --workspace

test-hardware:
	cargo test --workspace -- --include-ignored

clippy:
	cargo clippy --workspace --all-targets -- -D warnings

build:
	cargo build --workspace --release

verify-tla:
	@command -v tlc >/dev/null 2>&1 || { echo "TLC not found. Install: https://github.com/tlaplus/tlaplus/releases"; exit 1; }
	tlc tla/TokenCoherence.tla -config tla/TokenCoherence.cfg

benchmark:
	@command -v python3 >/dev/null 2>&1 || { echo "python3 required"; exit 1; }
	python3 scripts/baseline_benchmark.py --seed 42 --output bench_results/baseline_$$(date +%Y-%m-%d).json

bench:
	mkdir -p bench_results
	cargo bench -p bench --bench serving -- --save-baseline phantom_m4 2>&1 | tee bench_results/phantom_$(shell date +%Y-%m-%d).txt

clean:
	cargo clean

verify-dafny:
	@command -v dafny >/dev/null 2>&1 || { echo "Dafny not found. Install: brew install dafny"; exit 1; }
	dafny verify dafny/KvInvariant.dfy

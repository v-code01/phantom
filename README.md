# PHANTOM

Zero-copy multi-agent LLM KV-cache serving on Apple Silicon Unified Memory.

## What it does

Multiple inference agents share a single KV-cache slab in Metal Unified Memory. When a new request arrives, PHANTOM finds the longest cached token prefix across all registered artifacts and serves it without copying — agents read directly from the slab. Cache coherence is maintained with a MESI protocol backed by a TLA+-verified state machine.

## Architecture

```
HTTP (axum)
    └── Scheduler
            ├── Router  →  CoherenceEngine::lookup  (O(k) radix trie)
            └── CoherenceEngine
                    ├── RoutingIndex<B>          token-prefix trie, xxh3-keyed
                    ├── KvCache<B>               slab + DualRadixTrie
                    └── MESI per-artifact state  Exclusive / Shared / Modified / Invalid
```

**Request flow:**
1. Router walks `RoutingIndex` to find the deepest cached prefix match — O(k) where k = matched blocks.
2. On hit: agent reads blocks directly; Scheduler returns `cache_hit: true`.
3. On miss: Scheduler registers a new artifact, writes KV data to the slab, returns `cache_hit: false`.
4. K-bound enforcement: if an agent's last-seen version is more than `k_bound` writes stale, it is rejected and must re-read.

**Concurrency model:** `SyncEngine` wraps `CoherenceEngine` in an `Arc<RwLock<...>>`. The Write lock is held only during cold-miss registration. All hot-path operations (lookup, read, writeback) hold only the Read lock plus one per-artifact `Mutex`, so different artifacts are fully concurrent.

## Crates

| Crate | Role |
|---|---|
| `kv` | Block slab (`BlockSlab`), `KvCache<B>`, `DualRadixTrie` |
| `coherence` | `CoherenceEngine<B>`, `RoutingIndex<B>`, MESI state machine, `SyncEngine<B>` |
| `router` | Thin wrapper — calls `SyncEngine::lookup` |
| `scheduler` | Request dispatch: route → read or register → k-bound recovery |
| `api` | axum HTTP server: `POST /v1/serve`, `GET /metrics`, `GET /health` |
| `acs` | Apple Metal device setup and sanity checks |
| `bench` | Criterion benchmarks for serving throughput and routing scaling |
| `phantom` | Binary entry point |

## API

```
POST /v1/serve
  { "tokens": [u32], "agent_id": u64, "kv_data": [u8] }
  → { "cache_hit": bool, "block_ids": [u64], "latency_ns": u64 }

GET /health   → "ok"
GET /metrics  → Prometheus text (hits, misses, blocks used/total)
```

`tokens.len()` must be a non-zero multiple of `B` (default 16). `kv_data.len()` must equal `n_blocks * B * element_stride`.

## Building

```bash
cargo build --release
cargo test --workspace
cargo bench -p bench --bench serving
cargo bench -p bench --bench coherence
```

Requires Rust stable. Metal is used for GPU-backed slab allocation on macOS; the `new_heap` constructors bypass Metal for CPU-only use (tests, CI).

## Block size

The block size `B` is a compile-time const generic (default 16 tokens). Changing it recompiles the entire pipeline without runtime branching.

## License

MIT

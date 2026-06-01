#!/usr/bin/env bash
# Measure PHANTOM cache-hit serving throughput using wrk.
#
# Requires: cargo, wrk, curl, python3 (stdlib only)
# Usage:    ./scripts/bench_throughput.sh [--port 8080] [--duration 15s]
#
# What it does:
#   1. cargo build --release
#   2. Start phantom in the background
#   3. Prime the cache with a 32-token artifact (B=16, 2 blocks)
#   4. Verify the second identical request is a cache hit
#   5. Run wrk at 4 threads / 32 connections
#   6. Run wrk at 8 threads / 128 connections
#   7. Kill the server and clean up

set -euo pipefail

PORT="${PORT:-8080}"
DURATION="${DURATION:-15s}"
BASE_URL="http://127.0.0.1:${PORT}"

# ── helpers ──────────────────────────────────────────────────────────────────

die() { echo "ERROR: $*" >&2; exit 1; }

wait_for_health() {
    local deadline=$((SECONDS + 15))
    while [[ $SECONDS -lt $deadline ]]; do
        if curl -sf "${BASE_URL}/health" >/dev/null 2>&1; then return 0; fi
        sleep 0.2
    done
    die "server did not become healthy within 15 seconds"
}

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -f "${LUA_SCRIPT:-}"
}
trap cleanup EXIT

# ── build ─────────────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> cargo build --release"
cargo build --release -p phantom 2>&1 | tail -3

BINARY="./target/release/phantom"
[[ -x "$BINARY" ]] || die "binary not found at $BINARY"

# ── start server ──────────────────────────────────────────────────────────────

echo "==> starting phantom on port ${PORT}"
"$BINARY" &
SERVER_PID=$!
wait_for_health
echo "    server healthy (pid ${SERVER_PID})"

# ── prime cache (cold miss) ────────────────────────────────────────────────────
#
# B=16, 32 tokens → 2 blocks; kv_data = 2 * 16 * 64 = 2048 bytes (all 0x01).
# We build the JSON bodies with Python to avoid bash array arithmetic.

PRIME_BODY=$(python3 - <<'PYEOF'
import json, sys
tokens  = list(range(32))
kv_data = [1] * (2 * 16 * 64)   # 2 blocks * B * STRIDE = 2048 bytes
print(json.dumps({"tokens": tokens, "agent_id": 0, "kv_data": kv_data}))
PYEOF
)

HIT_BODY=$(python3 - <<'PYEOF'
import json
tokens  = list(range(32))
kv_data = [1] * (2 * 16 * 64)
print(json.dumps({"tokens": tokens, "agent_id": 1, "kv_data": kv_data}))
PYEOF
)

echo "==> priming cache (cold miss)"
PRIME_RESP=$(curl -sf -X POST "${BASE_URL}/v1/serve" \
    -H "Content-Type: application/json" \
    -d "$PRIME_BODY")
CACHE_HIT=$(echo "$PRIME_RESP" | python3 -c "import json,sys; print(json.load(sys.stdin)['cache_hit'])")
[[ "$CACHE_HIT" == "False" ]] || die "expected cold miss, got cache_hit=${CACHE_HIT}"
echo "    cold miss confirmed"

echo "==> verifying cache hit"
HIT_RESP=$(curl -sf -X POST "${BASE_URL}/v1/serve" \
    -H "Content-Type: application/json" \
    -d "$HIT_BODY")
CACHE_HIT=$(echo "$HIT_RESP" | python3 -c "import json,sys; print(json.load(sys.stdin)['cache_hit'])")
[[ "$CACHE_HIT" == "True" ]] || die "expected cache hit, got cache_hit=${CACHE_HIT}"
echo "    cache hit confirmed"

# ── wrk Lua script ────────────────────────────────────────────────────────────
#
# Every wrk request uses agent_id from a thread-local counter so concurrent
# agents all land on the same cached artifact (all are cache hits).

LUA_SCRIPT=$(mktemp /tmp/phantom_bench_XXXXXX.lua)

python3 - "$LUA_SCRIPT" <<'PYEOF'
import json, sys
body = json.dumps({"tokens": list(range(32)), "agent_id": 2, "kv_data": [1] * (2 * 16 * 64)})
lua = f"""
wrk.method  = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.body    = '{body}'
"""
with open(sys.argv[1], "w") as f:
    f.write(lua)
PYEOF

# ── throughput runs ───────────────────────────────────────────────────────────

which wrk >/dev/null 2>&1 || die "wrk not found — install with: brew install wrk"

echo ""
echo "==> wrk  4 threads / 32 connections / ${DURATION}"
wrk -t4 -c32 -d"${DURATION}" -s "$LUA_SCRIPT" "${BASE_URL}/v1/serve"

echo ""
echo "==> wrk  8 threads / 128 connections / ${DURATION}"
wrk -t8 -c128 -d"${DURATION}" -s "$LUA_SCRIPT" "${BASE_URL}/v1/serve"

echo ""
echo "==> done"

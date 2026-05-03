#!/usr/bin/env python3
"""
PHANTOM Baseline Benchmark
Records current SOTA multi-agent serving performance before PHANTOM exists.
All numbers produced here become Table 3 comparison baselines for MLSys 2027.

Hardware: Apple M4 Max, 128 GB unified memory, macOS Sequoia 15.x
Model: Llama 3.1 8B Q4_K_M (10-agent workflow, 8K context per agent)
Reproducible: make benchmark (seed=42 fixed)

Usage:
    python3 scripts/baseline_benchmark.py --seed 42 --output bench_results/baseline_$(date +%Y-%m-%d).json
    python3 scripts/baseline_benchmark.py --dry-run   # connectivity check only
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import concurrent.futures
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import requests


SYSTEMS = {
    "ollama":    {"base_url": "http://localhost:11434/v1",  "model": "llama3.1:8b-instruct-q4_K_M"},
    "vllm-mlx":  {"base_url": "http://localhost:8000/v1",   "model": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"},
    "llama-cpp": {"base_url": "http://localhost:8080/v1",   "model": "llama3.1-8b-q4"},
}

SHARED_DOCUMENT = (
    "The Unified Memory Architecture (UMA) of Apple Silicon integrates CPU, GPU, "
    "and Neural Engine into a single coherent memory pool. This eliminates PCIe "
    "transfer overhead present in discrete-GPU systems. " * 80
)

AGENT_PROMPTS = [
    "Summarize the key performance implications of UMA for LLM inference.",
    "What are the latency benefits of zero-copy memory access for tool calls?",
    "Compare UMA bandwidth (546 GB/s M4 Max) to PCIe 4.0 x16 (64 GB/s).",
    "How does KV cache management differ between discrete GPU and UMA systems?",
    "Explain how the Neural Engine can run concurrently with GPU decode on M4 Max.",
    "What is the MESI cache coherence protocol and how does it apply to multi-agent LLMs?",
    "Describe the DualRadixTree data structure for copy-on-write KV cache.",
    "What is the token overhead in a 10-agent flat topology without coherence?",
    "How does IOSurface enable zero-copy GPU-to-ANE data transfer?",
    "What are the Dafny-verifiable invariants for a unified memory agent scheduler?",
]


def host_info() -> dict:
    mac_ver = platform.mac_ver()[0]
    try:
        mem_bytes = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], stderr=subprocess.DEVNULL
        ).strip())
        mem_gb = mem_bytes // (1024 ** 3)
    except Exception:
        mem_gb = None
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        chip = platform.processor()
    return {"macos": mac_ver, "chip": chip, "memory_gb": mem_gb}


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


@dataclass
class AgentResult:
    agent_id: int
    prompt_tokens: int        # word-split approximation; BPE ≈ 1.3x word count
    completion_tokens: int    # word-split approximation; BPE ≈ 1.3x word count
    ttft_ms: float
    total_ms: float


@dataclass
class BenchResult:
    system: str
    model: str
    seed: int
    num_agents: int
    estimated_context_tokens: int  # approximate BPE token count for shared document
    timestamp: str
    host: dict
    script_git_sha: str            # git SHA of benchmark script at run time
    agent_results: list
    p50_ttft_ms: float
    p95_ttft_ms: float
    p99_ttft_ms: float
    p50_total_ms: float
    p95_total_ms: float
    total_tokens: int
    throughput_tok_per_sec: float  # completion tokens / wall-clock seconds at peak concurrency
    failed_agents: int             # number of agent requests that raised exceptions
    broadcast_duplication_pct: float


def check_server(base_url: str, timeout: float = 2.0) -> bool:
    try:
        r = requests.get(f"{base_url}/models", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def warmup(base_url: str, model: str) -> None:
    """Single warmup request to prime model cache and Metal kernels."""
    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "stream": False,
        }
        requests.post(f"{base_url}/chat/completions", json=payload, timeout=60)
    except Exception:
        pass  # warmup failure is non-fatal


def run_agent(base_url: str, model: str, agent_id: int,
              shared_doc: str, prompt: str, seed: int) -> AgentResult:
    messages = [
        {"role": "system",
         "content": f"You are agent {agent_id}. Reference document:\n\n{shared_doc}"},
        {"role": "user", "content": prompt},
    ]
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.0,
        "seed": seed + agent_id,
        "stream": True,
    }
    t0 = time.perf_counter()
    ttft_ms: Optional[float] = None
    chunks = []
    with requests.post(f"{base_url}/chat/completions", json=payload,
                       stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or line == b"data: [DONE]":
                continue
            if line.startswith(b"data: "):
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta and ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000
                chunks.append(delta)
    total_ms = (time.perf_counter() - t0) * 1000
    completion = "".join(chunks)
    # word-split approximation; BPE ≈ 1.3x word count
    prompt_tokens = len(shared_doc.split()) + len(prompt.split())
    return AgentResult(
        agent_id=agent_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=len(completion.split()),
        ttft_ms=ttft_ms or total_ms,
        total_ms=total_ms,
    )


def broadcast_duplication_pct(results: list[AgentResult]) -> float:
    shared_tokens = len(SHARED_DOCUMENT.split())
    total_tokens = sum(r.prompt_tokens for r in results)
    duplicated = shared_tokens * (len(results) - 1)
    return min(100.0, duplicated / total_tokens * 100) if total_tokens > 0 else 0.0


def benchmark_system(name: str, cfg: dict, seed: int, sha: str,
                     dry_run: bool, host: dict) -> Optional[BenchResult]:
    print(f"\n{'='*60}\nSystem: {name}  Model: {cfg['model']}\n{'='*60}")
    if not check_server(cfg["base_url"]):
        print(f"  [SKIP] {name} not reachable at {cfg['base_url']}")
        return None
    if dry_run:
        print(f"  [DRY RUN] server reachable, skipping inference")
        return None

    print(f"  Warming up {name}...")
    warmup(cfg["base_url"], cfg["model"])

    results = []
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(AGENT_PROMPTS)) as executor:
        future_to_id = {
            executor.submit(run_agent, cfg["base_url"], cfg["model"], i,
                            SHARED_DOCUMENT, AGENT_PROMPTS[i], seed): i
            for i in range(len(AGENT_PROMPTS))
        }
        for future in concurrent.futures.as_completed(future_to_id):
            agent_id = future_to_id[future]
            try:
                r = future.result()
                results.append(r)
                print(f"  Agent {agent_id+1:02d}: TTFT={r.ttft_ms:.0f}ms  total={r.total_ms:.0f}ms")
            except Exception as e:
                failed += 1
                print(f"  Agent {agent_id+1:02d}: ERROR: {e}")

    if not results:
        return None

    ttfts_arr = np.array([r.ttft_ms for r in results])
    totals_arr = np.array([r.total_ms for r in results])

    # wall-clock is determined by the slowest concurrent agent (peak concurrency window)
    wall_clock_s = float(np.max(totals_arr)) / 1000.0
    total_completion_tokens = sum(r.completion_tokens for r in results)
    throughput = total_completion_tokens / wall_clock_s if wall_clock_s > 0 else 0.0

    return BenchResult(
        system=name,
        model=cfg["model"],
        seed=seed,
        num_agents=len(results),
        estimated_context_tokens=int(len(SHARED_DOCUMENT.split()) * 1.3),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        host=host,
        script_git_sha=sha,
        agent_results=[asdict(r) for r in results],
        p50_ttft_ms=float(np.percentile(ttfts_arr, 50)),
        p95_ttft_ms=float(np.percentile(ttfts_arr, 95)),
        p99_ttft_ms=float(np.percentile(ttfts_arr, 99)),
        p50_total_ms=float(np.percentile(totals_arr, 50)),
        p95_total_ms=float(np.percentile(totals_arr, 95)),
        total_tokens=total_completion_tokens,
        throughput_tok_per_sec=throughput,
        failed_agents=failed,
        broadcast_duplication_pct=broadcast_duplication_pct(results),
    )


def main():
    parser = argparse.ArgumentParser(description="PHANTOM baseline benchmark")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--output",  type=str,
                        default=f"bench_results/baseline_{time.strftime('%Y-%m-%d')}.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--system",  type=str, default=None,
                        help=f"one of: {list(SYSTEMS)}")
    args = parser.parse_args()

    systems = SYSTEMS
    if args.system:
        if args.system not in SYSTEMS:
            print(f"Unknown system '{args.system}'. Choose from: {list(SYSTEMS)}")
            sys.exit(1)
        systems = {args.system: SYSTEMS[args.system]}

    host = host_info()
    sha = git_sha()
    print(f"PHANTOM Baseline Benchmark")
    print(f"Host:   {host['chip']}, {host['memory_gb']} GB, macOS {host['macos']}")
    print(f"Agents: {len(AGENT_PROMPTS)}  Context: ~{int(len(SHARED_DOCUMENT.split()) * 1.3)} tokens (estimated)  Seed: {args.seed}")
    print(f"Model:  Llama 3.1 8B Q4_K_M  Dry-run: {args.dry_run}")
    print(f"SHA:    {sha}")

    all_results = []
    for name, cfg in systems.items():
        result = benchmark_system(name, cfg, args.seed, sha, args.dry_run, host)
        if result:
            all_results.append(asdict(result))

    if not all_results:
        print("\nNo results (servers offline or dry-run). To run:")
        print("  brew install ollama && ollama pull llama3.1:8b-instruct-q4_K_M")
        print("  ollama serve &")
        print("  make benchmark")
        sys.exit(0)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"phantom_baseline": all_results}, f, indent=2)
    print(f"\nResults → {args.output}")
    print("\nSummary:")
    for r in all_results:
        print(f"  {r['system']:12s}  p50_ttft={r['p50_ttft_ms']:.0f}ms  "
              f"throughput={r['throughput_tok_per_sec']:.1f} tok/s  "
              f"broadcast_dup={r['broadcast_duplication_pct']:.1f}%")


if __name__ == "__main__":
    main()

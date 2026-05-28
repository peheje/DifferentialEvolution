"""DE tuner for llama.cpp Multi-Token Prediction speculative decoding params.

Optimizes MTP hyperparameters to maximize tokens/sec on a fixed prompt
while maintaining output quality (token count near baseline).

Usage:
  python -m torch_de.llama_mtp_tuner --smoke     # validate server + DE works
  python -m torch_de.llama_mtp_tuner --quick     # 5-gen quick sweep
  python -m torch_de.llama_mtp_tuner --tune      # full optimization run
"""

import argparse
import json
import math
import subprocess
import sys
import time
import urllib.request

import numpy as np

# ── server control ──────────────────────────────────────────────────

SERVER_URL = "http://127.0.0.1:8080"
LLAMA_SERVER_BIN = "/home/phj/llama.cpp/build-cuda/bin/llama-server"
MODEL_FILE = ("/home/phj/models/gguf/"
              "Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-"
              "Q4_K_M.gguf")
SERVER_LOG = "/tmp/qwen-de-tuner.log"
SERVER_PID_FILE = "/tmp/qwen-de-tuner.pid"


def server_running():
    try:
        req = urllib.request.Request(f"{SERVER_URL}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read()).get("status") == "ok"
    except Exception:
        return False


def stop_server():
    """Kill any running server from our PID file."""
    try:
        with open(SERVER_PID_FILE) as f:
            pid = int(f.read().strip())
        subprocess.run(["kill", str(pid)], capture_output=True)
        time.sleep(2)
    except Exception:
        pass


def start_server(params, prompt="Say hello in 50 words.", max_tokens=100):
    """Start llama-server with given MTP params. Returns True if healthy."""
    stop_server()

    draft_n_max = int(params[0])
    draft_n_min = int(params[1])
    draft_p_min = float(params[2])
    ngram_match = int(params[3])
    ngram_min = int(params[4])
    ngram_max = int(params[5])
    ngram_k4v_n = int(params[6])
    ngram_k4v_m = int(params[7])
    ngram_k4v_hits = int(params[8])
    fit_target = int(params[9])
    batch_size = int(params[10])
    ubatch_size = int(params[11])
    temp = float(params[12])
    top_p = float(params[13])

    spec_type = "draft-mtp,ngram-mod"
    if ngram_k4v_hits > 0:
        spec_type += ",ngram-map-k4v"

    cmd = [
        LLAMA_SERVER_BIN,
        "-m", MODEL_FILE,
        "-fitt", str(fit_target),
        "--spec-type", spec_type,
        "--spec-draft-n-max", str(draft_n_max),
        "--spec-draft-n-min", str(draft_n_min),
        "--spec-draft-p-min", str(draft_p_min),
        "--spec-ngram-mod-n-match", str(ngram_match),
        "--spec-ngram-mod-n-min", str(ngram_min),
        "--spec-ngram-mod-n-max", str(ngram_max),
        "--spec-ngram-map-k4v-size-n", str(ngram_k4v_n),
        "--spec-ngram-map-k4v-size-m", str(ngram_k4v_m),
        "--spec-ngram-map-k4v-min-hits", str(ngram_k4v_hits),
        "-c", "8192",
        "-b", str(batch_size),
        "-ub", str(ubatch_size),
        "-fa", "on",
        "--temp", f"{temp:.2f}",
        "--top-p", f"{top_p:.2f}",
        "--top-k", "20",
        "--min-p", "0",
        "--presence-penalty", "0.0",
        "--no-mmap", "--mlock", "--no-warmup",
        "--no-mmproj",
        "--reasoning", "off",
        "--chat-template-kwargs", '{"enable_thinking": false}',
        "--host", "127.0.0.1", "--port", "8080",
    ]

    with open(SERVER_LOG, "w") as log:
        subprocess.Popen(cmd, stdout=log, stderr=log,
                         start_new_session=True)

    # Record PID (llama-server daemonizes; find via pgrep)
    for _ in range(5):
        time.sleep(2)
        try:
            result = subprocess.run(
                ["pgrep", "-f", "llama-server"], capture_output=True, text=True)
            if result.stdout.strip():
                pid = result.stdout.strip().split("\n")[0]
                with open(SERVER_PID_FILE, "w") as f:
                    f.write(pid)
                break
        except Exception:
            pass

    # Wait for /health
    for _ in range(60):
        if server_running():
            return True
        time.sleep(2)
    return False


def benchmark_server(prompt, max_tokens=512, timeout=60):
    """Send prompt, measure wall time and tokens. Returns (tok_s, completion_tokens)."""
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{SERVER_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        return 0.0, 0, str(e)

    wall_s = time.perf_counter() - t0
    tokens = data.get("usage", {}).get("completion_tokens", 0)
    tok_s = tokens / wall_s if wall_s > 0 else 0.0
    return tok_s, tokens, None


# ── benchmark prompt ────────────────────────────────────────────────

BENCH_PROMPT = (
    "Write a complete Python implementation of the classic Snake game "
    "using pygame. Include a Game class with update, draw, handle_input, "
    "and reset methods. Include scoring, speed increase, and game over "
    "handling. Write clean, well-structured code with docstrings."
)

# Baseline token count (measured once at current server config)
BASELINE_TOKENS = None


def get_baseline_tokens():
    global BASELINE_TOKENS
    if BASELINE_TOKENS is not None:
        return BASELINE_TOKENS
    if not server_running():
        print("ERROR: server not running for baseline measurement")
        return 300
    _, tokens, err = benchmark_server(BENCH_PROMPT, max_tokens=512)
    if err:
        print(f"Baseline error: {err}")
        return 300
    BASELINE_TOKENS = tokens
    print(f"Baseline: {tokens} tokens at current config")
    return tokens


# ── DE objective ────────────────────────────────────────────────────

def _clamp_int(x, lo, hi):
    return max(lo, min(hi, int(round(x))))


def _clamp_float(x, lo, hi):
    return max(lo, min(hi, float(x)))


def objective(params_vec):
    """DE objective: given param vector [D], start server, benchmark, return -tok/s.

    We minimize, so return negative tokens/sec. Heavy penalty on failure.
    """
    D = len(params_vec)
    xs = [0.0] * D

    # Integer params
    xs[0] = _clamp_int(params_vec[0], 1, 3)      # draft_n_max
    xs[1] = _clamp_int(params_vec[1], 0, 1)      # draft_n_min
    xs[3] = _clamp_int(params_vec[3], 8, 64)      # ngram_match
    xs[4] = _clamp_int(params_vec[4], 16, 128)     # ngram_min
    xs[5] = _clamp_int(params_vec[5], 32, 256)     # ngram_max
    xs[6] = _clamp_int(params_vec[6], 8, 64)       # k4v_n
    xs[7] = _clamp_int(params_vec[7], 8, 64)       # k4v_m
    xs[8] = _clamp_int(params_vec[8], 0, 8)        # k4v_hits
    xs[9] = _clamp_int(params_vec[9], 256, 2048)   # fit_target
    xs[10] = _clamp_int(params_vec[10], 256, 4096)  # batch_size
    xs[11] = _clamp_int(params_vec[11], 64, 2048)   # ubatch_size

    # Float params
    xs[2] = _clamp_float(params_vec[2], 0.0, 0.9)   # draft_p_min
    xs[12] = _clamp_float(params_vec[12], 0.1, 1.5)  # temp
    xs[13] = _clamp_float(params_vec[13], 0.5, 1.0)  # top_p

    # Enforce ngram_min < ngram_max
    if xs[4] >= xs[5]:
        xs[5] = xs[4] + 8

    # Enforce ubatch <= batch
    if xs[11] > xs[10]:
        xs[11] = xs[10]

    # Start server
    if not start_server(xs):
        return 1e6  # huge penalty for failure

    # Warmup
    _, _, _ = benchmark_server("Hello", max_tokens=10, timeout=30)

    # Benchmark
    tok_s, comp_tokens, err = benchmark_server(BENCH_PROMPT, max_tokens=512)
    stop_server()

    if err or tok_s <= 0:
        return 1e6

    # Quality check: penalize if token count deviates too much from baseline
    baseline = get_baseline_tokens()
    quality_ratio = min(comp_tokens / max(baseline, 1), 2.0)
    if quality_ratio < 0.5:
        # Heavy penalty for degenerate output
        return 1e5 - tok_s

    # Minimize negative tokens/sec (i.e., maximize tok/s)
    # Small quality bonus for higher token counts
    score = -tok_s - 0.01 * quality_ratio
    return float(score)


# ── DE wrapper (population-level objective) ─────────────────────────

def de_objective(pop):
    """Vectorized DE objective: pop is [P, D] torch tensor, return losses [P]."""
    import torch
    pop_np = pop.cpu().numpy()
    losses = np.zeros(pop_np.shape[0], dtype=np.float64)
    for i in range(pop_np.shape[0]):
        losses[i] = objective(pop_np[i])
    return torch.from_numpy(losses).to(dtype=pop.dtype, device=pop.device)


# ── CLI ─────────────────────────────────────────────────────────────

def _main():
    parser = argparse.ArgumentParser(
        description="DE tuner for llama.cpp MTP spec decoding")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: validate server + single DE eval")
    parser.add_argument("--quick", action="store_true",
                        help="Quick 5-gen optimization run")
    parser.add_argument("--tune", action="store_true",
                        help="Full optimization (30-60 min)")
    args = parser.parse_args()

    # Param vector (14 dims, mixed int/float):
    # [draft_n_max, draft_n_min, draft_p_min, ngram_match, ngram_min,
    #  ngram_max, k4v_n, k4v_m, k4v_hits, fit_target, batch_size,
    #  ubatch_size, temp, top_p]
    bounds_lo = [1, 0, 0.0, 8, 16, 32, 8, 8, 0, 256, 256, 64, 0.1, 0.5]
    bounds_hi = [3, 1, 0.9, 64, 128, 256, 64, 64, 8, 2048, 4096, 2048, 1.5, 1.0]

    if not server_running():
        print("NOTE: Server not running. First eval will start it.")

    baseline = get_baseline_tokens()
    print(f"Baseline: {baseline} tokens\n")

    if args.smoke:
        print("=== SMOKE TEST ===\n")
        # Test single eval with default-ish params
        test_params = np.array([2, 0, 0.0, 32, 64, 96, 16, 24, 1,
                                1536, 4096, 1024, 0.6, 0.95], dtype=np.float64)
        print("Testing single eval with default params...")
        score = objective(test_params)
        print(f"  score (neg_tok_s) = {score:.2f}")
        print(f"  (negative means positive tok/s: tok_s ≈ {-score:.1f})")
        print("\nTesting single eval with variant params...")
        variant = test_params.copy()
        variant[0] = 3  # more draft tokens
        variant[2] = 0.3  # higher acceptance threshold
        score2 = objective(variant)
        print(f"  score = {score2:.2f} (tok_s ≈ {-score2:.1f})")
        print("\nSMOKE PASS" if score < 1e5 and score2 < 1e5 else "\nSMOKE FAIL")
        return

    from .de import de_solve
    import torch

    D = len(bounds_lo)
    if args.quick:
        popsize = 10
        gens = 5
    else:
        popsize = 20
        gens = 15

    print(f"DE MTP tuner: pop={popsize} gen={gens}")
    print(f"Estimated time: ~{popsize * gens * 45}s = {popsize * gens * 0.75:.0f} min\n")

    best = torch.zeros(D, dtype=torch.float64)
    t0 = time.perf_counter()
    loss = de_solve(D, popsize, popsize, gens,
                    bounds_lo, bounds_hi, de_objective, 42, best,
                    mutation_strategy=0, device="cpu", dtype=torch.float64)
    wall = time.perf_counter() - t0

    # Print best config
    b = best.numpy()
    print(f"\n{'='*60}")
    print(f"Best config found (wall={wall:.0f}s, tok_s ≈ {-loss:.1f}):")
    print(f"  draft_n_max={int(b[0])}  draft_n_min={int(b[1])}")
    print(f"  draft_p_min={b[2]:.2f}")
    print(f"  ngram: match={int(b[3])}  min={int(b[4])}  max={int(b[5])}")
    print(f"  k4v: n={int(b[6])}  m={int(b[7])}  hits={int(b[8])}")
    print(f"  fit={int(b[9])}  batch={int(b[10])}  ubatch={int(b[11])}")
    print(f"  temp={b[12]:.2f}  top_p={b[13]:.2f}")

    # Re-benchmark best config for final measurement
    print("\nFinal benchmark of best config...")
    stop_server()
    if start_server(b):
        tok_s, tokens, err = benchmark_server(BENCH_PROMPT, max_tokens=512)
        print(f"  tok/s={tok_s:.1f}  tokens={tokens}")
    stop_server()


if __name__ == "__main__":
    _main()

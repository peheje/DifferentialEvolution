"""DE tuner for llama.cpp MTP speculative decoding params.

Black-box optimization: maximize tokens/sec on a fixed benchmark prompt
by tuning MTP spec decoding hyperparameters. Each evaluation starts a
llama-server, benchmarks, and stops it (~45s/cycle).

Usage:
  python -m torch_de.llama_mtp_tuner --test     # dry-run + 1 real eval
  python -m torch_de.llama_mtp_tuner --quick    # 5-gen sweep (~8 min on fake, ~40min real)
  python -m torch_de.llama_mtp_tuner --tune     # full run (~4hrs)

Safety: stops user's server at start, restores it on exit via try/finally.
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
import urllib.request

import numpy as np

# ── config ──────────────────────────────────────────────────────────

LLAMA_SERVER = "/home/phj/llama.cpp/build-cuda/bin/llama-server"
MODEL = ("/home/phj/models/gguf/"
         "Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-Q4_K_M.gguf")
SERVER_PORT = 8080
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"
SERVER_LOG = "/tmp/qwen-de-tuner.log"
RESTORE_SCRIPT = "/home/phj/ai-scripts/qwen-server-start.sh"
STOP_SCRIPT = "/home/phj/ai-scripts/qwen-server-stop.sh"
RESULT_LOG = "/tmp/qwen-de-tuner-results.jsonl"

BENCH_PROMPT = (
    "Write a complete Python implementation of the classic Snake game "
    "using pygame. Include a Game class with update, draw, handle_input, "
    "and reset methods. Include scoring, speed increase, and game over "
    "handling. Write clean, well-structured code with docstrings."
)
BENCH_MAX_TOKENS = 512
BENCH_TIMEOUT = 90

# Fixed sampling (not evolved)
FIXED_TEMP = 0.6
FIXED_TOP_P = 0.95
FIXED_TOP_K = 20
FIXED_MIN_P = 0.0

# ── parameter definitions ───────────────────────────────────────────

# (name, type, lo, hi, default)
# type: "int" or "float"
PARAM_DEFS = [
    ("draft_n_max",       "int",   1,   3,   2),
    ("draft_n_min",       "int",   0,   1,   0),
    ("draft_p_min",       "float", 0.0, 0.9, 0.0),
    ("ngram_match",       "int",   8,  64,  32),
    ("ngram_min",         "int",  16, 128,  64),
    ("ngram_max",         "int",  32, 256,  96),
    ("k4v_n",             "int",   8,  64,  16),
    ("k4v_m",             "int",   8,  64,  24),
    ("k4v_hits",          "int",   0,   8,   1),
    ("fit_target",        "int", 256, 2048, 1536),
    ("batch_size",        "int", 256, 4096, 4096),
    ("ubatch_size",       "int",  64, 2048, 1024),
]

D = len(PARAM_DEFS)
BOUNDS_LO = np.array([p[2] for p in PARAM_DEFS], dtype=np.float64)
BOUNDS_HI = np.array([p[3] for p in PARAM_DEFS], dtype=np.float64)
IS_INT = np.array([p[1] == "int" for p in PARAM_DEFS], dtype=bool)
DEFAULTS = np.array([p[4] for p in PARAM_DEFS], dtype=np.float64)

# ── server control ──────────────────────────────────────────────────

_server_proc = None


def server_health():
    try:
        req = urllib.request.Request(f"{SERVER_URL}/health")
        with urllib.request.urlopen(req, timeout=3) as r:
            return json.loads(r.read()).get("status") == "ok"
    except Exception:
        return False


def kill_tuner_server():
    global _server_proc
    if _server_proc is not None:
        try:
            os.killpg(os.getpgid(_server_proc.pid), signal.SIGTERM)
        except Exception:
            pass
        try:
            _server_proc.wait(timeout=5)
        except Exception:
            try:
                os.killpg(os.getpgid(_server_proc.pid), signal.SIGKILL)
            except Exception:
                pass
        _server_proc = None
    time.sleep(2)


def stop_all_servers():
    """Stop any llama-server on our port."""
    kill_tuner_server()
    try:
        subprocess.run(["pkill", "-f", f"llama-server.*--port {SERVER_PORT}"],
                       capture_output=True)
        time.sleep(2)
    except Exception:
        pass


def start_server(params_vec):
    """Start llama-server with given params. Returns True if healthy."""
    global _server_proc
    kill_tuner_server()

    xs = _decode_params(params_vec)
    draft_n_max = int(xs[0]); draft_n_min = int(xs[1])
    draft_p_min = float(xs[2])
    ngram_match = int(xs[3]); ngram_min = int(xs[4]); ngram_max = int(xs[5])
    k4v_n = int(xs[6]); k4v_m = int(xs[7]); k4v_hits = int(xs[8])
    fit_target = int(xs[9])
    batch_size = int(xs[10]); ubatch_size = int(xs[11])

    spec_type = "draft-mtp,ngram-mod"
    if k4v_hits > 0:
        spec_type += ",ngram-map-k4v"

    cmd = [
        LLAMA_SERVER,
        "-m", MODEL,
        "-fitt", str(fit_target),
        "--spec-type", spec_type,
        "--spec-draft-n-max", str(draft_n_max),
        "--spec-draft-n-min", str(draft_n_min),
        "--spec-draft-p-min", str(draft_p_min),
        "--spec-ngram-mod-n-match", str(ngram_match),
        "--spec-ngram-mod-n-min", str(ngram_min),
        "--spec-ngram-mod-n-max", str(ngram_max),
        "--spec-ngram-map-k4v-size-n", str(k4v_n),
        "--spec-ngram-map-k4v-size-m", str(k4v_m),
        "--spec-ngram-map-k4v-min-hits", str(k4v_hits),
        "-c", "8192",
        "-b", str(batch_size), "-ub", str(ubatch_size),
        "-fa", "on",
        "--temp", f"{FIXED_TEMP:.2f}",
        "--top-p", f"{FIXED_TOP_P:.2f}",
        "--top-k", str(FIXED_TOP_K),
        "--min-p", f"{FIXED_MIN_P:.2f}",
        "--presence-penalty", "0.0",
        "--no-mmap", "--mlock", "--no-warmup", "--no-mmproj",
        "--reasoning", "off",
        "--chat-template-kwargs", '{"enable_thinking": false}',
        "--host", "127.0.0.1", "--port", str(SERVER_PORT),
    ]

    with open(SERVER_LOG, "w") as log:
        _server_proc = subprocess.Popen(
            cmd, stdout=log, stderr=log,
            start_new_session=True)

    for _ in range(90):
        time.sleep(2)
        if server_health():
            return True
        if _server_proc.poll() is not None:
            return False
    return False


# ── benchmarking ────────────────────────────────────────────────────

def benchmark_server(prompt=BENCH_PROMPT, max_tokens=BENCH_MAX_TOKENS,
                     timeout=BENCH_TIMEOUT):
    """Send prompt, measure tok/s. Returns dict with metrics or None on error."""
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": FIXED_TEMP,
        "top_p": FIXED_TOP_P,
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
        return {"error": str(e), "wall_s": time.perf_counter() - t0}

    wall_s = time.perf_counter() - t0
    usage = data.get("usage", {})
    # llama.cpp puts timings at top-level or under usage.timings
    timings = data.get("timings") or usage.get("timings", {})
    comp_tokens = usage.get("completion_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)
    tok_s = comp_tokens / wall_s if wall_s > 0 else 0.0

    content = ""
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        pass

    return {
        "wall_s": wall_s,
        "completion_tokens": comp_tokens,
        "prompt_tokens": prompt_tokens,
        "tok_s": tok_s,
        "predicted_tok_s": timings.get("predicted_per_second", 0),
        "prompt_tok_s": timings.get("prompt_per_second", 0),
        "draft_n": timings.get("draft_n", 0),
        "draft_accepted": timings.get("draft_n_accepted", 0),
        "output_chars": len(content),
        "output_hash": hash(content) & 0xFFFFFFFF,
        "timings_source": "top_level" if data.get("timings") else "usage",
    }


# ── baseline measurement ────────────────────────────────────────────

BASELINE = None


def measure_baseline():
    """Measure baseline using tuner's own server with DEFAULTS config.

    Stops any existing server on our port, starts a fresh one with DEFAULTS,
    measures 3 reps, caches, and leaves the server running.
    """
    global BASELINE
    if BASELINE is not None:
        return BASELINE

    print("Measuring baseline (tuner server with DEFAULTS)...")
    kill_tuner_server()
    stop_all_servers()
    time.sleep(3)

    if not start_server(DEFAULTS):
        print("ERROR: failed to start baseline server")
        BASELINE = {"tok_s": 50, "completion_tokens": 300, "predicted_tok_s": 55}
        return BASELINE

    results = []
    for i in range(3):
        r = benchmark_server()
        if r and "error" not in r:
            results.append(r)
            print(f"  baseline run {i+1}: {r['tok_s']:.1f} tok/s, "
                  f"{r['completion_tokens']} tokens, "
                  f"draft_n={r.get('draft_n',0)} "
                  f"draft_accepted={r.get('draft_accepted',0)} "
                  f"predicted={r.get('predicted_tok_s',0):.1f} tok/s, "
                  f"timings_src={r.get('timings_source','?')}")
        time.sleep(1)

    if not results:
        print("ERROR: all baseline measurements failed")
        BASELINE = {"tok_s": 50, "completion_tokens": 300, "predicted_tok_s": 55}
        return BASELINE

    tok_s_vals = [r["tok_s"] for r in results]
    tok_vals = [r["completion_tokens"] for r in results]
    pred_vals = [r.get("predicted_tok_s", 0) for r in results]

    BASELINE = {
        "tok_s": float(np.median(tok_s_vals)),
        "completion_tokens": int(np.median(tok_vals)),
        "predicted_tok_s": float(np.median(pred_vals)),
        "output_hash": results[0].get("output_hash", 0),
        "output_chars": results[0].get("output_chars", 0),
        "repeats": results,
    }
    print(f"  baseline: {BASELINE['tok_s']:.1f} tok/s, "
          f"{BASELINE['completion_tokens']} tokens\n")
    return BASELINE


# ── objective ───────────────────────────────────────────────────────

def _decode_params(vec):
    """Round integer params, clamp to bounds, enforce constraints."""
    xs = np.clip(vec, BOUNDS_LO, BOUNDS_HI).copy()
    xs[IS_INT] = np.round(xs[IS_INT])
    # Enforce ngram_min < ngram_max
    if xs[4] >= xs[5]:
        xs[5] = xs[4] + 8
    # Enforce ubatch <= batch
    if xs[11] > xs[10]:
        xs[11] = xs[10]
    return xs


def _params_to_dict(xs):
    return {PARAM_DEFS[i][0]: (int(xs[i]) if IS_INT[i] else float(xs[i]))
            for i in range(D)}


def evaluate_params(params_vec, result_file=None):
    """Full eval cycle: start server, benchmark, stop. Returns (score, metrics_dict)."""
    xs = _decode_params(params_vec)
    cfg = _params_to_dict(xs)

    t0 = time.perf_counter()
    ok = start_server(xs)
    start_s = time.perf_counter() - t0 if ok else -1

    if not ok:
        kill_tuner_server()
        result = {"error": "server_start_failed", "wall_s": time.perf_counter() - t0,
                  "score": 1e6, "config": cfg}
        _log_result(result, result_file)
        return 1e6, result

    # Warmup
    benchmark_server("Hello", max_tokens=10, timeout=30)

    # Benchmark
    r = benchmark_server()
    kill_tuner_server()

    if r is None or "error" in r:
        result = {"error": r.get("error", "benchmark_failed") if r else "None",
                  "wall_s": time.perf_counter() - t0, "score": 1e6, "config": cfg}
        _log_result(result, result_file)
        return 1e6, result

    bl = measure_baseline()
    tok_s = r["tok_s"]
    tokens = r["completion_tokens"]
    bl_tokens = bl["completion_tokens"]

    # Quality guardrail: require token_ratio >= 0.85
    quality_ratio = tokens / max(bl_tokens, 1)
    if quality_ratio < 0.85:
        # Heavy penalty: score as if tok_s = 0 + penalty proportional to shortfall
        quality_penalty = 100.0 * (0.85 - quality_ratio)
        score = quality_penalty
    else:
        score = -tok_s

    result = {
        "score": float(score),
        "tok_s": tok_s,
        "completion_tokens": tokens,
        "predicted_tok_s": r.get("predicted_tok_s", 0),
        "draft_n": r.get("draft_n", 0),
        "draft_accepted": r.get("draft_accepted", 0),
        "draft_acceptance": (r.get("draft_accepted", 0) / max(r.get("draft_n", 1), 1)),
        "output_chars": r.get("output_chars", 0),
        "output_hash": r.get("output_hash", 0),
        "wall_s": time.perf_counter() - t0,
        "server_start_s": start_s,
        "config": cfg,
        "quality_ratio": quality_ratio,
    }
    _log_result(result, result_file)
    return float(score), result


def _log_result(result, result_file=None):
    path = result_file or RESULT_LOG
    try:
        with open(path, "a") as f:
            f.write(json.dumps(result) + "\n")
    except Exception:
        pass


# ── DE loop (per-dim bounds, mixed int/float) ───────────────────────

def _de_tune(objective_fn, popsize=20, generations=15, seed=42,
             result_file=None):
    """Simple DE/rand/1/bin with per-dimension bounds and jDE adaptation."""
    rng = np.random.Generator(np.random.PCG64(seed))
    P = popsize
    lo = BOUNDS_LO
    hi = BOUNDS_HI

    # Init population
    pop = rng.uniform(0, 1, size=(P, D)) * (hi - lo) + lo
    Fi = np.full(P, 0.5)
    CRi = rng.uniform(0, 1, size=P)
    scores = np.full(P, np.inf)

    for i in range(P):
        scores[i] = objective_fn(pop[i])

    best_idx = np.argmin(scores)
    best_score = scores[best_idx]
    best_x = pop[best_idx].copy()

    for g in range(generations):
        for i in range(P):
            # jDE adaptation
            if rng.random() < 0.1:
                Fi[i] = rng.uniform(0.1, 0.9)
            if rng.random() < 0.1:
                CRi[i] = rng.uniform(0, 1)

            # rand/1: pick 3 distinct != i
            candidates = [j for j in range(P) if j != i]
            r1, r2, r3 = rng.choice(candidates, size=3, replace=False)

            mutant = pop[r1] + Fi[i] * (pop[r2] - pop[r3])
            mutant = np.clip(mutant, lo, hi)

            # Binomial crossover
            jrand = rng.integers(0, D)
            trial = pop[i].copy()
            for j in range(D):
                if j == jrand or rng.random() < CRi[i]:
                    trial[j] = mutant[j]

            trial_score = objective_fn(trial)

            if trial_score < scores[i]:
                pop[i] = trial
                scores[i] = trial_score
                if trial_score < best_score:
                    best_score = trial_score
                    best_x = trial.copy()

        print(f"  gen {g+1}/{generations}: best_score={best_score:.2f} "
              f"(tok_s≈{-best_score:.1f}) pop_best={scores.min():.2f}")

    return best_score, best_x


# ── integration test ────────────────────────────────────────────────

def _test_fake_objective():
    """Dry-run DE with a cheap fake objective — proves bounds, int rounding, etc."""
    print("─── Integration test: fake objective ───")

    call_count = [0]

    def fake_obj(vec):
        call_count[0] += 1
        xs = _decode_params(vec)
        # Verify bounds
        assert (xs >= BOUNDS_LO).all(), f"below lo: {xs[xs < BOUNDS_LO]}"
        assert (xs <= BOUNDS_HI).all(), f"above hi: {xs[xs > BOUNDS_HI]}"
        # Verify int params are integers
        for j in range(D):
            if IS_INT[j]:
                assert xs[j] == round(xs[j]), f"param {j} not int: {xs[j]}"
        # Simple objective: prefer larger draft_n_max, smaller draft_p_min
        return -xs[0] + xs[2] * 0.1

    score, best = _de_tune(fake_obj, popsize=4, generations=2, seed=42)
    assert call_count[0] >= 4 * 3, f"too few evals: {call_count[0]}"
    print(f"  PASS: {call_count[0]} evals, best_score={score:.3f}\n")


def _test_one_real():
    """One real eval using default params — proves server cycle works."""
    print("─── Integration test: 1 real eval ───")
    score, result = evaluate_params(DEFAULTS)
    ok = ("error" not in result and result.get("tok_s", 0) > 0
          and result.get("completion_tokens", 0) > 0)
    print(f"  tok_s={result.get('tok_s',0):.1f} "
          f"tokens={result.get('completion_tokens',0)} "
          f"ok={ok}")
    if not ok:
        print(f"  ERROR: {result.get('error', 'unknown')}")
        sys.exit(1)
    print("  PASS\n")


def _test_de_smoke():
    """2-gen DE with 3-pop on fake objective, then 1 real eval with defaults."""
    _test_fake_objective()
    _test_one_real()
    print("─── All integration tests passed ───")


# ── CLI ─────────────────────────────────────────────────────────────

def _main():
    parser = argparse.ArgumentParser(
        description="DE tuner for llama.cpp MTP spec decoding")
    parser.add_argument("--test", action="store_true",
                        help="Integration test: fake DE + 1 real eval")
    parser.add_argument("--quick", action="store_true",
                        help="Quick optimization (pop=10, gen=5, ~40 min)")
    parser.add_argument("--tune", action="store_true",
                        help="Full optimization (pop=20, gen=15, ~4 hrs)")
    args = parser.parse_args()

    if not (args.test or args.quick or args.tune):
        parser.print_help()
        return

    # Save original server state
    was_running = server_health()
    print(f"Server running before experiment: {was_running}")

    if args.test:
        # Fake objective test first (no server needed)
        _test_fake_objective()
        # Stop user server, run real eval, restore
        try:
            if was_running:
                print("Stopping user server for real eval test...")
                stop_all_servers()
                time.sleep(3)
            _test_one_real()
        finally:
            kill_tuner_server()
            if was_running:
                print("Restoring user server...")
                subprocess.run([RESTORE_SCRIPT, "32768", "off"],
                               capture_output=True, timeout=120)
                for _ in range(45):
                    time.sleep(2)
                    if server_health():
                        print("  User server restored.")
                        break
        return

    # Stop user server, measure baseline with tuner's own DEFAULTS server
    if was_running:
        print("Stopping user server...")
    stop_all_servers()
    time.sleep(3)

    try:
        baseline = measure_baseline()
        orig_tok_s = baseline["tok_s"]

        popsize = 10 if args.quick else 20
        generations = 5 if args.quick else 15
        n_evals = popsize * (1 + generations)
        est_min = n_evals * 0.75
        print(f"\nDE MTP tuner: pop={popsize} gen={generations}")
        print(f"Evaluations: {popsize} initial + {popsize}×{generations} = {n_evals}")
        print(f"Estimated time: ~{est_min:.0f} min")
        print(f"Results log: {RESULT_LOG}\n")

        obj_wrapper = lambda v: evaluate_params(v, result_file=RESULT_LOG)[0]
        score, best_x = _de_tune(obj_wrapper, popsize=popsize,
                                 generations=generations, seed=42,
                                 result_file=RESULT_LOG)

        # Final benchmark of best config
        print(f"\n{'='*60}")
        print(f"Final benchmark of best config...")
        print(f"Baseline tok/s: {orig_tok_s:.1f}")
        stop_all_servers()
        time.sleep(2)

        _, result = evaluate_params(best_x)
        best_tok_s = result.get("tok_s", 0)
        speedup = best_tok_s / orig_tok_s if orig_tok_s > 0 else 1.0

        xs = _decode_params(best_x)
        print(f"\nBest config (tok_s={best_tok_s:.1f}, {speedup:.2f}x vs baseline):")
        for i in range(D):
            val = int(xs[i]) if IS_INT[i] else float(xs[i])
            print(f"  {PARAM_DEFS[i][0]:<20} = {val}")
        print(f"\nResults: {RESULT_LOG}")

    finally:
        kill_tuner_server()
        if was_running:
            print("Restoring user server...")
            subprocess.run([RESTORE_SCRIPT, "32768", "off"],
                           capture_output=True, timeout=120)
            for _ in range(45):
                time.sleep(2)
                if server_health():
                    print("User server restored.")
                    break


if __name__ == "__main__":
    _main()

"""Benchmark runner for DE-trained NN experiments.

Matches the C de_seq.c task configurations (same target functions,
same architecture, same DE budgets).  Data draws are from independent
RNG streams; exact numerical comparison with C requires a fixture export.
"""

import argparse
import math
import sys
import time

import torch

from . import mlp
from . import tasks as task_mod
from .de import de_solve


def _lpsr_population_size(initial, min_pop, generations, gen):
    """Duplicate of de._lpsr_population_size for eval budget computation."""
    if min_pop >= initial or generations <= 0:
        return initial
    t = (gen + 1) / generations
    nxt = int(initial - (initial - min_pop) * t + 0.5)
    if nxt < min_pop:
        nxt = min_pop
    if nxt > initial:
        nxt = initial
    return nxt


def eval_budget(initial_popsize, min_popsize, generations):
    """Compute total trial evaluations per run (matching C)."""
    evals = initial_popsize  # initial population
    active = initial_popsize
    for g in range(generations):
        evals += active  # one trial per active member
        active = _lpsr_population_size(initial_popsize, min_popsize,
                                       generations, g)
    return evals


def target_variance(targets):
    """Population variance of targets (matching C var/total, not sample var)."""
    return float(torch.var(targets, correction=0))


def mse_loss(predictions, targets):
    """MSE loss over all outputs. predictions: [P, S, O], targets: [S]"""
    t = targets.view(1, -1, 1)
    e = predictions - t
    return (e * e).mean(dim=(1, 2))  # [P]


def huber_loss(predictions, targets, delta=0.5):
    """Huber loss with configurable delta. predictions: [P, S, O], targets: [S]"""
    t = targets.view(1, -1, 1)
    e = predictions - t
    ae = torch.abs(e)
    mse_part = 0.5 * e * e
    linear_part = delta * (ae - 0.5 * delta)
    return torch.where(ae <= delta, mse_part, linear_part).mean(dim=(1, 2))  # [P]


# Benchmark configurations matching C TASKS[] entries
BENCHMARKS = {
    "pair1": {
        "task": "pair1",
        "hidden_layers": 1,
        "hidden": 12,
        "popsize": 384,
        "min_popsize": 192,
        "generations": 6000,
        "strategy": 0,       # rand/1
        "pbest_percent": 0,
        "loss": "mse",
        "huber_delta": None,
        "bounds": (-5.0, 5.0),
    },
    "pair1-best1": {
        "task": "pair1",
        "hidden_layers": 1,
        "hidden": 12,
        "popsize": 384,
        "min_popsize": 192,
        "generations": 6000,
        "strategy": 1,       # best/1
        "pbest_percent": 0,
        "loss": "mse",
        "huber_delta": None,
        "bounds": (-5.0, 5.0),
    },
    "pair1-basepbest50": {
        "task": "pair1",
        "hidden_layers": 1,
        "hidden": 12,
        "popsize": 384,
        "min_popsize": 192,
        "generations": 6000,
        "strategy": 5,       # basepbest/1
        "pbest_percent": 50,
        "loss": "mse",
        "huber_delta": None,
        "bounds": (-5.0, 5.0),
    },
    "pair5-huber0.5": {
        "task": "pair5-huber0.5",
        "hidden_layers": 1,
        "hidden": 16,
        "popsize": 768,
        "min_popsize": 384,
        "generations": 12000,
        "strategy": 0,       # rand/1
        "pbest_percent": 0,
        "loss": "huber",
        "huber_delta": 0.5,
        "bounds": (-5.0, 5.0),
    },
    "pair5-huber0.5-basepbest50": {
        "task": "pair5-huber0.5",
        "hidden_layers": 1,
        "hidden": 16,
        "popsize": 768,
        "min_popsize": 384,
        "generations": 12000,
        "strategy": 5,
        "pbest_percent": 50,
        "loss": "huber",
        "huber_delta": 0.5,
        "bounds": (-5.0, 5.0),
    },
    "pair5-hard": {
        "task": "pair5-hard",
        "hidden_layers": 1,
        "hidden": 16,
        "popsize": 768,
        "min_popsize": 384,
        "generations": 12000,
        "strategy": 0,
        "pbest_percent": 0,
        "loss": "mse",
        "huber_delta": None,
        "bounds": (-5.0, 5.0),
    },
    "pair5-hard-basepbest50": {
        "task": "pair5-hard",
        "hidden_layers": 1,
        "hidden": 16,
        "popsize": 768,
        "min_popsize": 384,
        "generations": 12000,
        "strategy": 5,
        "pbest_percent": 50,
        "loss": "mse",
        "huber_delta": None,
        "bounds": (-5.0, 5.0),
    },
    # Scale benchmarks: same task, larger architectures
    "pair5-hard-wider": {
        "task": "pair5-hard",
        "hidden_layers": 1,
        "hidden": 64,
        "popsize": 768,
        "min_popsize": 384,
        "generations": 12000,
        "strategy": 0,
        "pbest_percent": 0,
        "loss": "mse",
        "huber_delta": None,
        "bounds": (-5.0, 5.0),
    },
    "pair5-hard-deeper": {
        "task": "pair5-hard",
        "hidden_layers": 2,
        "hidden": 64,
        "popsize": 768,
        "min_popsize": 384,
        "generations": 12000,
        "strategy": 0,
        "pbest_percent": 0,
        "loss": "mse",
        "huber_delta": None,
        "bounds": (-5.0, 5.0),
    },
    "pair5-hard-wider-basepbest50": {
        "task": "pair5-hard",
        "hidden_layers": 1,
        "hidden": 64,
        "popsize": 768,
        "min_popsize": 384,
        "generations": 12000,
        "strategy": 5,
        "pbest_percent": 50,
        "loss": "mse",
        "huber_delta": None,
        "bounds": (-5.0, 5.0),
    },
    "pair5-hard-deeper-basepbest50": {
        "task": "pair5-hard",
        "hidden_layers": 2,
        "hidden": 64,
        "popsize": 768,
        "min_popsize": 384,
        "generations": 12000,
        "strategy": 5,
        "pbest_percent": 50,
        "loss": "mse",
        "huber_delta": None,
        "bounds": (-5.0, 5.0),
    },
}


def run_benchmark(name, runs=5, device="cpu", dtype=torch.float32,
                  train_samples=512, test_samples=256):
    """Run a named benchmark and return results dict."""
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {name}. "
                         f"Available: {list(BENCHMARKS.keys())}")

    cfg = BENCHMARKS[name]
    task_name = cfg["task"]
    task_spec = task_mod.TASK_SPECS[task_name]
    n_dims = task_spec["n_dims"]
    target_fn = task_spec["target_fn"]
    hidden_layers = cfg["hidden_layers"]
    hidden = cfg["hidden"]
    outputs = task_spec["outputs"]
    popsize = cfg["popsize"]
    min_popsize = cfg["min_popsize"]
    generations = cfg["generations"]
    strategy = cfg["strategy"]
    pbest_pct = cfg["pbest_percent"]
    loss_type = cfg["loss"]
    huber_delta = cfg["huber_delta"]
    bounds_min, bounds_max = cfg["bounds"]

    model = mlp.MLP(n_dims, hidden_layers, hidden, outputs)
    D = model.total_weights

    # Generate data once (deterministic per task, independent of C RNG)
    train_in, train_tgt = task_mod.generate_data(
        train_samples, n_dims, target_fn, device=device, dtype=dtype,
        seed=0xD1FF3A9E2B1C4D77)
    test_in, test_tgt = task_mod.generate_data(
        test_samples, n_dims, target_fn, device=device, dtype=dtype,
        seed=0x7E57C0FFEE5EED)

    test_var = target_variance(test_tgt)

    budget = eval_budget(popsize, min_popsize, generations)

    print(f"{name}: inputs={n_dims} hidden={hidden} outputs={outputs} "
          f"weights={D} pop={popsize}->{min_popsize} gens={generations} "
          f"evals/run={budget} runs={runs}")
    print(f"  device={device} dtype={dtype} activation=sigmoid "
          f"(same target family, independent data draw vs C)")
    print(f"{'run':<5} {'loss':<14} {'ms':<10}")

    def make_objective():
        return lambda pop: _compute_loss(
            pop, model, train_in, train_tgt, loss_type, huber_delta)

    best_loss = float("inf")
    best_seed_val = 0
    best_ms = 0
    best_weights = torch.zeros(D, dtype=dtype, device=device)
    run_weights = torch.zeros(D, dtype=dtype, device=device)

    wall_t0 = time.time()
    base_seed = 0xD1FF3A9E2B1C4D77

    for r in range(runs):
        seed = base_seed + r * 0x9e3779b97f4a7c15
        objective = make_objective()

        t0 = time.time()
        loss = de_solve(
            D, popsize, min_popsize, generations,
            bounds_min, bounds_max,
            objective, seed, run_weights,
            mutation_strategy=strategy,
            pbest_percent=pbest_pct,
            device=device,
            dtype=dtype,
        )
        t1 = time.time()
        ms = (t1 - t0) * 1000

        if loss < best_loss:
            best_loss = loss
            best_seed_val = seed
            best_ms = ms
            best_weights.copy_(run_weights)

        print(f"{r:<5} {loss:<14.8g} {ms:<10.0f}")

    wall_t1 = time.time()
    wall_ms = (wall_t1 - wall_t0) * 1000

    print(f"best loss {best_loss:.8g} seed 0x{best_seed_val:016x} "
          f"ms {best_ms:.0f}  wall {wall_ms:.0f}ms")

    # Evaluate best model on test set
    with torch.no_grad():
        train_pred = model.forward(best_weights, train_in)
        test_pred = model.forward(best_weights, test_in)

        test_mse = float(mse_loss(test_pred.unsqueeze(0), test_tgt))
        train_mse = float(mse_loss(train_pred.unsqueeze(0), train_tgt))

    print(f"train MSE {train_mse:.8g}")
    print(f"test  MSE {test_mse:.8g}")
    if test_var > 0:
        print(f"test  nMSE {test_mse / test_var:.8g}  target_var {test_var:.8g}")
    print()

    return {
        "name": name,
        "best_loss": best_loss,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "test_nmse": test_mse / test_var if test_var > 0 else float("inf"),
        "target_var": test_var,
        "wall_ms": wall_ms,
        "best_ms": best_ms,
        "weights": best_weights,
        "evals_per_run": budget,
    }


def _compute_loss(pop, model, inputs, targets, loss_type, huber_delta=None):
    """Objective function for DE: population -> per-member loss."""
    preds = model.population_forward(pop, inputs)  # [P, S, O]
    if loss_type == "huber":
        return huber_loss(preds, targets, huber_delta)
    else:
        return mse_loss(preds, targets)


def _main():
    parser = argparse.ArgumentParser(
        description="PyTorch DE benchmark runner")
    parser.add_argument("benchmarks", nargs="*",
                        help="Benchmark names to run, e.g. pair1 pair1-best1")
    parser.add_argument("--device", default="cuda",
                        help="Device: cuda or cpu (default: cuda)")
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "float64"],
                        help="Tensor dtype (default: float32)")
    parser.add_argument("--runs", type=int, default=5,
                        help="Runs per benchmark (default: 5)")
    parser.add_argument("--train-samples", type=int, default=512,
                        help="Training samples (default: 512)")
    parser.add_argument("--test-samples", type=int, default=256,
                        help="Test samples (default: 256)")
    parser.add_argument("--list", action="store_true",
                        help="List available benchmarks and exit")
    args = parser.parse_args()

    if args.list:
        print("Available benchmarks:")
        for name in BENCHMARKS:
            cfg = BENCHMARKS[name]
            print(f"  {name:<32} strategy={cfg['strategy']} "
                  f"loss={cfg['loss']} pop={cfg['popsize']}->{cfg['min_popsize']}")
        return

    if not args.benchmarks:
        parser.print_help()
        sys.exit(1)

    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: --device cuda but torch.cuda.is_available() == False")
        print("  Use --device cpu or install a CUDA-capable PyTorch.")
        sys.exit(1)

    unknown = [b for b in args.benchmarks if b not in BENCHMARKS]
    if unknown:
        print(f"Unknown benchmarks: {unknown}")
        print(f"Available: {list(BENCHMARKS.keys())}")
        sys.exit(1)

    print("=" * 80)
    print(f"PyTorch DE Benchmarks: device={args.device} dtype={args.dtype} "
          f"runs={args.runs}")
    print("=" * 80)
    print()

    all_results = []
    total_wall = 0.0

    for name in args.benchmarks:
        t0 = time.time()
        result = run_benchmark(name, runs=args.runs, device=args.device,
                               dtype=dtype,
                               train_samples=args.train_samples,
                               test_samples=args.test_samples)
        t1 = time.time()
        w = (t1 - t0) * 1000
        total_wall += w
        all_results.append(result)

    # Summary table
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f'{"Task":<30} {"Train MSE":<14} {"Test MSE":<14} '
          f'{"nMSE":<12} {"Wall (ms)":<12}')
    print("-" * 82)
    for r in all_results:
        print(f'{r["name"]:<30} {r["train_mse"]:<14.8g} {r["test_mse"]:<14.8g} '
              f'{r["test_nmse"]:<12.8g} {r["wall_ms"]:<12.0f}')
    print("-" * 82)
    print(f"Total wall: {total_wall:.0f}ms ({total_wall / 1000:.1f}s)")
    print()
    print("NOTE: Same target/model/DE budget family as C de_seq.c, independent data")
    print("      draws (RNG differs, torch sigmoid vs C cached-sigmoid LUT).")
    print("      Not directly comparable to C leaderboard without fixture export.")


if __name__ == "__main__":
    _main()

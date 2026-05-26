"""Benchmark runner for DE-trained NN experiments.

Matches the C de_seq.c task configurations.
"""

import math
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
    """Compute variance of target values for nMSE normalization."""
    return float(torch.var(targets))


def mse_loss(predictions, targets):
    """MSE loss over all outputs. predictions: [P, S, O], targets: [S]"""
    # targets: [S] -> [1, S, 1]
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

    model = mlp.MLP(n_dims, hidden_layers, hidden, outputs,
                    activation=torch.tanh)
    D = model.total_weights

    # Generate data once (deterministic per task)
    train_in, train_tgt = task_mod.generate_data(
        train_samples, n_dims, target_fn, device=device, dtype=dtype,
        seed=0xD1FF3A9E2B1C4D77)
    test_in, test_tgt = task_mod.generate_data(
        test_samples, n_dims, target_fn, device=device, dtype=dtype,
        seed=0x7E57C0FFEE5EED)

    # also generate separate test data with a different seed for unbiased evaluation
    # for now use same as train generation pattern in C
    test_var = target_variance(test_tgt)

    budget = eval_budget(popsize, min_popsize, generations)

    print(f"{name}: inputs={n_dims} hidden={hidden} outputs={outputs} "
          f"weights={D} pop={popsize}->{min_popsize} gens={generations} "
          f"evals/run={budget} runs={runs}")
    print(f"  device={device} dtype={dtype}")
    print(f"{'run':<5} {'loss':<14} {'ms':<10}")

    # We need to pass n_dims, hidden_layers, hidden, outputs to the objective
    # The objective receives a flat population tensor [P, D] and returns [P] losses

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

        if loss_type == "huber":
            train_loss = float(huber_loss(
                train_pred.unsqueeze(0), train_tgt, huber_delta))
        else:
            train_loss = float(mse_loss(
                train_pred.unsqueeze(0), train_tgt))

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
    """Objective function for DE: population -> per-member loss.

    Args:
        pop: [P, D] population tensor
        model: MLP instance
        inputs: [S, I]
        targets: [S]
        loss_type: "mse" or "huber"
        huber_delta: delta for Huber loss

    Returns:
        [P] loss per population member
    """
    preds = model.population_forward(pop, inputs)  # [P, S, O]
    if loss_type == "huber":
        return huber_loss(preds, targets, huber_delta)
    else:
        return mse_loss(preds, targets)

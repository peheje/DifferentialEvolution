"""Memetic DE experiment: DE proposals plus short Adam polishing.

This is intentionally a small experiment runner, not a general solver.
It compares, under a hard wall-clock cap:

  - Adam-only
  - DE-only basepbest50
  - DE basepbest50 + K Adam polish steps on the top-Q trial population

The default task is pair5-hard with a 5->64->1 sigmoid MLP.
"""

import argparse
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from . import mlp
from . import tasks as task_mod
from .adam_baseline import AdamMLP
from .bench import mse_loss, target_variance
from .de import (
    JDE_F_HI,
    JDE_F_LO,
    JDE_TAU_CR,
    JDE_TAU_F,
    _lpsr_population_size,
    _pick_all_indices,
)


def get_data(task_name, n_train, n_test, device, dtype):
    spec = task_mod.TASK_SPECS[task_name]
    train_in, train_tgt = task_mod.generate_data(
        n_train,
        spec["n_dims"],
        spec["target_fn"],
        seed=0xD1FF3A9E2B1C4D77,
        device=device,
        dtype=dtype,
    )
    test_in, test_tgt = task_mod.generate_data(
        n_test,
        spec["n_dims"],
        spec["target_fn"],
        seed=0x7E57C0FFEE5EED,
        device=device,
        dtype=dtype,
    )
    return train_in, train_tgt, test_in, test_tgt


def eval_weights(model, weights, train_in, train_tgt, test_in, test_tgt):
    with torch.no_grad():
        train_pred = model.forward(weights, train_in)
        test_pred = model.forward(weights, test_in)
        train_mse = float(mse_loss(train_pred.unsqueeze(0), train_tgt)[0])
        test_mse = float(mse_loss(test_pred.unsqueeze(0), test_tgt)[0])
    return train_mse, test_mse


def pop_loss(pop, model, train_in, train_tgt):
    return mse_loss(model.population_forward(pop, train_in), train_tgt)


def adam_polish_population(pop, model, train_in, train_tgt, steps, lr):
    """Return a polished copy of pop using independent local Adam state."""
    if steps <= 0:
        return pop

    x = pop.detach().clone().requires_grad_(True)
    m = torch.zeros_like(x)
    v = torch.zeros_like(x)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    for t in range(1, steps + 1):
        if x.grad is not None:
            x.grad = None
        # Sum gives each candidate the same gradient it would get if polished
        # independently, while still using one batched forward/backward.
        loss = pop_loss(x, model, train_in, train_tgt).sum()
        loss.backward()
        with torch.no_grad():
            g = x.grad
            m.mul_(beta1).add_(g, alpha=1.0 - beta1)
            v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
            mh = m / (1.0 - beta1 ** t)
            vh = v / (1.0 - beta2 ** t)
            x.addcdiv_(mh, torch.sqrt(vh).add_(eps), value=-lr)
            x.clamp_(-5.0, 5.0)

    return x.detach()


def run_adam_only(task, arch, train_in, train_tgt, test_in, test_tgt,
                  test_var, wall_s, lr, seed, device, dtype):
    torch.manual_seed(seed)
    model = AdamMLP(arch).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best = {
        "train_mse": float("inf"),
        "test_mse": float("inf"),
        "step": 0,
    }
    steps = 0
    start = time.perf_counter()
    while True:
        if time.perf_counter() - start >= wall_s:
            break
        steps += 1
        optimizer.zero_grad(set_to_none=True)
        pred = model(train_in).squeeze(-1)
        loss = criterion(pred, train_tgt)
        loss.backward()
        optimizer.step()

        if steps % 50 == 0:
            with torch.no_grad():
                train_pred = model(train_in).squeeze(-1)
                test_pred = model(test_in).squeeze(-1)
                train_mse = float(nn.functional.mse_loss(train_pred, train_tgt))
                test_mse = float(nn.functional.mse_loss(test_pred, test_tgt))
            if test_mse < best["test_mse"]:
                best.update(train_mse=train_mse, test_mse=test_mse, step=steps)

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "method": "adam",
        "train_mse": best["train_mse"],
        "test_mse": best["test_mse"],
        "nmse": best["test_mse"] / test_var,
        "wall_s": elapsed,
        "steps": steps,
        "best_step": best["step"],
        "steps_per_sec": steps / elapsed if elapsed > 0 else float("inf"),
    }


def run_de_memetic(task, arch, train_in, train_tgt, test_in, test_tgt,
                   test_var, wall_s, seed, device, dtype, polish_steps,
                   polish_top_q, polish_lr, popsize, min_popsize,
                   pbest_percent):
    rng = np.random.Generator(np.random.PCG64(seed))
    torch.manual_seed(int(seed))

    model = mlp.MLP(arch[0], len(arch) - 2, arch[1], arch[-1])
    D = model.total_weights
    P0 = popsize
    P = popsize
    generation = 0

    pop_np = rng.uniform(-5.0, 5.0, size=(P, D))
    pop = torch.from_numpy(pop_np).to(device=device, dtype=dtype)
    Fi = torch.from_numpy(
        rng.uniform(JDE_F_LO, JDE_F_HI, size=P)
    ).to(device=device, dtype=dtype)
    CRi = torch.from_numpy(rng.uniform(0.0, 1.0, size=P)).to(
        device=device, dtype=dtype
    )
    scores = pop_loss(pop, model, train_in, train_tgt)

    best_idx = int(torch.argmin(scores).item())
    best_weights = pop[best_idx].detach().clone()
    best_train = float(scores[best_idx])
    best_test = eval_weights(model, best_weights, train_in, train_tgt,
                             test_in, test_tgt)[1]
    best_generation = 0

    start = time.perf_counter()
    trial_evals = P
    polished_evals = 0

    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= wall_s:
            break

        active = P
        sorted_cpu = torch.argsort(scores).cpu().numpy()
        pbest_count = max(1, (active * pbest_percent + 99) // 100)

        new_F = JDE_F_LO + torch.rand(active, device=device, dtype=dtype) * (
            JDE_F_HI - JDE_F_LO
        )
        new_CR = torch.rand(active, device=device, dtype=dtype)
        re_F = torch.rand(active, device=device) < JDE_TAU_F
        re_CR = torch.rand(active, device=device) < JDE_TAU_CR
        Fi_try = torch.where(re_F, new_F, Fi)
        CRi_try = torch.where(re_CR, new_CR, CRi)

        pbest_sel = sorted_cpu[rng.integers(0, pbest_count, size=active)]
        if pbest_count > 1:
            row_idx = np.arange(active)
            self_mask = pbest_sel == row_idx
            for _ in range(8):
                if not self_mask.any():
                    break
                pbest_sel[self_mask] = sorted_cpu[
                    rng.integers(0, pbest_count, size=self_mask.sum())
                ]
                self_mask = pbest_sel == row_idx

        idx_all = _pick_all_indices(rng, active, 2, pbest_sel)
        pb_t = torch.from_numpy(pbest_sel).to(device=device)
        r1_t = torch.from_numpy(idx_all[:, 0]).to(device=device)
        r2_t = torch.from_numpy(idx_all[:, 1]).to(device=device)

        mutant = pop[pb_t] + Fi_try.unsqueeze(1) * (pop[r1_t] - pop[r2_t])
        cross_mask = torch.rand(active, D, dtype=dtype, device=device) < CRi_try.unsqueeze(1)
        jrand_vals = torch.randint(0, D, (active,), device=device)
        cross_mask[torch.arange(active, device=device), jrand_vals] = True
        trials = torch.where(cross_mask, mutant, pop).clamp_(-5.0, 5.0)

        trial_scores = pop_loss(trials, model, train_in, train_tgt)
        trial_evals += active

        if polish_steps > 0 and polish_top_q > 0:
            # Polish only promising trials, then keep polished candidates only
            # when their train loss beats the unpolished trial loss.
            if time.perf_counter() - start >= wall_s:
                break
            q = min(polish_top_q, active)
            top_trial_idx = torch.argsort(trial_scores)[:q]
            polished = adam_polish_population(
                trials[top_trial_idx], model, train_in, train_tgt,
                polish_steps, polish_lr
            )
            polished_scores = pop_loss(polished, model, train_in, train_tgt)
            polished_evals += q * polish_steps
            better_polished = polished_scores < trial_scores[top_trial_idx]
            if better_polished.any():
                replace_idx = top_trial_idx[better_polished]
                trials[replace_idx] = polished[better_polished]
                trial_scores[replace_idx] = polished_scores[better_polished]

        improved = trial_scores < scores
        improved_idx = improved.nonzero(as_tuple=False).squeeze(-1)
        if improved_idx.numel() > 0:
            pop[improved_idx] = trials[improved_idx]
            scores[improved_idx] = trial_scores[improved_idx]
            Fi[improved_idx] = Fi_try[improved_idx]
            CRi[improved_idx] = CRi_try[improved_idx]

        generation += 1
        cur_best_idx = int(torch.argmin(scores).item())
        cur_best_train = float(scores[cur_best_idx])
        if cur_best_train < best_train:
            cand = pop[cur_best_idx].detach().clone()
            cand_train, cand_test = eval_weights(
                model, cand, train_in, train_tgt, test_in, test_tgt
            )
            best_weights = cand
            best_train = cand_train
            best_test = cand_test
            best_generation = generation

        next_p = _lpsr_population_size(P0, min_popsize, 12000, generation)
        if next_p < P:
            keep = torch.argsort(scores)[:next_p]
            pop = pop[keep]
            scores = scores[keep]
            Fi = Fi[keep]
            CRi = CRi[keep]
            P = next_p

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "method": (
            f"de+adam{polish_steps}top{polish_top_q}"
            if polish_steps and polish_top_q else "de"
        ),
        "train_mse": best_train,
        "test_mse": best_test,
        "nmse": best_test / test_var,
        "wall_s": elapsed,
        "generations": generation,
        "candidate_evals": trial_evals,
        "polished_evals": polished_evals,
        "best_generation": best_generation,
        "evals_per_sec": trial_evals / elapsed if elapsed > 0 else float("inf"),
    }


def run_suite(args):
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"

    arch = [int(x.strip()) for x in args.arch.split(",")]
    train_in, train_tgt, test_in, test_tgt = get_data(
        args.task, args.train_samples, args.test_samples, device, dtype
    )
    test_var = target_variance(test_tgt)

    methods = []
    if args.methods in ("all", "adam"):
        methods.append(("adam", None))
    if args.methods in ("all", "de"):
        methods.append(("de", 0))
    if args.methods in ("all", "hybrid"):
        for k in args.polish_steps:
            for q in args.polish_top_q:
                methods.append(("hybrid", (k, q)))

    print("=" * 88)
    print("Memetic DE Experiment")
    print("=" * 88)
    print(f"task={args.task} arch={'->'.join(map(str, arch))} "
          f"device={device} dtype={dtype} wall_cap={args.wall_seconds}s")
    print(f"train/test={args.train_samples}/{args.test_samples} "
          f"seed={args.seed} pbest={args.pbest_percent} pop={args.popsize}->{args.min_popsize}")
    print()

    rows = []
    for kind, k in methods:
        if kind == "adam":
            row = run_adam_only(
                args.task, arch, train_in, train_tgt, test_in, test_tgt,
                test_var, args.wall_seconds, args.adam_lr, args.seed,
                device, dtype,
            )
        else:
            polish_steps = k[0] if isinstance(k, tuple) else k
            polish_top_q = k[1] if isinstance(k, tuple) else args.polish_top_q[0]
            row = run_de_memetic(
                args.task, arch, train_in, train_tgt, test_in, test_tgt,
                test_var, args.wall_seconds, args.seed, device, dtype,
                polish_steps, polish_top_q, args.polish_lr, args.popsize,
                args.min_popsize, args.pbest_percent,
            )
        rows.append(row)
        print_row(row)

    print()
    print(f"{'method':<12} {'train_mse':>11} {'test_mse':>11} {'nMSE':>11} "
          f"{'wall_s':>8} {'work':>16} {'speed':>14}")
    print("-" * 88)
    for row in rows:
        work = row.get("steps", row.get("candidate_evals", 0))
        speed = row.get("steps_per_sec", row.get("evals_per_sec", 0.0))
        print(f"{row['method']:<12} {row['train_mse']:>11.5g} "
              f"{row['test_mse']:>11.5g} {row['nmse']:>11.5g} "
              f"{row['wall_s']:>8.2f} {work:>16} {speed:>14.1f}")
    return rows


def print_row(row):
    if row["method"] == "adam":
        print(f"{row['method']}: train={row['train_mse']:.6g} "
              f"test={row['test_mse']:.6g} nMSE={row['nmse']:.6g} "
              f"wall={row['wall_s']:.2f}s steps={row['steps']} "
              f"best_step={row['best_step']}")
    else:
        print(f"{row['method']}: train={row['train_mse']:.6g} "
              f"test={row['test_mse']:.6g} nMSE={row['nmse']:.6g} "
              f"wall={row['wall_s']:.2f}s gens={row['generations']} "
              f"evals={row['candidate_evals']} polished={row['polished_evals']} "
              f"best_gen={row['best_generation']}")


def _main():
    parser = argparse.ArgumentParser(description="Memetic DE + Adam polish experiment")
    parser.add_argument("--task", default="pair5-hard")
    parser.add_argument("--arch", default="5,64,1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--wall-seconds", type=float, default=60.0)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", default="all", choices=["all", "adam", "de", "hybrid"])
    parser.add_argument("--polish-steps", type=int, nargs="+", default=[5])
    parser.add_argument("--polish-top-q", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--adam-lr", type=float, default=3e-3)
    parser.add_argument("--polish-lr", type=float, default=1e-3)
    parser.add_argument("--popsize", type=int, default=384)
    parser.add_argument("--min-popsize", type=int, default=192)
    parser.add_argument("--pbest-percent", type=int, default=50)
    args = parser.parse_args()
    run_suite(args)


if __name__ == "__main__":
    _main()

"""Block-wise DE experiment: full-vector DE vs block-wise DE vs Adam.

Compares under a hard wall-clock cap on pair5-hard 5->64->1.

Variants:
  - full: standard basepbest50 full-vector DE
  - block-N: basepbest50 with block_size=N (mutation constrained to one block/trial)
  - block-N-slice: block-wise + 30% hard-example slice scoring
  - adam: Adam reference

CLI:
  python -m torch_de.block_de_experiment --device cuda --wall-seconds 60
"""

import argparse
import math
import sys
import time

import numpy as np
import torch

from . import mlp
from . import tasks as task_mod
from .bench import mse_loss, target_variance
from .de import (
    JDE_F_HI,
    JDE_F_LO,
    JDE_TAU_CR,
    JDE_TAU_F,
    _pick_all_indices,
)


# ── data ────────────────────────────────────────────────────────────

def get_data(task_name, n_train, n_test, device, dtype):
    spec = task_mod.TASK_SPECS[task_name]
    train_in, train_tgt = task_mod.generate_data(
        n_train, spec["n_dims"], spec["target_fn"],
        seed=0xD1FF3A9E2B1C4D77, device=device, dtype=dtype)
    test_in, test_tgt = task_mod.generate_data(
        n_test, spec["n_dims"], spec["target_fn"],
        seed=0x7E57C0FFEE5EED, device=device, dtype=dtype)
    return train_in, train_tgt, test_in, test_tgt


def eval_single(model, weights, train_in, train_tgt, test_in, test_tgt):
    with torch.no_grad():
        train_pred = model.forward(weights, train_in)
        test_pred = model.forward(weights, test_in)
        train_mse = float(mse_loss(train_pred.unsqueeze(0), train_tgt)[0])
        test_mse = float(mse_loss(test_pred.unsqueeze(0), test_tgt)[0])
    return train_mse, test_mse


def pop_loss_obj(pop, model, train_in, train_tgt):
    return mse_loss(model.population_forward(pop, train_in), train_tgt)


# ── block mask helper ───────────────────────────────────────────────

def _gen_block_mask(active, D, block_size, device):
    """Generate [active, D] bool mask with one random contiguous block per row."""
    if block_size <= 0 or block_size >= D:
        return None
    offsets = torch.randint(0, D - block_size + 1, (active,), device=device)
    mask = torch.zeros(active, D, dtype=torch.bool, device=device)
    idx_d = torch.arange(block_size, device=device)
    mask[torch.arange(active, device=device).unsqueeze(1),
         (offsets.unsqueeze(1) + idx_d).clamp(0, D - 1)] = True
    return mask


# ── slice-aware scoring ─────────────────────────────────────────────

def _slice_scores(trials, model, train_in, train_tgt, best_weights,
                  slice_frac=0.25, slice_weight=0.3):
    """Compute mixed score: (1-w)*full_mse + w*hard_slice_mse.

    Hard slice = top slice_frac samples by |error| under best_weights.
    """
    with torch.no_grad():
        best_pred = model.forward(best_weights, train_in)  # [S]
        best_err = (best_pred.squeeze(-1) - train_tgt).abs()

    n_hard = max(1, int(train_in.shape[0] * slice_frac))
    _, hard_idx = torch.topk(best_err, n_hard)
    hard_in = train_in[hard_idx]
    hard_tgt = train_tgt[hard_idx]

    full_loss = mse_loss(model.population_forward(trials, train_in), train_tgt)
    hard_loss = mse_loss(model.population_forward(trials, hard_in), hard_tgt)

    return (1.0 - slice_weight) * full_loss + slice_weight * hard_loss


# ── DE wall-clock runner ────────────────────────────────────────────

def run_de_wallclock(model, D, train_in, train_tgt, test_in, test_tgt,
                     test_var, wall_s, seed, device, dtype,
                     block_size=0, slice_scoring=False,
                     popsize=384, min_popsize=192, pbest_percent=50):
    """Run DE until wall clock expires. Returns best metrics dict."""
    rng = np.random.Generator(np.random.PCG64(seed))
    torch.manual_seed(int(seed) & 0x7FFF_FFFF_FFFF_FFFF)

    P = popsize
    pop_np = rng.uniform(-5.0, 5.0, size=(P, D))
    pop = torch.from_numpy(pop_np).to(device=device, dtype=dtype)
    Fi = torch.from_numpy(rng.uniform(JDE_F_LO, JDE_F_HI, size=P)).to(
        device=device, dtype=dtype)
    CRi = torch.from_numpy(rng.uniform(0.0, 1.0, size=P)).to(
        device=device, dtype=dtype)
    scores = pop_loss_obj(pop, model, train_in, train_tgt)

    best_idx = int(torch.argmin(scores).item())
    best_weights = pop[best_idx].detach().clone()
    best_train, best_test = eval_single(
        model, best_weights, train_in, train_tgt, test_in, test_tgt)
    best_generation = 0

    start = time.perf_counter()
    trial_evals = P
    generation = 0

    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= wall_s:
            break

        active = P
        sorted_cpu = torch.argsort(scores).cpu().numpy()
        pbest_count = max(1, (active * pbest_percent + 99) // 100)

        new_F = JDE_F_LO + torch.rand(active, device=device, dtype=dtype) * (
            JDE_F_HI - JDE_F_LO)
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
                    rng.integers(0, pbest_count, size=self_mask.sum())]
                self_mask = pbest_sel == row_idx

        idx_all = _pick_all_indices(rng, active, 2, pbest_sel)
        pb_t = torch.from_numpy(pbest_sel).to(device=device)
        r1_t = torch.from_numpy(idx_all[:, 0]).to(device=device)
        r2_t = torch.from_numpy(idx_all[:, 1]).to(device=device)

        mutant = pop[pb_t] + Fi_try.unsqueeze(1) * (pop[r1_t] - pop[r2_t])
        cross_mask = torch.rand(active, D, dtype=dtype, device=device) < CRi_try.unsqueeze(1)
        jrand_vals = torch.randint(0, D, (active,), device=device)
        cross_mask[torch.arange(active, device=device), jrand_vals] = True
        trials = torch.where(cross_mask, mutant, pop)

        # Block mask: constrain mutation to one random block per trial
        block_mask = _gen_block_mask(active, D, block_size, device)
        if block_mask is not None:
            trials = torch.where(block_mask, trials, pop)

        trials = trials.clamp_(-5.0, 5.0)

        if slice_scoring:
            trial_scores = _slice_scores(
                trials, model, train_in, train_tgt, best_weights)
        else:
            trial_scores = pop_loss_obj(trials, model, train_in, train_tgt)
        trial_evals += active

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
            cand_train, cand_test = eval_single(
                model, cand, train_in, train_tgt, test_in, test_tgt)
            if cand_train < best_train:
                best_train = cand_train
                best_test = cand_test
                best_weights = cand
                best_generation = generation

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    label = "DE full"
    if block_size > 0:
        label = f"DE block-{block_size}"
        if slice_scoring:
            label += " slice"
    return {
        "method": label,
        "train_mse": best_train,
        "test_mse": best_test,
        "nmse": best_test / test_var if test_var > 0 else float("inf"),
        "wall_s": elapsed,
        "generations": generation,
        "trial_evals": trial_evals,
        "evals_per_sec": trial_evals / elapsed if elapsed > 0 else 0,
        "best_generation": best_generation,
    }


# ── Adam reference ──────────────────────────────────────────────────

def run_adam_wallclock(train_in, train_tgt, test_in, test_tgt, test_var,
                       wall_s, arch, seed, lr, device, dtype):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(arch[0], arch[1]),
        torch.nn.Sigmoid(),
        torch.nn.Linear(arch[1], arch[-1]),
    ).to(device=device, dtype=dtype)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.MSELoss()

    best = {"train_mse": float("inf"), "test_mse": float("inf"), "step": 0}
    steps = 0
    start = time.perf_counter()
    while True:
        if time.perf_counter() - start >= wall_s:
            break
        steps += 1
        opt.zero_grad(set_to_none=True)
        pred = model(train_in).squeeze(-1)
        crit(pred, train_tgt).backward()
        opt.step()

        if steps % 50 == 0:
            with torch.no_grad():
                t_pred = model(train_in).squeeze(-1)
                v_pred = model(test_in).squeeze(-1)
                t_mse = float(torch.nn.functional.mse_loss(t_pred, train_tgt))
                v_mse = float(torch.nn.functional.mse_loss(v_pred, test_tgt))
            if v_mse < best["test_mse"]:
                best.update(train_mse=t_mse, test_mse=v_mse, step=steps)

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
        "steps_per_sec": steps / elapsed if elapsed > 0 else 0,
    }


# ── main ────────────────────────────────────────────────────────────

def _main():
    parser = argparse.ArgumentParser(
        description="Block-wise DE vs full-vector DE vs Adam")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wall-seconds", type=int, default=60)
    parser.add_argument("--task", default="pair5-hard")
    parser.add_argument("--arch", default="5,64,1")
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=256)
    parser.add_argument("--popsize", type=int, default=384)
    parser.add_argument("--min-popsize", type=int, default=192)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--adam-lr", type=float, default=0.003)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-adam", action="store_true")
    parser.add_argument("--no-slice", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    device = args.device
    dtype = torch.float32
    arch = [int(x) for x in args.arch.split(",")]
    wall_s = args.wall_seconds

    I = arch[0]; H = arch[1]; O = arch[-1]
    L = len(arch) - 2

    if args.smoke:
        wall_s = 5
        print(f"=== SMOKE ({wall_s}s wall clock) ===\n")

    n_train = args.train_samples
    n_test = args.test_samples
    train_in, train_tgt, test_in, test_tgt = get_data(
        args.task, n_train, n_test, device, dtype)
    test_var = target_variance(test_tgt)

    model = mlp.MLP(I, L, H, O)
    D = model.total_weights

    if args.smoke:
        popsize_smoke = 32
        minpop_smoke = 32
        # Quick smoke: verify each variant runs 1 generation
        print("Smoke full DE...")
        r = run_de_wallclock(
            model, D, train_in, train_tgt, test_in, test_tgt, test_var,
            wall_s, 42, device, dtype, block_size=0,
            popsize=popsize_smoke, min_popsize=minpop_smoke)
        print(f"  full: gen={r['generations']} train={r['train_mse']:.4g} "
              f"test={r['test_mse']:.4g} ok={math.isfinite(r['test_mse'])}")

        for bs in args.block_sizes:
            print(f"Smoke block-{bs}...")
            r = run_de_wallclock(
                model, D, train_in, train_tgt, test_in, test_tgt, test_var,
                wall_s, 42, device, dtype, block_size=bs,
                popsize=popsize_smoke, min_popsize=minpop_smoke)
            print(f"  block-{bs}: gen={r['generations']} train={r['train_mse']:.4g} "
                  f"test={r['test_mse']:.4g} ok={math.isfinite(r['test_mse'])}")

        if not args.no_slice:
            for bs in args.block_sizes:
                print(f"Smoke block-{bs}-slice...")
                r = run_de_wallclock(
                    model, D, train_in, train_tgt, test_in, test_tgt, test_var,
                    wall_s, 42, device, dtype, block_size=bs, slice_scoring=True,
                    popsize=popsize_smoke, min_popsize=minpop_smoke)
                print(f"  block-{bs}-slice: gen={r['generations']} "
                      f"train={r['train_mse']:.4g} test={r['test_mse']:.4g} "
                      f"ok={math.isfinite(r['test_mse'])}")

        print("\nSMOKE PASS")
        return

    print(f"{'='*70}")
    print(f"Block-wise DE Experiment")
    print(f"Task: {args.task}  Arch: {I}->{'->'.join([str(H)]*L)}->{O}  "
          f"Weights: {D}  Wall: {wall_s}s")
    print(f"Device: {device}  dtype: {dtype}")
    print(f"Pop: {args.popsize}->{args.min_popsize}  Train/test: "
          f"{n_train}/{n_test}")
    print(f"{'='*70}\n")

    results = []

    # Full-vector DE
    print("Running full-vector DE...")
    r = run_de_wallclock(
        model, D, train_in, train_tgt, test_in, test_tgt, test_var,
        wall_s, 0, device, dtype, block_size=0,
        popsize=args.popsize, min_popsize=args.min_popsize)
    results.append(r)
    print(f"  {r['method']}: gen={r['generations']} evals={r['trial_evals']} "
          f"train={r['train_mse']:.4g} test={r['test_mse']:.4g} "
          f"nMSE={r['nmse']:.4g}\n")

    # Block-wise DE
    for bs in args.block_sizes:
        print(f"Running block-{bs} DE...")
        r = run_de_wallclock(
            model, D, train_in, train_tgt, test_in, test_tgt, test_var,
            wall_s, 0, device, dtype, block_size=bs,
            popsize=args.popsize, min_popsize=args.min_popsize)
        results.append(r)
        print(f"  {r['method']}: gen={r['generations']} evals={r['trial_evals']} "
              f"train={r['train_mse']:.4g} test={r['test_mse']:.4g} "
              f"nMSE={r['nmse']:.4g}\n")

    # Block-wise DE + slice scoring
    if not args.no_slice:
        for bs in args.block_sizes:
            print(f"Running block-{bs} slice...")
            r = run_de_wallclock(
                model, D, train_in, train_tgt, test_in, test_tgt, test_var,
                wall_s, 0, device, dtype, block_size=bs, slice_scoring=True,
                popsize=args.popsize, min_popsize=args.min_popsize)
            results.append(r)
            print(f"  {r['method']}: gen={r['generations']} "
                  f"evals={r['trial_evals']} train={r['train_mse']:.4g} "
                  f"test={r['test_mse']:.4g} nMSE={r['nmse']:.4g}\n")

    # Adam reference
    if not args.no_adam:
        print("Running Adam...")
        for lr in [args.adam_lr]:
            r = run_adam_wallclock(
                train_in, train_tgt, test_in, test_tgt, test_var,
                wall_s, arch, 0, lr, device, dtype)
            results.append(r)
            print(f"  adam lr={lr}: steps={r['steps']} train={r['train_mse']:.4g} "
                  f"test={r['test_mse']:.4g} nMSE={r['nmse']:.4g}\n")

    # Summary table
    print(f"\n{'='*85}")
    print(f"Results ({wall_s}s wall clock)")
    print(f"{'='*85}")
    print(f"{'Method':<22} {'Train MSE':<12} {'Test MSE':<12} {'nMSE':<10} "
          f"{'Work':<22} {'Wall':<8}")
    print("-" * 85)
    for r in results:
        if "trial_evals" in r:
            work = f"{r['trial_evals']:,} evals"
        elif "steps" in r:
            work = f"{r['steps']:,} steps"
        else:
            work = ""
        print(f"{r['method']:<22} {r['train_mse']:<12.4g} {r['test_mse']:<12.4g} "
              f"{r['nmse']:<10.4g} {work:<22} {r['wall_s']:<8.1f}s")

    # Winner check
    de_best = min((r for r in results if "trial_evals" in r),
                  key=lambda r: r["test_mse"])
    print(f"\nBest DE variant: {de_best['method']} test MSE={de_best['test_mse']:.4g}")

    full_ref = next((r for r in results if r["method"] == "DE full"), None)
    if full_ref:
        ratio = de_best["test_mse"] / full_ref["test_mse"] if full_ref["test_mse"] > 0 else float("inf")
        direction = "better" if ratio < 1.0 else "worse"
        print(f"Block-wise vs full-vector: {ratio:.2f}x {direction}")

    adam_ref = next((r for r in results if "steps" in r), None)
    if adam_ref:
        print(f"Adam reference: test MSE={adam_ref['test_mse']:.4g}")


if __name__ == "__main__":
    _main()

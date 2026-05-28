"""Block-wise DE experiment: sweep of localized mutation strategies.

Compares full-vector DE against block/neuron/mixed/hybrid variants
under equal wall-clock on pair5-hard 5->64->1.

CLI:
  python -m torch_de.block_de_experiment --device cuda --phase quick   # 10s
  python -m torch_de.block_de_experiment --device cuda --phase full    # 60s
  python -m torch_de.block_de_experiment --device cuda --smoke
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
    JDE_F_HI, JDE_F_LO, JDE_TAU_CR, JDE_TAU_F,
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


def pop_loss(pop, model, train_in, train_tgt):
    return mse_loss(model.population_forward(pop, train_in), train_tgt)


# ── block generators ────────────────────────────────────────────────

def _contiguous_block(active, D, block_size, device):
    """One random contiguous block per trial. Returns (mask, jrand_range_start)."""
    if block_size <= 0 or block_size >= D:
        return None, 0, 0
    offsets = torch.randint(0, D - block_size + 1, (active,), device=device)
    mask = torch.zeros(active, D, dtype=torch.bool, device=device)
    idx_d = torch.arange(block_size, device=device)
    mask[torch.arange(active, device=device).unsqueeze(1),
         (offsets.unsqueeze(1) + idx_d)] = True
    return mask, offsets, block_size


def _random_index_block(active, D, block_size, device):
    """Non-contiguous random indices per trial."""
    if block_size <= 0 or block_size >= D:
        return None, 0, 0
    mask = torch.zeros(active, D, dtype=torch.bool, device=device)
    for i in range(active):
        cols = torch.randperm(D, device=device)[:block_size]
        mask[i, cols] = True
    # jrand from the first block position (arbitrary stable choice)
    return mask, torch.zeros(active, dtype=torch.long, device=device), block_size


def _neuron_block(active, D, neuron_params, device):
    """Per-neuron block: hidden bias+weights + output weight (+output bias for n0)."""
    n_neurons, pn = neuron_params
    neuron_idx = torch.randint(0, n_neurons, (active,), device=device)
    mask = torch.zeros(active, D, dtype=torch.bool, device=device)
    row_idx = torch.arange(active, device=device)
    for j in range(pn):
        mask[row_idx, neuron_idx * pn + j] = True
    hidden_total = n_neurons * pn
    mask[row_idx, hidden_total + 1 + neuron_idx] = True         # output weight
    mask[neuron_idx == 0, hidden_total] = True                   # output bias
    return mask, neuron_idx * pn, pn


def _layer_block(active, D, I, H, O, device):
    """Block = entire hidden layer OR entire output layer (random per trial)."""
    hidden_total = H * (1 + I)  # bias + input weights per neuron
    mask = torch.zeros(active, D, dtype=torch.bool, device=device)
    pick_hidden = torch.rand(active, device=device) < 0.5
    row_idx = torch.arange(active, device=device)
    # Hidden layer block
    mask[pick_hidden, :hidden_total] = True
    # Output layer block
    mask[~pick_hidden, hidden_total:] = True
    offset = torch.where(pick_hidden,
                         torch.zeros(active, dtype=torch.long, device=device),
                         torch.full((active,), hidden_total, dtype=torch.long, device=device))
    jrand_range = torch.where(pick_hidden,
                              torch.full((active,), hidden_total, dtype=torch.long, device=device),
                              torch.full((active,), D - hidden_total, dtype=torch.long, device=device))
    return mask, offset, jrand_range


# ── block schedule ──────────────────────────────────────────────────

def _get_block_config(active, D, generation, config, neuron_params, I, H, O, device):
    """Return (block_mask, jrand_offset, jrand_range) or None for full-vector.

    config keys:
      type: "contiguous", "random", "neuron", "layer", "mixed", "adaptive", "occasional_full"
      size: block size
      mix_sizes: list of sizes for "mixed" type
      full_prob: probability of full-vector trial for "occasional_full"
      initial_frac, final_frac: for "adaptive"
      total_gens: for "adaptive"
    """
    ctype = config.get("type", "contiguous")

    if ctype == "full":
        return None, 0, 0

    if ctype == "mixed":
        sizes = config.get("mix_sizes", [32, 112, 224, D])
        bs = int(np.random.choice(sizes))
        if bs >= D:
            return None, 0, 0
        return _contiguous_block(active, D, bs, device)

    if ctype == "occasional_full":
        full_prob = config.get("full_prob", 0.15)
        is_full = torch.rand(active, device=device) < full_prob
        bs = config.get("size", 112)
        mask, offset, jr = _contiguous_block(active, D, bs, device)
        if mask is not None:
            mask[is_full] = True  # full-vector for these rows
        return mask, offset, jr

    if ctype == "adaptive":
        total_gens = config.get("total_gens", 20000)
        init_f = config.get("initial_frac", 0.75)
        final_f = config.get("final_frac", 0.25)
        t = min(generation / max(total_gens, 1), 1.0)
        frac = init_f - t * (init_f - final_f)
        bs = max(int(D * frac), 4)
        if bs >= D:
            return None, 0, 0
        return _contiguous_block(active, D, bs, device)

    # Simple fixed-size variants
    bs = config.get("size", 112)
    if ctype == "contiguous":
        return _contiguous_block(active, D, bs, device)
    elif ctype == "random":
        return _random_index_block(active, D, bs, device)
    elif ctype == "neuron":
        if neuron_params is None:
            return None, 0, 0
        return _neuron_block(active, D, neuron_params, device)
    elif ctype == "layer":
        return _layer_block(active, D, I, H, O, device)

    return None, 0, 0


# ── slice scoring ───────────────────────────────────────────────────

def _slice_scores(trials, model, train_in, train_tgt, best_weights,
                  slice_frac=0.25, slice_weight=0.3):
    with torch.no_grad():
        best_err = (model.forward(best_weights, train_in).squeeze(-1) - train_tgt).abs()
    n_hard = max(1, int(train_in.shape[0] * slice_frac))
    _, hard_idx = torch.topk(best_err, n_hard)
    hard_in, hard_tgt = train_in[hard_idx], train_tgt[hard_idx]
    full = mse_loss(model.population_forward(trials, train_in), train_tgt)
    hard = mse_loss(model.population_forward(trials, hard_in), hard_tgt)
    return (1.0 - slice_weight) * full + slice_weight * hard


# ── DE wall-clock runner ────────────────────────────────────────────

def run_de_wallclock(model, D, train_in, train_tgt, test_in, test_tgt,
                     test_var, wall_s, seed, device, dtype,
                     block_config=None, slice_scoring=False,
                     popsize=384, pbest_percent=50):
    """Run DE until wall clock expires. Returns best metrics dict."""
    rng = np.random.Generator(np.random.PCG64(seed))
    torch.manual_seed(int(seed) & 0x7FFF_FFFF_FFFF_FFFF)

    P = popsize
    pop_np = rng.uniform(-5.0, 5.0, size=(P, D))
    pop = torch.from_numpy(pop_np).to(device=device, dtype=dtype)
    Fi = torch.from_numpy(rng.uniform(JDE_F_LO, JDE_F_HI, size=P)).to(device=device, dtype=dtype)
    CRi = torch.from_numpy(rng.uniform(0.0, 1.0, size=P)).to(device=device, dtype=dtype)
    scores = pop_loss(pop, model, train_in, train_tgt)

    best_idx = int(torch.argmin(scores).item())
    best_weights = pop[best_idx].detach().clone()
    best_train, best_test = eval_single(model, best_weights, train_in, train_tgt, test_in, test_tgt)
    best_generation = 0

    # Neuron params for 5->64->1 (used by "neuron" block type)
    neuron_params = (model.hidden, model.inputs + 1)  # (64, 6)

    # Estimate total generations for adaptive scheduling
    if block_config and block_config.get("type") == "adaptive":
        block_config["total_gens"] = int(wall_s * 400)

    start = time.perf_counter()
    trial_evals = P
    generation = 0

    while True:
        if time.perf_counter() - start >= wall_s:
            break

        active = P
        sorted_cpu = torch.argsort(scores).cpu().numpy()
        pbest_count = max(1, (active * pbest_percent + 99) // 100)

        new_F = JDE_F_LO + torch.rand(active, device=device, dtype=dtype) * (JDE_F_HI - JDE_F_LO)
        new_CR = torch.rand(active, device=device, dtype=dtype)
        re_F = torch.rand(active, device=device) < JDE_TAU_F
        re_CR = torch.rand(active, device=device) < JDE_TAU_CR
        Fi_try = torch.where(re_F, new_F, Fi)
        CRi_try = torch.where(re_CR, new_CR, CRi)

        pbest_sel = sorted_cpu[rng.integers(0, pbest_count, size=active)]
        if pbest_count > 1:
            row_idx_arr = np.arange(active)
            sm = pbest_sel == row_idx_arr
            for _ in range(8):
                if not sm.any():
                    break
                pbest_sel[sm] = sorted_cpu[rng.integers(0, pbest_count, size=sm.sum())]
                sm = pbest_sel == row_idx_arr

        idx_all = _pick_all_indices(rng, active, 2, pbest_sel)
        pb_t = torch.from_numpy(pbest_sel).to(device=device)
        r1_t = torch.from_numpy(idx_all[:, 0]).to(device=device)
        r2_t = torch.from_numpy(idx_all[:, 1]).to(device=device)

        mutant = pop[pb_t] + Fi_try.unsqueeze(1) * (pop[r1_t] - pop[r2_t])

        # ---- crossover with optional block mask ----
        cross_mask = torch.rand(active, D, dtype=dtype, device=device) < CRi_try.unsqueeze(1)

        block_mask, jrand_offset, jrand_range = _get_block_config(
            active, D, generation, block_config or {},
            neuron_params, model.inputs, model.hidden, model.outputs, device)

        if block_mask is not None:
            # Force jrand inside block before computing trials
            if isinstance(jrand_range, torch.Tensor):
                # Per-row jrand ranges: sample uniformly in [offset, offset+range)
                jr_i = jrand_range.int().clamp(min=1)
                jrand_vals = (jrand_offset.float() +
                    torch.rand(active, device=device) * jr_i.float()).long().clamp(0, D - 1)
            elif isinstance(jrand_range, int) and jrand_range > 0:
                jrand_vals = jrand_offset + torch.randint(
                    0, jrand_range, (active,), device=device)
                jrand_vals = jrand_vals.clamp(0, D - 1)
            else:
                jrand_vals = torch.randint(0, D, (active,), device=device)
            cross_mask[torch.arange(active, device=device), jrand_vals] = True

        # Compute trials (with jrand already set)
        trials = torch.where(cross_mask, mutant, pop)

        # Apply block mask to constrain mutation to block
        if block_mask is not None:
            trials = torch.where(block_mask, trials, pop)

        trials = trials.clamp_(-5.0, 5.0)

        if slice_scoring:
            trial_scores = _slice_scores(trials, model, train_in, train_tgt, best_weights)
        else:
            trial_scores = pop_loss(trials, model, train_in, train_tgt)
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
            cand_train, cand_test = eval_single(model, cand, train_in, train_tgt, test_in, test_tgt)
            if cand_train < best_train:
                best_train = cand_train
                best_test = cand_test
                best_weights = cand
                best_generation = generation

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "train_mse": best_train,
        "test_mse": best_test,
        "nmse": best_test / test_var if test_var > 0 else float("inf"),
        "wall_s": elapsed,
        "generations": generation,
        "trial_evals": trial_evals,
        "best_generation": best_generation,
    }


# ── Adam reference ──────────────────────────────────────────────────

def run_adam_wallclock(train_in, train_tgt, test_in, test_tgt, test_var,
                       wall_s, arch, seed, lr, device, dtype):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(arch[0], arch[1]), torch.nn.Sigmoid(),
        torch.nn.Linear(arch[1], arch[-1]),
    ).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.MSELoss()
    best = {"train_mse": float("inf"), "test_mse": float("inf"), "step": 0}
    steps = 0
    start = time.perf_counter()
    while time.perf_counter() - start < wall_s:
        steps += 1
        opt.zero_grad(set_to_none=True)
        crit(model(train_in).squeeze(-1), train_tgt).backward()
        opt.step()
        if steps % 50 == 0:
            with torch.no_grad():
                t = float(torch.nn.functional.mse_loss(model(train_in).squeeze(-1), train_tgt))
                v = float(torch.nn.functional.mse_loss(model(test_in).squeeze(-1), test_tgt))
            if v < best["test_mse"]:
                best.update(train_mse=t, test_mse=v, step=steps)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {"method": "adam", "train_mse": best["train_mse"], "test_mse": best["test_mse"],
            "nmse": best["test_mse"] / test_var, "wall_s": elapsed, "steps": steps}


# ── sweep ───────────────────────────────────────────────────────────

def run_sweep(device, dtype, wall_s, phase_label, extra_seeds=False):
    """Run the full sweep and return results list."""
    arch = [5, 64, 1]
    I, H, O = arch[0], arch[1], arch[-1]
    L = len(arch) - 2
    n_train, n_test = 512, 256

    train_in, train_tgt, test_in, test_tgt = get_data("pair5-hard", n_train, n_test, device, dtype)
    test_var = target_variance(test_tgt)
    model = mlp.MLP(I, L, H, O)
    D = model.total_weights

    variants = [
        # (label, block_config, slice_scoring)
        ("full", {"type": "full"}, False),
        ("block-32", {"type": "contiguous", "size": 32}, False),
        ("block-64", {"type": "contiguous", "size": 64}, False),
        ("block-112", {"type": "contiguous", "size": 112}, False),
        ("block-224", {"type": "contiguous", "size": 224}, False),
        ("block-336", {"type": "contiguous", "size": 336}, False),
        ("neuron", {"type": "neuron"}, False),
        ("layer", {"type": "layer"}, False),
        ("randidx-112", {"type": "random", "size": 112}, False),
        ("mixed", {"type": "mixed"}, False),
        ("occasional-full-112", {"type": "occasional_full", "size": 112, "full_prob": 0.15}, False),
        ("adaptive", {"type": "adaptive"}, False),
    ]

    # Slice variants (fewer, on promising block sizes)
    slice_variants = [
        ("block-112-slice", {"type": "contiguous", "size": 112}, True),
        ("block-224-slice", {"type": "contiguous", "size": 224}, True),
        ("mixed-slice", {"type": "mixed"}, True),
    ]

    all_variants = variants + slice_variants

    print(f"{'='*80}")
    print(f"Block DE Sweep: {phase_label} ({wall_s}s wall clock)")
    print(f"Task: pair5-hard  Arch: {I}->{H}->{O}  Weights: {D}")
    print(f"Variants: {len(all_variants)}")
    print(f"{'='*80}\n")

    results = []
    seeds_used = [0]
    if extra_seeds:
        seeds_used.append(42)

    for label, block_cfg, slice_sc in all_variants:
        best_across_seeds = None
        for seed in seeds_used:
            t0 = time.perf_counter()
            r = run_de_wallclock(
                model, D, train_in, train_tgt, test_in, test_tgt, test_var,
                wall_s, seed, device, dtype,
                block_config=block_cfg, slice_scoring=slice_sc,
                popsize=384, pbest_percent=50)
            r["method"] = label
            r["seed"] = seed
            r["block_cfg"] = str(block_cfg.get("type", "full"))
            if best_across_seeds is None or r["test_mse"] < best_across_seeds["test_mse"]:
                best_across_seeds = r
            wt = time.perf_counter() - t0
            print(f"  {label:<22} seed={seed} gen={r['generations']:>6} "
                  f"train={r['train_mse']:.4g} test={r['test_mse']:.4g} "
                  f"nMSE={r['nmse']:.4g} wall={wt:.1f}s")
        results.append(best_across_seeds)
        print()

    # Adam
    print("  adam...")
    r_adam = run_adam_wallclock(train_in, train_tgt, test_in, test_tgt, test_var,
                                wall_s, arch, 0, 0.003, device, dtype)
    print(f"  adam lr=0.003: steps={r_adam['steps']} train={r_adam['train_mse']:.4g} "
          f"test={r_adam['test_mse']:.4g}\n")

    # Summary
    print(f"\n{'='*80}")
    print(f"Results ({phase_label}, {wall_s}s)")
    print(f"{'='*80}")
    print(f"{'Method':<25} {'Train MSE':<12} {'Test MSE':<12} {'nMSE':<10} "
          f"{'Gen':<8} {'Evals':<12} {'vs Full':<10}")
    print("-" * 90)

    full_ref = next((r for r in results if r["method"] == "full"), None)
    full_test = full_ref["test_mse"] if full_ref else float("inf")

    for r in sorted(results, key=lambda x: x["test_mse"]):
        ratio = r["test_mse"] / full_test if full_test > 0 else float("inf")
        ratio_str = f"{ratio:.1f}x" if ratio > 1.1 else f"{ratio:.2f}x"
        print(f"{r['method']:<25} {r['train_mse']:<12.4g} {r['test_mse']:<12.4g} "
              f"{r['nmse']:<10.4g} {r['generations']:<8} {r['trial_evals']:<12,} "
              f"{ratio_str:<10}")

    print(f"{'adam':<25} {r_adam['train_mse']:<12.4g} {r_adam['test_mse']:<12.4g} "
          f"{r_adam['nmse']:<10.4g} {'--':<8} {r_adam['steps']:<12,} --")
    print()

    # Best non-full DE variant
    de_only = [r for r in results if r["method"] != "full"]
    if de_only:
        best_de = min(de_only, key=lambda r: r["test_mse"])
        print(f"Best non-full variant: {best_de['method']} "
              f"test={best_de['test_mse']:.4g} "
              f"({best_de['test_mse']/full_test:.1f}x vs full)")
        if best_de["test_mse"] / full_test < 1.0:
            print("*** Block variant BEATS full-vector DE! ***")

    return results, r_adam


# ── smoke test ──────────────────────────────────────────────────────

def smoke(device, dtype):
    print("=== SMOKE ===\n")
    arch = [5, 64, 1]
    I, H, O = arch[0], arch[1], arch[-1]
    model = mlp.MLP(I, 1, H, O)
    D = model.total_weights
    train_in, train_tgt, test_in, test_tgt = get_data("pair5-hard", 64, 32, device, dtype)
    test_var = target_variance(test_tgt)

    smoke_configs = [
        ("full", {"type": "full"}, False),
        ("block-32", {"type": "contiguous", "size": 32}, False),
        ("neuron", {"type": "neuron"}, False),
        ("layer", {"type": "layer"}, False),
        ("mixed", {"type": "mixed"}, False),
        ("adaptive", {"type": "adaptive"}, False),
    ]

    for label, cfg, sl in smoke_configs:
        r = run_de_wallclock(model, D, train_in, train_tgt, test_in, test_tgt,
                            test_var, 2.0, 42, device, dtype,
                            block_config=cfg, slice_scoring=sl,
                            popsize=16, pbest_percent=50)
        ok = math.isfinite(r["test_mse"]) and r["generations"] > 0
        print(f"  {label:<20} gen={r['generations']:>4} test={r['test_mse']:.4g} "
              f"ok={ok}")
    print("\nSMOKE PASS\n")


# ── CLI ─────────────────────────────────────────────────────────────

def _main():
    parser = argparse.ArgumentParser(description="Block DE sweep experiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--phase", default="quick",
                        choices=["quick", "full"],
                        help="quick=10s elimination, full=60s main run")
    parser.add_argument("--extra-seeds", action="store_true",
                        help="Run best-of-2 seeds per variant")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    device = args.device
    dtype = torch.float32

    if args.smoke:
        smoke(device, dtype)
        return

    wall_s = 10 if args.phase == "quick" else 60
    label = f"quick-{wall_s}s" if args.phase == "quick" else f"full-{wall_s}s"

    results, r_adam = run_sweep(device, dtype, wall_s, label,
                                extra_seeds=args.extra_seeds)


if __name__ == "__main__":
    _main()

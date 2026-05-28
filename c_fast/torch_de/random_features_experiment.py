"""Random nonlinear features + closed-form ridge regression.

Tests whether randomly projected features + ridge can compete with
trained NNs (Adam/DE) on pair1 and pair5-hard.

Feature families: sigmoid, tanh, fourier (sin+cos), mixed.
Parameters: feature count, input weight scale, ridge alpha.
Hyperparameter selection on validation split, final eval on test set.

CLI:
  python -m torch_de.random_features_experiment --device cuda --phase smoke
  python -m torch_de.random_features_experiment --device cuda --phase sweep
  python -m torch_de.random_features_experiment --device cuda --phase deep
"""

import argparse
import math
import sys
import time

import numpy as np
import torch

from . import tasks as task_mod
from .bench import target_variance as _torch_target_var


# ── data ────────────────────────────────────────────────────────────

def get_data(task_name, n_total, device="cpu", dtype=torch.float32):
    """Get train+test data. We split train into fit/val ourselves."""
    spec = task_mod.TASK_SPECS[task_name]
    train_in, train_tgt = task_mod.generate_data(
        n_total, spec["n_dims"], spec["target_fn"],
        seed=0xD1FF3A9E2B1C4D77, device=device, dtype=dtype)
    test_in, test_tgt = task_mod.generate_data(
        max(n_total // 2, 256), spec["n_dims"], spec["target_fn"],
        seed=0x7E57C0FFEE5EED, device=device, dtype=dtype)
    return (train_in.cpu().numpy(), train_tgt.cpu().numpy(),
            test_in.cpu().numpy(), test_tgt.cpu().numpy())


def split_fit_val(X, y, val_frac=0.25, seed=12345):
    """Deterministic split of training data into fit/validation."""
    rng = np.random.Generator(np.random.PCG64(seed))
    n = len(y)
    n_val = max(1, int(n * val_frac))
    idx = rng.permutation(n)
    val_idx = idx[:n_val]
    fit_idx = idx[n_val:]
    return X[fit_idx], y[fit_idx], X[val_idx], y[val_idx]


# ── feature generators ──────────────────────────────────────────────

def _generate_random_weights(n_features, n_inputs, scale, seed, family):
    """Generate W [n_features, n_inputs] and b [n_features]."""
    rng = np.random.Generator(np.random.PCG64(seed))
    W = rng.normal(0, scale, size=(n_features, n_inputs)).astype(np.float64)

    if family in ("fourier", "mixed"):
        # Uniform in [-pi, pi] for sin/cos
        b = rng.uniform(-np.pi, np.pi, size=n_features).astype(np.float64)
    else:
        b = rng.uniform(-1.0, 1.0, size=n_features).astype(np.float64)
    return W, b


def build_features(X, n_features, family="sigmoid", scale=1.0, seed=0,
                   include_raw=True):
    """Build design matrix Phi for input X [N, D].

    Returns Phi [N, n_features] (plus raw inputs if include_raw).
    """
    N, D = X.shape
    X = np.asarray(X, dtype=np.float64)
    parts = []

    if include_raw:
        parts.append(X)

    if family in ("sigmoid", "tanh", "fourier", "mixed"):
        nf = max(n_features - D * include_raw, 0)
        if family == "mixed":
            # Split features among sigmoid, tanh, fourier
            n_each = max(nf // 3, 1)
            W1, b1 = _generate_random_weights(n_each, D, scale, seed, "sigmoid")
            W2, b2 = _generate_random_weights(n_each, D, scale, seed + 1, "tanh")
            W3, b3 = _generate_random_weights(n_each, D, scale, seed + 2, "fourier")
            a1 = 1.0 / (1.0 + np.exp(-(X @ W1.T + b1)))
            a2 = np.tanh(X @ W2.T + b2)
            a3 = np.sin(X @ W3.T + b3)
            parts.extend([a1, a2, a3])
        elif family == "fourier":
            W, b = _generate_random_weights(nf // 2, D, scale, seed, "fourier")
            sin_part = np.sin(X @ W.T + b)
            cos_part = np.cos(X @ W.T + b)
            parts.extend([sin_part, cos_part])
        else:
            W, b = _generate_random_weights(nf, D, scale, seed, family)
            proj = X @ W.T + b
            if family == "sigmoid":
                parts.append(1.0 / (1.0 + np.exp(-proj)))
            elif family == "tanh":
                parts.append(np.tanh(proj))

    Phi = np.column_stack(parts) if parts else np.zeros((N, 0))
    return Phi


# ── ridge solver ────────────────────────────────────────────────────

def ridge_solve(Phi, y, alpha=1e-6):
    """Stable ridge: w = (Phi^T Phi + alpha I)^{-1} Phi^T y."""
    D = Phi.shape[1]
    A = Phi.T @ Phi + alpha * np.eye(D)
    b = Phi.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(Phi, y, rcond=None)[0]
    return w


# ── single config eval ──────────────────────────────────────────────

def eval_config(X_fit, y_fit, X_val, y_val, X_test, y_test, cfg):
    """Build features, fit ridge, eval all splits. Returns metrics dict."""
    nf = cfg["n_features"]
    family = cfg["family"]
    scale = cfg["scale"]
    alpha = cfg["alpha"]
    seed = cfg.get("seed", 0)
    include_raw = cfg.get("include_raw", True)
    standardize = cfg.get("standardize", True)

    t0 = time.perf_counter()

    # Build features for all splits with same random weights
    Phi_fit = build_features(X_fit, nf, family, scale, seed, include_raw)
    Phi_val = build_features(X_val, nf, family, scale, seed, include_raw)
    Phi_test = build_features(X_test, nf, family, scale, seed, include_raw)

    # Standardize: center + scale using fit-set statistics
    if standardize and Phi_fit.shape[1] > 0:
        mu = Phi_fit.mean(axis=0)
        sigma = Phi_fit.std(axis=0)
        sigma[sigma < 1e-12] = 1.0
        Phi_fit = (Phi_fit - mu) / sigma
        Phi_val = (Phi_val - mu) / sigma
        Phi_test = (Phi_test - mu) / sigma

    # Center y
    y_mean = y_fit.mean()
    y_fit_c = y_fit - y_mean
    y_val_c = y_val - y_mean
    y_test_c = y_test - y_mean

    # Solve ridge
    w = ridge_solve(Phi_fit, y_fit_c, alpha)

    # Predict
    yp_fit = Phi_fit @ w + y_mean
    yp_val = Phi_val @ w + y_mean
    yp_test = Phi_test @ w + y_mean

    train_mse = float(np.mean((y_fit - yp_fit) ** 2))
    val_mse = float(np.mean((y_val - yp_val) ** 2))
    test_mse = float(np.mean((y_test - yp_test) ** 2))
    wall_s = time.perf_counter() - t0

    return {
        "train_mse": train_mse,
        "val_mse": val_mse,
        "test_mse": test_mse,
        "wall_s": wall_s,
        "actual_n_features": Phi_fit.shape[1],
    }


# ── sweep runner ────────────────────────────────────────────────────

def run_sweep(task, n_total, families, n_features_list, scales, alphas,
              seed=0, include_raw=True, standardize=True, phase="sweep"):
    """Grid sweep over all hyperparameters. Returns best configs."""
    X_train, y_train, X_test, y_test = get_data(task, n_total)
    X_fit, y_fit, X_val, y_val = split_fit_val(X_train, y_train)

    test_var = float(np.var(y_test))
    n_inputs = X_train.shape[1]

    results = []
    total = len(families) * len(n_features_list) * len(scales) * len(alphas)
    count = 0

    for family in families:
        for nf in n_features_list:
            for scale in scales:
                for alpha in alphas:
                    cfg = {
                        "n_features": nf, "family": family,
                        "scale": scale, "alpha": alpha,
                        "seed": seed, "include_raw": include_raw,
                        "standardize": standardize,
                    }
                    r = eval_config(X_fit, y_fit, X_val, y_val, X_test, y_test, cfg)
                    r.update(cfg)
                    r["nmse"] = r["test_mse"] / test_var
                    r["test_var"] = test_var
                    r["task"] = task
                    results.append(r)
                    count += 1
                    if phase == "smoke" or count % max(total // 10, 1) == 0:
                        print(f"  [{count}/{total}] {family} nf={nf} "
                              f"scale={scale} alpha={alpha:.0e}: "
                              f"val={r['val_mse']:.4g} test={r['test_mse']:.4g} "
                              f"nMSE={r['nmse']:.4g} {r['wall_s']:.2f}s")

    return results


def print_best(results, top_n=10):
    """Print top-N configs by val MSE."""
    sorted_r = sorted(results, key=lambda r: r["val_mse"])
    print(f"\n{'='*100}")
    print(f"Top {top_n} configs (by validation MSE):")
    print(f"{'='*100}")
    hdr = (f"{'Task':<14} {'Family':<10} {'nFeat':<8} {'Scale':<8} "
           f"{'Alpha':<10} {'Train':<12} {'Val':<12} {'Test':<12} "
           f"{'nMSE':<10} {'Wall':<8}")
    print(hdr)
    print("-" * 100)
    for r in sorted_r[:top_n]:
        print(f"{r['task']:<14} {r['family']:<10} {r['n_features']:<8} "
              f"{r['scale']:<8} {r['alpha']:<10.0e} "
              f"{r['train_mse']:<12.4g} {r['val_mse']:<12.4g} "
              f"{r['test_mse']:<12.4g} {r['nmse']:<10.4g} "
              f"{r['wall_s']:<8.2f}")
    print()

    # Best per task
    for task in sorted(set(r["task"] for r in sorted_r)):
        task_results = [r for r in sorted_r if r["task"] == task]
        if not task_results:
            continue
        best = min(task_results, key=lambda r: r["val_mse"])
        print(f"Best {task}: family={best['family']} nf={best['n_features']} "
              f"scale={best['scale']} alpha={best['alpha']:.0e} "
              f"test MSE={best['test_mse']:.4g} nMSE={best['nmse']:.4g} "
              f"wall={best['wall_s']:.2f}s")


# ── deep run (multi-seed best configs) ──────────────────────────────

def run_deep(task, n_total, best_configs, n_seeds=3):
    """Rerun best configs across multiple seeds. Returns aggregated results."""
    X_train, y_train, X_test, y_test = get_data(task, n_total)
    test_var = float(np.var(y_test))
    all_results = []

    for cfg_base in best_configs:
        seed_results = []
        for seed in range(n_seeds):
            X_fit, y_fit, X_val, y_val = split_fit_val(
                X_train, y_train, seed=12345 + seed * 1000)
            cfg = dict(cfg_base)
            cfg["seed"] = seed
            r = eval_config(X_fit, y_fit, X_val, y_val, X_test, y_test, cfg)
            r["nmse"] = r["test_mse"] / test_var
            r["task"] = task
            r.update({k: cfg[k] for k in ["family", "n_features", "scale", "alpha"]})
            seed_results.append(r)
            print(f"  seed={seed} val={r['val_mse']:.4g} "
                  f"test={r['test_mse']:.4g} nMSE={r['nmse']:.4g}")

        # Best by val across seeds
        best = min(seed_results, key=lambda r: r["val_mse"])
        mean_test = np.mean([r["test_mse"] for r in seed_results])
        std_test = np.std([r["test_mse"] for r in seed_results])
        print(f"  => best val: test={best['test_mse']:.4g} nMSE={best['nmse']:.4g} "
              f"mean_test={mean_test:.4g}+-{std_test:.4g}\n")
        all_results.append({
            "task": task, "family": cfg_base["family"],
            "n_features": cfg_base["n_features"], "scale": cfg_base["scale"],
            "alpha": cfg_base["alpha"], "best_test_mse": best["test_mse"],
            "best_nmse": best["nmse"], "mean_test_mse": mean_test,
            "std_test_mse": std_test,
        })
    return all_results


# ── CLI ─────────────────────────────────────────────────────────────

def _main():
    parser = argparse.ArgumentParser(
        description="Random features + ridge regression experiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--phase", default="smoke",
                        choices=["smoke", "sweep", "deep"],
                        help="Experiment phase")
    args = parser.parse_args()

    if args.phase == "smoke":
        print("=== SMOKE: pair5-hard, 64/256 sigmoid features ===\n")
        results = run_sweep("pair5-hard", 256,
                            families=["sigmoid"],
                            n_features_list=[64, 256],
                            scales=[1.0],
                            alphas=[1e-4, 1e-2],
                            phase="smoke")
        print_best(results, top_n=10)
        print("SMOKE PASS\n")
        return

    if args.phase == "sweep":
        tasks = ["pair1", "pair5-hard"]
        n_total = 512
        families = ["sigmoid", "tanh", "fourier", "mixed"]
        n_features_list = [64, 256, 1024, 4096]
        scales = [0.5, 1.0, 2.0, 4.0]
        alphas = [1e-8, 1e-6, 1e-4, 1e-2, 1e0]

        all_results = []
        for task in tasks:
            print(f"\n{'='*60}")
            print(f"SWEEP: {task} | {len(families)} families × "
                  f"{len(n_features_list)} counts × {len(scales)} scales × "
                  f"{len(alphas)} alphas = "
                  f"{len(families)*len(n_features_list)*len(scales)*len(alphas)} configs")
            print(f"{'='*60}")
            results = run_sweep(task, n_total, families, n_features_list,
                               scales, alphas, phase="sweep")
            all_results.extend(results)

        print_best(all_results, top_n=20)

        # Count how many beat Adam
        adam_nmse = {"pair1": 0.000585, "pair5-hard": 0.00116}
        for task in tasks:
            task_results = [r for r in all_results if r["task"] == task]
            best = min(task_results, key=lambda r: r["val_mse"])
            beats = sum(1 for r in task_results
                       if r["test_mse"] < adam_nmse.get(task, 1.0) * 2)
            print(f"\n{task}: best random features nMSE={best['nmse']:.4g} "
                  f"vs Adam nMSE={adam_nmse.get(task, 0):.4g} "
                  f"({beats}/{len(task_results)} configs within 2x Adam)")
        return

    if args.phase == "deep":
        # Best configs from sweep results
        best_configs_pair5 = [
            {"family": "fourier", "n_features": 4096, "scale": 1.0, "alpha": 1e-6,
             "include_raw": True, "standardize": True},
            {"family": "mixed", "n_features": 4096, "scale": 1.0, "alpha": 1e-4,
             "include_raw": True, "standardize": True},
            {"family": "fourier", "n_features": 8192, "scale": 1.0, "alpha": 1e-6,
             "include_raw": True, "standardize": True},
            {"family": "mixed", "n_features": 8192, "scale": 1.0, "alpha": 1e-4,
             "include_raw": True, "standardize": True},
        ]
        best_configs_pair1 = [
            {"family": "fourier", "n_features": 4096, "scale": 1.0, "alpha": 1e-8,
             "include_raw": True, "standardize": True},
            {"family": "fourier", "n_features": 1024, "scale": 1.0, "alpha": 1e-8,
             "include_raw": True, "standardize": True},
        ]

        print("=== DEEP: multi-seed validation of best configs ===\n")
        all_deep = []
        for task, cfgs in [("pair5-hard", best_configs_pair5),
                           ("pair1", best_configs_pair1)]:
            print(f"\n{task}:")
            deep = run_deep(task, 512, cfgs, n_seeds=3)
            all_deep.extend(deep)

        print(f"\n{'='*80}")
        print("Final deep results:")
        for r in sorted(all_deep, key=lambda r: r["best_nmse"]):
            print(f"  {r['task']:<14} {r['family']:<10} nf={r['n_features']:<6} "
                  f"scale={r['scale']:<4} alpha={r['alpha']:.0e}: "
                  f"best_test={r['best_test_mse']:.4g} nMSE={r['best_nmse']:.4g} "
                  f"mean+-std={r['mean_test_mse']:.4g}+-{r['std_test_mse']:.4g}")
        return


if __name__ == "__main__":
    _main()

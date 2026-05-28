"""Symbolic basis discovery for pair5-hard via ridge-regressed term libraries.

Approach:
  1. Oracle: fit exact true terms → confirms coefficient-fitting works.
  2. Full library ridge: fit all candidates, sort by |coefficient|.
  3. Sequential selection: forward stepwise with ridge, complexity penalty.
  4. Evolutionary subset search: evolve which terms to include.

True target:
  y = sin(2.5*x0) + 0.7*sin(1.7*x1) + 0.35*x2*x3 + 0.25*sin(x4+x0) + 0.2*x2

CLI:
  python -m torch_de.symbolic_basis_experiment --device cuda
"""

import argparse
import math
import sys
import time

import numpy as np
import torch

from . import tasks as task_mod
from .bench import target_variance


# ── data ────────────────────────────────────────────────────────────

def get_data(task_name, n_train, n_test, device, dtype):
    spec = task_mod.TASK_SPECS[task_name]
    train_in, train_tgt = task_mod.generate_data(
        n_train, spec["n_dims"], spec["target_fn"],
        seed=0xD1FF3A9E2B1C4D77, device="cpu", dtype=dtype)
    test_in, test_tgt = task_mod.generate_data(
        n_test, spec["n_dims"], spec["target_fn"],
        seed=0x7E57C0FFEE5EED, device="cpu", dtype=dtype)
    return (train_in.numpy(), train_tgt.numpy(),
            test_in.numpy(), test_tgt.numpy())


# ── term library builder ────────────────────────────────────────────

def _build_term_library(X, n_dims=5):
    """Build a library of candidate basis functions evaluated on X.

    X: [N, 5] numpy array.
    Returns:
        Phi: [N, n_terms] design matrix
        names: list of term names
    """
    N = X.shape[0]
    phi_cols = []
    names = []

    def add(col, name):
        col = np.asarray(col, dtype=np.float64).ravel()
        # Skip constant/NaN columns
        if np.all(np.isfinite(col)) and np.std(col) > 1e-12:
            phi_cols.append(col)
            names.append(name)

    for i in range(n_dims):
        add(X[:, i], f"x{i}")

    # Pairwise products xi*xj
    for i in range(n_dims):
        for j in range(i + 1, n_dims):
            add(X[:, i] * X[:, j], f"x{i}*x{j}")

    # sin(a*xi) and cos(a*xi)
    freqs = [0.5, 1.0, 1.5, 1.7, 2.0, 2.5, 3.0]
    for a in freqs:
        for i in range(n_dims):
            add(np.sin(a * X[:, i]), f"sin({a}*x{i})")
            add(np.cos(a * X[:, i]), f"cos({a}*x{i})")

    # sin(xi+xj), sin(xi-xj)
    for i in range(n_dims):
        for j in range(i + 1, n_dims):
            add(np.sin(X[:, i] + X[:, j]), f"sin(x{i}+x{j})")
            add(np.sin(X[:, i] - X[:, j]), f"sin(x{i}-x{j})")

    # sin(a*xi + b*xj) for a,b in {1.0, 2.5}
    for a in [1.0, 2.5]:
        for b in [1.0, 2.5]:
            for i in range(n_dims):
                for j in range(n_dims):
                    if i == j:
                        continue
                    add(np.sin(a * X[:, i] + b * X[:, j]),
                        f"sin({a}*x{i}+{b}*x{j})")

    Phi = np.column_stack(phi_cols) if phi_cols else np.zeros((N, 0))
    return Phi, names


# ── oracle exact terms ──────────────────────────────────────────────

ORACLE_TERMS = {
    "pair5-hard": [
        ("sin(2.5*x0)", lambda X: np.sin(2.5 * X[:, 0])),
        ("0.7*sin(1.7*x1)", lambda X: 0.7 * np.sin(1.7 * X[:, 1])),
        ("0.35*x2*x3", lambda X: 0.35 * X[:, 2] * X[:, 3]),
        ("0.25*sin(x4+x0)", lambda X: 0.25 * np.sin(X[:, 4] + X[:, 0])),
        ("0.2*x2", lambda X: 0.2 * X[:, 2]),
    ],
    "pair1": [
        ("sin(2.5*x0)", lambda X: np.sin(2.5 * X[:, 0])),
        ("0.6*x1", lambda X: 0.6 * X[:, 1]),
    ],
}


def _oracle_terms_to_phi(X, task_name):
    """Build design matrix from known true term shapes (unknown coefficients)."""
    if task_name == "pair5-hard":
        terms = [
            ("sin(2.5*x0)", lambda X: np.sin(2.5 * X[:, 0])),
            ("sin(1.7*x1)", lambda X: np.sin(1.7 * X[:, 1])),
            ("x2*x3", lambda X: X[:, 2] * X[:, 3]),
            ("sin(x4+x0)", lambda X: np.sin(X[:, 4] + X[:, 0])),
            ("x2", lambda X: X[:, 2]),
        ]
    elif task_name == "pair1":
        terms = [
            ("sin(2.5*x0)", lambda X: np.sin(2.5 * X[:, 0])),
            ("x1", lambda X: X[:, 1]),
        ]
    else:
        terms = []
    cols = [fn(X) for _, fn in terms]
    names = [n for n, _ in terms]
    return np.column_stack(cols), names


# ── ridge regression ────────────────────────────────────────────────

def _ridge_fit(Phi, y, alpha=1e-6):
    """Closed-form ridge: w = (Phi^T Phi + alpha I)^{-1} Phi^T y."""
    D = Phi.shape[1]
    A = Phi.T @ Phi + alpha * np.eye(D)
    b = Phi.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(Phi + alpha * np.eye(*Phi.shape), y, rcond=None)[0]
    return w


def _evaluate_subset(Phi_train, Phi_test, y_train, y_test, mask,
                     alpha, complexity_penalty):
    """Fit ridge on selected terms, return (train_mse, test_mse, coeffs, score)."""
    if not mask.any():
        return float("inf"), float("inf"), np.zeros(0), float("inf")
    Phi_sel_train = Phi_train[:, mask]
    Phi_sel_test = Phi_test[:, mask]
    w = _ridge_fit(Phi_sel_train, y_train, alpha)
    y_pred_train = Phi_sel_train @ w
    y_pred_test = Phi_sel_test @ w
    train_mse = float(np.mean((y_train - y_pred_train) ** 2))
    test_mse = float(np.mean((y_test - y_pred_test) ** 2))
    n_terms = mask.sum()
    score = test_mse + complexity_penalty * n_terms
    return train_mse, test_mse, w, score


# ── evolutionary subset search ──────────────────────────────────────

def _evo_search(Phi_train, Phi_test, y_train, y_test, names, alpha=1e-6,
                complexity_penalty=1e-6, pop_size=200, generations=100,
                elite_frac=0.1, mut_prob=0.05, seed=42):
    """Evolve term subsets. Each individual = boolean mask over library."""
    rng = np.random.Generator(np.random.PCG64(seed))
    D = Phi_train.shape[1]
    P = pop_size
    n_elite = max(1, int(P * elite_frac))

    # Init population: random masks with ~5-15 terms
    pop = np.zeros((P, D), dtype=bool)
    for i in range(P):
        k = rng.integers(3, min(20, D + 1))
        idx = rng.choice(D, size=k, replace=False)
        pop[i, idx] = True

    scores = np.zeros(P)
    best_mask = None
    best_score = float("inf")
    best_train = float("inf")
    best_test = float("inf")
    best_w = None

    for gen in range(generations):
        # Evaluate
        for i in range(P):
            train_mse, test_mse, w, score = _evaluate_subset(
                Phi_train, Phi_test, y_train, y_test, pop[i], alpha, complexity_penalty)
            scores[i] = score
            if score < best_score:
                best_score = score
                best_mask = pop[i].copy()
                best_train = train_mse
                best_test = test_mse
                best_w = w

        # Sort
        order = np.argsort(scores)
        pop = pop[order]
        scores = scores[order]

        # Elite carryover
        new_pop = [pop[:n_elite]]

        # Fill rest with tournament selection + crossover + mutation
        while sum(len(x) for x in new_pop) < P:
            # Tournament select 2 parents
            t1 = rng.integers(0, P // 2)  # from top half
            t2 = rng.integers(0, P // 2)
            p1 = pop[t1]
            p2 = pop[t2]
            # Uniform crossover
            child = np.where(rng.random(D) < 0.5, p1, p2)
            # Mutation: flip random terms
            mut = rng.random(D) < mut_prob
            if mut.any():
                child[mut] = ~child[mut]
            # Ensure at least 1 term
            if not child.any():
                k = rng.integers(1, min(5, D + 1))
                idx = rng.choice(D, size=k, replace=False)
                child[idx] = True
            new_pop.append(child.reshape(1, -1))

        pop = np.concatenate(new_pop)[:P]

    return best_mask, best_train, best_test, best_w


# ── sequential forward selection ────────────────────────────────────

def _forward_select(Phi_train, Phi_test, y_train, y_test, names, alpha=1e-6,
                    complexity_penalty=1e-6, max_terms=20):
    """Greedy forward selection: add best term at each step."""
    D = Phi_train.shape[1]
    selected = np.zeros(D, dtype=bool)
    available = np.ones(D, dtype=bool)
    best_overall_score = float("inf")
    best_overall_mask = None
    best_overall_test = float("inf")
    best_overall_w = None

    for step in range(max_terms):
        best_score = float("inf")
        best_j = -1
        for j in np.where(available)[0]:
            sel = selected.copy()
            sel[j] = True
            _, _, _, score = _evaluate_subset(
                Phi_train, Phi_test, y_train, y_test, sel, alpha, complexity_penalty)
            if score < best_score:
                best_score = score
                best_j = j

        if best_j < 0:
            break
        selected[best_j] = True
        available[best_j] = False
        _, test_mse, w, score = _evaluate_subset(
            Phi_train, Phi_test, y_train, y_test, selected, alpha, complexity_penalty)
        if score < best_overall_score:
            best_overall_score = score
            best_overall_mask = selected.copy()
            best_overall_test = test_mse
            best_overall_w = w

    return best_overall_mask, best_overall_test, best_overall_w


# ── reporting ───────────────────────────────────────────────────────

def _report_formula(names, mask, coeffs):
    """Print a human-readable formula from selected terms and coefficients."""
    if mask is None or not mask.any():
        return "0"
    idx = np.where(mask)[0]
    parts = []
    for j, w in zip(idx, coeffs):
        w = float(w)
        if abs(w) < 1e-10:
            continue
        sign = "+" if w >= 0 else "-"
        aw = abs(w)
        if j == 0 and w >= 0:
            prefix = ""
        else:
            prefix = f" {sign} "
        parts.append(f"{prefix}{aw:.4f}*{names[j]}")
    return "".join(parts) if parts else "0"


def _print_result(label, train_mse, test_mse, nmse, n_terms, formula, wall_s, extra=""):
    print(f"  {label}:")
    print(f"    train MSE={train_mse:.6g}  test MSE={test_mse:.6g}  nMSE={nmse:.6g}")
    print(f"    terms={n_terms}  wall={wall_s:.2f}s {extra}")
    print(f"    formula: {formula[:120]}")
    print()


# ── main experiment ─────────────────────────────────────────────────

def run_experiment(device="cpu", task="pair5-hard", n_train=512, n_test=256):
    dtype = torch.float32
    X_train, y_train, X_test, y_test = get_data(task, n_train, n_test, device, dtype)
    test_var = float(np.var(y_test))  # population variance

    print(f"{'='*70}")
    print(f"Symbolic Basis Discovery: {task}")
    print(f"Train: {n_train}  Test: {n_test}  test var: {test_var:.6g}")
    print(f"{'='*70}\n")

    # ── 1. Oracle sanity check ────────────────────────────────────
    print("─── 1. Oracle: exact true terms, ridge fit coefficients ───")
    oracle_phi, oracle_names = _oracle_terms_to_phi(X_train, task)
    oracle_phi_test, _ = _oracle_terms_to_phi(X_test, task)

    for alpha in [0.0, 1e-8, 1e-6, 1e-4, 1e-2]:
        w = _ridge_fit(oracle_phi, y_train, alpha)
        yp_train = oracle_phi @ w
        yp_test = oracle_phi_test @ w
        train_mse = float(np.mean((y_train - yp_train) ** 2))
        test_mse = float(np.mean((y_test - yp_test) ** 2))
        formula = _report_formula(oracle_names, np.ones(len(oracle_names), dtype=bool), w)
        _print_result(f"oracle alpha={alpha:.0e}", train_mse, test_mse,
                      test_mse / test_var, len(oracle_names), formula, 0)

    # ── 2. Build term library ────────────────────────────────────
    print("─── 2. Build candidate term library ───")
    Phi_train, lib_names = _build_term_library(X_train, n_dims=5)
    Phi_test, _ = _build_term_library(X_test, n_dims=5)
    D = Phi_train.shape[1]
    print(f"  Library size: {D} terms\n")

    # ── 3. Full library ridge (all terms) ─────────────────────────
    print("─── 3. Full library ridge (all terms fit at once) ───")
    for alpha in [1e-6, 1e-4, 1e-2, 1.0, 100.0]:
        w = _ridge_fit(Phi_train, y_train, alpha)
        yp_train = Phi_train @ w
        yp_test = Phi_test @ w
        train_mse = float(np.mean((y_train - yp_train) ** 2))
        test_mse = float(np.mean((y_test - yp_test) ** 2))
        # Show top terms by |coefficient|
        order = np.argsort(-np.abs(w))
        top_n = min(10, D)
        top_formula = _report_formula(lib_names,
                                       np.isin(np.arange(D), order[:top_n]),
                                       w[order[:top_n]])
        _print_result(f"full-lib alpha={alpha:.0e}", train_mse, test_mse,
                      test_mse / test_var, D, f"[top {top_n}]: {top_formula}", 0)

    # ── 4. Forward selection ─────────────────────────────────────
    print("─── 4. Forward sequential selection ───")
    for alpha in [1e-6, 1e-4]:
        for cp in [0.0, 1e-8, 1e-6, 1e-4]:
            t0 = time.perf_counter()
            mask, test_mse, w = _forward_select(
                Phi_train, Phi_test, y_train, y_test, lib_names,
                alpha=alpha, complexity_penalty=cp, max_terms=30)
            wall_s = time.perf_counter() - t0
            if mask is not None and mask.any():
                formula = _report_formula(lib_names, mask, w)
            else:
                formula = "0"
            _print_result(f"fwd-sel alpha={alpha:.0e} cp={cp:.0e}",
                          float(np.mean((y_train - Phi_train[:, mask] @ w) ** 2)),
                          test_mse, test_mse / test_var,
                          mask.sum() if mask is not None else 0,
                          formula, wall_s)

    # ── 5. Evolutionary search ───────────────────────────────────
    print("─── 5. Evolutionary subset search ───")
    evo_configs = [
        (1e-6, 1e-6, 200, 100),
        (1e-6, 1e-4, 200, 100),
        (1e-4, 1e-6, 200, 100),
        (1e-6, 1e-8, 400, 200),
        (1e-6, 1e-6, 400, 200),
    ]
    for alpha, cp, pop, gens in evo_configs:
        t0 = time.perf_counter()
        mask, train_mse, test_mse, w = _evo_search(
            Phi_train, Phi_test, y_train, y_test, lib_names,
            alpha=alpha, complexity_penalty=cp,
            pop_size=pop, generations=gens, seed=42)
        wall_s = time.perf_counter() - t0
        formula = _report_formula(lib_names, mask, w) if mask is not None else "0"
        _print_result(f"evo alpha={alpha:.0e} cp={cp:.0e} pop={pop} gen={gens}",
                      train_mse, test_mse, test_mse / test_var,
                      mask.sum() if mask is not None else 0,
                      formula, wall_s)

    # ── 6. Best formula summary ──────────────────────────────────
    print("─── Summary ───")
    print(f"True pair5-hard:")
    print(f"  sin(2.5*x0) + 0.7*sin(1.7*x1) + 0.35*x2*x3 + 0.25*sin(x4+x0) + 0.2*x2")
    print()


# ── CLI ─────────────────────────────────────────────────────────────

def _main():
    parser = argparse.ArgumentParser(description="Symbolic basis discovery experiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", default="pair5-hard")
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=256)
    args = parser.parse_args()

    run_experiment(device=args.device, task=args.task,
                   n_train=args.train_samples, n_test=args.test_samples)


if __name__ == "__main__":
    _main()

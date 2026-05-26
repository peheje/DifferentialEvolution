"""Differential Evolution solver with jDE, LPSR, and multiple mutation strategies.

Strategies:
  0 = rand/1/bin          v = x_r1 + F * (x_r2 - x_r3)
  1 = best/1/bin          v = x_best + F * (x_r1 - x_r2)
  5 = basepbest/1/bin     v = x_pbest + F * (x_r1 - x_r2)

All use binomial crossover with forced jrand and jDE self-adaptive F/CR.

Vectorized: all trial vectors are generated and evaluated in a single
batch per generation. The only Python loop is over generations.
"""

import numpy as np
import torch


# jDE constants (matching C de_solver.h)
JDE_F_LO = 0.1
JDE_F_HI = 0.9
JDE_TAU_F = 0.1
JDE_TAU_CR = 0.1


def _lpsr_population_size(initial_popsize, min_popsize, generations, generation):
    """Linear population size reduction."""
    if min_popsize >= initial_popsize or generations <= 0:
        return initial_popsize
    t = (generation + 1) / generations
    nxt = int(initial_popsize - (initial_popsize - min_popsize) * t + 0.5)
    if nxt < min_popsize:
        nxt = min_popsize
    if nxt > initial_popsize:
        nxt = initial_popsize
    return nxt


def _pick_indices(rng, P, i, n, extra_exclude=None):
    """Pick n distinct random indices in [0, P), excluding i and extra_exclude.

    Args:
        rng: numpy Generator
        P: population size
        i: current individual index (always excluded)
        n: number of distinct indices to pick
        extra_exclude: optional int or iterable of additional excluded indices

    Returns:
        list of n int indices, mutually distinct and != i and != extra_exclude
    """
    excl = {int(i)}
    if extra_exclude is not None:
        if isinstance(extra_exclude, (int, np.integer)):
            excl.add(int(extra_exclude))
        else:
            for x in extra_exclude:
                excl.add(int(x))

    result = []
    for _ in range(n):
        while True:
            idx = int(rng.integers(0, P))
            if idx not in excl:
                excl.add(idx)
                result.append(idx)
                break
    return result


def de_solve(
    params,
    initial_popsize,
    min_popsize,
    generations,
    bounds_min,
    bounds_max,
    objective_fn,
    seed,
    best_out=None,
    mutation_strategy=0,
    pbest_percent=10,
    device="cpu",
    dtype=torch.float32,
):
    """Run Differential Evolution.

    Args:
        params: number of parameters (D)
        initial_popsize: starting population size
        min_popsize: minimum population size (for LPSR; set = initial_popsize
            to disable)
        generations: number of generations
        bounds_min, bounds_max: parameter bounds (scalars)
        objective_fn: callable(population_tensor) -> losses tensor [P]
        seed: RNG seed (int)
        best_out: if provided, tensor to copy the best solution into
        mutation_strategy: 0=rand/1, 1=best/1, 5=basepbest/1
        pbest_percent: top p% for pbest strategies (only strategy 5)

    Returns:
        best_loss (float)
    """
    if params <= 0 or initial_popsize < 4 or min_popsize < 4:
        return float("inf")
    if min_popsize > initial_popsize:
        min_popsize = initial_popsize

    use_lpsr = min_popsize < initial_popsize
    P = initial_popsize
    D = params

    # RNG (numpy PCG64 for fast integer generation)
    rng = np.random.Generator(np.random.PCG64(seed))
    bound_lo = float(bounds_min)
    bound_hi = float(bounds_max)

    # Initialize population
    pop_np = rng.uniform(bound_lo, bound_hi, size=(P, D))
    pop = torch.from_numpy(pop_np).to(dtype=dtype, device=device)

    # jDE parameters per individual
    Fi = torch.from_numpy(
        rng.uniform(JDE_F_LO, JDE_F_HI, size=P)).to(dtype=dtype, device=device)
    CRi = torch.from_numpy(
        rng.uniform(0.0, 1.0, size=P)).to(dtype=dtype, device=device)

    # Evaluate initial population
    scores = objective_fn(pop)

    best_idx = int(torch.argmin(scores).item())
    best_loss = float(scores[best_idx])

    for g in range(generations):
        active = P

        # === ranking ===
        sorted_indices = torch.argsort(scores)
        best_idx = int(sorted_indices[0].item())

        pbest_count = 1
        if mutation_strategy == 5:
            pbest_count = max(1, (active * pbest_percent + 99) // 100)

        # === jDE parameter re-sampling: produce Fi_try, CRi_try ===
        Fi_np = Fi.cpu().numpy().copy()
        CRi_np = CRi.cpu().numpy().copy()

        re_F = rng.random(active) < JDE_TAU_F
        re_CR = rng.random(active) < JDE_TAU_CR
        if re_F.any():
            Fi_np[re_F] = rng.uniform(JDE_F_LO, JDE_F_HI, size=re_F.sum())
        if re_CR.any():
            CRi_np[re_CR] = rng.uniform(0.0, 1.0, size=re_CR.sum())

        Fi_try = torch.from_numpy(Fi_np).to(dtype=dtype, device=device)
        CRi_try = torch.from_numpy(CRi_np).to(dtype=dtype, device=device)

        # === generate mutation base and difference vectors ===
        # Pre-compute all random indices
        idx_r1 = np.zeros(active, dtype=np.int32)
        idx_r2 = np.zeros(active, dtype=np.int32)
        idx_r3 = np.zeros(active, dtype=np.int32)
        pbest_sel = np.zeros(active, dtype=np.int32)

        for i in range(active):
            if mutation_strategy == 0:
                # rand/1: pick 3 distinct != i
                r1, r2, r3 = _pick_indices(rng, active, i, 3)
                idx_r1[i] = r1
                idx_r2[i] = r2
                idx_r3[i] = r3

            elif mutation_strategy == 1:
                # best/1: pick 2 distinct != i and != best_idx
                r1, r2 = _pick_indices(rng, active, i, 2,
                                       extra_exclude=best_idx)
                idx_r1[i] = r1
                idx_r2[i] = r2

            elif mutation_strategy == 5:
                # basepbest/1: pbest from top p%, then 2 distinct != i and != pbest
                pbest_pos = int(rng.integers(0, pbest_count))
                pb = int(sorted_indices[pbest_pos].item())
                if pbest_count > 1:
                    guard = 0
                    while pb == i and guard < 8:
                        pbest_pos = int(rng.integers(0, pbest_count))
                        pb = int(sorted_indices[pbest_pos].item())
                        guard += 1
                pbest_sel[i] = pb

                r1, r2 = _pick_indices(rng, active, i, 2,
                                       extra_exclude=pb)
                idx_r1[i] = r1
                idx_r2[i] = r2

        # === build mutant tensors ===
        if mutation_strategy == 0:
            # rand/1: v = x_r1 + F * (x_r2 - x_r3)
            r1_t = torch.from_numpy(idx_r1).to(device=device)
            r2_t = torch.from_numpy(idx_r2).to(device=device)
            r3_t = torch.from_numpy(idx_r3).to(device=device)
            base = pop[r1_t]                      # [P, D]
            diff = pop[r2_t] - pop[r3_t]           # [P, D]

        elif mutation_strategy == 1:
            # best/1: v = x_best + F * (x_r1 - x_r2)
            r1_t = torch.from_numpy(idx_r1).to(device=device)
            r2_t = torch.from_numpy(idx_r2).to(device=device)
            base = pop[best_idx].unsqueeze(0).expand(active, -1)  # [P, D]
            diff = pop[r1_t] - pop[r2_t]

        elif mutation_strategy == 5:
            # basepbest/1: v = x_pbest + F * (x_r1 - x_r2)
            pb_t = torch.from_numpy(pbest_sel).to(device=device)
            r1_t = torch.from_numpy(idx_r1).to(device=device)
            r2_t = torch.from_numpy(idx_r2).to(device=device)
            base = pop[pb_t]                      # [P, D]
            diff = pop[r1_t] - pop[r2_t]

        # === mutation (vectorized) ===
        mutant = base + Fi_try.unsqueeze(1) * diff  # [P, D]

        # === binomial crossover (vectorized) ===
        cross_rand = torch.from_numpy(
            rng.random((active, D)).astype(np.float32)
        ).to(dtype=dtype, device=device)
        cross_mask = cross_rand < CRi_try.unsqueeze(1)  # [P, D]

        # forced jrand
        jrand_vals = torch.from_numpy(
            rng.integers(0, D, size=active)
        ).to(device=device)
        cross_mask[torch.arange(active, device=device), jrand_vals] = True

        # trial = parent for dimensions not crossed, mutant for crossed
        trials = torch.where(cross_mask, mutant, pop)

        # clamp to bounds
        trials = torch.clamp(trials, bound_lo, bound_hi)

        # === evaluate all trials ===
        trial_scores = objective_fn(trials)  # [P]

        # === selection (vectorized) ===
        improved = trial_scores < scores
        improved_idx = improved.nonzero(as_tuple=False).squeeze(-1)
        if improved_idx.numel() > 0:
            pop[improved_idx] = trials[improved_idx]
            scores[improved_idx] = trial_scores[improved_idx]
            Fi[improved_idx] = Fi_try[improved_idx]
            CRi[improved_idx] = CRi_try[improved_idx]

        # === LPSR ===
        if use_lpsr:
            next_p = _lpsr_population_size(initial_popsize, min_popsize,
                                          generations, g)
            if next_p < P:
                sorted_idx = torch.argsort(scores)
                keep = sorted_idx[:next_p]
                pop = pop[keep]
                scores = scores[keep]
                Fi = Fi[keep]
                CRi = CRi[keep]
                P = next_p

    # Final best
    best_idx = int(torch.argmin(scores).item())
    best_loss = float(scores[best_idx])
    if best_out is not None:
        best_out.copy_(pop[best_idx])

    return best_loss

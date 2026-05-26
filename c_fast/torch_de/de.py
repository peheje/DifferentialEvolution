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


def _pick_many_distinct(rng, P, n_indices, n_pick, exclude_self=True):
    """Pick n_pick distinct random indices in [0, P) for each of n_indices rows.

    Args:
        rng: numpy Generator
        P: population size
        n_indices: number of rows (population size)
        n_pick: number of distinct indices to pick per row
        exclude_self: if True, exclude row index i from picks

    Returns:
        [n_indices, n_pick] numpy array of indices
    """
    result = np.zeros((n_indices, n_pick), dtype=np.int32)
    for i in range(n_indices):
        # Build exclusion set
        excl = {i} if exclude_self else set()
        for k in range(n_pick):
            while True:
                idx = int(rng.integers(0, P))
                if idx not in excl:
                    excl.add(idx)
                    result[i, k] = idx
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
        min_popsize: minimum population size (for LPSR; set = initial_popsize to disable)
        generations: number of generations
        bounds_min, bounds_max: parameter bounds (scalars)
        objective_fn: callable(population_tensor) -> losses tensor [P]
        seed: RNG seed (int)
        best_out: if provided, a tensor to copy the best solution into
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

        # === jDE parameter re-sampling (vectorized) ===
        re_F = torch.from_numpy(
            rng.random(active) < JDE_TAU_F).to(device=device)
        re_CR = torch.from_numpy(
            rng.random(active) < JDE_TAU_CR).to(device=device)
        if re_F.any():
            new_F = torch.from_numpy(
                rng.uniform(JDE_F_LO, JDE_F_HI, size=active)
            ).to(dtype=dtype, device=device)
            Fi = torch.where(re_F, new_F, Fi)
        if re_CR.any():
            new_CR = torch.from_numpy(
                rng.uniform(0.0, 1.0, size=active)
            ).to(dtype=dtype, device=device)
            CRi = torch.where(re_CR, new_CR, CRi)

        Fi_np = Fi.cpu().numpy()
        CRi_np = CRi.cpu().numpy()

        # === generate mutation base and difference vectors ===
        if mutation_strategy == 0:
            # rand/1: v = x_r1 + F * (x_r2 - x_r3)
            idx_r1 = _pick_many_distinct(rng, active, active, 1, True)
            idx_r2 = _pick_many_distinct(rng, active, active, 1, True)
            idx_r3 = _pick_many_distinct(rng, active, active, 1, True)
            # Ensure r2 != r1 etc.
            for i in range(active):
                while idx_r2[i, 0] == idx_r1[i, 0]:
                    idx_r2[i, 0] = int(rng.integers(0, active))
                    if idx_r2[i, 0] == i:
                        idx_r2[i, 0] = (i + 1) % active
                while idx_r3[i, 0] == idx_r1[i, 0] or idx_r3[i, 0] == idx_r2[i, 0]:
                    idx_r3[i, 0] = int(rng.integers(0, active))
                    if idx_r3[i, 0] == i:
                        idx_r3[i, 0] = (i + 2) % active

            r1_t = torch.from_numpy(idx_r1[:, 0]).to(device=device)
            r2_t = torch.from_numpy(idx_r2[:, 0]).to(device=device)
            r3_t = torch.from_numpy(idx_r3[:, 0]).to(device=device)

            base = pop[r1_t]   # [P, D]
            diff = pop[r2_t] - pop[r3_t]  # [P, D]

        elif mutation_strategy == 1:
            # best/1: v = x_best + F * (x_r1 - x_r2)
            idx_r1 = _pick_many_distinct(rng, active, active, 1, True)
            idx_r2 = _pick_many_distinct(rng, active, active, 1, True)
            for i in range(active):
                # Also exclude best_idx
                if idx_r1[i, 0] == best_idx:
                    idx_r1[i, 0] = (best_idx + 1 + i) % active
                while idx_r2[i, 0] == idx_r1[i, 0] or idx_r2[i, 0] == best_idx:
                    idx_r2[i, 0] = int(rng.integers(0, active))
                    if idx_r2[i, 0] == i:
                        idx_r2[i, 0] = (i + 1) % active

            r1_t = torch.from_numpy(idx_r1[:, 0]).to(device=device)
            r2_t = torch.from_numpy(idx_r2[:, 0]).to(device=device)

            base = pop[best_idx].unsqueeze(0).expand(active, -1)  # [P, D]
            diff = pop[r1_t] - pop[r2_t]

        elif mutation_strategy == 5:
            # basepbest/1: v = x_pbest + F * (x_r1 - x_r2)
            # Pick pbest index for each individual from top pbest_count
            pbest_idx_np = rng.integers(0, pbest_count, size=active)
            pbest_idx = sorted_indices[
                torch.from_numpy(pbest_idx_np).to(device=device)
            ]  # [P]

            # Exclude self from pbest picks (guard loop as in C)
            pbest_np = pbest_idx.cpu().numpy()
            for i in range(active):
                guard = 0
                while pbest_np[i] == i and guard < 8:
                    pbest_np[i] = sorted_indices[
                        int(rng.integers(0, pbest_count))
                    ].item()
                    guard += 1
            pbest_idx = torch.from_numpy(pbest_np).to(device=device)

            idx_r1 = _pick_many_distinct(rng, active, active, 1, True)
            idx_r2 = _pick_many_distinct(rng, active, active, 1, True)
            for i in range(active):
                pb = int(pbest_np[i])
                if idx_r1[i, 0] == pb:
                    idx_r1[i, 0] = (pb + 1 + i) % active
                while idx_r2[i, 0] == idx_r1[i, 0] or idx_r2[i, 0] == pb:
                    idx_r2[i, 0] = int(rng.integers(0, active))
                    if idx_r2[i, 0] == i:
                        idx_r2[i, 0] = (i + 1) % active

            r1_t = torch.from_numpy(idx_r1[:, 0]).to(device=device)
            r2_t = torch.from_numpy(idx_r2[:, 0]).to(device=device)

            base = pop[pbest_idx]  # [P, D]
            diff = pop[r1_t] - pop[r2_t]
        else:
            raise ValueError(f"Unknown mutation strategy: {mutation_strategy}")

        # === mutation (vectorized) ===
        F_t = Fi.to(device=device)
        mutant = base + F_t.unsqueeze(1) * diff  # [P, D]

        # === binomial crossover (vectorized) ===
        cross_rand = torch.from_numpy(
            rng.random((active, D)).astype(np.float32)
        ).to(dtype=dtype, device=device)
        CR_t = CRi.to(device=device)
        cross_mask = cross_rand < CR_t.unsqueeze(1)  # [P, D]

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
            Fi[improved_idx] = F_t[improved_idx]
            CRi[improved_idx] = CR_t[improved_idx]

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

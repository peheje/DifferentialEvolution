# LPSR experiment — TODO

**Population size linear reduction** (LPSR) from L-SHADE (Tanabe & Fukunaga 2014).

## The idea

Start with a large population (`N_init`) and linearly reduce it to a small
population (`N_min`) over the generations. At each reduction step, discard
the worst individuals.

Why: early generations need exploration (large population = diverse search).
Late generations need exploitation (small population = fast convergence).
LPSR provides this shift structurally, without changing the mutation strategy.

Combined with jDE + DE/rand/1, this should:
- Keep multimodal performance (rand/1 exploration in early gens)
- Improve Rosenbrock (small population exploits the valley in late gens)
- Potentially win back what the archive experiment lost

## Parameters (from L-SHADE)

```
N_init  = 18 * sqrt(dims)   ≈ 360 for d=200, ≈ 570 for d=1000
N_min   = 4
```

Reduction is linear: at generation g, N_g = round(N_init - (N_init - N_min) * g / G_max)

When N decreases, discard the (N_old - N_new) worst individuals.

## Implementation sketch

In `de_solver.h` `de_solve()`:

1. Add an explicit `de_solve_lpsr()` entry point
2. Allocate pop/prev_pop/scores/Fi/CRi for `N_init` capacity
3. `int N = N_init;` — current active population size
4. Each generation: compute new N, if smaller, sort & discard worst, compact survivors
5. All `#pragma omp for` loops use `N` instead of fixed capacity
6. RNG index selection uses `N` instead of fixed capacity

## Open questions

- Sorting the population each reduction step is O(N log N). Alternatives:
  - Partial sort / quickselect (only need to find the N_new best)
  - Track a heap — more complex, maybe not worth it
  - Sort only when N actually changes (every ~G_max/(N_init-N_min) gens)
- Does LPSR change the jDE dynamics? Smaller population = fewer F/CR values
  in the gene pool — could reduce parameter diversity.
- Thread safety: sorting and discarding needs a barrier. Already have one
  at end of `#pragma omp for`.

## Expected outcome

Based on literature: L-SHADE with LPSR achieves near-zero on CEC benchmarks.
Combined with our jDE, it should improve Rosenbrock (our only remaining
weakness) without the multimodal regression the archive caused.

## Initial results

Controlled first pass: fixed jDE `200->200` versus LPSR schedules starting at
the same initial population (`200->32`, `200->16`, `200->4`), 3 runs each.

```
objective    best fixed jDE        best LPSR result
sphere       1.7e-12, 3/3          9.9e-06, 0/3  (fails 1e-6 threshold)
rastrigin    0, 3/3                3.2e-10, 3/3
rosenbrock   226, 2/3              193, 3/3      (`200->16`)
ackley       1.8e-14, 3/3          3.2e-14, 3/3
griewank     1.0e-50, 3/3          2.6e-42, 3/3
```

LPSR is faster and helps Rosenbrock success in this small sweep, but the
aggressive reduction hurts high-dimensional sphere enough that it is not yet
a better default general-purpose setting.

Second pass: less aggressive floors (`200->128`, `200->96`, `200->64`,
`200->32`), 3 runs each.

```
objective    fixed jDE             best candidate
sphere       1.7e-12, 3/3          2.2e-13, 3/3  (`200->128`)
rastrigin    0, 3/3                0, 3/3        (`200->128` or `200->96`)
rosenbrock   226, 2/3              170, 3/3      (`200->128`)
ackley       1.8e-14, 3/3          1.8e-14, 3/3  (`200->128` or `200->96`)
griewank     1.0e-50, 3/3          2.3e-51, 3/3  (`200->128`)
```

`200->128` is the current best general-purpose candidate: it preserves all
success thresholds, improves Rosenbrock reliability, and reduces runtime by
roughly 15-20% on this benchmark sweep. More aggressive floors are faster but
start failing the sphere threshold.

## Status

☑ Initial implementation and two benchmark sweeps done.

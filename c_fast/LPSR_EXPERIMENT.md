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

In `de_bench.c` `run_case()`:

1. Compute `N_init`, `N_min` from problem dimensions
2. Allocate pop/trials/scores/Fi/CRi for `N_init` (not fixed POPSIZE)
3. `int N = N_init;` — current population size
4. Each generation: compute new N, if smaller, sort & discard worst, update N
5. All `#pragma omp for` loops use `N` instead of `POPSIZE`
6. RNG index selection uses `N` instead of `POPSIZE`

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

## Status

☐ Not started.

# Differential Evolution Experiments

This log records benchmark methodology and results so we can resume without re-learning the same lessons.

## Methodology

- Benchmark the C implementation with `c_fast/de_bench`.
- Rebuild before each benchmark after changing `c_fast/de_bench.c`.
- Run outside the sandbox when comparing performance.
- Check for stale benchmark processes before and after each run:
  - `pgrep -af '(^|/)de_bench($| )'`
  - terminate old `de_bench` processes before starting a new run.
- Prefer an explicit OpenMP thread count for comparable runs:
  - `OMP_NUM_THREADS=16 /usr/bin/time -f 'wall_seconds %e' ./de_bench`
- Change one algorithm idea at a time and compare against the committed baseline under the same runtime conditions.

## Current C Baseline

Date: 2026-04-25

Command:

```sh
cd c_fast
OMP_NUM_THREADS=16 /usr/bin/time -f 'wall_seconds %e' ./de_bench
```

Result from clean committed baseline:

```text
objective      dims generations  runs     best_score    best_ms  median_ms   worst_ms  success
sphere         1000      20000     3    3.49347e-10       2554       2887       3380   3/3
rastrigin       200       8000     3        541.412        416        426        440   0/3
rosenbrock      200       8000     3        188.515        240        274        275   1/3
ackley          200       8000     3    1.82609e-12        300        339        346   3/3
griewank        200       8000     3    1.66712e-23        608        663        731   3/3
wall_seconds 13.87
```

Earlier comparable clean baseline runs were around `13.87s` to `16.53s` wall time depending on machine load.

## Rejected Trial: Forced Crossover Dimension

Idea: ensure every trial vector mutates at least one dimension by selecting one forced crossover index per candidate.

Patch shape:

```c
int forced_j = rand_int(state, params);

for (int j = 0; j < params; ++j) {
    if (j == forced_j || rand01(state) < crossover) {
        trial[j] = clamp(x0[j] + (x1[j] - x2[j]) * mutate, bench->bounds_min, bench->bounds_max);
    } else {
        trial[j] = xt[j];
    }
}
```

Clean outside-sandbox result with `OMP_NUM_THREADS=16`:

```text
objective      dims generations  runs     best_score    best_ms  median_ms   worst_ms  success
sphere         1000      20000     3    5.73061e-10      22058      25730      28428   3/3
rastrigin       200       8000     3         553.16       1562       1568       1608   0/3
rosenbrock      200       8000     3        235.558       1581       1589       1609   2/3
ackley          200       8000     3    2.36611e-12       1576       1606       1899   3/3
griewank        200       8000     3    6.40979e-24       1499       1602       1876   3/3
wall_seconds 95.79
```

Conclusion: forced crossover dimension was much slower in this implementation and did not produce enough quality improvement to justify the cost. Keep it out of the committed baseline. The trial patch is preserved in the local stash as `forced-crossover-dimension`.

## Benchmark Hygiene Lesson

One misleading slow run happened while two `de_bench` processes were active. The correct response is to verify and clear existing benchmark processes, then rerun the committed baseline and the candidate change under identical conditions. Do not attribute a performance regression to OpenMP or runtime behavior until the candidate change has been stashed or reverted and the baseline has been rerun.

## Next Experiments

- Establish baseline first.
- Pick one algorithm idea.
- Implement it in both `c_fast/main.c` and `c_fast/de_bench.c`.
- Run the same benchmark command.
- Keep the change only if it improves the speed/quality tradeoff clearly.

## AVX2 RNG, staged commits and false sharing in `de_solver.h` (2026-10-24)

Goal: make the parallel jDE solver (`c_fast/de_solver.h`) faster without changing
what it computes. Check used throughout: the final score has to match the old
solver exactly (it does, at 1, 3 and 8 threads, for D = 200, 202 and 51 and for
sphere and rastrigin).

Changes:

- **AVX2 splitmix64.** The generator is counter based, so 4 draws are computed
  at once and come out in the same order as 4 sequential draws. The CR test is
  done on integers, `(draw >> 11) < ceil(CR * 2^53)`, and the choice between
  mutant and parent is a blend. A scalar tail handles D not divisible by 4.
- **Stage and commit.** Parents are read from `pop`, trials are written to a
  staging row (`prev_pop`), and only accepted rows are copied back. It used to
  copy the whole population every generation.
- **Private RNG state per thread.** All threads' states lived in one array
  (`rng_states[]`), so every draw made them fight over one cache line. This went
  unnoticed while the loop was slow: right after the first two changes, 2 to 8
  threads were *slower* than before (8 threads: 53 -> 140 ms) although 1 thread
  was 5x faster. Copying the state into a local at the start of the parallel
  region fixed it (8 threads: 16 ms for D=200, NP=100, 3000 generations).
- **Thread cap.** The default is now at most `DE_MAX_THREADS` (8) threads unless
  `OMP_NUM_THREADS` is set. The 16-thread penalty was not consistent, so this is
  a precaution more than a measured win.

Timings (ms, Ryzen 7 5800X, 8 physical cores, `-O3 -march=native -ffast-math`):

| case                       | 8 threads old -> new | 1 thread old -> new |
|----------------------------|---------------------:|--------------------:|
| sphere D=50 NP=50 10k gen  |             33 -> 16 |            147 -> 45 |
| sphere D=200 NP=100 10k    |            173 -> 47 |          1041 -> 187 |
| sphere D=1000 NP=200 2k    |            319 -> 66 |          2021 -> 368 |
| sphere D=4000 NP=400 500   |           605 -> 116 |          4056 -> 757 |
| rastrigin D=200 NP=100 3k  |             46 -> 27 |           233 -> 116 |
| rastrigin D=1000 NP=200 1k |            167 -> 69 |          1023 -> 379 |

Against `CSharp/Program.cs` (D=1000, NP=200, 20000 generations, sphere): about
3.7 s for C# (3.6 s with its SIMD sphere, so the loss is not its bottleneck),
3.2 s for the old C solver, 0.61 s for the new one. The algorithms differ (the
C# version updates the population in place and uses one F and CR per
generation), so this compares speed, not solution quality.

Measuring tip: `perf` is not available here (`perf_event_paranoid` is 4), so all
of this is wall-clock timing, median of several runs.

### Does this help training neural networks? No.

The speedup is in DE's own bookkeeping, which only dominates when the loss is
cheap. A network loss (population x samples x forward pass) is thousands of
times more work than building a trial vector, so a faster DE changes the total
by a few percent (estimate, not measured). A faster machine would speed up Adam
by the same factor, and a gradient gives D numbers per step where a DE
evaluation gives one, which is why `c_fast/NN_EXPERIMENTS.md` and
`PYTORCH_DE_HANDOFF.md` on the `nn-genann-experiments` branch find Adam far
ahead on smooth tasks. Remaining niches: non-differentiable objectives, cheap
communication (only seeds and scores), tuning system parameters.

### Other optimizers tried (D=200, 1M evaluations, median of 7 seeds)

| optimizer     | sphere | rastrigin | rosenbrock | ackley |
|---------------|-------:|----------:|-----------:|-------:|
| DE rand/1/bin | 6e-19  | 240       | 292        | 1.3    |
| jDE           | 8e-60  | 0         | 325        | 8e-15  |
| JADE          | 9e-72  | 4e-13     | 111        | 2.3    |
| PSO           | 1e-15  | 1810      | 368        | 5.5    |

jDE (already the algorithm in `de_solver.h`) solves rastrigin and ackley where
plain DE gets stuck; JADE is best on rosenbrock. sep-CMA-ES is not listed: the
first run had a bug (the covariance update multiplied by C twice), and the
corrected run did not finish in the time I gave it. The harness was a throwaway
and is not in the repo.

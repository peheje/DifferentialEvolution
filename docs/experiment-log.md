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

## Zeroth-order Adam vs tuned DE (2026-10-24)

Files: `canonical/c/zo_adam.c` (Adam on an SPSA gradient estimate: m random
+-1 directions per step, each evaluated both ways), `canonical/c/de_tune.c`
(DE/rand/1/bin with NP, F, CR as arguments, plus jDE), `canonical/c/losses.h`.
Budget: 1M loss evaluations for everything; median of 5 seeds (3 for the
ellipsoid); bounds [-10, 10]. DE got 52 settings (NP 20/50/100/200 x F
0.3/0.5/0.8 x CR 0.1/0.5/0.9/0.99, plus jDE per NP); zeroth-order Adam got 12
(m 1/8/32 x step 0.3/0.03 x probe radius 1x/10x the step). Best setting per
case, chosen on the same seeds (optimistic for both):

| case            | DE default | best DE  | best DE setting    | best ZO-Adam |
|-----------------|-----------:|---------:|--------------------|-------------:|
| sphere 200      | 1e-17      | 7e-122   | NP=20 F=.5 CR=.1   | 5e-11        |
| ackley 200      | 1.4        | 1.5e-14  | NP=50 F=.3 CR=.1   | 1.3e-5       |
| rastrigin 200   | 252        | 0        | jDE NP=100         | 34           |
| rosenbrock 200  | 406        | 183      | jDE NP=200         | **1.2**      |
| ellipsoid 200   | 111        | 33.5     | NP=20 F=.5 CR=.5   | **4.9e-9**   |
| sphere 1000     | 172        | 1.7e-20  | NP=50 F=.3 CR=.1   | 1.8e-9       |
| ackley 1000     | 5.5        | 1.6e-8   | NP=50 F=.3 CR=.1   | 2.2e-5       |
| rastrigin 1000  | 4430       | 1260     | NP=20 F=.3 CR=.1   | 831          |
| rosenbrock 1000 | 1.8e8      | 1040     | NP=100 F=.3 CR=.1  | 974          |

Read: tuning DE changes the picture. The DE default (CR=0.9) was a poor
choice; the winning settings almost all use CR=0.1, which changes only a few
coordinates per trial. That is very strong on *separable* functions, where each
coordinate can be fixed on its own (sphere, ackley, rastrigin), and tuned DE
wins those. Zeroth-order Adam wins clearly where the function is *not*
separable: the rotated ellipsoid (by 10^10) and rosenbrock at D=200 (by 150x);
at D=1000 rastrigin and rosenbrock are about even. A network loss is not
separable, so the non-separable rows are the ones that matter for that.

Caveats: zeroth-order Adam's step size decays by a fixed 1e-5 over the run,
which caps how precise it can get (its sphere/ackley results stop around
1e-9..1e-5); that schedule was not tuned. The D=200 ellipsoid is one rotation.

Also: the staged random-subspace DE at D=1000 (k=200, S=20) was worse than
default DE on sphere, rastrigin and ackley and better only on rosenbrock, so the
subspace idea is dropped.

Wall clock (same runs, ms per 1M-evaluation run, 12 runs sharing the machine,
so roughly +-20%): both methods take about the same time. On the expensive
ellipsoid it is ~30 s for both (all loss evaluation), so for expensive losses
such as a network, equal evaluations means equal time. On cheap losses
zeroth-order Adam was up to ~3x faster (sphere D=200: 388 vs 1056 ms; rosenbrock
D=200: 428 vs 1467 ms) because it draws one random number per 64 coordinates
and `de_tune.c` draws one per coordinate; neither program is speed-optimized
(the AVX2 tricks from `de_fast.c` would cut DE's overhead ~2.4x).

Open next step: rotated rastrigin and rotated ackley, to check directly whether
tuned DE's wins come from separability.

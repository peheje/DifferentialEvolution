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

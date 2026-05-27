# PyTorch DE Handoff

## Memetic DE reality check

Implemented `torch_de.memetic_experiment` to compare, under the same hard
wall-clock cap:

- Adam-only
- DE-only `basepbest50`
- DE `basepbest50` plus Adam polishing on every trial population

Command:

```bash
.venv/bin/python -m torch_de.memetic_experiment \
  --device cuda \
  --wall-seconds 60 \
  --methods all \
  --polish-steps 1 5 \
  --task pair5-hard \
  --arch 5,64,1 \
  --popsize 384 \
  --min-popsize 192 \
  --adam-lr 0.003 \
  --polish-lr 0.001
```

Setup:

- Task: `pair5-hard`
- Architecture: `5->64->1`
- Data: 512 train / 256 test
- dtype: CUDA float32
- Seed: 0
- Wall cap: 60 seconds per method

Results:

| Method | Train MSE | Test MSE | nMSE | Wall | Work |
|---|---:|---:|---:|---:|---:|
| Adam | 3.79e-05 | 0.00160 | 0.00120 | 60.0s | 112,837 steps |
| DE basepbest50 | 0.00128 | 0.0397 | 0.0299 | 60.0s | 8,971,104 candidate evals |
| DE + Adam polish 1 | 0.00405 | 0.0494 | 0.0373 | 60.0s | 5,362,464 candidate evals |
| DE + Adam polish 5 | 0.0195 | 0.268 | 0.202 | 60.0s | 2,016,336 candidate evals |

Interpretation:

- Adam-only decisively wins this smooth supervised regression setup.
- Naive memetic DE, where every trial is polished with Adam, is worse than
  DE-only here. The polish cost cuts generations/evaluations heavily, and it
  spends gradient work on many low-quality trial candidates.
- If hybrid evolution remains interesting, the next version should polish only
  a small selected subset of trials/elites, not the whole population.

PICKUP: Do not continue with "Adam polish every trial" as the hybrid design.
Try "DE proposes many -> polish top q trials/elites -> select" only if we still
want a memetic route. For smooth supervised tasks, Adam remains the baseline to
beat by wall-clock and test MSE.

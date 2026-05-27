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

## Selective top-Q Adam polish

Updated `torch_de.memetic_experiment` so DE first evaluates all unpolished
trials, then polishes only the top-Q trials by train loss. The polished trial is
kept only if it improves over the unpolished trial. This preserves broad DE
exploration while spending gradient work only on promising candidates.

Command:

```bash
.venv/bin/python -m torch_de.memetic_experiment \
  --device cuda \
  --wall-seconds 60 \
  --methods all \
  --polish-steps 5 \
  --polish-top-q 8 16 \
  --task pair5-hard \
  --arch 5,64,1 \
  --popsize 384 \
  --min-popsize 192 \
  --adam-lr 0.003 \
  --polish-lr 0.001
```

Results:

| Method | Train MSE | Test MSE | nMSE | Wall | Work |
|---|---:|---:|---:|---:|---:|
| Adam | 3.38e-05 | 0.00159 | 0.00120 | 60.0s | 114,623 steps |
| DE basepbest50 | 0.00130 | 0.0398 | 0.0300 | 60.0s | 8,830,752 candidate evals |
| DE + Adam polish top 8 | 0.00299 | 0.0972 | 0.0734 | 60.0s | 2,801,112 candidate evals + 358,800 polish evals |
| DE + Adam polish top 16 | 0.00128 | 0.0200 | 0.0151 | 60.0s | 2,716,452 candidate evals + 689,760 polish evals |

Interpretation:

- Selective top-16 polishing is much better than polishing every trial and
  improves over DE-only on test MSE by about 2x.
- Top-8 is too narrow/noisy in this run and hurts generalization.
- Adam-only still dominates by a large margin on this smooth supervised task.
- The hybrid idea has some signal, but not enough to justify more work on this
  task unless the objective becomes less gradient-friendly.

PICKUP: Top-Q polishing is the right memetic shape if we continue hybrids:
`DE proposes many -> polish top q trials -> select`. On `pair5-hard 5->64->1`,
`q=16,k=5` helps DE but remains far behind Adam. Next hybrid work should move
to non-smooth/rollout/sparse-reward objectives, not smooth MSE regression.

## Non-NN baselines

Added `torch_de.non_nn_baselines` to compare Adam MLPs with tree/kernel models
and genetic-programming symbolic regression on the same deterministic fixtures.
This installs/uses `scikit-learn` and `gplearn` in the local venv.

Command, current hard task:

```bash
.venv/bin/python -m torch_de.non_nn_baselines \
  --device cuda \
  --task pair5-hard \
  --arch 5,64,1 \
  --train-samples 512 \
  --test-samples 256 \
  --adam-steps 5000 \
  --trees 500 \
  --hgb-iter 1000 \
  --gp-population 2000 \
  --gp-generations 25 \
  --jobs -1
```

Results:

| Method | Train MSE | Test MSE | nMSE | Wall |
|---|---:|---:|---:|---:|
| Adam MLP 5,64,1 | 0.000745 | 0.00582 | 0.00439 | 2.99s |
| HistGradientBoosting | 0.000176 | 0.108 | 0.0812 | 0.73s |
| KernelRidge RBF grid | 3.77e-12 | 0.263 | 0.198 | 0.46s |
| ExtraTrees | 0 | 0.264 | 0.199 | 0.24s |
| RandomForest | 0.0423 | 0.283 | 0.214 | 0.38s |
| Symbolic GP | 0.782 | 0.849 | 0.640 | 25.7s |

Command, easier 2D task:

```bash
.venv/bin/python -m torch_de.non_nn_baselines \
  --device cuda \
  --task pair1 \
  --arch 2,12,1 \
  --train-samples 512 \
  --test-samples 256 \
  --adam-steps 5000 \
  --trees 500 \
  --hgb-iter 1000 \
  --gp-population 2000 \
  --gp-generations 25 \
  --jobs -1
```

Results:

| Method | Train MSE | Test MSE | nMSE | Wall |
|---|---:|---:|---:|---:|
| KernelRidge RBF grid | 1.19e-07 | 1.99e-05 | 1.51e-05 | 1.31s |
| Adam MLP 2,12,1 | 0.000439 | 0.000770 | 0.000585 | 2.90s |
| Symbolic GP | 0.00387 | 0.00386 | 0.00293 | 30.9s |
| HistGradientBoosting | 0.000621 | 0.0102 | 0.00778 | 0.66s |
| ExtraTrees | 0 | 0.0167 | 0.0127 | 0.23s |
| RandomForest | 0.00214 | 0.0192 | 0.0146 | 0.39s |

Interpretation:

- The current 5D hard task still favors Adam strongly.
- Trees overfit the sparse 512-sample continuous function and do not compete
  with Adam here.
- Kernel methods are extremely strong on the simpler 2D smooth recurrence, but
  do not scale cleanly to the harder 5D function with the same sample count.
- Symbolic GP found a recognizable pair1 formula:
  `sin(2.553*x0) + x1/1.607`, close to the true
  `sin(2.5*x0) + 0.6*x1`. It did not solve pair5-hard at this budget.

PICKUP: Non-NN baselines are worth keeping. Kernel ridge is a serious baseline
on small smooth low-dimensional tasks, and symbolic GP can rediscover simple
equations, but neither beats Adam on `pair5-hard` yet.

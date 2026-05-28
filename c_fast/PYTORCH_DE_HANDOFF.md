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

## Block-wise DE experiment

Implemented `torch_de.block_de_experiment` and added `block_size` parameter to
`de_solve`. Each trial mutates only one random contiguous block, leaving the
rest of the genome unchanged. Optional slice scoring: 70% full MSE + 30% MSE
on top-25% hardest training samples under current best.

Command:

```bash
.venv/bin/python -m torch_de.block_de_experiment \
  --device cuda \
  --wall-seconds 60 \
  --task pair5-hard \
  --arch 5,64,1 \
  --block-sizes 16 32 64
```

Results (60s wall clock, pair5-hard 5->64->1):

| Method | Train MSE | Test MSE | nMSE | Gen | Evals |
|---|---:|---:|---:|---:|---:|
| DE full | 0.00118 | 0.0447 | 0.0337 | 39,499 | 15.2M |
| DE block-16 | 1.52 | 3.96 | 2.99 | 42,943 | 16.5M |
| DE block-32 | 1.27 | 3.60 | 2.71 | 43,460 | 16.7M |
| DE block-64 | 1.60 | 5.82 | 4.39 | 42,122 | 16.2M |
| DE block-16 slice | 3.97 | 7.53 | 5.68 | 26,291 | 10.1M |
| DE block-32 slice | 4.25 | 8.46 | 6.38 | 26,378 | 10.1M |
| DE block-64 slice | 3.86 | 8.01 | 6.04 | 26,292 | 10.1M |
| Adam | 3.24e-05 | 0.00154 | 0.00116 | — | 120K steps |

Interpretation:

- Full-vector DE dominates block-wise DE by ~80-130x on test MSE.
- Blocks touching only 3.5-14% of params per trial prevent effective
  parameter-space exploration. Each generation is slightly faster (more
  generations fit under the wall clock), but the exploration handicap
  overwhelms any benefit.
- Slice scoring reduces generations (costs extra forward passes) and does not
  improve test MSE here.
- The per-neuron block idea, where semantically related parameters are mutated
  together, is not explored here but might behave differently.

## Block-wise DE final sweep

Ran 15 block variants × 2 seeds × 60s wall-clock (+ 10s elimination pass).
All variants use basepbest50, same population budget as full DE.

Command:
```bash
.venv/bin/python -m torch_de.block_de_experiment --device cuda --phase full --extra-seeds
.venv/bin/python -m torch_de.block_de_experiment --device cuda --phase quick
```

Final results (60s, pair5-hard 5->64->1, best-of-2-seeds):

| Method | Test MSE | nMSE | vs Full |
|---|---:|---:|---:|
| DE full | 0.0318 | 0.0240 | 1.0x |
| DE layer-block | 2.74 | 2.07 | 86x |
| DE occasional-full-112 | 3.42 | 2.58 | 108x |
| DE adaptive (75%→25%) | 3.52 | 2.66 | 111x |
| DE block-32 | 3.79 | 2.86 | 119x |
| DE block-224 | 4.14 | 3.12 | 130x |
| DE neuron-block | 4.23 | 3.19 | 133x |
| DE block-64 | 4.42 | 3.34 | 139x |
| DE block-112 | 4.60 | 3.47 | 145x |
| DE mixed (32/112/224/full) | 4.97 | 3.75 | 157x |
| DE block-336 | 5.37 | 4.05 | 169x |
| DE random-index-112 | 6.02 | 4.54 | 190x |
| DE block-224-slice | 6.47 | 4.88 | 204x |
| DE mixed-slice | 7.33 | 5.53 | 231x |
| DE block-112-slice | 7.47 | 5.64 | 235x |
| Adam | 0.00155 | 0.00117 | — |

Conclusion: Every localized/block/slice DE variant is 86-235x worse than
full-vector DE on this NN regression task. The parameter couplings are
too strong for coordinate-descent-style block mutation. DE's differential
mutation `x_base + F*(x_r1 - x_r2)` only works when it operates on the
full correlated parameter vector.

The "smaller experiments give DE better signal" hypothesis is falsified
for random/local parameter subsets on coupled NN weights.

PICKUP: Block-wise DE is not useful for NN weight optimization. DE works
best as a full-vector optimizer on this problem class.

## Symbolic basis discovery — exact formula recovery for pair5-hard

Approach: build 185-term candidate library (linear, products, sin/cos at 7
frequencies, sin(xi±xj), sin(a*xi+b*xj)), then select terms via forward
selection + ridge regression.

Command:
```bash
.venv/bin/python -m torch_de.symbolic_basis_experiment --device cuda
```

Oracle sanity: ridge on exact true terms recovers coefficients perfectly
(nMSE ~4e-14). Confirms coefficient-fitting path is correct.

Forward selection with tiny complexity penalty (cp=1e-8) discovers the
EXACT true formula in 0.3 seconds:

  y = 0.2000*x2 + 0.3500*x2*x3 + 0.7000*sin(1.7*x1)
    + 1.0000*sin(2.5*x0) + 0.2500*sin(x0+x4)

Test nMSE = 3.4e-15 — 11 orders of magnitude better than Adam (0.00116).

Evolutionary subset search also recovers true structure but includes some
zero-coefficient terms.

Comparison:

| Method | Test MSE | nMSE | Terms | Wall |
|---|---:|---:|---:|---:|
| Symbolic + fwd-sel + ridge | 4.5e-15 | 3.4e-15 | 5 | 0.3s |
| Adam MLP 5->64->1 | 0.00154 | 0.00116 | 449 | 2.5s |
| DE basepbest50 | 0.0318 | 0.0240 | 449 | 60s |
| gplearn Symbolic GP | 0.849 | 0.640 | — | 25.7s |

Key finding: Symbolic structure discovery via forward selection + ridge
completely solves pair5-hard when true basis functions are in the library.
This is a fundamentally different result class from NN optimization —
exact formula recovery vs function approximation.

Limitation: success depends on having the exact term shapes in the library.
The sin(a*xi) grid at 0.5/1.0/1.5/1.7/2.0/2.5/3.0 happens to include
the exact frequencies (1.7, 2.5). Without exact frequency matches, the
ridge fit would be approximate.

PICKUP: Symbolic basis discovery with forward selection + ridge is the
strongest approach yet for pair5-hard. The library-then-select pattern
works beautifully when the true basis functions are in the search space.

## Random features + ridge regression

Command:
```bash
.venv/bin/python -m torch_de.random_features_experiment --device cuda --phase sweep
.venv/bin/python -m torch_de.random_features_experiment --device cuda --phase deep
```

Sweep: 640 configs (2 tasks × 4 families × 4 counts × 4 scales × 5 alphas).
Deep: 3-seed validation of best configs, including 8192 features.

Results:

| Task | Method | Test nMSE | Features | Wall |
|---|---:|---:|---:|---:|
| pair1 | Fourier random 1024 | **1.0e-10** | 1024 | 0.04s |
| pair1 | Fourier random 4096 | 7.1e-10 | 4096 | 0.65s |
| pair1 | Adam MLP 2->12->1 | 0.000585 | 49 | 2.9s |
| pair5-hard | Mixed random 8192 | 0.30 | 8192 | 1.3s |
| pair5-hard | Fourier random 8192 | 0.30 | 8192 | 1.3s |
| pair5-hard | Adam MLP 5->64->1 | **0.00116** | 449 | 2.5s |
| pair5-hard | DE basepbest50 | 0.024 | 449 | 60s |
| pair5-hard | Symbolic + fwd-sel | 3.4e-15 | 5 | 0.3s |

pair1: Fourier random features achieve nMSE ~1e-10 (585,000x better than
Adam). The target sin(2.5*x) + 0.6*x lives in the span of a moderate
random Fourier basis.

pair5-hard: Random features fail (nMSE=0.30, 260x worse than Adam). Even
8192 features cannot match 449 trained NN weights. The structured terms
(specific sin frequencies, x2*x3 interaction) need either the exact basis
in a symbolic library or learned features via gradient descent.

The gap between pair1 and pair5-hard shows that random features work when
the target is smooth/low-dimensional-enough for a random basis to cover
it, but fail on structured targets with specific interaction terms.

PICKUP: Random features + ridge is a strong baseline for smooth low-dim
tasks (Fourier excels on pair1) but cannot replace trained NNs on
structured targets like pair5-hard. Adding structured low-order features
(xi, xi*xj) does not rescue it — the sin terms require matched basis
functions, not random projections.

## Random features v2 — structured low-order + Fourier

Added pairwise product features xi*xj alongside Fourier random features.
90-config sweep on pair5-hard + structured-only baseline.

  Best v2: Fourier 8192 + raw + pairwise, nMSE=0.30
  Structured-only: raw + pairwise, nMSE=0.56
  Adam MLP: nMSE=0.00116

Structured features don't close the gap. The sin(2.5*x0), sin(1.7*x1),
sin(x4+x0) terms require matching basis functions — random sin/cos
projections can't efficiently approximate specific-frequency sinusoids.

Future DE research
should target non-differentiable/noisy objectives where gradient signal
is unavailable, not attempt to decompose the DE mutation operator.

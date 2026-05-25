# Neural network experiments — notes

## Current state

DE + Genann can train small generated tasks:

- 2-bit XOR solves quickly.
- 3-bit parity solves reliably.
- 4-bit parity solves with a 49-weight network in a few seconds.
- Generated circle classification works, and boundary-focused sampling improves
  test accuracy/loss versus uniform sampling.

This is enough to show feasibility for tiny networks, but scaling is visible:
the cost is dominated by repeated forward passes:

```
population * generations * samples * network_eval_cost
```

## Speculative / multi-output forward-pass idea

Recent inference work on speculative decoding and multi-token prediction speeds
up language-model inference by getting more useful output per forward pass.

The same general shape may matter for evolutionary learning:

- DE evaluates many candidate networks independently.
- Each candidate evaluation is mostly forward passes.
- If a model/task can produce multiple useful predictions per pass, or share
  intermediate work across related predictions, DE gets cheaper without changing
  the optimizer.

For NN experiments here, this suggests looking beyond "one candidate, one
sample, one scalar output" evaluation. Useful directions:

- Batch samples per candidate so the network implementation can exploit cache
  locality and vectorization.
- Batch candidates per sample if using a custom NN kernel instead of Genann.
- Predict multiple targets from one forward pass for generated tasks.
- Use auxiliary heads / multi-task losses so each evaluation gives richer
  fitness signal.
- For sequence-like toy tasks, test multi-step prediction loss rather than one
  next-step label.

## Future plan

1. Keep generated tasks first. They are fast, deterministic, and remove data
   loading noise.
2. Add a batched/custom evaluator for small dense networks, then compare against
   Genann on parity and circle tasks.
3. Try smarter sample weighting:
   - boundary-focused samples for geometry tasks
   - high-error sample replay
   - curriculum from easy uniform points to near-boundary points
4. Try multi-output generated tasks:
   - classify circle + predict radius bucket
   - parity + count of active bits
   - multiple Boolean functions from the same input
5. Only after that, move to small real datasets such as Iris.

Open question: whether DE benefits more from better evolutionary mechanics
or from richer/cheaper fitness evaluation. Current circle results point toward
fitness design and sample selection being the first lever to pull.

## Zoomed-out read

The current sequence work is best read as a study of whether DE can train small
NNs under different fitness/data designs, not as a claim that DE should compete
with gradient descent on ordinary smooth regression. The sine recurrence is a
smooth differentiable task, so GD has the natural advantage. DE becomes more
interesting when the objective is discontinuous, rank-based, simulation-based,
multi-objective, architecture-dependent, or otherwise awkward for gradients.

Useful signals so far:

- DE can train small NNs on generated tasks, but raw weight count matters a lot.
- Fitting one generated trajectory can look good while still not learning the
  full state-space function.
- Independent random state-pair coverage is a much more honest test of function
  learning.
- Extra output heads and MTP-style direct multi-horizon prediction hurt this
  Genann + DE setup, mostly by increasing search dimensionality.
- Carrying whole DE populations across stages works mechanically and may be
  useful infrastructure, but simple central-to-full curriculum did not improve
  the final full-box task.
- Loss choice matters: MAE was a modest improvement over MSE for average fit in
  one full-box run, while pure max-error loss was too blunt.

Less useful so far:

- More generations alone gave only marginal gains on the full random-pair task.
- Naive mixed easy/full training made full-box generalization worse.
- Rollout loss did not beat plain one-step training.
- ReLU/SiLU/GELU did not help this small smooth regression; cached sigmoid is
  still the best tradeoff in Genann because it uses a lookup table.

Most promising next steps:

- Add full-box error diagnostics by region: central vs edge, high-curvature
  `sin` zones, target extremes, and sign/phase bands.
- Try smoother robust objectives such as top-k average error or Huber-like loss.
- Try task-aware structure that reduces the DE search dimension, e.g.
  `output = NN(a) + c*b`, because the true recurrence has an explicit linear
  `b` term.
- Revisit multi-output/MTP only after the evaluator is faster or the model is
  structured enough that extra heads do not dominate the search cost.

## Sequence prediction experiments

Added `de_seq.c` to test generated sequence prediction:

```
x[t] = sin(2.5 * x[t-1]) + 0.6 * x[t-2]
```

The initial data generator used one long deterministic train trajectory and one
long deterministic test trajectory. This is useful as a first sanity check, but
the samples are correlated adjacent windows from one orbit through state space.

### Tasks tried

- `seq1`: one-step prediction, `2 -> 10 -> 1`.
- `seq1-wide`: one-step prediction, `2 -> 12 -> 1`.
- `seq2`: direct two-step prediction, `2 -> 12 -> 2`.
- `seq1-sinaux`: predict `x[t]` plus `sin(2.5 * x[t-1])`.
- `seq1-sinaux-light`: same auxiliary target, lower auxiliary loss weight.
- `seq1-dup`: two output heads both predicting `x[t]`.
- `seq1-61w`: one-output width control with roughly the same weight count as
  the two-output models.
- `seq1-roll2`: one-output model whose fitness includes recursive two-step
  rollout error.
- `seq1-roll2-light`: same rollout fitness, lower rollout weight.

### Current sequence results

Representative default-budget results, `env OMP_NUM_THREADS=8 ./de_seq <task>`:

```
seq1-wide:
  test step0 MSE:           0.00377
  recursive test step1 MSE: 0.00981

seq2:
  test output[0] MSE:       0.04962
  test output[1] MSE:       0.20435

seq2, 1 run, 96000 generations:
  test output[0] MSE:       0.02943
  test output[1] MSE:       0.09837

seq1-sinaux:
  test output[0] MSE:       0.02711

seq1-sinaux-light:
  test output[0] MSE:       0.01625

seq1-61w:
  test step0 MSE:           0.00720
  recursive test step1 MSE: 0.03099

seq1-dup:
  test output[0] MSE:       0.01148

seq1-roll2:
  test step0 MSE:           0.00413
  recursive test step1 MSE: 0.01221

seq1-roll2-light:
  test step0 MSE:           0.00766
  recursive test step1 MSE: 0.02234
```

Interpretation so far:

- Plain one-step prediction plus recursive rollout is the best result on this
  trajectory-based setup.
- Extra output heads hurt DE search substantially, even when the second output
  is a duplicate of the first. This suggests the added parameter dimensions are
  expensive for this optimizer/network scale.
- Rollout loss is a better richer-fitness idea because it avoids extra weights,
  but it did not beat plain `seq1-wide` and costs about 2x per fitness
  evaluation.
- Multi-token/multi-output prediction is not ruled out; revisit it after
  improving data coverage and/or using a faster batched evaluator.

### Immediate next question

The path-based sequence data may only test whether the NN can fit one orbit.
The next experiment is to train/test on many independent random state pairs:

```
a, b ~ random bounded state values
target = sin(2.5 * a) + 0.6 * b
[a, b] -> target
```

This tests whether DE learns the recurrence function across state space rather
than the local path visited by one generated sequence.

### Random state-pair results

Added random-pair tasks:

- `pair1`: one-step prediction on independent state pairs sampled from
  `[-2.5, 2.5] x [-2.5, 2.5]`.
- `pair1-mid`: one-step prediction on independent state pairs sampled from
  `[-1.5, 1.5] x [-1.5, 1.5]`.
- `pair1-2p0`: one-step prediction on independent state pairs sampled from
  `[-2.0, 2.0] x [-2.0, 2.0]`.
- `pair1-mix25`: training set with 75% `[-1.5, 1.5]^2` samples and 25%
  `[-2.5, 2.5]^2` samples, tested on the full `[-2.5, 2.5]^2` box.
- `pair1-mix50`: training set with 50% `[-1.5, 1.5]^2` samples and 50%
  `[-2.5, 2.5]^2` samples, tested on the full `[-2.5, 2.5]^2` box.
- `pair1-roll2`: one-output model with recursive two-step rollout loss on the
  same random state-pair distribution.
- `pair2`: direct two-step prediction on random state pairs.

Representative results:

```
pair1:
  test step0 MSE:           0.04930
  recursive test step1 MSE: 0.12732

pair1, 1 run, 24000 generations:
  test step0 MSE:           0.04812
  recursive test step1 MSE: 0.14433

pair1-mid:
  test step0 MSE:           0.00565
  recursive test step1 MSE: 0.01849

pair1-2p0:
  test step0 MSE:           0.01634
  recursive test step1 MSE: 0.06866

pair1-mix25, tested on full box:
  test step0 MSE:           0.06232
  recursive test step1 MSE: 0.18887

pair1-mix50, tested on full box:
  test step0 MSE:           0.08063
  recursive test step1 MSE: 0.20159

pair1-roll2:
  test step0 MSE:           0.10223
  recursive test step1 MSE: 0.19240

pair2:
  test output[0] MSE:       0.04705
  test output[1] MSE:       0.28170
```

Interpretation:

- Random state-pair coverage is much harder than fitting one trajectory. The
  trajectory result was partly an orbit-fitting result, not full state-space
  function learning.
- More generations alone did not materially improve `pair1`.
- The mid-range random-pair task works well, so the difficulty jump comes from
  asking one small DE-trained network to cover the full invariant box at once.
- Medium-range `[-2.0, 2.0]^2` training is intermediate: learnable, but already
  much harder than `[-1.5, 1.5]^2`.
- Simple mixed easy/full coverage did not improve full-box test error. The
  easier samples appear to dominate selection pressure, so mixed coverage likely
  needs explicit sample weighting or staged optimization to act like a real
  curriculum.
- Rollout loss became worse under broad random coverage.
- Direct two-output prediction still does not help; `pair2` learns the first
  head about as well as `pair1`, but the second horizon is poor.

Next likely iteration:

- Try actual staged coverage: start from `[-1.5, 1.5]^2`, then widen toward
  `[-2.5, 2.5]^2` while preserving useful candidates/population state.
- If staying with one-shot fitness, weight hard/full-box samples more heavily
  instead of mixing them uniformly with easy samples.
- Alternatively split the full box into difficulty bands and compare whether
  DE fails mostly near high-curvature `sin` regions or near target extremes.

### Population-carrying curriculum

Extended `de_solver.h` with a reusable `DeState` so the final population can be
carried across training stages. Existing `de_solve*` entry points still allocate
and free their own state internally.

Curriculum modes in `de_seq.c`:

- `curriculum`: evolve the same fixed population through
  `pair1-mid -> pair1-2p0 -> pair1`.
- `curriculum2`: evolve through `pair1-2p0 -> pair1`.
- `fixedfull`: control run using the same fixed-population state API, trained
  only on `pair1`.

Results:

```
curriculum, 6000 generations per stage:
  final full-box test step0 MSE: 0.05324
  final recursive test step1 MSE: 0.13823

curriculum2, 6000 generations per stage:
  final full-box test step0 MSE: 0.06298
  final recursive test step1 MSE: 0.17755

fixedfull, 18000 generations:
  final full-box test step0 MSE: 0.04538
  final recursive test step1 MSE: 0.15584

pair1, LPSR, 18000 generations:
  final full-box test step0 MSE: 0.04579
  final recursive test step1 MSE: 0.15712
```

Interpretation:

- Carrying the whole DE population works mechanically and gives a fair baseline.
- The staged curricula tried here did not improve final full-box error.
- The issue is not the new fixed-population API: `fixedfull` matches the older
  LPSR full-box result closely.
- Central-region pretraining appears to bias the population toward solutions
  that are not better starting points for the full box.

Next likely iteration:

- Analyze full-box errors by region/difficulty. If failures are localized,
  target those regions directly instead of using broad central-to-full staging.
- Try hard-sample weighting or hard-region oversampling, but keep the test set
  full-box and independent.

### Activation sweep

Genann defaults to cached sigmoid hidden activations. Added task variants using
local hidden activations on the full random-pair task:

- `pair1-tanh`
- `pair1-relu`
- `pair1-silu`
- `pair1-gelu`

The output activation remains linear for all regression tasks.

Results:

```
pair1, cached sigmoid, 5 runs:
  test step0 MSE:           0.04930
  ms/run:                   ~6000

pair1-tanh, 5 runs:
  test step0 MSE:           0.04998
  ms/run:                   ~23100

pair1-relu, 5 runs:
  test step0 MSE:           0.13715
  ms/run:                   ~4300

pair1-silu, 2 runs:
  test step0 MSE:           0.11945
  ms/run:                   ~10100

pair1-gelu, 2 runs:
  test step0 MSE:           0.14562
  ms/run:                   ~25300
```

Interpretation:

- Cached sigmoid remains the best tradeoff in this small Genann + DE setup.
- Tanh reaches similar error but is much slower because it calls `tanh` per
  activation instead of using Genann's sigmoid lookup table.
- ReLU is fast but a poor fit for this smooth signed sine regression.
- SiLU/GELU are popular in modern transformer-style models, but here they are
  slower and worse under the current DE budget.
- Activation choice is still worth revisiting if the evaluator becomes batched
  or if activations get lookup-table implementations.

### Robust loss sweep

Added full random-pair loss variants:

- `pair1`: average MSE objective.
- `pair1-mae`: average MAE objective.
- `pair1-max`: maximum squared error objective over the training samples.

Also added p95 absolute error and max absolute error diagnostics for task
reports.

Results:

```
pair1, MSE objective:
  test MSE:                 0.06092
  test MAE:                 0.18864
  test abs p95:             0.51050
  test abs max:             0.88812
  recursive test step1 MSE: 0.18820

pair1-mae, MAE objective:
  test MSE:                 0.05369
  test MAE:                 0.16672
  test abs p95:             0.57537
  test abs max:             0.79437
  recursive test step1 MSE: 0.14040

pair1-max, max squared error objective:
  test MSE:                 0.19885
  test MAE:                 0.39472
  test abs p95:             0.71451
  test abs max:             0.82468
  recursive test step1 MSE: 0.42124
```

Interpretation:

- MAE objective is a modest improvement for average fit and recursive rollout,
  and it lowers the worst observed test error in this run.
- MAE worsened p95 absolute error, so it is not a clean tail-risk win.
- Direct max-error optimization is too blunt here: it slightly reduces max
  error compared with this MSE run, but badly damages average and p95 behavior.
- A smoother robust objective, such as top-k average error or Huber-like loss,
  is more likely to be useful than pure max loss.

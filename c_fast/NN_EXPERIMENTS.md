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

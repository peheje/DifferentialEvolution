# Archive experiment — April 2026

We tested adding an **external archive** of replaced solutions (from SHADE,
Tanabe & Fukunaga 2013) on top of jDE + DE/rand/1/bin.

## The idea

Whenever a trial beats its parent, push the old parent into a fixed-size
archive (random replacement when full). During mutation, occasionally draw
a vector from the archive instead of the population. This injects diversity
from *formerly competitive* solutions — not random noise, but "used to be good."

## Variants tested

| Variant | Mutation | Archive usage |
|---|---|---|
| **jDE-only** | DE/rand/1 | none |
| archive + rand/1 | DE/rand/1 | x₂ sometimes from archive |
| archive + ctb | DE/current-to-best/1 | guide = best or archive (50/50) |
| archive + pbest | DE/current-to-pbest/1 (tournament) | guide = pbest or archive; x₂ sometimes from archive |

## Results (3 runs each)

```
                    Sphere        Rastrigin    Rosenbrock   Ackley       Griewank
                    ───────────   ──────────   ──────────   ──────────   ───────────
jDE-only            8.1×10⁻²⁶ ✓   0.0    ✓     219    ✓    1.8×10⁻¹⁴ ✓   3.2×10⁻⁵⁹ ✓
+archive (rand/1)   2.3×10⁻³¹ ✓   0.0    ✓     199    ✓    1.8×10⁻¹⁴ ✓   8.3×10⁻⁵⁷ ✓
+archive (ctb)      8.5      ✗   80     ✓     0.15   ✓    4.1      ✗   1.2×10⁻¹⁶ ✓
+archive (pbest)    3.9      ✗   59     ✓     1.5    ✓    2.8      ✗   1.7×10⁻²⁰ ✓
```

## What we learned

1. **Archive + rand/1 is neutral.** The archive is redundant when every vector
   is already random — it doesn't hurt, but doesn't help either.

2. **Archive + exploitative mutation helps Rosenbrock dramatically** (0.15 vs 219)
   but **destroys multimodal performance** (sphere, ackley regress by 20+ orders).
   The exploitation pressure collapses diversity; the archive can't fully
   compensate.

3. **jDE-only is the better general-purpose algorithm.** It passes all five
   benchmarks. The archive is a Rosenbrock specialist — useful if that's your
   only problem, harmful otherwise.

## Reproduction note

The temporary harness used to generate these numbers was removed after the
shared solver was extracted into `de_solver.h`. Recreate archive variants in a
short-lived experiment branch rather than keeping alternate DE implementations
in main.

## References

- Tanabe, R. & Fukunaga, A. (2013). "Success-History Based Parameter
  Adaptation for Differential Evolution." *IEEE CEC 2013.*
- The SHADE algorithm combines jDE-style parameter adaptation with an
  external archive and population size reduction (LPSR). We have jDE;
  the archive helps only with a matching exploitative mutation strategy
  (current-to-pbest/1), which conflicts with general-purpose optimization.

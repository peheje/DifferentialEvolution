# Canonical DE: Roc vs C vs C#

Plain DE/rand/1/bin (Storn & Price 1997), single-threaded, in three languages.
All three use the same parameters, the same RNG (splitmix64, seed 42) and the
same order of random draws, so they print the **same score bit for bit**. That
makes it a check that they compute the same thing, not just a speed race.

Parameters: sphere function, D = 200, NP = 100, 10 000 generations,
F = 0.5, CR = 0.9, bounds [-10, 10], generational replacement.

```
./bench.sh          # builds everything, prints median of 7 runs (N=… to change)
```

## Results

AMD Ryzen 7 7840HS, gcc 15.2, .NET 10.0.100, Roc alpha4 (LLVM, `--optimize`):

| variant        | median  | vs C  |
|----------------|--------:|------:|
| C (gcc -O3)    |  653 ms | 1.0×  |
| C fast (AVX2)  |  195 ms | 0.3×  |
| C# (.NET 10)   |  796 ms | 1.2×  |
| Roc (simple)   | 1838 ms | 2.8×  |
| Roc (in-place) | 1071 ms | 1.6×  |

(The C fast row was measured on a Ryzen 7 5800X, where `c/de.c` also takes
about 660 ms; the other rows are from the machine above.)

## C fast (`c/de_fast.c`)

Same algorithm, same parameters, same score bit for bit as `c/de.c`, but
3.4× faster (about 195 ms against 655 ms on a Ryzen 7 5800X, gcc 15.2). Needs
AVX2: build with `-O3 -march=native -ffp-contract=off`.

What it does (all keeps the exact draw order, so the output cannot change):

- **AVX2 splitmix64.** The generator is counter based (`state += GAMMA`), so a
  trial's 200 draws start at a known state and are computed 4 at a time.
  gcc does not vectorize this itself (the 64-bit multiply has to be emulated by
  hand), and `-fno-tree-vectorize` made no difference.
- **Branchless crossover.** `to_unit(u) < CR` becomes the integer test
  `(u >> 11) < ceil(CR * 2^53)`, which is identical; the choice between mutant
  and parent is a blend, not a branch (it mispredicts about 10% of the time).
- **Copy only accepted trials.** About 11% of trials are accepted, so rejected
  ones cost no copy; accepted rows are copied back after the generation.
- **Sphere fused into the loop, in index order.** Same summation order as
  `de.c`, hence the same bits.
- **2 trials in lock-step** to overlap their serial sum chains (about 7%).

What the speedup is made of (D=200, NP=100, 10 000 generations, median ms):

| variant                                              |  ms |
|------------------------------------------------------|----:|
| `de.c` (loss behind a function pointer: same, 659)   | 660 |
| generic fast DE (loss behind a function pointer)     | 274 |
| sphere fused into the loop, no early exit            | 197 |
| fused + 2-trial lock-step (`de_fast.c`)              | 195 |

So about 2.4× comes from DE's own bookkeeping and works with any loss; the
other 1.4× needs the loss built into the loop.

Tried and dropped:

- **Early exit.** Terms of the sphere are non-negative, so a trial can be
  abandoned once its running sum passes the parent's score. Trials run 44 of 50
  blocks on average before that happens, so it saved nothing (195 vs 197 ms).
- **Compiler flags.** `-O2` is slower (583 ms vs 355 ms before the AVX2 work);
  `-funroll-loops`, `-march=znver3` made no difference.
- **More than 2 trials in lock-step** got slower (register pressure).

Not done: FMA and a vectorized or reassociated sum would be faster but change
the last digits of the score.

Because the RNG is counter based, a generation's trials are independent, so a
multithreaded version would also give identical bits. Not built here.

## The two Roc versions

- `roc/main.roc`: the most direct version. It builds a new population list
  each generation. The 160 KB list is freed and re-mmapped every generation,
  and the kernel zeroing fresh pages costs ~0.5 s of sys time.
- `roc/main_inplace.roc`: the same algorithm, but it reuses two population
  buffers and one trial buffer. It is still pure code: Roc mutates a list in
  place when nothing else references it.

Pitfall found on the way: putting the lists in a record (`{ g & buf: … }`)
kept a second reference alive, so every `List.set` copied the whole list.
That version ran in 57 s. Passing the lists as plain function arguments fixed it.

The remaining gap to C, from the profile: a bounds check on every
`List.get`/`List.set`, a uniqueness (refcount) check on every `List.set`,
and some locals spilled to the stack.

## Notes

- C is built with `-ffp-contract=off`. Without it gcc fuses multiply-adds
  (FMA), which is slightly faster but changes the last digits of the score.
- Roc alpha4 is the old Rust/LLVM compiler. The new Zig compiler (nightlies)
  uses different syntax and is not used here.
- Roc platform: basic-cli 0.20.0 (the release that matches alpha4).

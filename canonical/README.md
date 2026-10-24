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
| C# (.NET 10)   |  796 ms | 1.2×  |
| Roc (simple)   | 1838 ms | 2.8×  |
| Roc (in-place) | 1071 ms | 1.6×  |

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

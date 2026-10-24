app [main!] { pf: platform "https://github.com/roc-lang/basic-cli/releases/download/0.20.0/X73hGh05nNTkDHU06FHC0YfFaQB1pimX7gncRcao5mU.tar.br" }

# Canonical differential evolution, DE/rand/1/bin (Storn & Price 1997).
# Same parameters, RNG (splitmix64) and draw order as ../c and ../csharp,
# so all three should print the same score.
#
# Variant: reuses two population buffers and one trial buffer. Still pure
# code - Roc mutates a list in place when nothing else references it.

import pf.Stdout
import pf.Utc

dim : U64
dim = 200

np : U64
np = 100

generations : U64
generations = 10000

f : F64
f = 0.5

cr : F64
cr = 0.9

lo : F64
lo = -10.0

hi : F64
hi = 10.0

seed : U64
seed = 42

# ── splitmix64 ──────────────────────────────────────────────

next_u64 : U64 -> { state : U64, value : U64 }
next_u64 = |state|
    s = Num.add_wrap(state, 0x9E3779B97F4A7C15u64)
    z1 = Num.mul_wrap(Num.bitwise_xor(s, Num.shift_right_zf_by(s, 30)), 0xBF58476D1CE4E5B9u64)
    z2 = Num.mul_wrap(Num.bitwise_xor(z1, Num.shift_right_zf_by(z1, 27)), 0x94D049BB133111EBu64)
    { state: s, value: Num.bitwise_xor(z2, Num.shift_right_zf_by(z2, 31)) }

# uniform in [0, 1) from the top 53 bits
to_unit : U64 -> F64
to_unit = |v| Num.to_f64(Num.shift_right_zf_by(v, 11)) * 1.1102230246251565e-16

# index in [0, n)
rand_index : U64, U64 -> { state : U64, value : U64 }
rand_index = |state, n|
    r = next_u64(state)
    { state: r.state, value: Num.rem(r.value, n) }

# ── objective ───────────────────────────────────────────────

sphere : List F64 -> F64
sphere = |xs| List.walk(xs, 0.0, |acc, x| acc + x * x)

# ── helpers ─────────────────────────────────────────────────

at : List F64, U64 -> F64
at = |xs, k| List.get(xs, k) |> Result.with_default(0.0)

clamp : F64 -> F64
clamp = |v|
    if v < lo then
        lo
    else if v > hi then
        hi
    else
        v

init_pop : U64, U64, List F64 -> { state : U64, pop : List F64 }
init_pop = |state, k, acc|
    if k == 0 then
        { state, pop: acc }
    else
        r = next_u64(state)
        init_pop(r.state, k - 1, List.append(acc, lo + to_unit(r.value) * (hi - lo)))

# draw an index in [0, np) different from x, y and z (np = "unused")
pick_distinct : U64, U64, U64, U64 -> { state : U64, value : U64 }
pick_distinct = |state, x, y, z|
    r = rand_index(state, np)
    if r.value == x or r.value == y or r.value == z then
        pick_distinct(r.state, x, y, z)
    else
        r

copy_into : List F64, U64, List F64, U64, U64 -> List F64
copy_into = |dst, dst_off, src, src_off, j|
    if j == dim then
        dst
    else
        copy_into(List.set(dst, dst_off + j, at(src, src_off + j)), dst_off, src, src_off, j + 1)

# fills `trial` in place (it is uniquely owned, so List.set does not copy)
build_trial : U64, List F64, U64, U64, U64, U64, U64, U64, List F64 -> { state : U64, trial : List F64 }
build_trial = |state, pop, i, r1, r2, r3, jrand, j, trial|
    if j == dim then
        { state, trial }
    else
        r = next_u64(state)
        v =
            if to_unit(r.value) < cr or j == jrand then
                clamp(at(pop, r1 * dim + j) + f * (at(pop, r2 * dim + j) - at(pop, r3 * dim + j)))
            else
                at(pop, i * dim + j)
        build_trial(r.state, pop, i, r1, r2, r3, jrand, j + 1, List.set(trial, j, v))

# one generation: read from `pop`, write survivors into `buf`.
# Lists are passed as plain arguments (not record fields) so each stays
# uniquely referenced and List.set can update it in place.
generation : U64, List F64, List F64, List F64, List F64, List F64, U64 -> (U64, List F64, List F64, List F64, List F64, List F64)
generation = |state, pop, scores, buf, buf_scores, trial, i|
    if i == np then
        (state, buf, buf_scores, pop, scores, trial)
    else
        a = pick_distinct(state, i, np, np)
        b = pick_distinct(a.state, i, a.value, np)
        c = pick_distinct(b.state, i, a.value, b.value)
        jr = rand_index(c.state, dim)
        t = build_trial(jr.state, pop, i, a.value, b.value, c.value, jr.value, 0, trial)
        trial_score = sphere(t.trial)
        old_score = at(scores, i)
        if trial_score <= old_score then
            new_buf = copy_into(buf, i * dim, t.trial, 0, 0)
            generation(t.state, pop, scores, new_buf, List.set(buf_scores, i, trial_score), t.trial, i + 1)
        else
            new_buf = copy_into(buf, i * dim, pop, i * dim, 0)
            generation(t.state, pop, scores, new_buf, List.set(buf_scores, i, old_score), t.trial, i + 1)

evolve : U64, List F64, List F64, List F64, List F64, List F64, U64 -> List F64
evolve = |state, pop, scores, buf, buf_scores, trial, n|
    if n == 0 then
        scores
    else
        (s, p, sc, b, bs, t) = generation(state, pop, scores, buf, buf_scores, trial, 0)
        evolve(s, p, sc, b, bs, t, n - 1)

run : {} -> F64
run = |{}|
    init = init_pop(seed, np * dim, List.with_capacity(np * dim))
    scores =
        List.range({ start: At 0, end: Before np })
        |> List.map(|i| sphere(List.sublist(init.pop, { start: i * dim, len: dim })))
    final_scores = evolve(
        init.state,
        init.pop,
        scores,
        List.repeat(0.0, np * dim),
        List.repeat(0.0, np),
        List.repeat(0.0, dim),
        generations,
    )
    List.walk(final_scores, Num.max_f64, Num.min)

main! = |_args|
    start = Utc.now!({})
    best = run({})
    end = Utc.now!({})
    Stdout.line!("score ${Num.to_str(best)}")?
    Stdout.line!("execution time ${Num.to_str(Utc.delta_as_millis(start, end))} ms")

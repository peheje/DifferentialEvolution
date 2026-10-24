/* DE in a random subspace, run in stages, against plain full-space DE.

   Full space ("full"): DE/rand/1/bin exactly as in de.c (same RNG, same draw
   order, generational replacement), but the loss is a function pointer. On the
   sphere with seed 42 it prints the same score as de.c.

   Subspace ("sub"): the genome is z in R^k, the weights are
       w = clamp(base + Q^T z)
   where the k rows of Q are random orthonormal directions. DE evolves z for S
   generations, then base moves to the best w, Q is re-drawn, and the next stage
   starts (the population restarts, with z = 0, i.e. the current best, as one
   member, so a stage can never make things worse). z is bounded to [-R, R];
   R follows the size of the last successful step.

   Both use the same budget of trial evaluations (NP * G).

   Build: gcc -O3 -march=native -ffp-contract=off -o de_subspace de_subspace.c -lm
   Run:   ./de_subspace <function> <full|sub> <k> <S> <seeds>
          function: sphere rastrigin rosenbrock ackley ellipsoid
   Prints the median final score and median wall time over the seeds. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { DIM = 200, NP = 100, GENERATIONS = 10000 };
static const double F = 0.5, CR = 0.9, LO = -10.0, HI = 10.0;
static const double TWO_PI = 6.283185307179586;

static uint64_t next_u64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static double to_unit(uint64_t v) { return (double)(v >> 11) * 1.1102230246251565e-16; }
static uint64_t rand_index(uint64_t *s, uint64_t n) { return next_u64(s) % n; }
static double gauss(uint64_t *s) {
    double u = to_unit(next_u64(s)), v = to_unit(next_u64(s));
    if (u < 1e-300) u = 1e-300;
    return sqrt(-2.0 * log(u)) * cos(TWO_PI * v);
}
static double clamp(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* ── loss functions ─────────────────────────────────────────────── */
typedef double (*loss_fn)(const double *x, int n);

static double sphere(const double *x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += x[i] * x[i];
    return s;
}
static double rastrigin(const double *x, int n) {
    double s = 10.0 * n;
    for (int i = 0; i < n; ++i) s += x[i] * x[i] - 10.0 * cos(TWO_PI * x[i]);
    return s;
}
static double rosenbrock(const double *x, int n) {
    double s = 0.0;
    for (int i = 0; i < n - 1; ++i) {
        double a = x[i + 1] - x[i] * x[i], b = 1.0 - x[i];
        s += 100.0 * a * a + b * b;
    }
    return s;
}
static double ackley(const double *x, int n) {
    double a = 0.0, b = 0.0;
    for (int i = 0; i < n; ++i) { a += x[i] * x[i]; b += cos(TWO_PI * x[i]); }
    return -20.0 * exp(-0.2 * sqrt(a / n)) - exp(b / n) + 20.0 + M_E;
}

/* Rotated ellipsoid: sum_i cond^(i/(n-1)) y_i^2 with y = R x, R a fixed random
   rotation. Not separable, so coordinate tricks don't help; every eval is O(n^2). */
static double rot[DIM][DIM];
static void gram_schmidt(double *rows, int n, int d) { /* orthonormalize n rows of length d, twice for stability */
    for (int pass = 0; pass < 2; ++pass)
        for (int i = 0; i < n; ++i) {
            double *ri = rows + (size_t)i * d;
            for (int j = 0; j < i; ++j) {
                const double *rj = rows + (size_t)j * d;
                double dot = 0.0;
                for (int t = 0; t < d; ++t) dot += ri[t] * rj[t];
                for (int t = 0; t < d; ++t) ri[t] -= dot * rj[t];
            }
            double nrm = 0.0;
            for (int t = 0; t < d; ++t) nrm += ri[t] * ri[t];
            nrm = sqrt(nrm);
            for (int t = 0; t < d; ++t) ri[t] /= nrm;
        }
}
static void init_rotation(void) {
    uint64_t s = 12345;
    for (int i = 0; i < DIM; ++i) for (int j = 0; j < DIM; ++j) rot[i][j] = gauss(&s);
    gram_schmidt(&rot[0][0], DIM, DIM);
}
static double ellipsoid(const double *x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        double y = 0.0;
        for (int j = 0; j < n; ++j) y += rot[i][j] * x[j];
        s += pow(1000.0, (double)i / (n - 1)) * y * y;
    }
    return s;
}

/* ── full-space DE/rand/1/bin (same as de.c) ────────────────────── */
static double run_full(loss_fn f, uint64_t seed) {
    uint64_t rng = seed;
    double *pop = malloc(sizeof(double) * NP * DIM), *next = malloc(sizeof(double) * NP * DIM);
    double scores[NP], next_scores[NP], trial[DIM];
    for (int k = 0; k < NP * DIM; ++k) pop[k] = LO + to_unit(next_u64(&rng)) * (HI - LO);
    for (int i = 0; i < NP; ++i) scores[i] = f(pop + i * DIM, DIM);
    for (int g = 0; g < GENERATIONS; ++g) {
        for (uint64_t i = 0; i < NP; ++i) {
            uint64_t r1, r2, r3;
            do r1 = rand_index(&rng, NP); while (r1 == i);
            do r2 = rand_index(&rng, NP); while (r2 == i || r2 == r1);
            do r3 = rand_index(&rng, NP); while (r3 == i || r3 == r1 || r3 == r2);
            uint64_t jrand = rand_index(&rng, DIM);
            const double *a = pop + r1 * DIM, *b = pop + r2 * DIM, *c = pop + r3 * DIM, *x = pop + i * DIM;
            for (uint64_t j = 0; j < DIM; ++j)
                trial[j] = (to_unit(next_u64(&rng)) < CR || j == jrand) ? clamp(a[j] + F * (b[j] - c[j]), LO, HI) : x[j];
            double ts = f(trial, DIM);
            if (ts <= scores[i]) { memcpy(next + i * DIM, trial, sizeof trial); next_scores[i] = ts; }
            else { memcpy(next + i * DIM, x, sizeof trial); next_scores[i] = scores[i]; }
        }
        double *t = pop; pop = next; next = t;
        memcpy(scores, next_scores, sizeof scores);
    }
    double best = scores[0];
    for (int i = 1; i < NP; ++i) if (scores[i] < best) best = scores[i];
    free(pop); free(next);
    return best;
}

/* ── staged random-subspace DE ──────────────────────────────────── */
static void expand(const double *base, const double *Q, int k, const double *z, double *w) {
    memcpy(w, base, DIM * sizeof(double));
    for (int a = 0; a < k; ++a) {
        const double *qa = Q + (size_t)a * DIM;
        double za = z[a];
        for (int t = 0; t < DIM; ++t) w[t] += za * qa[t];
    }
    for (int t = 0; t < DIM; ++t) w[t] = clamp(w[t], LO, HI);
}

static double run_sub(loss_fn f, uint64_t seed, int k, int S) {
    uint64_t rng = seed;
    double *Q = malloc(sizeof(double) * (size_t)k * DIM);
    double *pop = malloc(sizeof(double) * NP * k), *next = malloc(sizeof(double) * NP * k);
    double scores[NP], next_scores[NP], trial[k], w[DIM], base[DIM];
    for (int t = 0; t < DIM; ++t) base[t] = LO + to_unit(next_u64(&rng)) * (HI - LO);
    double best = f(base, DIM), R = 10.0;
    const int stages = GENERATIONS / (S + 1); /* each stage: NP initial evals + S generations of NP trials */
    for (int st = 0; st < stages; ++st) {
        for (int a = 0; a < k; ++a) for (int t = 0; t < DIM; ++t) Q[(size_t)a * DIM + t] = gauss(&rng);
        gram_schmidt(Q, k, DIM);
        for (int i = 0; i < NP; ++i)
            for (int a = 0; a < k; ++a) pop[i * k + a] = i == 0 ? 0.0 : -R + 2.0 * R * to_unit(next_u64(&rng));
        for (int i = 0; i < NP; ++i) { expand(base, Q, k, pop + i * k, w); scores[i] = f(w, DIM); }
        for (int g = 0; g < S; ++g) {
            for (uint64_t i = 0; i < NP; ++i) {
                uint64_t r1, r2, r3;
                do r1 = rand_index(&rng, NP); while (r1 == i);
                do r2 = rand_index(&rng, NP); while (r2 == i || r2 == r1);
                do r3 = rand_index(&rng, NP); while (r3 == i || r3 == r1 || r3 == r2);
                uint64_t jrand = rand_index(&rng, (uint64_t)k);
                const double *a = pop + r1 * k, *b = pop + r2 * k, *c = pop + r3 * k, *x = pop + i * k;
                for (uint64_t j = 0; j < (uint64_t)k; ++j)
                    trial[j] = (to_unit(next_u64(&rng)) < CR || j == jrand) ? clamp(a[j] + F * (b[j] - c[j]), -R, R) : x[j];
                expand(base, Q, k, trial, w);
                double ts = f(w, DIM);
                if (ts <= scores[i]) { memcpy(next + i * k, trial, sizeof trial); next_scores[i] = ts; }
                else { memcpy(next + i * k, x, sizeof trial); next_scores[i] = scores[i]; }
            }
            double *t = pop; pop = next; next = t;
            memcpy(scores, next_scores, sizeof scores);
        }
        int bi = 0;
        for (int i = 1; i < NP; ++i) if (scores[i] < scores[bi]) bi = i;
        double zmax = 0.0;
        for (int a = 0; a < k; ++a) zmax = fmax(zmax, fabs(pop[bi * k + a]));
        expand(base, Q, k, pop + bi * k, w);
        memcpy(base, w, sizeof base);
        best = scores[bi];
        R = fmax(fmax(2.0 * zmax, 0.5 * R), 1e-300);
    }
    free(Q); free(pop); free(next);
    return best;
}

static int cmp_double(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s <sphere|rastrigin|rosenbrock|ackley|ellipsoid> <full|sub> <k> <S> <seeds>\n", argv[0]); return 1; }
    struct { const char *name; loss_fn fn; } fns[] = {
        {"sphere", sphere}, {"rastrigin", rastrigin}, {"rosenbrock", rosenbrock}, {"ackley", ackley}, {"ellipsoid", ellipsoid}};
    loss_fn f = NULL;
    for (unsigned i = 0; i < sizeof fns / sizeof *fns; ++i) if (!strcmp(argv[1], fns[i].name)) f = fns[i].fn;
    if (!f) { fprintf(stderr, "unknown function %s\n", argv[1]); return 1; }
    int full = !strcmp(argv[2], "full"), k = atoi(argv[3]), S = atoi(argv[4]), seeds = atoi(argv[5]);
    if (!full && (k < 1 || k > DIM || S < 1)) { fprintf(stderr, "need 1 <= k <= %d, S >= 1\n", DIM); return 1; }
    init_rotation();
    double scores[64], ms[64];
    if (seeds > 64) seeds = 64;
    for (int s = 0; s < seeds; ++s) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        uint64_t seed = 42 + (uint64_t)s;
        scores[s] = full ? run_full(f, seed) : run_sub(f, seed, k, S);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        ms[s] = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
        if (seeds == 1) printf("score %.17g\n", scores[s]);
    }
    qsort(scores, seeds, sizeof(double), cmp_double);
    qsort(ms, seeds, sizeof(double), cmp_double);
    printf("%-10s %-4s k=%-3d S=%-4d median %.4g   best %.4g   worst %.4g   %.0f ms/run\n",
           argv[1], argv[2], full ? 0 : k, full ? 0 : S, scores[seeds / 2], scores[0], scores[seeds - 1], ms[seeds / 2]);
    return 0;
}

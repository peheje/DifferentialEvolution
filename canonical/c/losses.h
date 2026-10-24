/* Shared by de_subspace.c and zo_adam.c: RNG, bounds and the loss functions.
   DIM can be overridden at build time (-DDIM=1000). */
#ifndef LOSSES_H
#define LOSSES_H
#include <math.h>
#include <stdint.h>
#include <string.h>

#ifndef DIM
#define DIM 200
#endif
static const double LO = -10.0, HI = 10.0;
static const double TWO_PI = 6.283185307179586;

static uint64_t next_u64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static double to_unit(uint64_t v) { return (double)(v >> 11) * 1.1102230246251565e-16; }
static inline uint64_t rand_index(uint64_t *s, uint64_t n) { return next_u64(s) % n; }
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


static loss_fn find_loss(const char *name) {
    static const struct { const char *name; loss_fn fn; } fns[] = {
        {"sphere", sphere}, {"rastrigin", rastrigin}, {"rosenbrock", rosenbrock}, {"ackley", ackley}, {"ellipsoid", ellipsoid}};
    for (unsigned i = 0; i < sizeof fns / sizeof *fns; ++i)
        if (!strcmp(name, fns[i].name)) return fns[i].fn;
    return NULL;
}

static int cmp_double(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
#endif

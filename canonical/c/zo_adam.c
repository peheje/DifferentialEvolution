/* Zeroth-order Adam: Adam driven by a gradient estimated from loss evaluations
   only (no backward pass), to compare against DE at an equal evaluation budget.

   Each step draws m random directions u with entries +-1 (SPSA style) and
   evaluates each one both ways (antithetic):
       g ~= (1/m) * sum_r (f(x + s*u_r) - f(x - s*u_r)) / (2s) * u_r
   g then goes through a standard Adam update. The step size a and the probe
   radius s = sigma_mult * a decay geometrically by 1e-5 over the run.
   The reported score is the best loss seen at any evaluated point.

   Budget: 1,000,000 evaluations, the same as de_subspace.c (NP 100 x 10000).

   Build: gcc -O3 -march=native -ffp-contract=off [-DDIM=1000] -o zo_adam zo_adam.c -lm
   Run:   ./zo_adam <function> <m> <alpha0> <sigma_mult> <seeds>
          function: sphere rastrigin rosenbrock ackley ellipsoid */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "losses.h"

enum { BUDGET = 1000000 };
static const double BETA1 = 0.9, BETA2 = 0.999, EPS = 1e-8, DECAY = 1e-5;

static double run_zo(loss_fn f, uint64_t seed, int m, double alpha0, double sigma_mult) {
    uint64_t rng = seed;
    static double x[DIM], mo[DIM], v[DIM], g[DIM], xp[DIM], xm[DIM];
    static signed char u[DIM];
    for (int t = 0; t < DIM; ++t) { x[t] = LO + to_unit(next_u64(&rng)) * (HI - LO); mo[t] = v[t] = 0.0; }
    double best = f(x, DIM);
    const long steps = (BUDGET - 1) / (2L * m);
    for (long step = 0; step < steps; ++step) {
        const double a = alpha0 * pow(DECAY, (double)step / steps), s = sigma_mult * a;
        memset(g, 0, sizeof g);
        for (int r = 0; r < m; ++r) {
            for (int t = 0; t < DIM; t += 64) {
                uint64_t bits = next_u64(&rng);
                for (int b = 0; b < 64 && t + b < DIM; ++b) u[t + b] = (bits >> b & 1) ? 1 : -1;
            }
            for (int t = 0; t < DIM; ++t) { xp[t] = clamp(x[t] + s * u[t], LO, HI); xm[t] = clamp(x[t] - s * u[t], LO, HI); }
            const double fp = f(xp, DIM), fm = f(xm, DIM);
            if (fp < best) best = fp;
            if (fm < best) best = fm;
            const double d = (fp - fm) / (2.0 * s * m);
            for (int t = 0; t < DIM; ++t) g[t] += d * u[t];
        }
        const double c1 = 1.0 - pow(BETA1, (double)(step + 1)), c2 = 1.0 - pow(BETA2, (double)(step + 1));
        for (int t = 0; t < DIM; ++t) {
            mo[t] = BETA1 * mo[t] + (1.0 - BETA1) * g[t];
            v[t] = BETA2 * v[t] + (1.0 - BETA2) * g[t] * g[t];
            x[t] = clamp(x[t] - a * (mo[t] / c1) / (sqrt(v[t] / c2) + EPS), LO, HI);
        }
    }
    return best;
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s <sphere|rastrigin|rosenbrock|ackley|ellipsoid> <m> <alpha0> <sigma_mult> <seeds>\n", argv[0]); return 1; }
    loss_fn f = find_loss(argv[1]);
    if (!f) { fprintf(stderr, "unknown function %s\n", argv[1]); return 1; }
    int m = atoi(argv[2]), seeds = atoi(argv[5]);
    double alpha0 = atof(argv[3]), sigma_mult = atof(argv[4]);
    if (m < 1 || alpha0 <= 0 || sigma_mult <= 0) { fprintf(stderr, "need m >= 1, alpha0 > 0, sigma_mult > 0\n"); return 1; }
    init_rotation();
    double scores[64], ms[64];
    if (seeds > 64) seeds = 64;
    for (int s = 0; s < seeds; ++s) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        scores[s] = run_zo(f, 42 + (uint64_t)s, m, alpha0, sigma_mult);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        ms[s] = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    }
    qsort(scores, seeds, sizeof(double), cmp_double);
    qsort(ms, seeds, sizeof(double), cmp_double);
    printf("%-10s zo   D=%-4d m=%-3d a0=%-5g sm=%-4g median %.4g   best %.4g   worst %.4g   %.0f ms/run\n",
           argv[1], DIM, m, alpha0, sigma_mult, scores[seeds / 2], scores[0], scores[seeds - 1], ms[seeds / 2]);
    return 0;
}

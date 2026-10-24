/* Tunable DE for a fair comparison with zo_adam.c at the same budget of
   1,000,000 trial evaluations (generations = budget / NP).

   de:  DE/rand/1/bin with given NP, F, CR, as in de.c. With NP=100, F=0.5,
        CR=0.9 and seed 42 on the sphere it prints de.c's exact score.
   jde: jDE (Brest et al. 2006), DE/rand/1/bin where each individual carries
        its own F and CR: with probability 0.1 each is re-drawn before a trial
        (F ~ U(0.1, 1.0), CR ~ U(0, 1)); they survive with a successful trial.
        The F and CR arguments are ignored.

   Build: gcc -O3 -march=native -ffp-contract=off [-DDIM=1000] -o de_tune de_tune.c -lm
   Run:   ./de_tune <function> <de|jde> <NP> <F> <CR> <seeds> */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "losses.h"

enum { BUDGET = 1000000 };

static double run_de(loss_fn f, uint64_t seed, int np, double F, double CR, int jde) {
    uint64_t rng = seed;
    const int generations = BUDGET / np;
    double *pop = malloc(sizeof(double) * np * DIM), *next = malloc(sizeof(double) * np * DIM);
    double *scores = malloc(sizeof(double) * np), *next_scores = malloc(sizeof(double) * np);
    double *Fi = malloc(sizeof(double) * np), *CRi = malloc(sizeof(double) * np);
    double *nFi = malloc(sizeof(double) * np), *nCRi = malloc(sizeof(double) * np);
    double trial[DIM];
    for (int k = 0; k < np * DIM; ++k) pop[k] = LO + to_unit(next_u64(&rng)) * (HI - LO);
    for (int i = 0; i < np; ++i) { scores[i] = f(pop + i * DIM, DIM); Fi[i] = 0.5; CRi[i] = 0.9; }
    for (int g = 0; g < generations; ++g) {
        for (uint64_t i = 0; i < (uint64_t)np; ++i) {
            double Ft = F, CRt = CR;
            if (jde) {
                Ft = to_unit(next_u64(&rng)) < 0.1 ? 0.1 + 0.9 * to_unit(next_u64(&rng)) : Fi[i];
                CRt = to_unit(next_u64(&rng)) < 0.1 ? to_unit(next_u64(&rng)) : CRi[i];
            }
            uint64_t r1, r2, r3;
            do r1 = rand_index(&rng, np); while (r1 == i);
            do r2 = rand_index(&rng, np); while (r2 == i || r2 == r1);
            do r3 = rand_index(&rng, np); while (r3 == i || r3 == r1 || r3 == r2);
            uint64_t jrand = rand_index(&rng, DIM);
            const double *a = pop + r1 * DIM, *b = pop + r2 * DIM, *c = pop + r3 * DIM, *x = pop + i * DIM;
            for (uint64_t j = 0; j < DIM; ++j)
                trial[j] = (to_unit(next_u64(&rng)) < CRt || j == jrand) ? clamp(a[j] + Ft * (b[j] - c[j]), LO, HI) : x[j];
            double ts = f(trial, DIM);
            if (ts <= scores[i]) { memcpy(next + i * DIM, trial, sizeof trial); next_scores[i] = ts; nFi[i] = Ft; nCRi[i] = CRt; }
            else { memcpy(next + i * DIM, x, sizeof trial); next_scores[i] = scores[i]; nFi[i] = Fi[i]; nCRi[i] = CRi[i]; }
        }
        double *t = pop; pop = next; next = t;
        t = scores; scores = next_scores; next_scores = t;
        t = Fi; Fi = nFi; nFi = t;
        t = CRi; CRi = nCRi; nCRi = t;
    }
    double best = scores[0];
    for (int i = 1; i < np; ++i) if (scores[i] < best) best = scores[i];
    free(pop); free(next); free(scores); free(next_scores); free(Fi); free(CRi); free(nFi); free(nCRi);
    return best;
}

int main(int argc, char **argv) {
    if (argc < 7) { fprintf(stderr, "usage: %s <sphere|rastrigin|rosenbrock|ackley|ellipsoid> <de|jde> <NP> <F> <CR> <seeds>\n", argv[0]); return 1; }
    loss_fn f = find_loss(argv[1]);
    if (!f) { fprintf(stderr, "unknown function %s\n", argv[1]); return 1; }
    int jde = !strcmp(argv[2], "jde"), np = atoi(argv[3]), seeds = atoi(argv[6]);
    double F = atof(argv[4]), CR = atof(argv[5]);
    if (np < 4) { fprintf(stderr, "need NP >= 4\n"); return 1; }
    init_rotation();
    double scores[64], ms[64];
    if (seeds > 64) seeds = 64;
    for (int s = 0; s < seeds; ++s) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        scores[s] = run_de(f, 42 + (uint64_t)s, np, F, CR, jde);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        ms[s] = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
        if (seeds == 1) printf("score %.17g\n", scores[s]);
    }
    qsort(scores, seeds, sizeof(double), cmp_double);
    qsort(ms, seeds, sizeof(double), cmp_double);
    printf("%-10s %-4s D=%-4d NP=%-3d F=%-4g CR=%-4g median %.4g   best %.4g   worst %.4g   %.0f ms/run\n",
           argv[1], argv[2], DIM, np, jde ? 0 : F, jde ? 0 : CR, scores[seeds / 2], scores[0], scores[seeds - 1], ms[seeds / 2]);
    return 0;
}

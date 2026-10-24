/* Canonical differential evolution, DE/rand/1/bin (Storn & Price 1997).
   Single-threaded. Same parameters, RNG (splitmix64) and draw order as
   ../roc and ../csharp, so all three should print the same score. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { DIM = 200, NP = 100, GENERATIONS = 10000 };
static const double F = 0.5, CR = 0.9, LO = -10.0, HI = 10.0;
static const uint64_t SEED = 42;

static uint64_t next_u64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static double to_unit(uint64_t v) { return (double)(v >> 11) * 1.1102230246251565e-16; }
static uint64_t rand_index(uint64_t *s, uint64_t n) { return next_u64(s) % n; }

static double sphere(const double *xs, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += xs[i] * xs[i];
    return sum;
}

static double clamp(double v) { return v < LO ? LO : (v > HI ? HI : v); }

int main(void) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    uint64_t rng = SEED;
    double *pop = malloc(sizeof(double) * NP * DIM);
    double *next = malloc(sizeof(double) * NP * DIM);
    double scores[NP], next_scores[NP], trial[DIM];

    for (int k = 0; k < NP * DIM; ++k) pop[k] = LO + to_unit(next_u64(&rng)) * (HI - LO);
    for (int i = 0; i < NP; ++i) scores[i] = sphere(pop + i * DIM, DIM);

    for (int g = 0; g < GENERATIONS; ++g) {
        for (uint64_t i = 0; i < NP; ++i) {
            uint64_t r1, r2, r3;
            do r1 = rand_index(&rng, NP); while (r1 == i);
            do r2 = rand_index(&rng, NP); while (r2 == i || r2 == r1);
            do r3 = rand_index(&rng, NP); while (r3 == i || r3 == r1 || r3 == r2);
            uint64_t jrand = rand_index(&rng, DIM);

            const double *a = pop + r1 * DIM, *b = pop + r2 * DIM, *c = pop + r3 * DIM, *x = pop + i * DIM;
            for (uint64_t j = 0; j < DIM; ++j) {
                if (to_unit(next_u64(&rng)) < CR || j == jrand)
                    trial[j] = clamp(a[j] + F * (b[j] - c[j]));
                else
                    trial[j] = x[j];
            }

            double ts = sphere(trial, DIM);
            if (ts <= scores[i]) {
                memcpy(next + i * DIM, trial, sizeof trial);
                next_scores[i] = ts;
            } else {
                memcpy(next + i * DIM, x, sizeof trial);
                next_scores[i] = scores[i];
            }
        }
        double *tmp = pop; pop = next; next = tmp;
        memcpy(scores, next_scores, sizeof scores);
    }

    double best = scores[0];
    for (int i = 1; i < NP; ++i) if (scores[i] < best) best = scores[i];

    clock_gettime(CLOCK_MONOTONIC, &t1);
    long ms = (t1.tv_sec - t0.tv_sec) * 1000L + (t1.tv_nsec - t0.tv_nsec) / 1000000L;
    printf("score %.17g\n", best);
    printf("execution time %ld ms\n", ms);
    free(pop); free(next);
    return 0;
}

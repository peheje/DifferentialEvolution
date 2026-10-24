/* Canonical DE/rand/1/bin, optimized. Bit-identical to de.c:
   - splitmix64 is counter based (state += GAMMA), so draws are batched and a
     rejected trial skips the rest of its draws in O(1).
   - CR test is done on the integer (u>>11) < ceil(CR*2^53): same as the double test.
   - Sphere terms are >= 0, so the in-order partial sum is monotone: once it
     exceeds scores[i] the trial can't be accepted (same order => same bits). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { DIM = 200, NP = 100, GENERATIONS = 10000, BLK = 4 };
static const double F = 0.5, CR = 0.9, LO = -10.0, HI = 10.0;
#define GAMMA 0x9E3779B97F4A7C15ULL

static inline uint64_t mix(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline uint64_t next_u64(uint64_t *s) { return mix(*s += GAMMA); }
static inline double to_unit(uint64_t v) { return (double)(v >> 11) * 1.1102230246251565e-16; }
static inline uint64_t rand_index(uint64_t *s, uint64_t n) { return next_u64(s) % n; }
static inline double clamp(double v) { return v < LO ? LO : (v > HI ? HI : v); }

int main(void) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    const uint64_t T = (uint64_t)__builtin_ceil(CR * 9007199254740992.0);

    uint64_t rng = 42;
    double *pop = aligned_alloc(64, sizeof(double) * NP * DIM);
    double *next = aligned_alloc(64, sizeof(double) * NP * DIM);
    double scores[NP];

    for (int k = 0; k < NP * DIM; ++k) pop[k] = LO + to_unit(next_u64(&rng)) * (HI - LO);
    for (int i = 0; i < NP; ++i) {
        double s = 0; for (int j = 0; j < DIM; ++j) s += pop[i*DIM+j] * pop[i*DIM+j];
        scores[i] = s;
    }

    for (int g = 0; g < GENERATIONS; ++g) {
        for (uint64_t i = 0; i < NP; ++i) {
            uint64_t r1, r2, r3;
            do r1 = rand_index(&rng, NP); while (r1 == i);
            do r2 = rand_index(&rng, NP); while (r2 == i || r2 == r1);
            do r3 = rand_index(&rng, NP); while (r3 == i || r3 == r1 || r3 == r2);
            uint64_t jrand = rand_index(&rng, DIM);

            const double *a = pop + r1 * DIM, *b = pop + r2 * DIM, *c = pop + r3 * DIM, *x = pop + i * DIM;
            double *out = next + i * DIM;
            const double lim = scores[i];
            double sum = 0.0;
            int j = 0, rejected = 0;
            for (; j < DIM; j += BLK) {
                uint64_t u[BLK];
                for (int k = 0; k < BLK; ++k) u[k] = mix(rng + (uint64_t)(k + 1) * GAMMA);
                rng += BLK * GAMMA;
                double t[BLK];
                for (int k = 0; k < BLK; ++k) {
                    int take = (u[k] >> 11) < T || (uint64_t)(j + k) == jrand;
                    double v = clamp(a[j+k] + F * (b[j+k] - c[j+k]));
                    t[k] = take ? v : x[j+k];
                }
                for (int k = 0; k < BLK; ++k) { out[j+k] = t[k]; sum += t[k] * t[k]; }
                if (sum > lim) { rng += (uint64_t)(DIM - j - BLK) * GAMMA; rejected = 1; break; }
            }
            if (rejected || sum > lim) {
                memcpy(out, x, DIM * sizeof(double));
                /* score unchanged */
            } else {
                scores[i] = sum;
            }
        }
        double *tmp = pop; pop = next; next = tmp;
    }

    double best = scores[0];
    for (int i = 1; i < NP; ++i) if (scores[i] < best) best = scores[i];
    clock_gettime(CLOCK_MONOTONIC, &t1);
    long ms = (t1.tv_sec - t0.tv_sec) * 1000L + (t1.tv_nsec - t0.tv_nsec) / 1000000L;
    printf("score %.17g\nexecution time %ld ms\n", best, ms);
    return 0;
}

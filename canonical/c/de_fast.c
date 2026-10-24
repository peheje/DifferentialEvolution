/* Canonical DE/rand/1/bin, optimized (AVX2). Bit-identical to de.c:
   - splitmix64 is counter based (state += GAMMA), so each trial's 200 draws
     start at a known state and are computed 4 at a time with AVX2.
   - CR test is done on the integer (u>>11) < ceil(CR*2^53): same as the double test.
   - The sphere sum is fused into the loop, in index order (same bits as de.c).
   - K trials are evaluated in lock-step to overlap their serial sum chains.
   - Accepted trials are copied back after the generation; rejected ones cost no copy.
   Build: gcc -O3 -march=native -ffp-contract=off */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
static inline __m256i mul64(__m256i a, __m256i b) { /* low 64 bits of a*b */
    __m256i bswap = _mm256_shuffle_epi32(b, 0xB1);
    __m256i prod = _mm256_mullo_epi32(a, bswap);
    __m256i zero = _mm256_setzero_si256();
    __m256i sum = _mm256_hadd_epi32(prod, zero);
    __m256i hi = _mm256_shuffle_epi32(sum, 0x73);
    return _mm256_add_epi64(_mm256_mul_epu32(a, b), hi);
}

enum { DIM = 200, NP = 100, GENERATIONS = 10000, BLK = 4, K = 2 };
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
    const __m256i Tv = _mm256_set1_epi64x((long long)T), four = _mm256_set1_epi64x(4);
    const __m256i M1 = _mm256_set1_epi64x((long long)0xBF58476D1CE4E5B9ULL), M2 = _mm256_set1_epi64x((long long)0x94D049BB133111EBULL);
    const __m256d Fv = _mm256_set1_pd(F), Hv = _mm256_set1_pd(HI), Lv = _mm256_set1_pd(LO);
    const __m256i ctr = _mm256_set_epi64x((long long)(4 * GAMMA), (long long)(3 * GAMMA), (long long)(2 * GAMMA), (long long)GAMMA);

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
        int acc[NP], na = 0;
        for (int i0 = 0; i0 < NP; i0 += K) {
            const double *A[K], *B[K], *C[K], *X[K];
            double *O[K], lim[K], sum[K];
            __m256i base[K], jr[K];
            for (int k = 0; k < K; ++k) {
                uint64_t i = (uint64_t)(i0 + k), r1, r2, r3;
                do r1 = rand_index(&rng, NP); while (r1 == i);
                do r2 = rand_index(&rng, NP); while (r2 == i || r2 == r1);
                do r3 = rand_index(&rng, NP); while (r3 == i || r3 == r1 || r3 == r2);
                uint64_t jrand = rand_index(&rng, DIM);
                A[k] = pop + r1 * DIM; B[k] = pop + r2 * DIM; C[k] = pop + r3 * DIM; X[k] = pop + i * DIM;
                O[k] = next + i * DIM; lim[k] = scores[i]; sum[k] = 0.0;
                base[k] = _mm256_add_epi64(_mm256_set1_epi64x((long long)rng), ctr);
                jr[k] = _mm256_set1_epi64x((long long)jrand);
                rng += (uint64_t)DIM * GAMMA;
            }
            __m256i jv = _mm256_set_epi64x(3, 2, 1, 0);
            const __m256i gstep = _mm256_set1_epi64x((long long)(4 * GAMMA));
            for (int j = 0; j < DIM; j += 4) {
                #pragma GCC unroll 8
                for (int k = 0; k < K; ++k) {
                    __m256i z = base[k];
                    base[k] = _mm256_add_epi64(z, gstep);
                    z = mul64(_mm256_xor_si256(z, _mm256_srli_epi64(z, 30)), M1);
                    z = mul64(_mm256_xor_si256(z, _mm256_srli_epi64(z, 27)), M2);
                    z = _mm256_xor_si256(z, _mm256_srli_epi64(z, 31));
                    __m256i take = _mm256_or_si256(_mm256_cmpgt_epi64(Tv, _mm256_srli_epi64(z, 11)), _mm256_cmpeq_epi64(jv, jr[k]));
                    __m256d va = _mm256_load_pd(A[k] + j), vb = _mm256_load_pd(B[k] + j), vc = _mm256_load_pd(C[k] + j), vx = _mm256_load_pd(X[k] + j);
                    __m256d v = _mm256_add_pd(va, _mm256_mul_pd(Fv, _mm256_sub_pd(vb, vc)));
                    v = _mm256_max_pd(_mm256_min_pd(v, Hv), Lv);
                    __m256d r = _mm256_blendv_pd(vx, v, _mm256_castsi256_pd(take));
                    _mm256_store_pd(O[k] + j, r);
                    double sq[4] __attribute__((aligned(32)));
                    _mm256_store_pd(sq, _mm256_mul_pd(r, r));
                    double s = sum[k];
                    s += sq[0]; s += sq[1]; s += sq[2]; s += sq[3];
                    sum[k] = s;
                }
                jv = _mm256_add_epi64(jv, four);
            }
            for (int k = 0; k < K; ++k)
                if (sum[k] <= lim[k]) { scores[i0 + k] = sum[k]; acc[na++] = i0 + k; }
        }
        for (int k = 0; k < na; ++k) memcpy(pop + acc[k] * DIM, next + acc[k] * DIM, DIM * sizeof(double));
    }

    double best = scores[0];
    for (int i = 1; i < NP; ++i) if (scores[i] < best) best = scores[i];
    clock_gettime(CLOCK_MONOTONIC, &t1);
    long ms = (t1.tv_sec - t0.tv_sec) * 1000L + (t1.tv_nsec - t0.tv_nsec) / 1000000L;
    printf("score %.17g\nexecution time %ld ms\n", best, ms);
    return 0;
}

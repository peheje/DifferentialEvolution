#ifndef DE_SOLVER_H
#define DE_SOLVER_H

#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ── jDE self-adaptive parameter control (Brest et al. 2006) ─────
   Each individual carries its own F_i and CR_i. Before mutation:
     • with probability TAU_F: re-sample F_i  ~ U(F_LO, F_HI)
     • with probability TAU_CR: re-sample CR_i ~ U(0, 1)
   Winning parameters survive with the replacement; losing ones are
   re-rolled next generation. Selection pressure tunes them for free.
   To override: change these constants or wrap in #ifndef / -D.    */
static const double JDE_F_LO  = 0.1;
static const double JDE_F_HI  = 0.9;
static const double JDE_TAU_F = 0.1;
static const double JDE_TAU_CR = 0.1;

/* ── splitmix64 RNG ────────────────────────────────────────────── */
static inline uint64_t de_splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}


/* ── AVX2 splitmix64: 4 consecutive draws at once ─────────────────
   splitmix64 is counter based (state += GAMMA), so draws k=1..4 from state s
   are mix(s + k*GAMMA); a block of 4 yields exactly the same numbers, in the
   same order, as 4 calls to de_splitmix64. */
static inline __m256i de_mul64(__m256i a, __m256i b) { /* low 64 bits of a*b */
    __m256i bswap = _mm256_shuffle_epi32(b, 0xB1);
    __m256i prod = _mm256_mullo_epi32(a, bswap);
    __m256i sum = _mm256_hadd_epi32(prod, _mm256_setzero_si256());
    __m256i hi = _mm256_shuffle_epi32(sum, 0x73);
    return _mm256_add_epi64(_mm256_mul_epu32(a, b), hi);
}
static inline __m256i de_splitmix64x4(uint64_t *state) {
    const __m256i ctr = _mm256_set_epi64x((long long)(4 * 0x9e3779b97f4a7c15ULL), (long long)(3 * 0x9e3779b97f4a7c15ULL),
                                          (long long)(2 * 0x9e3779b97f4a7c15ULL), (long long)0x9e3779b97f4a7c15ULL);
    __m256i z = _mm256_add_epi64(_mm256_set1_epi64x((long long)*state), ctr);
    *state += 4 * 0x9e3779b97f4a7c15ULL;
    z = de_mul64(_mm256_xor_si256(z, _mm256_srli_epi64(z, 30)), _mm256_set1_epi64x((long long)0xbf58476d1ce4e5b9ULL));
    z = de_mul64(_mm256_xor_si256(z, _mm256_srli_epi64(z, 27)), _mm256_set1_epi64x((long long)0x94d049bb133111ebULL));
    return _mm256_xor_si256(z, _mm256_srli_epi64(z, 31));
}

static inline double de_rand01(uint64_t *state) {
    return (de_splitmix64(state) >> 11) * (1.0 / 9007199254740992.0);
}

static inline int de_rand_int(uint64_t *state, int n) {
    return (int)(de_splitmix64(state) % (uint64_t)n);
}

static inline double de_rand_range(uint64_t *state, double lo, double hi) {
    return de_rand01(state) * (hi - lo) + lo;
}

static inline double de_clamp(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static inline void de_pick3(uint64_t *state, int popsize, int i,
                            int *r0, int *r1, int *r2) {
    do { *r0 = de_rand_int(state, popsize); } while (*r0 == i);
    do { *r1 = de_rand_int(state, popsize); } while (*r1 == i || *r1 == *r0);
    do { *r2 = de_rand_int(state, popsize); } while (*r2 == i || *r2 == *r0 || *r2 == *r1);
}

typedef struct {
    double score;
    int index;
} DeRank;

static int de_rank_compare(const void *a, const void *b) {
    const DeRank *ra = (const DeRank *)a;
    const DeRank *rb = (const DeRank *)b;
    return (ra->score > rb->score) - (ra->score < rb->score);
}

static inline int de_lpsr_population_size(int initial_popsize, int min_popsize,
                                          int generations, int generation) {
    if (min_popsize >= initial_popsize || generations <= 0)
        return initial_popsize;

    const double t = (double)(generation + 1) / (double)generations;
    int next = (int)((double)initial_popsize
        - (double)(initial_popsize - min_popsize) * t + 0.5);
    if (next < min_popsize) next = min_popsize;
    if (next > initial_popsize) next = initial_popsize;
    return next;
}

/* ── DE/rand/1/bin with jDE ──────────────────────────────────────
   Returns the best objective value found. The population converges
   in-place; only the final best score is reported.               */
static double de_solve_impl(
    int      params,
    int      initial_popsize,
    int      min_popsize,
    int      generations,
    double   bounds_min,
    double   bounds_max,
    double (*objective)(const double *xs, int n),
    uint64_t seed)
{
    if (params <= 0 || initial_popsize < 4 || min_popsize < 4 ||
        min_popsize > initial_popsize || generations < 0 || !objective)
        return INFINITY;

    const size_t agent_len = (size_t)params;
    const size_t capacity_len = (size_t)initial_popsize * agent_len;
    const int threads = omp_get_max_threads();
    const int use_lpsr = min_popsize < initial_popsize;

    double *pop           = (double *)malloc(capacity_len * sizeof(double));
    double *prev_pop      = (double *)malloc(capacity_len * sizeof(double));
    double *trial_scratch = (double *)malloc((size_t)threads * agent_len * sizeof(double));
    double *scores        = (double *)malloc((size_t)initial_popsize * sizeof(double));
    unsigned char *accepted = (unsigned char *)calloc((size_t)initial_popsize, 1);
    double *Fi            = (double *)malloc((size_t)initial_popsize * sizeof(double));
    double *CRi           = (double *)malloc((size_t)initial_popsize * sizeof(double));
    double *tmp_scores    = use_lpsr ? (double *)malloc((size_t)initial_popsize * sizeof(double)) : NULL;
    double *tmp_Fi        = use_lpsr ? (double *)malloc((size_t)initial_popsize * sizeof(double)) : NULL;
    double *tmp_CRi       = use_lpsr ? (double *)malloc((size_t)initial_popsize * sizeof(double)) : NULL;
    DeRank *rank          = use_lpsr ? (DeRank *)malloc((size_t)initial_popsize * sizeof(DeRank)) : NULL;
    if (!pop || !prev_pop || !trial_scratch || !accepted || !scores || !Fi || !CRi ||
        (use_lpsr && (!tmp_scores || !tmp_Fi || !tmp_CRi || !rank))) {
        free(pop); free(prev_pop); free(trial_scratch);
        free(scores); free(Fi); free(CRi);
        free(tmp_scores); free(tmp_Fi); free(tmp_CRi); free(rank);
        return INFINITY;
    }

    uint64_t *rng_states = (uint64_t *)malloc((size_t)threads * sizeof(uint64_t));
    if (!rng_states) {
        free(pop); free(prev_pop); free(trial_scratch);
        free(scores); free(Fi); free(CRi);
        free(tmp_scores); free(tmp_Fi); free(tmp_CRi); free(rank);
        return INFINITY;
    }
    for (int t = 0; t < threads; ++t) {
        uint64_t s = seed + (uint64_t)t * 0x9e3779b97f4a7c15ULL;
        rng_states[t] = de_splitmix64(&s);
    }

    /* ── initialise population ───────────────────────────────── */
    for (int i = 0; i < initial_popsize; ++i) {
        double *agent = pop + (size_t)i * agent_len;
        for (int j = 0; j < params; ++j)
            agent[j] = de_rand_range(&rng_states[0], bounds_min, bounds_max);
        scores[i] = objective(agent, params);
        Fi[i]  = JDE_F_LO + de_rand01(&rng_states[0]) * (JDE_F_HI - JDE_F_LO);
        CRi[i] = de_rand01(&rng_states[0]);
    }

    int active_popsize = initial_popsize;

    /* ── main loop ───────────────────────────────────────────── */
#pragma omp parallel default(none) shared(params, agent_len, pop, prev_pop, accepted, trial_scratch, scores, Fi, CRi, tmp_scores, tmp_Fi, tmp_CRi, rank, rng_states, active_popsize, initial_popsize, min_popsize, use_lpsr, generations, bounds_min, bounds_max, objective)
    {
        const int tid = omp_get_thread_num();
        uint64_t local_state = rng_states[tid]; /* private copy: adjacent rng_states entries would false-share */
        uint64_t *state = &local_state;

        for (int g = 0; g < generations; ++g) {
            const int active = active_popsize;

#pragma omp for schedule(static)
            for (int i = 0; i < active; ++i) {
                /* ── jDE: maybe re-sample F_i / CR_i ──────────── */
                double Fi_try  = Fi[i];
                double CRi_try = CRi[i];
                if (de_rand01(state) < JDE_TAU_F)
                    Fi_try = JDE_F_LO + de_rand01(state) * (JDE_F_HI - JDE_F_LO);
                if (de_rand01(state) < JDE_TAU_CR)
                    CRi_try = de_rand01(state);

                /* ── DE/rand/1/bin ───────────────────────────── */
                int i0, i1, i2;
                de_pick3(state, active, i, &i0, &i1, &i2);

                const double *x0 = pop + (size_t)i0 * agent_len;
                const double *x1 = pop + (size_t)i1 * agent_len;
                const double *x2 = pop + (size_t)i2 * agent_len;
                const double *xt_prev = pop + (size_t)i * agent_len;
                double *trial = prev_pop + (size_t)i * agent_len; /* staging row for individual i */

                /* de_rand01(state) < CR  <=>  (draw >> 11) < ceil(CR * 2^53) */
                const __m256i thr = _mm256_set1_epi64x((long long)ceil(CRi_try * 9007199254740992.0));
                const __m256d vF = _mm256_set1_pd(Fi_try);
                const __m256d vlo = _mm256_set1_pd(bounds_min), vhi = _mm256_set1_pd(bounds_max);
                int j = 0;
                for (; j + 4 <= params; j += 4) {
                    __m256i u = de_splitmix64x4(state);
                    __m256d take = _mm256_castsi256_pd(_mm256_cmpgt_epi64(thr, _mm256_srli_epi64(u, 11)));
                    __m256d v = _mm256_add_pd(_mm256_loadu_pd(x0 + j),
                                              _mm256_mul_pd(_mm256_sub_pd(_mm256_loadu_pd(x1 + j), _mm256_loadu_pd(x2 + j)), vF));
                    v = _mm256_max_pd(_mm256_min_pd(v, vhi), vlo);
                    _mm256_storeu_pd(trial + j, _mm256_blendv_pd(_mm256_loadu_pd(xt_prev + j), v, take));
                }
                for (; j < params; ++j) {
                    if (de_rand01(state) < CRi_try)
                        trial[j] = de_clamp(x0[j] + (x1[j] - x2[j]) * Fi_try, bounds_min, bounds_max);
                    else
                        trial[j] = xt_prev[j];
                }

                const double trial_score = objective(trial, params);
                if (trial_score < scores[i]) {
                    accepted[i] = 1;
                    scores[i] = trial_score;
                    Fi[i]  = Fi_try;
                    CRi[i] = CRi_try;
                }
            }

            /* commit accepted trials (all parents were read from the old population above) */
#pragma omp for schedule(static)
            for (int i = 0; i < active; ++i) {
                if (accepted[i]) {
                    memcpy(pop + (size_t)i * agent_len, prev_pop + (size_t)i * agent_len, agent_len * sizeof(double));
                    accepted[i] = 0;
                }
            }

            if (use_lpsr) {
#pragma omp single
                {
                    const int next_active = de_lpsr_population_size(initial_popsize,
                                                                   min_popsize,
                                                                   generations,
                                                                   g);
                    if (next_active < active_popsize) {
                        for (int i = 0; i < active_popsize; ++i) {
                            rank[i].score = scores[i];
                            rank[i].index = i;
                        }
                        qsort(rank, (size_t)active_popsize, sizeof(rank[0]), de_rank_compare);

                        for (int i = 0; i < next_active; ++i) {
                            const int old = rank[i].index;
                            memcpy(prev_pop + (size_t)i * agent_len,
                                   pop + (size_t)old * agent_len,
                                   agent_len * sizeof(double));
                            tmp_scores[i] = scores[old];
                            tmp_Fi[i] = Fi[old];
                            tmp_CRi[i] = CRi[old];
                        }

                        memcpy(pop, prev_pop, (size_t)next_active * agent_len * sizeof(double));
                        memcpy(scores, tmp_scores, (size_t)next_active * sizeof(double));
                        memcpy(Fi, tmp_Fi, (size_t)next_active * sizeof(double));
                        memcpy(CRi, tmp_CRi, (size_t)next_active * sizeof(double));
                        active_popsize = next_active;
                    }
                }
            }
        }
    }

    double best = scores[0];
    for (int i = 1; i < active_popsize; ++i)
        if (scores[i] < best) best = scores[i];

    free(rng_states);
    free(pop);
    free(prev_pop);
    free(trial_scratch);
    free(accepted);
    free(scores);
    free(Fi);
    free(CRi);
    free(tmp_scores);
    free(tmp_Fi);
    free(tmp_CRi);
    free(rank);
    return best;
}

static inline double de_solve(
    int      params,
    int      popsize,
    int      generations,
    double   bounds_min,
    double   bounds_max,
    double (*objective)(const double *xs, int n),
    uint64_t seed)
{
    return de_solve_impl(params, popsize, popsize, generations,
                         bounds_min, bounds_max, objective, seed);
}

static inline double de_solve_lpsr(
    int      params,
    int      initial_popsize,
    int      min_popsize,
    int      generations,
    double   bounds_min,
    double   bounds_max,
    double (*objective)(const double *xs, int n),
    uint64_t seed)
{
    return de_solve_impl(params, initial_popsize, min_popsize, generations,
                         bounds_min, bounds_max, objective, seed);
}

#endif /* DE_SOLVER_H */

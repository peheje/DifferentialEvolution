#ifndef DE_SOLVER_H
#define DE_SOLVER_H

#include <math.h>
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

/* ── DE/rand/1/bin with jDE ──────────────────────────────────────
   Returns the best objective value found. The population converges
   in-place; only the final best score is reported.               */
static double de_solve(
    int      params,
    int      popsize,
    int      generations,
    double   bounds_min,
    double   bounds_max,
    double (*objective)(const double *xs, int n),
    uint64_t seed)
{
    if (params <= 0 || popsize < 4 || generations < 0 || !objective)
        return INFINITY;

    const size_t agent_len = (size_t)params;
    const size_t pop_len   = (size_t)popsize * agent_len;
    const int threads = omp_get_max_threads();

    double *pop           = (double *)malloc(pop_len * sizeof(double));
    double *prev_pop      = (double *)malloc(pop_len * sizeof(double));
    double *trial_scratch = (double *)malloc((size_t)threads * agent_len * sizeof(double));
    double *scores        = (double *)malloc((size_t)popsize * sizeof(double));
    double *Fi            = (double *)malloc((size_t)popsize * sizeof(double));
    double *CRi           = (double *)malloc((size_t)popsize * sizeof(double));
    if (!pop || !prev_pop || !trial_scratch || !scores || !Fi || !CRi) {
        free(pop); free(prev_pop); free(trial_scratch);
        free(scores); free(Fi); free(CRi);
        return INFINITY;
    }

    uint64_t *rng_states = (uint64_t *)malloc((size_t)threads * sizeof(uint64_t));
    if (!rng_states) {
        free(pop); free(prev_pop); free(trial_scratch);
        free(scores); free(Fi); free(CRi);
        return INFINITY;
    }
    for (int t = 0; t < threads; ++t) {
        uint64_t s = seed + (uint64_t)t * 0x9e3779b97f4a7c15ULL;
        rng_states[t] = de_splitmix64(&s);
    }

    /* ── initialise population ───────────────────────────────── */
    for (int i = 0; i < popsize; ++i) {
        double *agent = pop + (size_t)i * agent_len;
        for (int j = 0; j < params; ++j)
            agent[j] = de_rand_range(&rng_states[0], bounds_min, bounds_max);
        scores[i] = objective(agent, params);
        Fi[i]  = JDE_F_LO + de_rand01(&rng_states[0]) * (JDE_F_HI - JDE_F_LO);
        CRi[i] = de_rand01(&rng_states[0]);
    }

    /* ── main loop ───────────────────────────────────────────── */
#pragma omp parallel default(none) shared(params, agent_len, pop_len, pop, prev_pop, trial_scratch, scores, Fi, CRi, rng_states, popsize, generations, bounds_min, bounds_max, objective)
    {
        const int tid = omp_get_thread_num();
        uint64_t *state = &rng_states[tid];
        double *trial = trial_scratch + (size_t)tid * agent_len;

        for (int g = 0; g < generations; ++g) {
#pragma omp for schedule(static)
            for (size_t k = 0; k < pop_len; ++k)
                prev_pop[k] = pop[k];

#pragma omp for schedule(static)
            for (int i = 0; i < popsize; ++i) {
                /* ── jDE: maybe re-sample F_i / CR_i ──────────── */
                double Fi_try  = Fi[i];
                double CRi_try = CRi[i];
                if (de_rand01(state) < JDE_TAU_F)
                    Fi_try = JDE_F_LO + de_rand01(state) * (JDE_F_HI - JDE_F_LO);
                if (de_rand01(state) < JDE_TAU_CR)
                    CRi_try = de_rand01(state);

                /* ── DE/rand/1/bin ───────────────────────────── */
                int i0, i1, i2;
                de_pick3(state, popsize, i, &i0, &i1, &i2);

                const double *x0 = prev_pop + (size_t)i0 * agent_len;
                const double *x1 = prev_pop + (size_t)i1 * agent_len;
                const double *x2 = prev_pop + (size_t)i2 * agent_len;
                const double *xt_prev = prev_pop + (size_t)i * agent_len;
                double *xt       = pop + (size_t)i  * agent_len;

                for (int j = 0; j < params; ++j) {
                    if (de_rand01(state) < CRi_try)
                        trial[j] = de_clamp(x0[j] + (x1[j] - x2[j]) * Fi_try,
                                            bounds_min, bounds_max);
                    else
                        trial[j] = xt_prev[j];
                }

                const double trial_score = objective(trial, params);
                if (trial_score < scores[i]) {
                    memcpy(xt, trial, agent_len * sizeof(double));
                    scores[i] = trial_score;
                    Fi[i]  = Fi_try;
                    CRi[i] = CRi_try;
                }
            }
        }
    }

    double best = scores[0];
    for (int i = 1; i < popsize; ++i)
        if (scores[i] < best) best = scores[i];

    free(rng_states);
    free(pop);
    free(prev_pop);
    free(trial_scratch);
    free(scores);
    free(Fi);
    free(CRi);
    return best;
}

#endif /* DE_SOLVER_H */

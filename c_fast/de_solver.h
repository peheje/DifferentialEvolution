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

static inline void de_pick2_excluding(uint64_t *state, int popsize, int i, int excl,
                                       int *r0, int *r1) {
    do { *r0 = de_rand_int(state, popsize); } while (*r0 == i || *r0 == excl);
    do { *r1 = de_rand_int(state, popsize); } while (*r1 == i || *r1 == excl || *r1 == *r0);
}

static inline void de_pick2(uint64_t *state, int popsize, int i,
                             int *r0, int *r1) {
    do { *r0 = de_rand_int(state, popsize); } while (*r0 == i);
    do { *r1 = de_rand_int(state, popsize); } while (*r1 == i || *r1 == *r0);
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

typedef struct {
    int params;
    int capacity_popsize;
    int active_popsize;
    double *pop;
    double *scores;
    double *Fi;
    double *CRi;
} DeState;

static int de_rank_compare(const void *a, const void *b) {
    const DeRank *ra = (const DeRank *)a;
    const DeRank *rb = (const DeRank *)b;
    return (ra->score > rb->score) - (ra->score < rb->score);
}

static void de_state_free(DeState *state) {
    if (!state)
        return;
    free(state->pop);
    free(state->scores);
    free(state->Fi);
    free(state->CRi);
    state->params = 0;
    state->capacity_popsize = 0;
    state->active_popsize = 0;
    state->pop = NULL;
    state->scores = NULL;
    state->Fi = NULL;
    state->CRi = NULL;
}

static int de_state_alloc(DeState *state, int params, int popsize) {
    if (!state || params <= 0 || popsize < 4)
        return 0;

    memset(state, 0, sizeof(*state));
    state->params = params;
    state->capacity_popsize = popsize;
    state->active_popsize = popsize;
    state->pop = (double *)malloc((size_t)popsize * (size_t)params * sizeof(double));
    state->scores = (double *)malloc((size_t)popsize * sizeof(double));
    state->Fi = (double *)malloc((size_t)popsize * sizeof(double));
    state->CRi = (double *)malloc((size_t)popsize * sizeof(double));
    if (!state->pop || !state->scores || !state->Fi || !state->CRi) {
        de_state_free(state);
        return 0;
    }
    return 1;
}

static void de_state_init_random(
    DeState *state,
    double bounds_min,
    double bounds_max,
    double (*objective)(const double *xs, int n),
    uint64_t seed)
{
    uint64_t rng = seed;
    const size_t agent_len = (size_t)state->params;
    state->active_popsize = state->capacity_popsize;
    for (int i = 0; i < state->active_popsize; ++i) {
        double *agent = state->pop + (size_t)i * agent_len;
        for (int j = 0; j < state->params; ++j)
            agent[j] = de_rand_range(&rng, bounds_min, bounds_max);
        state->scores[i] = objective(agent, state->params);
        state->Fi[i] = JDE_F_LO + de_rand01(&rng) * (JDE_F_HI - JDE_F_LO);
        state->CRi[i] = de_rand01(&rng);
    }
}

static void de_state_rescore(
    DeState *state,
    double (*objective)(const double *xs, int n))
{
    const size_t agent_len = (size_t)state->params;
    for (int i = 0; i < state->active_popsize; ++i)
        state->scores[i] = objective(state->pop + (size_t)i * agent_len,
                                     state->params);
}

static int de_state_best_index(const DeState *state) {
    int best_idx = 0;
    double best = state->scores[0];
    for (int i = 1; i < state->active_popsize; ++i) {
        if (state->scores[i] < best) {
            best = state->scores[i];
            best_idx = i;
        }
    }
    return best_idx;
}

static double de_state_best_score(const DeState *state, double *best_out) {
    const int best_idx = de_state_best_index(state);
    if (best_out)
        memcpy(best_out, state->pop + (size_t)best_idx * (size_t)state->params,
               (size_t)state->params * sizeof(double));
    return state->scores[best_idx];
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

static double de_state_evolve(
    DeState *de_state,
    int      min_popsize,
    int      generations,
    double   bounds_min,
    double   bounds_max,
    double (*objective)(const double *xs, int n),
    uint64_t seed,
    double  *best_out,
    int      mutation_strategy,
    int      pbest_percent)
{
    if (!de_state || de_state->params <= 0 || de_state->active_popsize < 4 ||
        min_popsize < 4 || min_popsize > de_state->active_popsize ||
        generations < 0 || !objective)
        return INFINITY;

    const int params = de_state->params;
    const int initial_popsize = de_state->active_popsize;
    const size_t agent_len = (size_t)params;
    const size_t capacity_len = (size_t)initial_popsize * agent_len;
    const int threads = omp_get_max_threads();
    const int use_lpsr = min_popsize < initial_popsize;
    double *prev_pop      = (double *)malloc(capacity_len * sizeof(double));
    double *trial_scratch = (double *)malloc((size_t)threads * agent_len * sizeof(double));
    double *tmp_scores    = use_lpsr ? (double *)malloc((size_t)initial_popsize * sizeof(double)) : NULL;
    double *tmp_Fi        = use_lpsr ? (double *)malloc((size_t)initial_popsize * sizeof(double)) : NULL;
    double *tmp_CRi       = use_lpsr ? (double *)malloc((size_t)initial_popsize * sizeof(double)) : NULL;
    const int use_rank = use_lpsr || mutation_strategy == 4;
    DeRank *rank          = use_rank ? (DeRank *)malloc((size_t)initial_popsize * sizeof(DeRank)) : NULL;
    if (!prev_pop || !trial_scratch ||
        ((use_lpsr && (!tmp_scores || !tmp_Fi || !tmp_CRi)) ||
         (use_rank && !rank))) {
        free(prev_pop); free(trial_scratch);
        free(tmp_scores); free(tmp_Fi); free(tmp_CRi); free(rank);
        return INFINITY;
    }

    uint64_t *rng_states = (uint64_t *)malloc((size_t)threads * sizeof(uint64_t));
    if (!rng_states) {
        free(prev_pop); free(trial_scratch);
        free(tmp_scores); free(tmp_Fi); free(tmp_CRi); free(rank);
        return INFINITY;
    }
    for (int t = 0; t < threads; ++t) {
        uint64_t s = seed + (uint64_t)t * 0x9e3779b97f4a7c15ULL;
        rng_states[t] = de_splitmix64(&s);
    }

    int active_popsize = initial_popsize;
    double *pop = de_state->pop;
    double *scores = de_state->scores;
    double *Fi = de_state->Fi;
    double *CRi = de_state->CRi;

    /* ── main loop ───────────────────────────────────────────── */
    int best_idx = 0;
#pragma omp parallel default(none) shared(params, agent_len, pop, prev_pop, trial_scratch, scores, Fi, CRi, tmp_scores, tmp_Fi, tmp_CRi, rank, rng_states, active_popsize, initial_popsize, min_popsize, use_lpsr, generations, bounds_min, bounds_max, objective, best_idx, mutation_strategy, pbest_percent)
    {
        const int tid = omp_get_thread_num();
        uint64_t *state = &rng_states[tid];
        double *trial = trial_scratch + (size_t)tid * agent_len;

        for (int g = 0; g < generations; ++g) {
            const int active = active_popsize;
            const size_t active_len = (size_t)active * agent_len;

#pragma omp for schedule(static)
            for (size_t k = 0; k < active_len; ++k)
                prev_pop[k] = pop[k];

            if (mutation_strategy) {
#pragma omp single
                {
                    best_idx = 0;
                    double best = scores[0];
                    for (int i = 1; i < active; ++i) {
                        if (scores[i] < best) {
                            best = scores[i];
                            best_idx = i;
                        }
                    }
                    if (mutation_strategy == 4) {
                        for (int i = 0; i < active; ++i) {
                            rank[i].score = scores[i];
                            rank[i].index = i;
                        }
                        qsort(rank, (size_t)active, sizeof(rank[0]), de_rank_compare);
                    }
                }
            }

#pragma omp for schedule(static)
            for (int i = 0; i < active; ++i) {
                /* ── jDE: maybe re-sample F_i / CR_i ──────────── */
                double Fi_try  = Fi[i];
                double CRi_try = CRi[i];
                if (de_rand01(state) < JDE_TAU_F)
                    Fi_try = JDE_F_LO + de_rand01(state) * (JDE_F_HI - JDE_F_LO);
                if (de_rand01(state) < JDE_TAU_CR)
                    CRi_try = de_rand01(state);

                const double *x0, *x1, *x2;
                const double *xt_prev = prev_pop + (size_t)i * agent_len;
                double *xt       = pop + (size_t)i  * agent_len;
                int is_curtobest = 0;
                const int jrand = de_rand_int(state, params);

                if (mutation_strategy == 1) {
                    /* ── DE/best/1/bin ────────────────────── */
                    int i1, i2;
                    de_pick2_excluding(state, active, i, best_idx, &i1, &i2);
                    x0 = prev_pop + (size_t)best_idx * agent_len;
                    x1 = prev_pop + (size_t)i1      * agent_len;
                    x2 = prev_pop + (size_t)i2      * agent_len;
                } else if (mutation_strategy == 2) {
                    /* ── DE/current-to-best/1/bin ─────────── */
                    int i1, i2;
                    de_pick2(state, active, i, &i1, &i2);
                    x0 = prev_pop + (size_t)best_idx * agent_len;
                    x1 = prev_pop + (size_t)i1      * agent_len;
                    x2 = prev_pop + (size_t)i2      * agent_len;
                    is_curtobest = 1;
                } else if (mutation_strategy == 3) {
                    /* ── DE/annealed-rand-to-best/1/bin ──────
                       λ = (gens-g)/gens, anneals rand→best  */
                    int ir, i1, i2;
                    de_pick2_excluding(state, active, i, best_idx, &ir, &i1);
                    do { i2 = de_rand_int(state, active); }
                        while (i2 == i || i2 == ir || i2 == i1);
                    x0 = prev_pop + (size_t)ir       * agent_len;
                    x1 = prev_pop + (size_t)i1       * agent_len;
                    x2 = prev_pop + (size_t)i2       * agent_len;
                    {
                        const double lam = (double)(generations - g)
                                         / (double)generations;
                        const double *xb = prev_pop
                                         + (size_t)best_idx * agent_len;
                        for (int j = 0; j < params; ++j) {
                            const double base = lam * x0[j]
                                              + (1.0 - lam) * xb[j];
                            if (j == jrand || de_rand01(state) < CRi_try)
                                trial[j] = de_clamp(
                                    base + (x1[j] - x2[j]) * Fi_try,
                                    bounds_min, bounds_max);
                            else
                                trial[j] = xt_prev[j];
                        }
                        /* skip the normal param loop below */
                        const double trial_score = objective(trial, params);
                        if (trial_score < scores[i]) {
                            memcpy(xt, trial, agent_len * sizeof(double));
                            scores[i] = trial_score;
                            Fi[i]  = Fi_try;
                            CRi[i] = CRi_try;
                        }
                        continue;
                    }
                } else if (mutation_strategy == 4) {
                    /* ── DE/current-to-pbest/1/bin ───────────
                       JADE-style p-best without external archive. */
                    int pbest_count = (active * pbest_percent + 99) / 100;
                    if (pbest_count < 1)
                        pbest_count = 1;
                    if (pbest_count > active)
                        pbest_count = active;

                    int pbest_idx = rank[de_rand_int(state, pbest_count)].index;
                    if (pbest_count > 1) {
                        int guard = 0;
                        while (pbest_idx == i && guard++ < 8)
                            pbest_idx = rank[de_rand_int(state, pbest_count)].index;
                    }

                    int i1, i2;
                    de_pick2_excluding(state, active, i, pbest_idx, &i1, &i2);
                    x0 = prev_pop + (size_t)pbest_idx * agent_len;
                    x1 = prev_pop + (size_t)i1        * agent_len;
                    x2 = prev_pop + (size_t)i2        * agent_len;
                    is_curtobest = 1;
                } else {
                    /* ── DE/rand/1/bin ────────────────────── */
                    int i0, i1, i2;
                    de_pick3(state, active, i, &i0, &i1, &i2);
                    x0 = prev_pop + (size_t)i0 * agent_len;
                    x1 = prev_pop + (size_t)i1 * agent_len;
                    x2 = prev_pop + (size_t)i2 * agent_len;
                }

                for (int j = 0; j < params; ++j) {
                    if (j == jrand || de_rand01(state) < CRi_try) {
                        if (is_curtobest)
                            trial[j] = de_clamp(
                                xt_prev[j]
                                + Fi_try * (x0[j] - xt_prev[j])
                                + Fi_try * (x1[j] - x2[j]),
                                bounds_min, bounds_max);
                        else
                            trial[j] = de_clamp(
                                x0[j] + (x1[j] - x2[j]) * Fi_try,
                                bounds_min, bounds_max);
                    } else {
                        trial[j] = xt_prev[j];
                    }
                }

                const double trial_score = objective(trial, params);
                if (trial_score < scores[i]) {
                    memcpy(xt, trial, agent_len * sizeof(double));
                    scores[i] = trial_score;
                    Fi[i]  = Fi_try;
                    CRi[i] = CRi_try;
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

    de_state->active_popsize = active_popsize;
    const double best = de_state_best_score(de_state, best_out);

    free(rng_states);
    free(prev_pop);
    free(trial_scratch);
    free(tmp_scores);
    free(tmp_Fi);
    free(tmp_CRi);
    free(rank);
    return best;
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
    uint64_t seed,
    double  *best_out,
    int      mutation_strategy,
    int      pbest_percent)
{
    if (params <= 0 || initial_popsize < 4 || min_popsize < 4 ||
        min_popsize > initial_popsize || generations < 0 || !objective)
        return INFINITY;

    DeState state;
    if (!de_state_alloc(&state, params, initial_popsize))
        return INFINITY;

    de_state_init_random(&state, bounds_min, bounds_max, objective, seed);
    const double best = de_state_evolve(&state, min_popsize, generations,
                                        bounds_min, bounds_max, objective,
                                        seed, best_out, mutation_strategy,
                                        pbest_percent);
    de_state_free(&state);
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
                         bounds_min, bounds_max, objective, seed, NULL, 0, 0);
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
                         bounds_min, bounds_max, objective, seed, NULL, 0, 0);
}

static inline double de_solve_lpsr_best(
    int      params,
    int      initial_popsize,
    int      min_popsize,
    int      generations,
    double   bounds_min,
    double   bounds_max,
    double (*objective)(const double *xs, int n),
    uint64_t seed,
    double  *best_out)
{
    return de_solve_impl(params, initial_popsize, min_popsize, generations,
                         bounds_min, bounds_max, objective, seed, best_out, 0, 0);
}

static inline double de_solve_lpsr_best1(
    int      params,
    int      initial_popsize,
    int      min_popsize,
    int      generations,
    double   bounds_min,
    double   bounds_max,
    double (*objective)(const double *xs, int n),
    uint64_t seed,
    double  *best_out)
{
    return de_solve_impl(params, initial_popsize, min_popsize, generations,
                         bounds_min, bounds_max, objective, seed, best_out, 1, 0);
}

static inline double de_solve_lpsr_curtobest1(
    int      params,
    int      initial_popsize,
    int      min_popsize,
    int      generations,
    double   bounds_min,
    double   bounds_max,
    double (*objective)(const double *xs, int n),
    uint64_t seed,
    double  *best_out)
{
    return de_solve_impl(params, initial_popsize, min_popsize, generations,
                         bounds_min, bounds_max, objective, seed, best_out, 2, 0);
}

static inline double de_solve_lpsr_anneal(
    int      params,
    int      initial_popsize,
    int      min_popsize,
    int      generations,
    double   bounds_min,
    double   bounds_max,
    double (*objective)(const double *xs, int n),
    uint64_t seed,
    double  *best_out)
{
    return de_solve_impl(params, initial_popsize, min_popsize, generations,
                         bounds_min, bounds_max, objective, seed, best_out, 3, 0);
}

static inline double de_solve_lpsr_pbest(
    int      params,
    int      initial_popsize,
    int      min_popsize,
    int      generations,
    double   bounds_min,
    double   bounds_max,
    double (*objective)(const double *xs, int n),
    uint64_t seed,
    double  *best_out,
    int      pbest_percent)
{
    if (pbest_percent < 1)
        pbest_percent = 1;
    if (pbest_percent > 100)
        pbest_percent = 100;
    return de_solve_impl(params, initial_popsize, min_popsize, generations,
                         bounds_min, bounds_max, objective, seed, best_out, 4,
                         pbest_percent);
}

#endif /* DE_SOLVER_H */

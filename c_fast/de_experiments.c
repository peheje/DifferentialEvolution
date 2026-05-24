#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum {
    POPSIZE = 200,
    PRINT_EVERY = 0
};

/* ── variant tags ───────────────────────────────────────────────── */
typedef enum {
    V_BASELINE = 0,           /* DE/rand/1/bin, random F/CR per-gen, clamp bounds   */
    V_CURRENT_TO_BEST,        /* DE/current-to-best/1/bin, random F/CR per-gen      */
    V_JDE,                    /* DE/rand/1/bin with jDE self-adaptive F/CR           */
    V_CTB_JDE,                /* DE/current-to-best/1/bin with jDE adaptive F/CR     */
    V_BOUNCE,                 /* DE/rand/1/bin, random F/CR, bounce-back bounds      */
    V_ARCHIVE,                /* DE/rand/1/bin with external archive (SHADE-like)    */
    V_CTB_ARCHIVE,            /* DE/current-to-best/1 with archive                   */
    V_VARIANT_COUNT
} Variant;

static const char *variant_names[] = {
    "baseline",
    "ctb",
    "jde",
    "ctb_jde",
    "bounce",
    "archive",
    "ctb_archive",
};

/* ── problem definitions ────────────────────────────────────────── */
typedef double (*objective_fn)(const double *xs, int n);

typedef struct {
    const char *name;
    objective_fn objective;
    int params;
    int generations;
    int runs;
    double bounds_min;
    double bounds_max;
    double success_threshold;
} BenchCase;

/* ── RNG ────────────────────────────────────────────────────────── */
static inline uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline double rand01(uint64_t *state) {
    return (splitmix64(state) >> 11) * (1.0 / 9007199254740992.0);
}

static inline int rand_int(uint64_t *state, int n) {
    return (int)(splitmix64(state) % (uint64_t)n);
}

static inline double rand_range(uint64_t *state, double minv, double maxv) {
    return rand01(state) * (maxv - minv) + minv;
}

static inline double rand_normal(uint64_t *state, double mean, double std) {
    /* Box-Muller */
    double u1 = rand01(state);
    double u2 = rand01(state);
    if (u1 < 1e-300) u1 = 1e-300;
    return mean + std * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static inline double clamp(double x, double minv, double maxv) {
    return x < minv ? minv : (x > maxv ? maxv : x);
}

/* ── benchmarks ─────────────────────────────────────────────────── */
static double sphere(const double *xs, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += xs[i] * xs[i];
    return sum;
}

static double rastrigin(const double *xs, int n) {
    double sum = 10.0 * (double)n;
    for (int i = 0; i < n; ++i) {
        double x = xs[i];
        sum += x * x - 10.0 * cos(2.0 * M_PI * x);
    }
    return sum;
}

static double rosenbrock(const double *xs, int n) {
    double sum = 0.0;
    for (int i = 0; i < n - 1; ++i) {
        double a = xs[i + 1] - xs[i] * xs[i];
        double b = 1.0 - xs[i];
        sum += 100.0 * a * a + b * b;
    }
    return sum;
}

static double ackley(const double *xs, int n) {
    double sum_sq = 0.0, sum_cos = 0.0;
    for (int i = 0; i < n; ++i) {
        double x = xs[i];
        sum_sq += x * x;
        sum_cos += cos(2.0 * M_PI * x);
    }
    return -20.0 * exp(-0.2 * sqrt(sum_sq / (double)n)) -
           exp(sum_cos / (double)n) + 20.0 + exp(1.0);
}

static double griewank(const double *xs, int n) {
    double sum = 0.0, product = 1.0;
    for (int i = 0; i < n; ++i) {
        double x = xs[i];
        sum += x * x / 4000.0;
        product *= cos(x / sqrt((double)(i + 1)));
    }
    return sum - product + 1.0;
}

/* ── mutation helpers ───────────────────────────────────────────── */

/* Pick 3 distinct random indices, all different from `i` */
static inline void pick3(uint64_t *state, int popsize, int i, int *r0, int *r1, int *r2) {
    do { *r0 = rand_int(state, popsize); } while (*r0 == i);
    do { *r1 = rand_int(state, popsize); } while (*r1 == i || *r1 == *r0);
    do { *r2 = rand_int(state, popsize); } while (*r2 == i || *r2 == *r0 || *r2 == *r1);
}

/* Pick an index from the union of population [0, popsize) + archive [0, arsize)
   that is != i. Returns 1 if from archive (index in ar_off), 0 if from population (index in r_off). */
static inline int pick_archived(uint64_t *state, int popsize, int arsize, int i,
                                int *r_off, int *ar_off) {
    int total = popsize + arsize;
    int pick = rand_int(state, total);
    if (pick < popsize) {
        if (pick == i) pick = (i + 1) % popsize;
        *r_off = pick;
        return 0;
    } else {
        *ar_off = pick - popsize;
        return 1;
    }
}

/* ── per-variant core loop ──────────────────────────────────────── */

static double run_baseline(const BenchCase *bench, uint64_t *rng_states, int threads) {
    const double CR_MIN = 0.1, CR_MAX = 1.0, F_MIN = 0.2, F_MAX = 0.95;
    int params = bench->params;
    size_t agent_len = (size_t)params;
    size_t pop_len = (size_t)POPSIZE * agent_len;

    double *pop = (double *)malloc(pop_len * sizeof(double));
    double *trials = (double *)malloc(pop_len * sizeof(double));
    double *scores = (double *)malloc((size_t)POPSIZE * sizeof(double));

    for (int i = 0; i < POPSIZE; ++i) {
        double *agent = pop + (size_t)i * agent_len;
        for (int j = 0; j < params; ++j)
            agent[j] = rand_range(&rng_states[0], bench->bounds_min, bench->bounds_max);
        scores[i] = bench->objective(agent, params);
    }

    double crossover = 0.0, mutate = 0.0;
#pragma omp parallel default(none) shared(bench, params, agent_len, pop, trials, scores, rng_states, crossover, mutate)
    {
        int tid = omp_get_thread_num();
        uint64_t *state = &rng_states[tid];

        for (int g = 0; g < bench->generations; ++g) {
#pragma omp single
            {
                crossover = rand_range(&rng_states[0], CR_MIN, CR_MAX);
                mutate = rand_range(&rng_states[0], F_MIN, F_MAX);
            }
#pragma omp for schedule(static)
            for (int i = 0; i < POPSIZE; ++i) {
                int i0, i1, i2;
                pick3(state, POPSIZE, i, &i0, &i1, &i2);
                const double *x0 = pop + (size_t)i0 * agent_len;
                const double *x1 = pop + (size_t)i1 * agent_len;
                const double *x2 = pop + (size_t)i2 * agent_len;
                double *xt = pop + (size_t)i * agent_len;
                double *trial = trials + (size_t)i * agent_len;

                for (int j = 0; j < params; ++j) {
                    if (rand01(state) < crossover)
                        trial[j] = clamp(x0[j] + (x1[j] - x2[j]) * mutate,
                                         bench->bounds_min, bench->bounds_max);
                    else
                        trial[j] = xt[j];
                }
                double ts = bench->objective(trial, params);
                if (ts < scores[i]) {
                    memcpy(xt, trial, agent_len * sizeof(double));
                    scores[i] = ts;
                }
            }
        }
    }
    double best = scores[0];
    for (int i = 1; i < POPSIZE; ++i) if (scores[i] < best) best = scores[i];
    free(pop); free(trials); free(scores);
    return best;
}

/* ── V_CURRENT_TO_BEST ─────────────────────────────────────────── */
static double run_current_to_best(const BenchCase *bench, uint64_t *rng_states, int threads) {
    const double CR_MIN = 0.1, CR_MAX = 1.0, F_MIN = 0.2, F_MAX = 0.95;
    int params = bench->params;
    size_t agent_len = (size_t)params;
    size_t pop_len = (size_t)POPSIZE * agent_len;

    double *pop = (double *)malloc(pop_len * sizeof(double));
    double *trials = (double *)malloc(pop_len * sizeof(double));
    double *scores = (double *)malloc((size_t)POPSIZE * sizeof(double));

    for (int i = 0; i < POPSIZE; ++i) {
        double *agent = pop + (size_t)i * agent_len;
        for (int j = 0; j < params; ++j)
            agent[j] = rand_range(&rng_states[0], bench->bounds_min, bench->bounds_max);
        scores[i] = bench->objective(agent, params);
    }

    double crossover = 0.0, mutate = 0.0;
#pragma omp parallel default(none) shared(bench, params, agent_len, pop, trials, scores, rng_states, crossover, mutate)
    {
        int tid = omp_get_thread_num();
        uint64_t *state = &rng_states[tid];

        for (int g = 0; g < bench->generations; ++g) {
            int best_idx = 0;
#pragma omp single
            {
                crossover = rand_range(&rng_states[0], CR_MIN, CR_MAX);
                mutate = rand_range(&rng_states[0], F_MIN, F_MAX);
                /* find current best (single-threaded is fine — cheap) */
                for (int i = 1; i < POPSIZE; ++i)
                    if (scores[i] < scores[best_idx]) best_idx = i;
            }
            const double *x_best = pop + (size_t)best_idx * agent_len;
#pragma omp for schedule(static)
            for (int i = 0; i < POPSIZE; ++i) {
                int i1, i2;
                do { i1 = rand_int(state, POPSIZE); } while (i1 == i);
                do { i2 = rand_int(state, POPSIZE); } while (i2 == i || i2 == i1);

                const double *x1 = pop + (size_t)i1 * agent_len;
                const double *x2 = pop + (size_t)i2 * agent_len;
                double *xt = pop + (size_t)i * agent_len;
                double *trial = trials + (size_t)i * agent_len;

                for (int j = 0; j < params; ++j) {
                    if (rand01(state) < crossover)
                        trial[j] = clamp(xt[j] + (x_best[j] - xt[j]) * mutate +
                                         (x1[j] - x2[j]) * mutate,
                                         bench->bounds_min, bench->bounds_max);
                    else
                        trial[j] = xt[j];
                }
                double ts = bench->objective(trial, params);
                if (ts < scores[i]) {
                    memcpy(xt, trial, agent_len * sizeof(double));
                    scores[i] = ts;
                }
            }
        }
    }
    double best = scores[0];
    for (int i = 1; i < POPSIZE; ++i) if (scores[i] < best) best = scores[i];
    free(pop); free(trials); free(scores);
    return best;
}

/* ── V_JDE (self-adaptive F/CR per individual) ─────────────────── */
static double run_jde(const BenchCase *bench, uint64_t *rng_states, int threads) {
    const double F_L = 0.1, F_U = 0.9;
    const double TAU1 = 0.1, TAU2 = 0.1;
    int params = bench->params;
    size_t agent_len = (size_t)params;
    size_t pop_len = (size_t)POPSIZE * agent_len;

    double *pop = (double *)malloc(pop_len * sizeof(double));
    double *trials = (double *)malloc(pop_len * sizeof(double));
    double *scores = (double *)malloc((size_t)POPSIZE * sizeof(double));
    double *Fi    = (double *)malloc((size_t)POPSIZE * sizeof(double));
    double *CRi   = (double *)malloc((size_t)POPSIZE * sizeof(double));

    for (int i = 0; i < POPSIZE; ++i) {
        double *agent = pop + (size_t)i * agent_len;
        for (int j = 0; j < params; ++j)
            agent[j] = rand_range(&rng_states[0], bench->bounds_min, bench->bounds_max);
        scores[i] = bench->objective(agent, params);
        Fi[i]  = F_L + rand01(&rng_states[0]) * (F_U - F_L);
        CRi[i] = rand01(&rng_states[0]);
    }

#pragma omp parallel default(none) shared(bench, params, agent_len, pop, trials, scores, Fi, CRi, rng_states)
    {
        int tid = omp_get_thread_num();
        uint64_t *state = &rng_states[tid];

        for (int g = 0; g < bench->generations; ++g) {
#pragma omp for schedule(static)
            for (int i = 0; i < POPSIZE; ++i) {
                /* jDE parameter adaptation */
                double Fi_new = Fi[i];
                double CRi_new = CRi[i];
                if (rand01(state) < TAU1)
                    Fi_new = F_L + rand01(state) * (F_U - F_L);
                if (rand01(state) < TAU2)
                    CRi_new = rand01(state);

                int i0, i1, i2;
                pick3(state, POPSIZE, i, &i0, &i1, &i2);
                const double *x0 = pop + (size_t)i0 * agent_len;
                const double *x1 = pop + (size_t)i1 * agent_len;
                const double *x2 = pop + (size_t)i2 * agent_len;
                double *xt = pop + (size_t)i * agent_len;
                double *trial = trials + (size_t)i * agent_len;

                for (int j = 0; j < params; ++j) {
                    if (rand01(state) < CRi_new)
                        trial[j] = clamp(x0[j] + (x1[j] - x2[j]) * Fi_new,
                                         bench->bounds_min, bench->bounds_max);
                    else
                        trial[j] = xt[j];
                }
                double ts = bench->objective(trial, params);
                if (ts < scores[i]) {
                    memcpy(xt, trial, agent_len * sizeof(double));
                    scores[i] = ts;
                    Fi[i]  = Fi_new;   /* keep winning parameters */
                    CRi[i] = CRi_new;
                }
            }
        }
    }
    double best = scores[0];
    for (int i = 1; i < POPSIZE; ++i) if (scores[i] < best) best = scores[i];
    free(pop); free(trials); free(scores); free(Fi); free(CRi);
    return best;
}

/* ── V_CTB_JDE (current-to-best + jDE adaptation) ──────────────── */
static double run_ctb_jde(const BenchCase *bench, uint64_t *rng_states, int threads) {
    const double F_L = 0.1, F_U = 0.9;
    const double TAU1 = 0.1, TAU2 = 0.1;
    int params = bench->params;
    size_t agent_len = (size_t)params;
    size_t pop_len = (size_t)POPSIZE * agent_len;

    double *pop = (double *)malloc(pop_len * sizeof(double));
    double *trials = (double *)malloc(pop_len * sizeof(double));
    double *scores = (double *)malloc((size_t)POPSIZE * sizeof(double));
    double *Fi    = (double *)malloc((size_t)POPSIZE * sizeof(double));
    double *CRi   = (double *)malloc((size_t)POPSIZE * sizeof(double));

    for (int i = 0; i < POPSIZE; ++i) {
        double *agent = pop + (size_t)i * agent_len;
        for (int j = 0; j < params; ++j)
            agent[j] = rand_range(&rng_states[0], bench->bounds_min, bench->bounds_max);
        scores[i] = bench->objective(agent, params);
        Fi[i]  = F_L + rand01(&rng_states[0]) * (F_U - F_L);
        CRi[i] = rand01(&rng_states[0]);
    }

#pragma omp parallel default(none) shared(bench, params, agent_len, pop, trials, scores, Fi, CRi, rng_states)
    {
        int tid = omp_get_thread_num();
        uint64_t *state = &rng_states[tid];

        for (int g = 0; g < bench->generations; ++g) {
            int best_idx = 0;
#pragma omp single
            {
                for (int i = 1; i < POPSIZE; ++i)
                    if (scores[i] < scores[best_idx]) best_idx = i;
            }
            const double *x_best = pop + (size_t)best_idx * agent_len;
#pragma omp for schedule(static)
            for (int i = 0; i < POPSIZE; ++i) {
                double Fi_new = Fi[i];
                double CRi_new = CRi[i];
                if (rand01(state) < TAU1)
                    Fi_new = F_L + rand01(state) * (F_U - F_L);
                if (rand01(state) < TAU2)
                    CRi_new = rand01(state);

                int i1, i2;
                do { i1 = rand_int(state, POPSIZE); } while (i1 == i);
                do { i2 = rand_int(state, POPSIZE); } while (i2 == i || i2 == i1);

                const double *x1 = pop + (size_t)i1 * agent_len;
                const double *x2 = pop + (size_t)i2 * agent_len;
                double *xt = pop + (size_t)i * agent_len;
                double *trial = trials + (size_t)i * agent_len;

                for (int j = 0; j < params; ++j) {
                    if (rand01(state) < CRi_new)
                        trial[j] = clamp(xt[j] + (x_best[j] - xt[j]) * Fi_new +
                                         (x1[j] - x2[j]) * Fi_new,
                                         bench->bounds_min, bench->bounds_max);
                    else
                        trial[j] = xt[j];
                }
                double ts = bench->objective(trial, params);
                if (ts < scores[i]) {
                    memcpy(xt, trial, agent_len * sizeof(double));
                    scores[i] = ts;
                    Fi[i]  = Fi_new;
                    CRi[i] = CRi_new;
                }
            }
        }
    }
    double best = scores[0];
    for (int i = 1; i < POPSIZE; ++i) if (scores[i] < best) best = scores[i];
    free(pop); free(trials); free(scores); free(Fi); free(CRi);
    return best;
}

/* ── V_BOUNCE (bounce-back boundary handling) ──────────────────── */
static inline double bounce(double x, double minv, double maxv) {
    while (x < minv || x > maxv) {
        if (x < minv) x = 2.0 * minv - x;
        if (x > maxv) x = 2.0 * maxv - x;
    }
    return x;
}

static double run_bounce(const BenchCase *bench, uint64_t *rng_states, int threads) {
    const double CR_MIN = 0.1, CR_MAX = 1.0, F_MIN = 0.2, F_MAX = 0.95;
    int params = bench->params;
    size_t agent_len = (size_t)params;
    size_t pop_len = (size_t)POPSIZE * agent_len;

    double *pop = (double *)malloc(pop_len * sizeof(double));
    double *trials = (double *)malloc(pop_len * sizeof(double));
    double *scores = (double *)malloc((size_t)POPSIZE * sizeof(double));

    for (int i = 0; i < POPSIZE; ++i) {
        double *agent = pop + (size_t)i * agent_len;
        for (int j = 0; j < params; ++j)
            agent[j] = rand_range(&rng_states[0], bench->bounds_min, bench->bounds_max);
        scores[i] = bench->objective(agent, params);
    }

    double crossover = 0.0, mutate = 0.0;
#pragma omp parallel default(none) shared(bench, params, agent_len, pop, trials, scores, rng_states, crossover, mutate)
    {
        int tid = omp_get_thread_num();
        uint64_t *state = &rng_states[tid];

        for (int g = 0; g < bench->generations; ++g) {
#pragma omp single
            {
                crossover = rand_range(&rng_states[0], CR_MIN, CR_MAX);
                mutate = rand_range(&rng_states[0], F_MIN, F_MAX);
            }
#pragma omp for schedule(static)
            for (int i = 0; i < POPSIZE; ++i) {
                int i0, i1, i2;
                pick3(state, POPSIZE, i, &i0, &i1, &i2);
                const double *x0 = pop + (size_t)i0 * agent_len;
                const double *x1 = pop + (size_t)i1 * agent_len;
                const double *x2 = pop + (size_t)i2 * agent_len;
                double *xt = pop + (size_t)i * agent_len;
                double *trial = trials + (size_t)i * agent_len;

                for (int j = 0; j < params; ++j) {
                    if (rand01(state) < crossover)
                        trial[j] = bounce(x0[j] + (x1[j] - x2[j]) * mutate,
                                          bench->bounds_min, bench->bounds_max);
                    else
                        trial[j] = xt[j];
                }
                double ts = bench->objective(trial, params);
                if (ts < scores[i]) {
                    memcpy(xt, trial, agent_len * sizeof(double));
                    scores[i] = ts;
                }
            }
        }
    }
    double best = scores[0];
    for (int i = 1; i < POPSIZE; ++i) if (scores[i] < best) best = scores[i];
    free(pop); free(trials); free(scores);
    return best;
}

/* ── V_ARCHIVE (external archive of replaced solutions) ────────── */
/* This implements a SHADE-like archive: replaced parents go into an
   external archive (size = popsize). When the archive is full, a random
   entry is overwritten. Mutation uses DE/current-to-rand-archive/1 when
   archive is non-empty, otherwise DE/rand/1. */
static double run_archive(const BenchCase *bench, uint64_t *rng_states, int threads) {
    const double CR_MIN = 0.1, CR_MAX = 1.0, F_MIN = 0.2, F_MAX = 0.95;
    int params = bench->params;
    size_t agent_len = (size_t)params;
    size_t pop_len = (size_t)POPSIZE * agent_len;

    double *pop = (double *)malloc(pop_len * sizeof(double));
    double *trials = (double *)malloc(pop_len * sizeof(double));
    double *scores = (double *)malloc((size_t)POPSIZE * sizeof(double));
    /* external archive */
    double *archive = (double *)calloc(pop_len, sizeof(double));
    int ar_size = 0;

    for (int i = 0; i < POPSIZE; ++i) {
        double *agent = pop + (size_t)i * agent_len;
        for (int j = 0; j < params; ++j)
            agent[j] = rand_range(&rng_states[0], bench->bounds_min, bench->bounds_max);
        scores[i] = bench->objective(agent, params);
    }

    double crossover = 0.0, mutate = 0.0;
#pragma omp parallel default(none) shared(bench, params, agent_len, pop, trials, scores, rng_states, archive, ar_size, crossover, mutate)
    {
        int tid = omp_get_thread_num();
        uint64_t *state = &rng_states[tid];
        int best_idx = 0;

        for (int g = 0; g < bench->generations; ++g) {
#pragma omp single
            {
                crossover = rand_range(&rng_states[0], CR_MIN, CR_MAX);
                mutate = rand_range(&rng_states[0], F_MIN, F_MAX);
                for (int i = 1; i < POPSIZE; ++i)
                    if (scores[i] < scores[best_idx]) best_idx = i;
            }
            const double *x_best = pop + (size_t)best_idx * agent_len;
#pragma omp for schedule(static)
            for (int i = 0; i < POPSIZE; ++i) {
                int i1, i2;
                do { i1 = rand_int(state, POPSIZE); } while (i1 == i);
                do { i2 = rand_int(state, POPSIZE); } while (i2 == i1);

                const double *x1 = pop + (size_t)i1 * agent_len;
                const double *x2 = pop + (size_t)i2 * agent_len;
                double *xt = pop + (size_t)i * agent_len;
                double *trial = trials + (size_t)i * agent_len;

                /* With 50% probability, replace x_best with a random archive member */
                const double *x_guide = x_best;
                if (ar_size > 0 && rand01(state) < 0.5) {
                    int ar_idx = rand_int(state, ar_size);
                    x_guide = archive + (size_t)ar_idx * agent_len;
                }

                for (int j = 0; j < params; ++j) {
                    if (rand01(state) < crossover)
                        trial[j] = clamp(xt[j] + (x_guide[j] - xt[j]) * mutate +
                                         (x1[j] - x2[j]) * mutate,
                                         bench->bounds_min, bench->bounds_max);
                    else
                        trial[j] = xt[j];
                }
                double ts = bench->objective(trial, params);
                if (ts < scores[i]) {
                    /* push replaced parent into archive */
                    int ar_pos;
#pragma omp critical
                    {
                        if (ar_size < POPSIZE) {
                            ar_pos = ar_size++;
                        } else {
                            ar_pos = rand_int(&rng_states[0], POPSIZE);
                        }
                        memcpy(archive + (size_t)ar_pos * agent_len, xt, agent_len * sizeof(double));
                    }
                    memcpy(xt, trial, agent_len * sizeof(double));
                    scores[i] = ts;
                }
            }
        }
    }
    double best = scores[0];
    for (int i = 1; i < POPSIZE; ++i) if (scores[i] < best) best = scores[i];
    free(archive); free(pop); free(trials); free(scores);
    return best;
}

/* ── V_CTB_ARCHIVE (current-to-best + archive) ─────────────────── */
static double run_ctb_archive(const BenchCase *bench, uint64_t *rng_states, int threads) {
    const double CR_MIN = 0.1, CR_MAX = 1.0, F_MIN = 0.2, F_MAX = 0.95;
    int params = bench->params;
    size_t agent_len = (size_t)params;
    size_t pop_len = (size_t)POPSIZE * agent_len;

    double *pop = (double *)malloc(pop_len * sizeof(double));
    double *trials = (double *)malloc(pop_len * sizeof(double));
    double *scores = (double *)malloc((size_t)POPSIZE * sizeof(double));
    double *archive = (double *)calloc(pop_len, sizeof(double));
    int ar_size = 0;

    for (int i = 0; i < POPSIZE; ++i) {
        double *agent = pop + (size_t)i * agent_len;
        for (int j = 0; j < params; ++j)
            agent[j] = rand_range(&rng_states[0], bench->bounds_min, bench->bounds_max);
        scores[i] = bench->objective(agent, params);
    }

    double crossover = 0.0, mutate = 0.0;
#pragma omp parallel default(none) shared(bench, params, agent_len, pop, trials, scores, rng_states, archive, ar_size, crossover, mutate)
    {
        int tid = omp_get_thread_num();
        uint64_t *state = &rng_states[tid];
        int best_idx = 0;

        for (int g = 0; g < bench->generations; ++g) {
#pragma omp single
            {
                crossover = rand_range(&rng_states[0], CR_MIN, CR_MAX);
                mutate = rand_range(&rng_states[0], F_MIN, F_MAX);
                for (int i = 1; i < POPSIZE; ++i)
                    if (scores[i] < scores[best_idx]) best_idx = i;
            }
            const double *x_best = pop + (size_t)best_idx * agent_len;
#pragma omp for schedule(static)
            for (int i = 0; i < POPSIZE; ++i) {
                int i1, i2;
                do { i1 = rand_int(state, POPSIZE); } while (i1 == i);
                do { i2 = rand_int(state, POPSIZE); } while (i2 == i || i2 == i1);

                const double *x1 = pop + (size_t)i1 * agent_len;
                const double *x2 = pop + (size_t)i2 * agent_len;
                double *xt = pop + (size_t)i * agent_len;
                double *trial = trials + (size_t)i * agent_len;

                const double *x_guide = x_best;
                if (ar_size > 0 && rand01(state) < 0.5) {
                    int ar_idx = rand_int(state, ar_size);
                    x_guide = archive + (size_t)ar_idx * agent_len;
                }

                for (int j = 0; j < params; ++j) {
                    if (rand01(state) < crossover)
                        trial[j] = clamp(xt[j] + (x_guide[j] - xt[j]) * mutate +
                                         (x1[j] - x2[j]) * mutate,
                                         bench->bounds_min, bench->bounds_max);
                    else
                        trial[j] = xt[j];
                }
                double ts = bench->objective(trial, params);
                if (ts < scores[i]) {
                    int ar_pos;
#pragma omp critical
                    {
                        if (ar_size < POPSIZE) {
                            ar_pos = ar_size++;
                        } else {
                            ar_pos = rand_int(&rng_states[0], POPSIZE);
                        }
                        memcpy(archive + (size_t)ar_pos * agent_len, xt, agent_len * sizeof(double));
                    }
                    memcpy(xt, trial, agent_len * sizeof(double));
                    scores[i] = ts;
                }
            }
        }
    }
    double best = scores[0];
    for (int i = 1; i < POPSIZE; ++i) if (scores[i] < best) best = scores[i];
    free(archive); free(pop); free(trials); free(scores);
    return best;
}

/* ── dispatch table ─────────────────────────────────────────────── */
typedef double (*variant_fn)(const BenchCase *, uint64_t *, int);

static variant_fn variant_table[] = {
    run_baseline,
    run_current_to_best,
    run_jde,
    run_ctb_jde,
    run_bounce,
    run_archive,
    run_ctb_archive,
};

/* ── timing ─────────────────────────────────────────────────────── */
static long elapsed_ms(struct timespec start, struct timespec end) {
    return (long)((end.tv_sec - start.tv_sec) * 1000L + (end.tv_nsec - start.tv_nsec) / 1000000L);
}

static int compare_long(const void *a, const void *b) {
    long x = *(const long *)a, y = *(const long *)b;
    return (x > y) - (x < y);
}

/* ── main ──────────────────────────────────────────────────────── */
int main(void) {
    BenchCase cases[] = {
        {"sphere",     sphere,     1000, 20000, 5, -10.0,    10.0,    1e-6},
        {"rastrigin",  rastrigin,  200,  8000,  5, -5.12,    5.12,    250.0},
        {"rosenbrock", rosenbrock, 200,  8000,  5, -5.0,     10.0,    250.0},
        {"ackley",     ackley,     200,  8000,  5, -32.768,  32.768,  1e-3},
        {"griewank",   griewank,   200,  8000,  5, -600.0,   600.0,   1e-3},
    };
    int case_count = (int)(sizeof(cases) / sizeof(cases[0]));

    uint64_t base_seed = 0xD1FF3A9E2B1C4D77ULL;

    printf("=== Comparing DE variants (%d runs each) ===\n\n", cases[0].runs);

    /* For each problem, run every variant */
    for (int c = 0; c < case_count; ++c) {
        const BenchCase *bench = &cases[c];
        printf("━━━ %s  (d=%d, gens=%d, threshold=%.4g) ━━━\n",
               bench->name, bench->params, bench->generations, bench->success_threshold);
        printf("%-16s %14s %10s %10s %10s %8s\n",
               "variant", "best_score", "best_ms", "median_ms", "worst_ms", "success");

        for (int v = 0; v < V_VARIANT_COUNT; ++v) {
            double best_overall = INFINITY;
            long times[16];
            int successes = 0;

            for (int r = 0; r < bench->runs; ++r) {
                uint64_t seed = base_seed
                    + (uint64_t)c  * 0x9e3779b97f4a7c15ULL
                    + (uint64_t)v  * 0xbf58476d1ce4e5b9ULL
                    + (uint64_t)r  * 0x94d049bb133111ebULL;

                int threads = omp_get_max_threads();
                uint64_t *rng_states = (uint64_t *)malloc((size_t)threads * sizeof(uint64_t));
                for (int t = 0; t < threads; ++t) {
                    uint64_t s = seed + (uint64_t)t * 0x9e3779b97f4a7c15ULL;
                    rng_states[t] = splitmix64(&s);
                }

                struct timespec t0, t1;
                clock_gettime(CLOCK_MONOTONIC, &t0);
                double score = variant_table[v](bench, rng_states, threads);
                clock_gettime(CLOCK_MONOTONIC, &t1);

                if (score < best_overall) best_overall = score;
                if (score <= bench->success_threshold) successes++;
                times[r] = elapsed_ms(t0, t1);
                free(rng_states);
            }

            qsort(times, (size_t)bench->runs, sizeof(times[0]), compare_long);
            printf("%-16s %14.6g %10ld %10ld %10ld %3d/%-4d\n",
                   variant_names[v],
                   best_overall,
                   times[0],
                   times[bench->runs / 2],
                   times[bench->runs - 1],
                   successes,
                   bench->runs);
        }
        printf("\n");
    }

    return 0;
}

#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    PARAMS = 1000,
    POPSIZE = 200,
    GENERATIONS = 20000,
    PRINT_EVERY = 20000
};

static const double BOUNDS_MIN = -10.0;
static const double BOUNDS_MAX = 10.0;
static const double CROSSOVER_MIN = 0.1;
static const double CROSSOVER_MAX = 1.0;
static const double MUTATE_MIN = 0.2;
static const double MUTATE_MAX = 0.95;

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

static inline double clamp(double x, double minv, double maxv) {
    return x < minv ? minv : (x > maxv ? maxv : x);
}

static inline double f1(const double *xs) {
    double sum = 0.0;
    for (int i = 0; i < PARAMS; ++i) {
        sum += xs[i] * xs[i];
    }
    return sum;
}

int main(void) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    double *pop = (double *)malloc((size_t)POPSIZE * PARAMS * sizeof(double));
    double *trials = (double *)malloc((size_t)POPSIZE * PARAMS * sizeof(double));
    double *scores = (double *)malloc((size_t)POPSIZE * sizeof(double));
    if (!pop || !trials || !scores) {
        fprintf(stderr, "allocation failed\n");
        free(pop);
        free(trials);
        free(scores);
        return 1;
    }

    int threads = omp_get_max_threads();
    uint64_t *rng_states = (uint64_t *)malloc((size_t)threads * sizeof(uint64_t));
    if (!rng_states) {
        fprintf(stderr, "rng allocation failed\n");
        free(pop);
        free(trials);
        free(scores);
        return 1;
    }

    uint64_t seed = (uint64_t)time(NULL) ^ 0xD1FF3A9E2B1C4D77ULL;
    for (int t = 0; t < threads; ++t) {
        uint64_t s = seed + (uint64_t)t * 0x9e3779b97f4a7c15ULL;
        rng_states[t] = splitmix64(&s);
    }

    for (int i = 0; i < POPSIZE; ++i) {
        double *agent = pop + (size_t)i * PARAMS;
        for (int j = 0; j < PARAMS; ++j) {
            agent[j] = rand_range(&rng_states[0], BOUNDS_MIN, BOUNDS_MAX);
        }
        scores[i] = f1(agent);
    }

    double crossover = 0.0;
    double mutate = 0.0;
#pragma omp parallel default(none) shared(pop, trials, scores, rng_states, crossover, mutate, stdout)
    {
        int tid = omp_get_thread_num();
        uint64_t *state = &rng_states[tid];

        for (int g = 0; g < GENERATIONS; ++g) {
#pragma omp single
            {
                crossover = rand_range(&rng_states[0], CROSSOVER_MIN, CROSSOVER_MAX);
                mutate = rand_range(&rng_states[0], MUTATE_MIN, MUTATE_MAX);
            }

#pragma omp for schedule(static)
            for (int i = 0; i < POPSIZE; ++i) {
                int i0 = rand_int(state, POPSIZE);
                int i1 = rand_int(state, POPSIZE);
                int i2 = rand_int(state, POPSIZE);

                const double *x0 = pop + (size_t)i0 * PARAMS;
                const double *x1 = pop + (size_t)i1 * PARAMS;
                const double *x2 = pop + (size_t)i2 * PARAMS;
                double *xt = pop + (size_t)i * PARAMS;
                double *trial = trials + (size_t)i * PARAMS;

                for (int j = 0; j < PARAMS; ++j) {
                    if (rand01(state) < crossover) {
                        trial[j] = clamp(x0[j] + (x1[j] - x2[j]) * mutate, BOUNDS_MIN, BOUNDS_MAX);
                    } else {
                        trial[j] = xt[j];
                    }
                }

                double trial_score = f1(trial);
                if (trial_score < scores[i]) {
                    memcpy(xt, trial, (size_t)PARAMS * sizeof(double));
                    scores[i] = trial_score;
                }
            }

#pragma omp single
            {
                if (g % PRINT_EVERY == 0 || g == GENERATIONS - 1) {
                    double sum = 0.0;
                    double minv = scores[0];
                    for (int i = 0; i < POPSIZE; ++i) {
                        sum += scores[i];
                        if (scores[i] < minv) minv = scores[i];
                    }
                    printf("generation %d\n", g);
                    printf("mean %.6f\n", sum / POPSIZE);
                    printf("minimum %.6f\n", minv);
                }
            }
        }
    }

    int best_idx = 0;
    for (int i = 1; i < POPSIZE; ++i) {
        if (scores[i] < scores[best_idx]) best_idx = i;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    long ms = (long)((t1.tv_sec - t0.tv_sec) * 1000L + (t1.tv_nsec - t0.tv_nsec) / 1000000L);

    printf("score %.6f\n", scores[best_idx]);
    printf("execution time %ld ms\n", ms);

    free(rng_states);
    free(pop);
    free(trials);
    free(scores);
    return 0;
}

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

static const double CROSSOVER_MIN = 0.1;
static const double CROSSOVER_MAX = 1.0;
static const double MUTATE_MIN = 0.2;
static const double MUTATE_MAX = 0.95;

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

typedef struct {
    double best_score;
    long elapsed_ms;
} BenchResult;

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

static long elapsed_ms(struct timespec start, struct timespec end) {
    return (long)((end.tv_sec - start.tv_sec) * 1000L + (end.tv_nsec - start.tv_nsec) / 1000000L);
}

static int compare_long(const void *a, const void *b) {
    long x = *(const long *)a;
    long y = *(const long *)b;
    return (x > y) - (x < y);
}

static double sphere(const double *xs, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += xs[i] * xs[i];
    }
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
    double sum_sq = 0.0;
    double sum_cos = 0.0;
    for (int i = 0; i < n; ++i) {
        double x = xs[i];
        sum_sq += x * x;
        sum_cos += cos(2.0 * M_PI * x);
    }
    return -20.0 * exp(-0.2 * sqrt(sum_sq / (double)n)) -
           exp(sum_cos / (double)n) +
           20.0 + exp(1.0);
}

static double griewank(const double *xs, int n) {
    double sum = 0.0;
    double product = 1.0;
    for (int i = 0; i < n; ++i) {
        double x = xs[i];
        sum += x * x / 4000.0;
        product *= cos(x / sqrt((double)(i + 1)));
    }
    return sum - product + 1.0;
}

static BenchResult run_case(const BenchCase *bench, uint64_t seed) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int params = bench->params;
    size_t agent_len = (size_t)params;
    size_t pop_len = (size_t)POPSIZE * agent_len;

    double *pop = (double *)malloc(pop_len * sizeof(double));
    double *trials = (double *)malloc(pop_len * sizeof(double));
    double *scores = (double *)malloc((size_t)POPSIZE * sizeof(double));
    if (!pop || !trials || !scores) {
        fprintf(stderr, "allocation failed\n");
        exit(1);
    }

    int threads = omp_get_max_threads();
    uint64_t *rng_states = (uint64_t *)malloc((size_t)threads * sizeof(uint64_t));
    if (!rng_states) {
        fprintf(stderr, "rng allocation failed\n");
        exit(1);
    }

    for (int t = 0; t < threads; ++t) {
        uint64_t s = seed + (uint64_t)t * 0x9e3779b97f4a7c15ULL;
        rng_states[t] = splitmix64(&s);
    }

    for (int i = 0; i < POPSIZE; ++i) {
        double *agent = pop + (size_t)i * agent_len;
        for (int j = 0; j < params; ++j) {
            agent[j] = rand_range(&rng_states[0], bench->bounds_min, bench->bounds_max);
        }
        scores[i] = bench->objective(agent, params);
    }

    double crossover = 0.0;
    double mutate = 0.0;
#pragma omp parallel default(none) shared(bench, params, agent_len, pop, trials, scores, rng_states, crossover, mutate, stdout)
    {
        int tid = omp_get_thread_num();
        uint64_t *state = &rng_states[tid];

        for (int g = 0; g < bench->generations; ++g) {
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

                const double *x0 = pop + (size_t)i0 * agent_len;
                const double *x1 = pop + (size_t)i1 * agent_len;
                const double *x2 = pop + (size_t)i2 * agent_len;
                double *xt = pop + (size_t)i * agent_len;
                double *trial = trials + (size_t)i * agent_len;

                for (int j = 0; j < params; ++j) {
                    if (rand01(state) < crossover) {
                        trial[j] = clamp(x0[j] + (x1[j] - x2[j]) * mutate, bench->bounds_min, bench->bounds_max);
                    } else {
                        trial[j] = xt[j];
                    }
                }

                double trial_score = bench->objective(trial, params);
                if (trial_score < scores[i]) {
                    memcpy(xt, trial, agent_len * sizeof(double));
                    scores[i] = trial_score;
                }
            }

#if PRINT_EVERY > 0
#pragma omp single
            {
                if (g % PRINT_EVERY == 0 || g == bench->generations - 1) {
                    double minv = scores[0];
                    for (int i = 1; i < POPSIZE; ++i) {
                        if (scores[i] < minv) minv = scores[i];
                    }
                    printf("%s generation %d minimum %.12g\n", bench->name, g, minv);
                }
            }
#endif
        }
    }

    double best_score = scores[0];
    for (int i = 1; i < POPSIZE; ++i) {
        if (scores[i] < best_score) best_score = scores[i];
    }

    if (!isfinite(best_score)) {
        fprintf(stderr, "%s produced non-finite best score\n", bench->name);
        exit(1);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    BenchResult result = {
        .best_score = best_score,
        .elapsed_ms = elapsed_ms(t0, t1),
    };

    free(rng_states);
    free(pop);
    free(trials);
    free(scores);
    return result;
}

int main(void) {
    BenchCase cases[] = {
        {"sphere", sphere, 1000, 20000, 3, -10.0, 10.0, 1e-6},
        {"rastrigin", rastrigin, 200, 8000, 3, -5.12, 5.12, 250.0},
        {"rosenbrock", rosenbrock, 200, 8000, 3, -5.0, 10.0, 250.0},
        {"ackley", ackley, 200, 8000, 3, -32.768, 32.768, 1e-3},
        {"griewank", griewank, 200, 8000, 3, -600.0, 600.0, 1e-3},
    };

    int case_count = (int)(sizeof(cases) / sizeof(cases[0]));
    uint64_t base_seed = 0xD1FF3A9E2B1C4D77ULL;

    printf("%-12s %6s %10s %5s %14s %10s %10s %10s %8s\n",
           "objective", "dims", "generations", "runs", "best_score", "best_ms", "median_ms", "worst_ms", "success");

    for (int c = 0; c < case_count; ++c) {
        const BenchCase *bench = &cases[c];
        double best_score = INFINITY;
        long times[16];
        int successes = 0;

        if (bench->runs > (int)(sizeof(times) / sizeof(times[0]))) {
            fprintf(stderr, "too many runs for %s\n", bench->name);
            return 1;
        }

        for (int r = 0; r < bench->runs; ++r) {
            uint64_t seed = base_seed + (uint64_t)c * 0x9e3779b97f4a7c15ULL + (uint64_t)r * 0xbf58476d1ce4e5b9ULL;
            BenchResult result = run_case(bench, seed);
            if (result.best_score < best_score) best_score = result.best_score;
            if (result.best_score <= bench->success_threshold) successes++;
            times[r] = result.elapsed_ms;
        }

        qsort(times, (size_t)bench->runs, sizeof(times[0]), compare_long);
        printf("%-12s %6d %10d %5d %14.6g %10ld %10ld %10ld %3d/%-4d\n",
               bench->name,
               bench->params,
               bench->generations,
               bench->runs,
               best_score,
               times[0],
               times[bench->runs / 2],
               times[bench->runs - 1],
               successes,
               bench->runs);
    }

    return 0;
}

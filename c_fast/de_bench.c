#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "de_solver.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── benchmark suite ───────────────────────────────────────────── */
typedef double (*objective_fn)(const double *xs, int n);

typedef struct {
    const char   *name;
    objective_fn  objective;
    int           params;
    int           generations;
    int           runs;
    double        bounds_min;
    double        bounds_max;
    double        success_threshold;
} BenchCase;

typedef struct {
    const char *name;
    int initial_popsize;
    int min_popsize;
} SolverCase;

/* ── test functions ────────────────────────────────────────────── */
static double sphere(const double *xs, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += xs[i] * xs[i];
    return sum;
}

static double rastrigin(const double *xs, int n) {
    double sum = 10.0 * (double)n;
    for (int i = 0; i < n; ++i) {
        const double x = xs[i];
        sum += x * x - 10.0 * cos(2.0 * M_PI * x);
    }
    return sum;
}

static double rosenbrock(const double *xs, int n) {
    double sum = 0.0;
    for (int i = 0; i < n - 1; ++i) {
        const double a = xs[i + 1] - xs[i] * xs[i];
        const double b = 1.0 - xs[i];
        sum += 100.0 * a * a + b * b;
    }
    return sum;
}

static double ackley(const double *xs, int n) {
    double sum_sq = 0.0, sum_cos = 0.0;
    for (int i = 0; i < n; ++i) {
        const double x = xs[i];
        sum_sq  += x * x;
        sum_cos += cos(2.0 * M_PI * x);
    }
    return -20.0 * exp(-0.2 * sqrt(sum_sq / (double)n))
           - exp(sum_cos / (double)n) + 20.0 + exp(1.0);
}

static double griewank(const double *xs, int n) {
    double sum = 0.0, product = 1.0;
    for (int i = 0; i < n; ++i) {
        const double x = xs[i];
        sum     += x * x / 4000.0;
        product *= cos(x / sqrt((double)(i + 1)));
    }
    return sum - product + 1.0;
}

/* ── timing helpers ────────────────────────────────────────────── */
static long elapsed_ms(struct timespec start, struct timespec end) {
    return (long)((end.tv_sec - start.tv_sec) * 1000L
                + (end.tv_nsec - start.tv_nsec) / 1000000L);
}

static int compare_long(const void *a, const void *b) {
    const long x = *(const long *)a;
    const long y = *(const long *)b;
    return (x > y) - (x < y);
}

/* ── main ──────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    enum { POPSIZE = 200 };
    setbuf(stdout, NULL);

    const int quick = argc > 1 && strcmp(argv[1], "quick") == 0;

    const BenchCase cases[] = {
        {"sphere",     sphere,     quick ? 200 : 1000,
         quick ? 2000 : 20000, quick ? 1 : 3, -10.0,   10.0,   1e-6},
        {"rastrigin",  rastrigin,  quick ? 50  : 200,
         quick ? 1000 : 8000,  quick ? 1 : 3, -5.12,   5.12,   250.0},
        {"rosenbrock", rosenbrock, quick ? 50  : 200,
         quick ? 1000 : 8000,  quick ? 1 : 3, -5.0,    10.0,   250.0},
        {"ackley",     ackley,     quick ? 50  : 200,
         quick ? 1000 : 8000,  quick ? 1 : 3, -32.768, 32.768, 1e-3},
        {"griewank",   griewank,   quick ? 50  : 200,
         quick ? 1000 : 8000,  quick ? 1 : 3, -600.0,  600.0,  1e-3},
    };
    const int case_count = (int)(sizeof(cases) / sizeof(cases[0]));
    const SolverCase solvers[] = {
        {"jde",     POPSIZE, POPSIZE},
        {"lpsr128", POPSIZE, 128},
        {"lpsr96",  POPSIZE, 96},
        {"lpsr64",  POPSIZE, 64},
        {"lpsr32",  POPSIZE, 32},
        {"best1",   POPSIZE, POPSIZE},
        {"pbest5",  POPSIZE, POPSIZE},
        {"pbest10", POPSIZE, POPSIZE},
        {"pbest20", POPSIZE, POPSIZE},
        {"pbest50", POPSIZE, POPSIZE},
    };
    const int solver_count = (int)(sizeof(solvers) / sizeof(solvers[0]));
    const uint64_t base_seed = 0xD1FF3A9E2B1C4D77ULL;

    printf("%-12s %-6s %6s %10s %5s %9s %14s %10s %10s %10s %8s\n",
           "objective", "solver", "dims", "generations", "runs", "pop",
           "best_score", "best_ms", "median_ms", "worst_ms", "success");

    for (int c = 0; c < case_count; ++c) {
        const BenchCase *b = &cases[c];
        for (int s = 0; s < solver_count; ++s) {
            const SolverCase *solver = &solvers[s];
            double best_score = INFINITY;
            long   times[16];
            int    successes = 0;

            for (int r = 0; r < b->runs; ++r) {
                const uint64_t seed = base_seed
                    + (uint64_t)c * 0x9e3779b97f4a7c15ULL
                    + (uint64_t)r * 0xbf58476d1ce4e5b9ULL;

                struct timespec t0, t1;
                clock_gettime(CLOCK_MONOTONIC, &t0);

                const char *pbest = strstr(solver->name, "pbest");
                const int use_best1 = strcmp(solver->name, "best1") == 0;
                const double score = use_best1
                    ? de_solve_lpsr_best1(b->params,
                                          solver->initial_popsize,
                                          solver->min_popsize,
                                          b->generations,
                                          b->bounds_min,
                                          b->bounds_max,
                                          b->objective,
                                          seed, NULL)
                    : pbest
                    ? de_solve_lpsr_pbest(b->params,
                                          solver->initial_popsize,
                                          solver->min_popsize,
                                          b->generations,
                                          b->bounds_min,
                                          b->bounds_max,
                                          b->objective,
                                          seed, NULL,
                                          atoi(pbest + 5))
                    : de_solve_lpsr(b->params,
                                    solver->initial_popsize,
                                    solver->min_popsize,
                                    b->generations,
                                    b->bounds_min,
                                    b->bounds_max,
                                    b->objective,
                                    seed);

                clock_gettime(CLOCK_MONOTONIC, &t1);

                if (score < best_score) best_score = score;
                if (score <= b->success_threshold) successes++;
                times[r] = elapsed_ms(t0, t1);
            }

            qsort(times, (size_t)b->runs, sizeof(times[0]), compare_long);
            printf("%-12s %-6s %6d %10d %5d %4d->%-4d %14.6g %10ld %10ld %10ld %3d/%-4d\n",
                   b->name, solver->name, b->params, b->generations, b->runs,
                   solver->initial_popsize, solver->min_popsize,
                   best_score,
                   times[0], times[b->runs / 2], times[b->runs - 1],
                   successes, b->runs);
        }
    }

    return 0;
}

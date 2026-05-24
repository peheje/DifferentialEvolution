#include <stdio.h>
#include <time.h>

#include "de_solver.h"

enum {
    PARAMS      = 1000,
    POPSIZE     = 200,
    GENERATIONS = 20000,
};

static const double BOUNDS_MIN = -10.0;
static const double BOUNDS_MAX =  10.0;

static double sphere(const double *xs, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
        sum += xs[i] * xs[i];
    return sum;
}

int main(void) {
    const uint64_t seed = (uint64_t)time(NULL) ^ 0xD1FF3A9E2B1C4D77ULL;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    const double score = de_solve(PARAMS, POPSIZE, GENERATIONS,
                                  BOUNDS_MIN, BOUNDS_MAX, sphere, seed);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    const long ms = (long)((t1.tv_sec - t0.tv_sec) * 1000L
                         + (t1.tv_nsec - t0.tv_nsec) / 1000000L);

    printf("score %.6f\n", score);
    printf("execution time %ld ms\n", ms);
    return 0;
}

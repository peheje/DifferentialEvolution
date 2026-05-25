#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "de_solver.h"
#include "third_party/genann/genann.h"

enum {
    OUTPUTS = 1,
    RUNS = 5
};

typedef struct {
    const char *name;
    int inputs;
    int hidden_layers;
    int hidden;
    int samples;
    int popsize;
    int min_popsize;
    int generations;
    double bounds_min;
    double bounds_max;
    double success_loss;
    const double *input;
    const double *target;
    int test_samples;
    const double *test_input;
    const double *test_target;
} NnTask;

static const double XOR_INPUT[] = {
    0.0, 0.0,
    0.0, 1.0,
    1.0, 0.0,
    1.0, 1.0,
};

static const double XOR_TARGET[] = {0.0, 1.0, 1.0, 0.0};

static const double PARITY3_INPUT[] = {
    0.0, 0.0, 0.0,
    0.0, 0.0, 1.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 1.0,
    1.0, 0.0, 0.0,
    1.0, 0.0, 1.0,
    1.0, 1.0, 0.0,
    1.0, 1.0, 1.0,
};

static const double PARITY3_TARGET[] = {0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0};

static const double PARITY4_INPUT[] = {
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 1.0, 1.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 1.0,
    0.0, 1.0, 1.0, 0.0,
    0.0, 1.0, 1.0, 1.0,
    1.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 1.0, 0.0,
    1.0, 0.0, 1.0, 1.0,
    1.0, 1.0, 0.0, 0.0,
    1.0, 1.0, 0.0, 1.0,
    1.0, 1.0, 1.0, 0.0,
    1.0, 1.0, 1.0, 1.0,
};

static const double PARITY4_TARGET[] = {
    0.0, 1.0, 1.0, 0.0,
    1.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 0.0, 1.0,
    0.0, 1.0, 1.0, 0.0,
};

enum {
    CIRCLE_TRAIN_SAMPLES = 256,
    CIRCLE_TEST_SAMPLES = 1024,
    CIRCLE_BOUNDARY_TRAIN_SAMPLES = 256
};

static double CIRCLE_TRAIN_INPUT[CIRCLE_TRAIN_SAMPLES * 2];
static double CIRCLE_TRAIN_TARGET[CIRCLE_TRAIN_SAMPLES];
static double CIRCLE_TEST_INPUT[CIRCLE_TEST_SAMPLES * 2];
static double CIRCLE_TEST_TARGET[CIRCLE_TEST_SAMPLES];
static double CIRCLE_BOUNDARY_INPUT[CIRCLE_BOUNDARY_TRAIN_SAMPLES * 2];
static double CIRCLE_BOUNDARY_TARGET[CIRCLE_BOUNDARY_TRAIN_SAMPLES];

static const NnTask TASKS[] = {
    {
        "xor",
        2, 1, 2, 4,
        256, 128, 4000,
        -10.0, 10.0, 1e-3,
        XOR_INPUT, XOR_TARGET,
        4, XOR_INPUT, XOR_TARGET,
    },
    {
        "parity3",
        3, 1, 6, 8,
        512, 256, 12000,
        -12.0, 12.0, 1e-3,
        PARITY3_INPUT, PARITY3_TARGET,
        8, PARITY3_INPUT, PARITY3_TARGET,
    },
    {
        "parity4-small",
        4, 1, 8, 16,
        768, 384, 20000,
        -14.0, 14.0, 1e-3,
        PARITY4_INPUT, PARITY4_TARGET,
        16, PARITY4_INPUT, PARITY4_TARGET,
    },
    {
        "parity4-wide",
        4, 1, 12, 16,
        1024, 512, 30000,
        -14.0, 14.0, 1e-3,
        PARITY4_INPUT, PARITY4_TARGET,
        16, PARITY4_INPUT, PARITY4_TARGET,
    },
    {
        "circle",
        2, 1, 12, CIRCLE_TRAIN_SAMPLES,
        512, 256, 12000,
        -10.0, 10.0, 0.04,
        CIRCLE_TRAIN_INPUT, CIRCLE_TRAIN_TARGET,
        CIRCLE_TEST_SAMPLES, CIRCLE_TEST_INPUT, CIRCLE_TEST_TARGET,
    },
    {
        "circle-boundary",
        2, 1, 12, CIRCLE_BOUNDARY_TRAIN_SAMPLES,
        512, 256, 12000,
        -10.0, 10.0, 0.04,
        CIRCLE_BOUNDARY_INPUT, CIRCLE_BOUNDARY_TARGET,
        CIRCLE_TEST_SAMPLES, CIRCLE_TEST_INPUT, CIRCLE_TEST_TARGET,
    },
};

static _Thread_local genann *thread_ann;
static _Thread_local const NnTask *thread_task;

static genann *task_ann(const NnTask *task) {
    if (thread_task != task) {
        genann_free(thread_ann);
        thread_ann = NULL;
        thread_task = task;
    }
    if (!thread_ann) {
        thread_ann = genann_init(task->inputs, task->hidden_layers,
                                 task->hidden, OUTPUTS);
        if (!thread_ann) {
            fprintf(stderr, "genann_init failed\n");
            abort();
        }
    }
    return thread_ann;
}

static const NnTask *current_task;

static double task_loss(const double *weights, int n) {
    const NnTask *task = current_task;
    genann *ann = task_ann(task);
    if (n != ann->total_weights)
        return INFINITY;

    memcpy(ann->weight, weights, (size_t)n * sizeof(double));

    double loss = 0.0;
    for (int i = 0; i < task->samples; ++i) {
        const double *input = task->input + (size_t)i * task->inputs;
        const double y = genann_run(ann, input)[0];
        const double e = y - task->target[i];
        loss += e * e;
    }
    return loss / (double)task->samples;
}

static long elapsed_ms(struct timespec start, struct timespec end) {
    return (long)((end.tv_sec - start.tv_sec) * 1000L
                + (end.tv_nsec - start.tv_nsec) / 1000000L);
}

static uint64_t local_splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static double local_rand01(uint64_t *state) {
    return (local_splitmix64(state) >> 11) * (1.0 / 9007199254740992.0);
}

static double local_rand_range(uint64_t *state, double lo, double hi) {
    return lo + (hi - lo) * local_rand01(state);
}

static void fill_circle_data(double *input, double *target, int samples, uint64_t seed) {
    const double radius = 0.55;
    for (int i = 0; i < samples; ++i) {
        const double x = local_rand_range(&seed, -1.0, 1.0);
        const double y = local_rand_range(&seed, -1.0, 1.0);
        input[(size_t)i * 2] = x;
        input[(size_t)i * 2 + 1] = y;
        target[i] = x * x + y * y <= radius * radius ? 1.0 : 0.0;
    }
}

static void fill_circle_boundary_data(double *input, double *target, int samples, uint64_t seed) {
    const double radius = 0.55;
    const int boundary_samples = samples * 3 / 4;
    int i = 0;

    for (; i < boundary_samples; ++i) {
        const double angle = local_rand_range(&seed, 0.0, 2.0 * M_PI);
        const double r = local_rand_range(&seed, radius - 0.18, radius + 0.18);
        const double x = r * cos(angle);
        const double y = r * sin(angle);
        input[(size_t)i * 2] = x;
        input[(size_t)i * 2 + 1] = y;
        target[i] = r <= radius ? 1.0 : 0.0;
    }

    for (; i < samples; ++i) {
        const double x = local_rand_range(&seed, -1.0, 1.0);
        const double y = local_rand_range(&seed, -1.0, 1.0);
        input[(size_t)i * 2] = x;
        input[(size_t)i * 2 + 1] = y;
        target[i] = x * x + y * y <= radius * radius ? 1.0 : 0.0;
    }
}

static void init_generated_data(void) {
    fill_circle_data(CIRCLE_TRAIN_INPUT, CIRCLE_TRAIN_TARGET,
                     CIRCLE_TRAIN_SAMPLES, 0xC1CC1E712A1B5EEDULL);
    fill_circle_data(CIRCLE_TEST_INPUT, CIRCLE_TEST_TARGET,
                     CIRCLE_TEST_SAMPLES, 0x7E57C1CC1E5EEDULL);
    fill_circle_boundary_data(CIRCLE_BOUNDARY_INPUT, CIRCLE_BOUNDARY_TARGET,
                              CIRCLE_BOUNDARY_TRAIN_SAMPLES,
                              0xB04D3A7C1ACC1E5ULL);
}

static void evaluate_task(genann *ann, const NnTask *task,
                          const double *input, const double *target,
                          int samples, double *loss_out, int *correct_out) {
    double loss = 0.0;
    int correct = 0;
    for (int i = 0; i < samples; ++i) {
        const double *xs = input + (size_t)i * task->inputs;
        const double y = genann_run(ann, xs)[0];
        const double e = y - target[i];
        const int predicted = y >= 0.5;
        loss += e * e;
        if (predicted == (int)target[i])
            ++correct;
    }
    *loss_out = loss / (double)samples;
    *correct_out = correct;
}

static int run_task(const NnTask *task) {
    genann *template = genann_init(task->inputs, task->hidden_layers,
                                   task->hidden, OUTPUTS);
    if (!template) {
        fprintf(stderr, "genann_init failed\n");
        return 1;
    }

    const int weights = template->total_weights;
    genann_free(template);
    current_task = task;

    const uint64_t base_seed = 0xD1FF3A9E2B1C4D77ULL;
    double best_loss = INFINITY;
    uint64_t best_seed = 0;
    long best_ms = 0;
    double *best_weights = (double *)malloc((size_t)weights * sizeof(double));
    double *run_weights = (double *)malloc((size_t)weights * sizeof(double));
    if (!best_weights || !run_weights) {
        fprintf(stderr, "allocation failed\n");
        free(best_weights);
        free(run_weights);
        return 1;
    }

    printf("%s topology: inputs=%d hidden_layers=%d hidden=%d outputs=%d, weights=%d, pop=%d->%d, generations=%d\n",
           task->name, task->inputs, task->hidden_layers, task->hidden, OUTPUTS,
           weights, task->popsize, task->min_popsize, task->generations);
    printf("%-5s %14s %10s\n", "run", "loss", "ms");

    for (int r = 0; r < RUNS; ++r) {
        const uint64_t seed = base_seed + (uint64_t)r * 0x9e3779b97f4a7c15ULL;
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        const double loss = de_solve_lpsr_best(weights, task->popsize,
                                               task->min_popsize,
                                               task->generations,
                                               task->bounds_min,
                                               task->bounds_max,
                                               task_loss, seed,
                                               run_weights);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        const long ms = elapsed_ms(t0, t1);
        if (loss < best_loss) {
            best_loss = loss;
            best_seed = seed;
            best_ms = ms;
            memcpy(best_weights, run_weights, (size_t)weights * sizeof(double));
        }
        printf("%-5d %14.8g %10ld\n", r, loss, ms);
    }

    printf("best loss %.8g seed 0x%016llx ms %ld\n",
           best_loss, (unsigned long long)best_seed, best_ms);

    genann *ann = genann_init(task->inputs, task->hidden_layers,
                              task->hidden, OUTPUTS);
    if (!ann) {
        fprintf(stderr, "genann_init failed\n");
        free(best_weights);
        free(run_weights);
        return 1;
    }
    memcpy(ann->weight, best_weights, (size_t)weights * sizeof(double));
    printf("%-14s %-9s %-9s\n", "input", "target", "output");
    int correct = 0;
    const int rows_to_print = task->samples <= 16 ? task->samples : 16;
    for (int i = 0; i < rows_to_print; ++i) {
        const double *input = task->input + (size_t)i * task->inputs;
        const double y = genann_run(ann, input)[0];
        const int predicted = y >= 0.5;
        if (predicted == (int)task->target[i])
            ++correct;
        for (int j = 0; j < task->inputs; ++j)
            printf("%.0f%s", input[j], j == task->inputs - 1 ? "" : " ");
        printf("%*s %.0f         %.6f\n",
               14 - (2 * task->inputs - 1), "",
               task->target[i], y);
    }
    if (rows_to_print < task->samples)
        printf("... %d more train rows omitted\n", task->samples - rows_to_print);

    double train_loss, test_loss;
    int train_correct, test_correct;
    evaluate_task(ann, task, task->input, task->target, task->samples,
                  &train_loss, &train_correct);
    evaluate_task(ann, task, task->test_input, task->test_target, task->test_samples,
                  &test_loss, &test_correct);
    printf("train accuracy %d/%d loss %.8g\n",
           train_correct, task->samples, train_loss);
    printf("test  accuracy %d/%d loss %.8g\n",
           test_correct, task->test_samples, test_loss);

    genann_free(ann);
    free(best_weights);
    free(run_weights);
    printf("\n");
    return best_loss < task->success_loss ? 0 : 1;
}

int main(int argc, char **argv) {
    init_generated_data();

    int failed = 0;
    const int task_count = (int)(sizeof(TASKS) / sizeof(TASKS[0]));
    for (int i = 0; i < task_count; ++i) {
        if (argc > 1 && strcmp(argv[1], TASKS[i].name) != 0)
            continue;
        failed += run_task(&TASKS[i]);
    }
    return failed ? 1 : 0;
}

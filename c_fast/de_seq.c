#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "de_solver.h"
#include "third_party/genann/genann.h"

enum { DEFAULT_RUNS = 5 };

/* ── recurrence ───────────────────────────────────────────────────
   x_{t+1} = sin(A·x_t) + B·x_{t-1}
   A=2.5, B=0.6  produces oscillatory, bounded dynamics in [-2.5, 2.5].
   Proof: |x_{t+1}| ≤ 1 + 0.6·|x_{t-1}|, so if |x| ≤ 2.5 then
   |x_{t+1}| ≤ 1 + 0.6·2.5 = 2.5.  The invariant holds from init.    */
static double recurrence_next(double xt, double xt1) {
    return sin(2.5 * xt) + 0.6 * xt1;
}

/* N-dimensional recurrence:
   target = sin(2.5*a_1) + sum_{i=2}^{N} 0.6 * 0.6^{i-2} * a_i
   The linear coefficients sum to < 1.5, so full-box random inputs can
   produce a wider target range than the 2D invariant sequence.    */
static double recurrence_nd(int n_dims, const double *xs) {
    double y = sin(2.5 * xs[0]);
    double coeff = 0.6;
    for (int i = 1; i < n_dims; ++i) {
        y += coeff * xs[i];
        coeff *= 0.6;
    }
    return y;
}

/* Nonlinear 5D variant: sin on a1 and a2, linear on rest.
   40% nonlinear (2/5 terms). Used to test DE strategy preference. */
static double recurrence_5d_nonlin(const double *xs) {
    return sin(2.5 * xs[0]) + sin(1.5 * xs[1])
         + 0.6 * xs[2] + 0.36 * xs[3] + 0.22 * xs[4];
}

/* Hard 5D regression: multi-sine + interaction term.
   Tests whether DE/NN can handle genuine nonlinear coupling.
   y = sin(2.5*x0) + 0.7*sin(1.7*x1) + 0.35*x2*x3
     + 0.25*sin(x4 + x0) + 0.2*x2                              */
static double recurrence_5d_hard(const double *xs) {
    return sin(2.5 * xs[0])
         + 0.7 * sin(1.7 * xs[1])
         + 0.35 * xs[2] * xs[3]
         + 0.25 * sin(xs[4] + xs[0])
         + 0.2 * xs[2];
}

/* ── local activation variants ─────────────────────────────────── */
static double act_tanh(const genann *ann, double a) {
    (void)ann;
    return tanh(a);
}

static double act_relu(const genann *ann, double a) {
    (void)ann;
    return a > 0.0 ? a : 0.0;
}

static double act_silu(const genann *ann, double a) {
    (void)ann;
    return a / (1.0 + exp(-a));
}

static double act_gelu(const genann *ann, double a) {
    (void)ann;
    return 0.5 * a * (1.0 + tanh(0.7978845608028654
        * (a + 0.044715 * a * a * a)));
}

/* ── data buffers ───────────────────────────────────────────────── */
enum {
    SEQ_TRAIN_SAMPLES = 512,
    SEQ_TEST_SAMPLES  = 256,
};

static double SEQ1_TRAIN_INPUT [SEQ_TRAIN_SAMPLES * 2];
static double SEQ1_TRAIN_TARGET[SEQ_TRAIN_SAMPLES];
static double SEQ2_TRAIN_INPUT [SEQ_TRAIN_SAMPLES * 2];
static double SEQ2_TRAIN_TARGET[SEQ_TRAIN_SAMPLES * 2];
static double SEQ_AUX_TRAIN_INPUT [SEQ_TRAIN_SAMPLES * 2];
static double SEQ_AUX_TRAIN_TARGET[SEQ_TRAIN_SAMPLES * 2];
static double SEQ_DUP_TRAIN_INPUT [SEQ_TRAIN_SAMPLES * 2];
static double SEQ_DUP_TRAIN_TARGET[SEQ_TRAIN_SAMPLES * 2];
static double SEQ1_TEST_INPUT [SEQ_TEST_SAMPLES * 2];
static double SEQ1_TEST_TARGET[SEQ_TEST_SAMPLES];
static double SEQ2_TEST_INPUT [SEQ_TEST_SAMPLES * 2];
static double SEQ2_TEST_TARGET[SEQ_TEST_SAMPLES * 2];
static double SEQ_AUX_TEST_INPUT [SEQ_TEST_SAMPLES * 2];
static double SEQ_AUX_TEST_TARGET[SEQ_TEST_SAMPLES * 2];
static double SEQ_DUP_TEST_INPUT [SEQ_TEST_SAMPLES * 2];
static double SEQ_DUP_TEST_TARGET[SEQ_TEST_SAMPLES * 2];
static double PAIR1_TRAIN_INPUT [SEQ_TRAIN_SAMPLES * 2];
static double PAIR1_TRAIN_TARGET[SEQ_TRAIN_SAMPLES];
static double PAIR2_TRAIN_INPUT [SEQ_TRAIN_SAMPLES * 2];
static double PAIR2_TRAIN_TARGET[SEQ_TRAIN_SAMPLES * 2];
static double PAIRMID1_TRAIN_INPUT [SEQ_TRAIN_SAMPLES * 2];
static double PAIRMID1_TRAIN_TARGET[SEQ_TRAIN_SAMPLES];
static double PAIRMID2_TRAIN_INPUT [SEQ_TRAIN_SAMPLES * 2];
static double PAIRMID2_TRAIN_TARGET[SEQ_TRAIN_SAMPLES * 2];
static double PAIR2P0_1_TRAIN_INPUT [SEQ_TRAIN_SAMPLES * 2];
static double PAIR2P0_1_TRAIN_TARGET[SEQ_TRAIN_SAMPLES];
static double PAIR2P0_2_TRAIN_INPUT [SEQ_TRAIN_SAMPLES * 2];
static double PAIR2P0_2_TRAIN_TARGET[SEQ_TRAIN_SAMPLES * 2];
static double PAIRMIX25_1_TRAIN_INPUT [SEQ_TRAIN_SAMPLES * 2];
static double PAIRMIX25_1_TRAIN_TARGET[SEQ_TRAIN_SAMPLES];
static double PAIRMIX25_2_TRAIN_INPUT [SEQ_TRAIN_SAMPLES * 2];
static double PAIRMIX25_2_TRAIN_TARGET[SEQ_TRAIN_SAMPLES * 2];
static double PAIRMIX50_1_TRAIN_INPUT [SEQ_TRAIN_SAMPLES * 2];
static double PAIRMIX50_1_TRAIN_TARGET[SEQ_TRAIN_SAMPLES];
static double PAIRMIX50_2_TRAIN_INPUT [SEQ_TRAIN_SAMPLES * 2];
static double PAIRMIX50_2_TRAIN_TARGET[SEQ_TRAIN_SAMPLES * 2];
static double PAIR1_TEST_INPUT [SEQ_TEST_SAMPLES * 2];
static double PAIR1_TEST_TARGET[SEQ_TEST_SAMPLES];
static double PAIR_SCALED_TRAIN_INPUT[SEQ_TRAIN_SAMPLES * 2];
static double PAIR_SCALED_TEST_INPUT[SEQ_TEST_SAMPLES * 2];
static double PAIR5_TRAIN_INPUT[SEQ_TRAIN_SAMPLES * 5];
static double PAIR5_TRAIN_TARGET[SEQ_TRAIN_SAMPLES];
static double PAIR5_TEST_INPUT[SEQ_TEST_SAMPLES * 5];
static double PAIR5_TEST_TARGET[SEQ_TEST_SAMPLES];
static double PAIR10_TRAIN_INPUT[SEQ_TRAIN_SAMPLES * 10];
static double PAIR10_TRAIN_TARGET[SEQ_TRAIN_SAMPLES];
static double PAIR10_TEST_INPUT[SEQ_TEST_SAMPLES * 10];
static double PAIR10_TEST_TARGET[SEQ_TEST_SAMPLES];
static double PAIR5NL_TRAIN_INPUT[SEQ_TRAIN_SAMPLES * 5];
static double PAIR5NL_TRAIN_TARGET[SEQ_TRAIN_SAMPLES];
static double PAIR5NL_TEST_INPUT[SEQ_TEST_SAMPLES * 5];
static double PAIR5NL_TEST_TARGET[SEQ_TEST_SAMPLES];
static double PAIR5H_TRAIN_INPUT[SEQ_TRAIN_SAMPLES * 5];
static double PAIR5H_TRAIN_TARGET[SEQ_TRAIN_SAMPLES];
static double PAIR5H_TEST_INPUT[SEQ_TEST_SAMPLES * 5];
static double PAIR5H_TEST_TARGET[SEQ_TEST_SAMPLES];
static double PAIR2_TEST_INPUT [SEQ_TEST_SAMPLES * 2];
static double PAIR2_TEST_TARGET[SEQ_TEST_SAMPLES * 2];
static double PAIRMID1_TEST_INPUT [SEQ_TEST_SAMPLES * 2];
static double PAIRMID1_TEST_TARGET[SEQ_TEST_SAMPLES];
static double PAIRMID2_TEST_INPUT [SEQ_TEST_SAMPLES * 2];
static double PAIRMID2_TEST_TARGET[SEQ_TEST_SAMPLES * 2];
static double PAIR2P0_1_TEST_INPUT [SEQ_TEST_SAMPLES * 2];
static double PAIR2P0_1_TEST_TARGET[SEQ_TEST_SAMPLES];
static double PAIR2P0_2_TEST_INPUT [SEQ_TEST_SAMPLES * 2];
static double PAIR2P0_2_TEST_TARGET[SEQ_TEST_SAMPLES * 2];

/* ── task descriptor ────────────────────────────────────────────── */
typedef struct {
    const char *name;
    int inputs;
    int hidden_layers;
    int hidden;
    int outputs;
    int samples;
    int popsize;
    int min_popsize;
    int generations;
    int target_stride;
    double aux_loss_weight;
    double rollout_loss_weight;
    double bounds_min;
    double bounds_max;
    double success_loss;
    const double *input;
    const double *target;
    int test_samples;
    const double *test_input;
    const double *test_target;
} NnTask;

static const NnTask TASKS[] = {
    {
        "seq1",
        2, 1, 10, 1,
        SEQ_TRAIN_SAMPLES,
        256, 128, 4000,
        1,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        SEQ1_TRAIN_INPUT, SEQ1_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, SEQ1_TEST_INPUT, SEQ1_TEST_TARGET,
    },
    {
        "seq1-wide",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        1,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        SEQ1_TRAIN_INPUT, SEQ1_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, SEQ1_TEST_INPUT, SEQ1_TEST_TARGET,
    },
    {
        "seq1-61w",
        2, 1, 15, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        1,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        SEQ1_TRAIN_INPUT, SEQ1_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, SEQ1_TEST_INPUT, SEQ1_TEST_TARGET,
    },
    {
        "seq2",
        2, 1, 12, 2,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        SEQ2_TRAIN_INPUT, SEQ2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, SEQ2_TEST_INPUT, SEQ2_TEST_TARGET,
    },
    {
        "seq1-sinaux",
        2, 1, 12, 2,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        SEQ_AUX_TRAIN_INPUT, SEQ_AUX_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, SEQ_AUX_TEST_INPUT, SEQ_AUX_TEST_TARGET,
    },
    {
        "seq1-sinaux-light",
        2, 1, 12, 2,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        0.25,
        0.0,
        -5.0, 5.0, 1e-4,
        SEQ_AUX_TRAIN_INPUT, SEQ_AUX_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, SEQ_AUX_TEST_INPUT, SEQ_AUX_TEST_TARGET,
    },
    {
        "seq1-dup",
        2, 1, 12, 2,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        SEQ_DUP_TRAIN_INPUT, SEQ_DUP_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, SEQ_DUP_TEST_INPUT, SEQ_DUP_TEST_TARGET,
    },
    {
        "seq1-roll2",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        1.0,
        -5.0, 5.0, 1e-4,
        SEQ2_TRAIN_INPUT, SEQ2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, SEQ2_TEST_INPUT, SEQ2_TEST_TARGET,
    },
    {
        "seq1-roll2-light",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.25,
        -5.0, 5.0, 1e-4,
        SEQ2_TRAIN_INPUT, SEQ2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, SEQ2_TEST_INPUT, SEQ2_TEST_TARGET,
    },
    {
        "pair1",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-mae",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-max",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-tanh",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-relu",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-silu",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-gelu",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-mid",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIRMID2_TRAIN_INPUT, PAIRMID2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIRMID2_TEST_INPUT, PAIRMID2_TEST_TARGET,
    },
    {
        "pair1-2p0",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2P0_2_TRAIN_INPUT, PAIR2P0_2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2P0_2_TEST_INPUT, PAIR2P0_2_TEST_TARGET,
    },
    {
        "pair1-mix25",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIRMIX25_2_TRAIN_INPUT, PAIRMIX25_2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-mix50",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIRMIX50_2_TRAIN_INPUT, PAIRMIX50_2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-roll2",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        1.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair2",
        2, 1, 12, 2,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-scale",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR_SCALED_TRAIN_INPUT, PAIR1_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR_SCALED_TEST_INPUT, PAIR1_TEST_TARGET,
    },
    {
        "pair1-mae-scale",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR_SCALED_TRAIN_INPUT, PAIR1_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR_SCALED_TEST_INPUT, PAIR1_TEST_TARGET,
    },
    {
        "pair1-ln",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-mae-ln",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-dw1",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-dw2",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-dw4",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-huber0.2",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-huber0.5",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-huber1.0",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-huber0.5-dw1",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-huber0.5-relu",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-huber0.5-scale-relu",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR_SCALED_TRAIN_INPUT, PAIR1_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR_SCALED_TEST_INPUT, PAIR1_TEST_TARGET,
    },
    {
        "pair1-huber0.5-scale-gelu",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR_SCALED_TRAIN_INPUT, PAIR1_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR_SCALED_TEST_INPUT, PAIR1_TEST_TARGET,
    },
    {
        "pair1-huber0.5-scale-silu",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR_SCALED_TRAIN_INPUT, PAIR1_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR_SCALED_TEST_INPUT, PAIR1_TEST_TARGET,
    },
    {
        "pair1-huber0.5-ln-tanh",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-huber0.5-silu-b10",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -10.0, 10.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-huber0.5-gelu-b10",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -10.0, 10.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-best1",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-pbest5",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-pbest10",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-pbest20",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-pbest50",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-basepbest5",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-basepbest10",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-basepbest20",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-basepbest50",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-huber0.5-best1",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-huber0.5-wide",
        2, 1, 24, 1,
        SEQ_TRAIN_SAMPLES,
        768, 384, 12000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-huber0.5-wide-eqbudget",
        2, 1, 24, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-huber0.5-wide-best1",
        2, 1, 24, 1,
        SEQ_TRAIN_SAMPLES,
        768, 384, 12000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-huber0.5-deep",
        2, 2, 12, 1,
        SEQ_TRAIN_SAMPLES,
        1536, 768, 25000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-huber0.5-deep-eqbudget",
        2, 2, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair1-huber0.5-deep-best1",
        2, 2, 12, 1,
        SEQ_TRAIN_SAMPLES,
        1536, 768, 25000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair5-huber0.5",
        5, 1, 16, 1,
        SEQ_TRAIN_SAMPLES,
        768, 384, 12000,
        1,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR5_TRAIN_INPUT, PAIR5_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR5_TEST_INPUT, PAIR5_TEST_TARGET,
    },
    {
        "pair5-huber0.5-best1",
        5, 1, 16, 1,
        SEQ_TRAIN_SAMPLES,
        768, 384, 12000,
        1,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR5_TRAIN_INPUT, PAIR5_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR5_TEST_INPUT, PAIR5_TEST_TARGET,
    },
    {
        "pair5-huber0.5-pbest5",
        5, 1, 16, 1,
        SEQ_TRAIN_SAMPLES,
        768, 384, 12000,
        1,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR5_TRAIN_INPUT, PAIR5_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR5_TEST_INPUT, PAIR5_TEST_TARGET,
    },
    {
        "pair5-huber0.5-pbest10",
        5, 1, 16, 1,
        SEQ_TRAIN_SAMPLES,
        768, 384, 12000,
        1,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR5_TRAIN_INPUT, PAIR5_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR5_TEST_INPUT, PAIR5_TEST_TARGET,
    },
    {
        "pair5-huber0.5-pbest20",
        5, 1, 16, 1,
        SEQ_TRAIN_SAMPLES,
        768, 384, 12000,
        1,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR5_TRAIN_INPUT, PAIR5_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR5_TEST_INPUT, PAIR5_TEST_TARGET,
    },
    {
        "pair5-huber0.5-pbest50",
        5, 1, 16, 1,
        SEQ_TRAIN_SAMPLES,
        768, 384, 12000,
        1,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR5_TRAIN_INPUT, PAIR5_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR5_TEST_INPUT, PAIR5_TEST_TARGET,
    },
    {
        "pair5-huber0.5-basepbest5",
        5, 1, 16, 1,
        SEQ_TRAIN_SAMPLES,
        768, 384, 12000,
        1,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR5_TRAIN_INPUT, PAIR5_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR5_TEST_INPUT, PAIR5_TEST_TARGET,
    },
    {
        "pair5-huber0.5-basepbest10",
        5, 1, 16, 1,
        SEQ_TRAIN_SAMPLES,
        768, 384, 12000,
        1,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR5_TRAIN_INPUT, PAIR5_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR5_TEST_INPUT, PAIR5_TEST_TARGET,
    },
    {
        "pair5-huber0.5-basepbest20",
        5, 1, 16, 1,
        SEQ_TRAIN_SAMPLES,
        768, 384, 12000,
        1,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR5_TRAIN_INPUT, PAIR5_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR5_TEST_INPUT, PAIR5_TEST_TARGET,
    },
    {
        "pair5-huber0.5-basepbest50",
        5, 1, 16, 1,
        SEQ_TRAIN_SAMPLES,
        768, 384, 12000,
        1,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR5_TRAIN_INPUT, PAIR5_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR5_TEST_INPUT, PAIR5_TEST_TARGET,
    },
    {
        "pair5-hard",
        5, 1, 16, 1,
        SEQ_TRAIN_SAMPLES,
        768, 384, 12000,
        1,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR5H_TRAIN_INPUT, PAIR5H_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR5H_TEST_INPUT, PAIR5H_TEST_TARGET,
    },
    {
        "pair5-hard-basepbest10",
        5, 1, 16, 1,
        SEQ_TRAIN_SAMPLES,
        768, 384, 12000,
        1,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR5H_TRAIN_INPUT, PAIR5H_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR5H_TEST_INPUT, PAIR5H_TEST_TARGET,
    },
    {
        "pair5-hard-basepbest20",
        5, 1, 16, 1,
        SEQ_TRAIN_SAMPLES,
        768, 384, 12000,
        1,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR5H_TRAIN_INPUT, PAIR5H_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR5H_TEST_INPUT, PAIR5H_TEST_TARGET,
    },
    {
        "pair5-hard-basepbest50",
        5, 1, 16, 1,
        SEQ_TRAIN_SAMPLES,
        768, 384, 12000,
        1,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR5H_TRAIN_INPUT, PAIR5H_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR5H_TEST_INPUT, PAIR5H_TEST_TARGET,
    },
    {
        "pair10-huber0.5",
        10, 1, 24, 1,
        SEQ_TRAIN_SAMPLES,
        1536, 768, 20000,
        1,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR10_TRAIN_INPUT, PAIR10_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR10_TEST_INPUT, PAIR10_TEST_TARGET,
    },
    {
        "pair10-huber0.5-best1",
        10, 1, 24, 1,
        SEQ_TRAIN_SAMPLES,
        1536, 768, 20000,
        1,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR10_TRAIN_INPUT, PAIR10_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR10_TEST_INPUT, PAIR10_TEST_TARGET,
    },
    {
        "pair5-nonlin-huber0.5",
        5, 1, 16, 1,
        SEQ_TRAIN_SAMPLES,
        768, 384, 12000,
        1,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR5NL_TRAIN_INPUT, PAIR5NL_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR5NL_TEST_INPUT, PAIR5NL_TEST_TARGET,
    },
    {
        "pair5-nonlin-huber0.5-best1",
        5, 1, 16, 1,
        SEQ_TRAIN_SAMPLES,
        768, 384, 12000,
        1,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR5NL_TRAIN_INPUT, PAIR5NL_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR5NL_TEST_INPUT, PAIR5NL_TEST_TARGET,
    },
    {
        "pair1-huber0.5-curtobest1",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair5-huber0.5-curtobest1",
        5, 1, 16, 1,
        SEQ_TRAIN_SAMPLES,
        768, 384, 12000,
        1,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR5_TRAIN_INPUT, PAIR5_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR5_TEST_INPUT, PAIR5_TEST_TARGET,
    },
    {
        "pair5-nonlin-huber0.5-curtobest1",
        5, 1, 16, 1,
        SEQ_TRAIN_SAMPLES,
        768, 384, 12000,
        1,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR5NL_TRAIN_INPUT, PAIR5NL_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR5NL_TEST_INPUT, PAIR5NL_TEST_TARGET,
    },
    {
        "pair1-huber0.5-anneal",
        2, 1, 12, 1,
        SEQ_TRAIN_SAMPLES,
        384, 192, 6000,
        2,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
    },
    {
        "pair5-huber0.5-anneal",
        5, 1, 16, 1,
        SEQ_TRAIN_SAMPLES,
        768, 384, 12000,
        1,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR5_TRAIN_INPUT, PAIR5_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR5_TEST_INPUT, PAIR5_TEST_TARGET,
    },
    {
        "pair5-nonlin-huber0.5-anneal",
        5, 1, 16, 1,
        SEQ_TRAIN_SAMPLES,
        768, 384, 12000,
        1,
        1.0,
        0.0,
        -5.0, 5.0, 1e-4,
        PAIR5NL_TRAIN_INPUT, PAIR5NL_TRAIN_TARGET,
        SEQ_TEST_SAMPLES, PAIR5NL_TEST_INPUT, PAIR5NL_TEST_TARGET,
    },
};

/* ── thread-local genann cache ──────────────────────────────────── */
static _Thread_local genann *thread_ann;
static _Thread_local const NnTask *thread_task;
static _Thread_local double *thread_ln_gamma;
static _Thread_local double *thread_ln_beta;
static _Thread_local int thread_ln_hidden;

static void clear_thread_layernorm(void) {
    free(thread_ln_gamma);
    free(thread_ln_beta);
    thread_ln_gamma = NULL;
    thread_ln_beta = NULL;
    thread_ln_hidden = 0;
}

static void configure_task_ann(genann *ann, const NnTask *task) {
    ann->activation_output = genann_act_linear;
    if (strstr(task->name, "tanh"))
        ann->activation_hidden = act_tanh;
    else if (strstr(task->name, "relu"))
        ann->activation_hidden = act_relu;
    else if (strstr(task->name, "silu"))
        ann->activation_hidden = act_silu;
    else if (strstr(task->name, "gelu"))
        ann->activation_hidden = act_gelu;
}

static int task_uses_layernorm(const NnTask *task) {
    return strstr(task->name, "-ln") != NULL;
}

static int task_total_params(const NnTask *task, int regular_weights) {
    if (task_uses_layernorm(task))
        return regular_weights + 2 * task->hidden * task->hidden_layers;
    return regular_weights;
}

/* ── forward pass with optional LayerNorm ──────────────────────────
   Thread-local: thread_ln_hidden, thread_ln_gamma, thread_ln_beta.
   When thread_ln_hidden == 0, behaves identically to genann_run.
   LN normalizes pre-activations across hidden neurons, then applies
   gamma * norm + beta before activation.                               */
static double const *genann_run_ln(genann const *ann, double const *inputs) {
    double const *w = ann->weight;
    double *o = ann->output + ann->inputs;
    double const *i = ann->output;

    memcpy(ann->output, inputs, sizeof(double) * ann->inputs);

    int h, j, k;

    if (!ann->hidden_layers) {
        double *ret = o;
        for (j = 0; j < ann->outputs; ++j) {
            double sum = *w++ * -1.0;
            for (k = 0; k < ann->inputs; ++k)
                sum += *w++ * i[k];
            *o++ = ann->activation_output(ann, sum);
        }
        return ret;
    }

    for (h = 0; h < ann->hidden_layers; ++h) {
        const int prev = (h == 0) ? ann->inputs : ann->hidden;

        if (thread_ln_hidden > h) {
            for (j = 0; j < ann->hidden; ++j) {
                double sum = *w++ * -1.0;
                for (k = 0; k < prev; ++k)
                    sum += *w++ * i[k];
                o[j] = sum;
            }
            double mean = 0.0;
            for (j = 0; j < ann->hidden; ++j) mean += o[j];
            mean /= (double)ann->hidden;
            double var = 0.0;
            for (j = 0; j < ann->hidden; ++j) {
                double d = o[j] - mean;
                var += d * d;
            }
            var /= (double)ann->hidden;
            double inv_std = 1.0 / sqrt(var + 1e-5);
            const int ln_off = h * ann->hidden;
            for (j = 0; j < ann->hidden; ++j) {
                double v = (o[j] - mean) * inv_std;
                v = thread_ln_gamma[ln_off + j] * v
                  + thread_ln_beta[ln_off + j];
                o[j] = ann->activation_hidden(ann, v);
            }
            o += ann->hidden;
        } else {
            for (j = 0; j < ann->hidden; ++j) {
                double sum = *w++ * -1.0;
                for (k = 0; k < prev; ++k)
                    sum += *w++ * i[k];
                *o++ = ann->activation_hidden(ann, sum);
            }
        }
        i += prev;
    }

    double const *ret = o;
    for (j = 0; j < ann->outputs; ++j) {
        double sum = *w++ * -1.0;
        for (k = 0; k < ann->hidden; ++k)
            sum += *w++ * i[k];
        *o++ = ann->activation_output(ann, sum);
    }

    return ret;
}

static genann *task_ann(const NnTask *task) {
    if (thread_task != task) {
        genann_free(thread_ann);
        thread_ann = NULL;
        clear_thread_layernorm();
        thread_task = task;
    }
    if (!thread_ann) {
        thread_ann = genann_init(task->inputs, task->hidden_layers,
                                 task->hidden, task->outputs);
        if (!thread_ann) {
            fprintf(stderr, "genann_init failed\n");
            abort();
        }
        configure_task_ann(thread_ann, task);
        if (task_uses_layernorm(task)) {
            const int n_ln = task->hidden * task->hidden_layers;
            thread_ln_gamma = (double *)malloc(
                (size_t)n_ln * sizeof(double));
            thread_ln_beta = (double *)malloc(
                (size_t)n_ln * sizeof(double));
            if (!thread_ln_gamma || !thread_ln_beta) {
                fprintf(stderr, "LN allocation failed\n");
                abort();
            }
            thread_ln_hidden = task->hidden_layers;
        }
    }
    return thread_ann;
}

/* ── objective ──────────────────────────────────────────────────── */
static const NnTask *current_task;

static int task_uses_mae_loss(const NnTask *task) {
    return strstr(task->name, "-mae") != NULL;
}

static int task_uses_max_loss(const NnTask *task) {
    return strstr(task->name, "-max") != NULL;
}

static double task_distance_weight_strength(const NnTask *task) {
    const char *p = strstr(task->name, "-dw");
    if (!p) return 0.0;
    return atof(p + 3);
}

static double task_huber_delta(const NnTask *task) {
    const char *p = strstr(task->name, "huber");
    if (!p) return -1.0;
    return atof(p + 5);
}

static int task_mutation_strategy(const NnTask *task) {
    if (strstr(task->name, "-basepbest"))  return 5;
    if (strstr(task->name, "-pbest"))      return 4;
    if (strstr(task->name, "-anneal"))     return 3;
    if (strstr(task->name, "-curtobest1")) return 2;
    if (strstr(task->name, "-best1"))      return 1;
    return 0;
}

static int task_pbest_percent(const NnTask *task) {
    const char *p = strstr(task->name, "-basepbest");
    int offset = 10;
    if (!p) {
        p = strstr(task->name, "-pbest");
        offset = 6;
    }
    if (!p)
        return 0;
    const int percent = atoi(p + offset);
    return percent > 0 ? percent : 10;
}

static int is_full_box_pair_task(const char *name) {
    if (strncmp(name, "pair1", 5) != 0) return 0;
    if (strstr(name, "mid")) return 0;
    if (strstr(name, "2p0")) return 0;
    if (strstr(name, "mix")) return 0;
    if (strstr(name, "roll2")) return 0;
    if (strstr(name, "scale")) return 0;
    return 1;
}

static double task_loss(const double *weights, int n) {
    const NnTask *task = current_task;
    genann *ann = task_ann(task);
    const int total = task_total_params(task, ann->total_weights);
    if (n != total)
        return INFINITY;

    const int n_reg = ann->total_weights;
    memcpy(ann->weight, weights, (size_t)n_reg * sizeof(double));
    if (task_uses_layernorm(task)) {
        const int n_ln = task->hidden * task->hidden_layers;
        memcpy(thread_ln_gamma, weights + n_reg,
               (size_t)n_ln * sizeof(double));
        memcpy(thread_ln_beta, weights + n_reg + n_ln,
               (size_t)n_ln * sizeof(double));
    }

    const double dw_strength = task_distance_weight_strength(task);
    const double huber_delta = task_huber_delta(task);
    const double r_max = 2.5 * 1.41421356237; /* sqrt(2) * 2.5 */
    double loss = 0.0;
    double max_sq = 0.0;
    double total_weight = 0.0;
    for (int i = 0; i < task->samples; ++i) {
        const double *input  = task->input + (size_t)i * task->inputs;
        const double *target = task->target + (size_t)i * task->target_stride;

        const double sw = dw_strength > 0.0
            ? 1.0 + dw_strength * (sqrt(input[0] * input[0]
                                      + input[1] * input[1]) / r_max)
            : 1.0;

        if (task->rollout_loss_weight > 0.0) {
            const double y0 = genann_run_ln(ann, input)[0];
            const double next_input[2] = { y0, input[0] };
            const double y1 = genann_run_ln(ann, next_input)[0];
            const double e0 = y0 - target[0];
            const double e1 = y1 - target[1];
            loss += sw * (e0 * e0 + task->rollout_loss_weight * e1 * e1);
            total_weight += sw * (1.0 + task->rollout_loss_weight);
            continue;
        }

        const double *output = genann_run_ln(ann, input);
        for (int o = 0; o < task->outputs; ++o) {
            const double w = o == 0 ? 1.0 : task->aux_loss_weight;
            const double e = output[o] - target[o];
            const double sq = e * e;
            if (task_uses_max_loss(task)) {
                if (sq > max_sq)
                    max_sq = sq;
            } else if (huber_delta > 0.0) {
                const double ae = fabs(e);
                if (ae <= huber_delta)
                    loss += sw * w * 0.5 * e * e;
                else
                    loss += sw * w * huber_delta * (ae - 0.5 * huber_delta);
                total_weight += sw * w;
            } else if (task_uses_mae_loss(task)) {
                loss += sw * w * fabs(e);
                total_weight += sw * w;
            } else {
                loss += sw * w * sq;
                total_weight += sw * w;
            }
        }
    }
    if (task_uses_max_loss(task))
        return max_sq;
    return total_weight > 0.0 ? loss / total_weight : loss;
}

/* ── helpers ────────────────────────────────────────────────────── */
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

/* ── sequence generation ────────────────────────────────────────── */
static void generate_sequence(double *seq, int length, uint64_t seed) {
    /* seed the first two values in [-0.8, 0.8] — well inside the
       recurrence's stable range so transients are short */
    seq[0] = -0.8 + local_rand01(&seed) * 1.6;
    seq[1] = -0.8 + local_rand01(&seed) * 1.6;
    for (int t = 2; t < length; ++t)
        seq[t] = recurrence_next(seq[t - 1], seq[t - 2]);
}

/* pack contiguous sequence into (input_window, target) samples
   single-step:  input = [seq[t-1], seq[t-2]]   target = [seq[t]]
   2-step:       input = [seq[t-1], seq[t-2]]   target = [seq[t], seq[t+1]] */
static void pack_seq1(const double *seq, int num, double *inp, double *tgt) {
    for (int i = 0; i < num; ++i) {
        const int t = i + 2;
        inp[i * 2]     = seq[t - 1];
        inp[i * 2 + 1] = seq[t - 2];
        tgt[i]         = seq[t];
    }
}

static void pack_seq2(const double *seq, int num, double *inp, double *tgt) {
    for (int i = 0; i < num; ++i) {
        const int t = i + 2;
        inp[i * 2]       = seq[t - 1];
        inp[i * 2 + 1]   = seq[t - 2];
        tgt[i * 2]       = seq[t];
        tgt[i * 2 + 1]   = seq[t + 1];
    }
}

static void pack_seq_sin_aux(const double *seq, int num, double *inp, double *tgt) {
    for (int i = 0; i < num; ++i) {
        const int t = i + 2;
        inp[i * 2]       = seq[t - 1];
        inp[i * 2 + 1]   = seq[t - 2];
        tgt[i * 2]       = seq[t];
        tgt[i * 2 + 1]   = sin(2.5 * seq[t - 1]);
    }
}

static void pack_seq_dup(const double *seq, int num, double *inp, double *tgt) {
    for (int i = 0; i < num; ++i) {
        const int t = i + 2;
        inp[i * 2]       = seq[t - 1];
        inp[i * 2 + 1]   = seq[t - 2];
        tgt[i * 2]       = seq[t];
        tgt[i * 2 + 1]   = seq[t];
    }
}

static void fill_random_pair_data(double *input1, double *target1,
                                  double *input2, double *target2,
                                  int samples, double lo, double hi,
                                  uint64_t seed) {
    for (int i = 0; i < samples; ++i) {
        const double a = local_rand_range(&seed, lo, hi);
        const double b = local_rand_range(&seed, lo, hi);
        const double y0 = recurrence_next(a, b);
        const double y1 = recurrence_next(y0, a);

        input1[(size_t)i * 2] = a;
        input1[(size_t)i * 2 + 1] = b;
        target1[i] = y0;

        input2[(size_t)i * 2] = a;
        input2[(size_t)i * 2 + 1] = b;
        target2[(size_t)i * 2] = y0;
        target2[(size_t)i * 2 + 1] = y1;
    }
}

static void fill_mixed_pair_data(double *input1, double *target1,
                                 double *input2, double *target2,
                                 int samples, int full_samples,
                                 uint64_t seed) {
    if (full_samples < 0)
        full_samples = 0;
    if (full_samples > samples)
        full_samples = samples;

    const int mid_samples = samples - full_samples;
    fill_random_pair_data(input1, target1, input2, target2,
                          mid_samples, -1.5, 1.5, seed);
    fill_random_pair_data(input1 + (size_t)mid_samples * 2,
                          target1 + mid_samples,
                          input2 + (size_t)mid_samples * 2,
                          target2 + (size_t)mid_samples * 2,
                          full_samples, -2.5, 2.5,
                          seed ^ 0xC0FFEE123456789ULL);
}

static void fill_nd_pair_data(double *input, double *target,
                               int samples, int n_dims,
                               double lo, double hi, uint64_t seed) {
    for (int i = 0; i < samples; ++i) {
        double *xs = input + (size_t)i * n_dims;
        for (int d = 0; d < n_dims; ++d)
            xs[d] = local_rand_range(&seed, lo, hi);
        target[i] = recurrence_nd(n_dims, xs);
    }
}

static void init_generated_data(void) {
    /* train sequence — seed A */
    enum { TRAIN_LEN = SEQ_TRAIN_SAMPLES + 4 };
    double *train_seq = (double *)malloc((size_t)TRAIN_LEN * sizeof(double));
    if (!train_seq) { fprintf(stderr, "malloc failed\n"); abort(); }
    generate_sequence(train_seq, TRAIN_LEN, 0x5EED5EEDC0FFEEULL);
    pack_seq1(train_seq, SEQ_TRAIN_SAMPLES, SEQ1_TRAIN_INPUT, SEQ1_TRAIN_TARGET);
    pack_seq2(train_seq, SEQ_TRAIN_SAMPLES, SEQ2_TRAIN_INPUT, SEQ2_TRAIN_TARGET);
    pack_seq_sin_aux(train_seq, SEQ_TRAIN_SAMPLES,
                     SEQ_AUX_TRAIN_INPUT, SEQ_AUX_TRAIN_TARGET);
    pack_seq_dup(train_seq, SEQ_TRAIN_SAMPLES,
                 SEQ_DUP_TRAIN_INPUT, SEQ_DUP_TRAIN_TARGET);
    free(train_seq);

    /* test sequence — seed B (independent) */
    enum { TEST_LEN = SEQ_TEST_SAMPLES + 4 };
    double *test_seq = (double *)malloc((size_t)TEST_LEN * sizeof(double));
    if (!test_seq) { fprintf(stderr, "malloc failed\n"); abort(); }
    generate_sequence(test_seq, TEST_LEN, 0x7E57C0FFEE5EEDULL);
    pack_seq1(test_seq, SEQ_TEST_SAMPLES, SEQ1_TEST_INPUT, SEQ1_TEST_TARGET);
    pack_seq2(test_seq, SEQ_TEST_SAMPLES, SEQ2_TEST_INPUT, SEQ2_TEST_TARGET);
    pack_seq_sin_aux(test_seq, SEQ_TEST_SAMPLES,
                     SEQ_AUX_TEST_INPUT, SEQ_AUX_TEST_TARGET);
    pack_seq_dup(test_seq, SEQ_TEST_SAMPLES,
                 SEQ_DUP_TEST_INPUT, SEQ_DUP_TEST_TARGET);
    free(test_seq);

    fill_random_pair_data(PAIR1_TRAIN_INPUT, PAIR1_TRAIN_TARGET,
                          PAIR2_TRAIN_INPUT, PAIR2_TRAIN_TARGET,
                          SEQ_TRAIN_SAMPLES, -2.5, 2.5,
                          0x9A1E5EEDC0FFEEULL);
    fill_random_pair_data(PAIR1_TEST_INPUT, PAIR1_TEST_TARGET,
                          PAIR2_TEST_INPUT, PAIR2_TEST_TARGET,
                          SEQ_TEST_SAMPLES, -2.5, 2.5,
                          0x0DD57A7E5EEDULL);

    /* scaled-input copies of pair1: inputs / 2.5 → ~[-1, 1] */
    for (int i = 0; i < SEQ_TRAIN_SAMPLES; ++i) {
        PAIR_SCALED_TRAIN_INPUT[i * 2]     = PAIR1_TRAIN_INPUT[i * 2] / 2.5;
        PAIR_SCALED_TRAIN_INPUT[i * 2 + 1] = PAIR1_TRAIN_INPUT[i * 2 + 1] / 2.5;
    }
    for (int i = 0; i < SEQ_TEST_SAMPLES; ++i) {
        PAIR_SCALED_TEST_INPUT[i * 2]     = PAIR1_TEST_INPUT[i * 2] / 2.5;
        PAIR_SCALED_TEST_INPUT[i * 2 + 1] = PAIR1_TEST_INPUT[i * 2 + 1] / 2.5;
    }
    fill_nd_pair_data(PAIR5_TRAIN_INPUT, PAIR5_TRAIN_TARGET,
                      SEQ_TRAIN_SAMPLES, 5, -2.5, 2.5,
                      0x50A1E5EEDC0FFEEULL);
    fill_nd_pair_data(PAIR5_TEST_INPUT, PAIR5_TEST_TARGET,
                      SEQ_TEST_SAMPLES, 5, -2.5, 2.5,
                      0x50A1E7E57C0FFEEULL);
    fill_nd_pair_data(PAIR10_TRAIN_INPUT, PAIR10_TRAIN_TARGET,
                      SEQ_TRAIN_SAMPLES, 10, -2.5, 2.5,
                      0x10A1E5EEDC0FFEEULL);
    fill_nd_pair_data(PAIR10_TEST_INPUT, PAIR10_TEST_TARGET,
                      SEQ_TEST_SAMPLES, 10, -2.5, 2.5,
                      0x10A1E7E57C0FFEEULL);
    /* nonlinear 5D: sin on a1 and a2 */
    {
        uint64_t s = 0x5A1E5EEDC0FFEEULL;
        for (int i = 0; i < SEQ_TRAIN_SAMPLES; ++i) {
            double *xs = PAIR5NL_TRAIN_INPUT + (size_t)i * 5;
            for (int d = 0; d < 5; ++d)
                xs[d] = local_rand_range(&s, -2.5, 2.5);
            PAIR5NL_TRAIN_TARGET[i] = recurrence_5d_nonlin(xs);
        }
    }
    {
        uint64_t s = 0x5A1E7E57C0FFEEULL;
        for (int i = 0; i < SEQ_TEST_SAMPLES; ++i) {
            double *xs = PAIR5NL_TEST_INPUT + (size_t)i * 5;
            for (int d = 0; d < 5; ++d)
                xs[d] = local_rand_range(&s, -2.5, 2.5);
            PAIR5NL_TEST_TARGET[i] = recurrence_5d_nonlin(xs);
        }
    }
    /* pair5-hard: multi-sine + interaction */
    {
        uint64_t s = 0x5A0E5EEDC0FFEEULL;
        for (int i = 0; i < SEQ_TRAIN_SAMPLES; ++i) {
            double *xs = PAIR5H_TRAIN_INPUT + (size_t)i * 5;
            for (int d = 0; d < 5; ++d)
                xs[d] = local_rand_range(&s, -2.5, 2.5);
            PAIR5H_TRAIN_TARGET[i] = recurrence_5d_hard(xs);
        }
    }
    {
        uint64_t s = 0x5A0E7E57C0FFEEULL;
        for (int i = 0; i < SEQ_TEST_SAMPLES; ++i) {
            double *xs = PAIR5H_TEST_INPUT + (size_t)i * 5;
            for (int d = 0; d < 5; ++d)
                xs[d] = local_rand_range(&s, -2.5, 2.5);
            PAIR5H_TEST_TARGET[i] = recurrence_5d_hard(xs);
        }
    }
    fill_random_pair_data(PAIRMID1_TRAIN_INPUT, PAIRMID1_TRAIN_TARGET,
                          PAIRMID2_TRAIN_INPUT, PAIRMID2_TRAIN_TARGET,
                          SEQ_TRAIN_SAMPLES, -1.5, 1.5,
                          0x4D1D5EEDC0FFEEULL);
    fill_random_pair_data(PAIRMID1_TEST_INPUT, PAIRMID1_TEST_TARGET,
                          PAIRMID2_TEST_INPUT, PAIRMID2_TEST_TARGET,
                          SEQ_TEST_SAMPLES, -1.5, 1.5,
                          0x4D1D7E57C0FFEEULL);
    fill_random_pair_data(PAIR2P0_1_TRAIN_INPUT, PAIR2P0_1_TRAIN_TARGET,
                          PAIR2P0_2_TRAIN_INPUT, PAIR2P0_2_TRAIN_TARGET,
                          SEQ_TRAIN_SAMPLES, -2.0, 2.0,
                          0x2A0E5EEDC0FFEEULL);
    fill_random_pair_data(PAIR2P0_1_TEST_INPUT, PAIR2P0_1_TEST_TARGET,
                          PAIR2P0_2_TEST_INPUT, PAIR2P0_2_TEST_TARGET,
                          SEQ_TEST_SAMPLES, -2.0, 2.0,
                          0x2A0E7E57C0FFEEULL);
    fill_mixed_pair_data(PAIRMIX25_1_TRAIN_INPUT, PAIRMIX25_1_TRAIN_TARGET,
                         PAIRMIX25_2_TRAIN_INPUT, PAIRMIX25_2_TRAIN_TARGET,
                         SEQ_TRAIN_SAMPLES, SEQ_TRAIN_SAMPLES / 4,
                         0x251E5EEDC0FFEEULL);
    fill_mixed_pair_data(PAIRMIX50_1_TRAIN_INPUT, PAIRMIX50_1_TRAIN_TARGET,
                         PAIRMIX50_2_TRAIN_INPUT, PAIRMIX50_2_TRAIN_TARGET,
                         SEQ_TRAIN_SAMPLES, SEQ_TRAIN_SAMPLES / 2,
                         0x501E5EEDC0FFEEULL);
}

/* ── evaluation ─────────────────────────────────────────────────── */
static void evaluate_task(genann *ann, const NnTask *task,
                          const double *input, const double *target,
                          int samples,
                          double *mse_out, double *mae_out) {
    double mse = 0.0, mae = 0.0;
    for (int i = 0; i < samples; ++i) {
        const double *xs = input + (size_t)i * task->inputs;
        const double *ts = target + (size_t)i * task->target_stride;
        const double *ys = genann_run_ln(ann, xs);
        for (int o = 0; o < task->outputs; ++o) {
            const double e = ys[o] - ts[o];
            mse += e * e;
            mae += fabs(e);
        }
    }
    const double denom = (double)(samples * task->outputs);
    *mse_out = mse / denom;
    *mae_out = mae / denom;
}

static void evaluate_task_by_output(genann *ann, const NnTask *task,
                                    const double *input, const double *target,
                                    int samples,
                                    double *mse_out, double *mae_out) {
    for (int o = 0; o < task->outputs; ++o) {
        mse_out[o] = 0.0;
        mae_out[o] = 0.0;
    }

    for (int i = 0; i < samples; ++i) {
        const double *xs = input + (size_t)i * task->inputs;
        const double *ts = target + (size_t)i * task->target_stride;
        const double *ys = genann_run_ln(ann, xs);
        for (int o = 0; o < task->outputs; ++o) {
            const double e = ys[o] - ts[o];
            mse_out[o] += e * e;
            mae_out[o] += fabs(e);
        }
    }

    for (int o = 0; o < task->outputs; ++o) {
        mse_out[o] /= (double)samples;
        mae_out[o] /= (double)samples;
    }
}

static double target_variance(const NnTask *task, const double *target,
                              int samples) {
    const int total = samples * task->outputs;
    double mean = 0.0;
    for (int i = 0; i < samples; ++i) {
        const double *ts = target + (size_t)i * task->target_stride;
        for (int o = 0; o < task->outputs; ++o)
            mean += ts[o];
    }
    mean /= (double)total;

    double var = 0.0;
    for (int i = 0; i < samples; ++i) {
        const double *ts = target + (size_t)i * task->target_stride;
        for (int o = 0; o < task->outputs; ++o) {
            const double d = ts[o] - mean;
            var += d * d;
        }
    }
    return var / (double)total;
}

/* ── regional error diagnostics ─────────────────────────────────── */
enum { REGIONAL_GRID = 200 };

typedef struct {
    double mse_sum;
    double mae_sum;
    double max_abs;
    int count;
} RegionalBin;

static void regional_bin_add(RegionalBin *b, double abs_err) {
    b->mse_sum += abs_err * abs_err;
    b->mae_sum += abs_err;
    if (abs_err > b->max_abs) b->max_abs = abs_err;
    b->count++;
}

static void regional_print(const char *title,
                           const RegionalBin *bins, int n,
                           const char *const *labels) {
    printf("\n  %s\n", title);
    printf("  %-22s %8s %10s %10s %10s\n",
           "region", "count", "MSE", "MAE", "max|e|");
    for (int i = 0; i < n; ++i) {
        if (bins[i].count == 0) continue;
        printf("  %-22s %8d %10.6f %10.6f %10.6f\n",
               labels[i], bins[i].count,
               bins[i].mse_sum / bins[i].count,
               bins[i].mae_sum / bins[i].count,
               bins[i].max_abs);
    }
}

static void evaluate_regional_errors(const genann *ann) {
    enum { DIST_BINS = 4, GRAD_BINS = 3, CURV_BINS = 3,
           QUAD_BINS = 4, TMAG_BINS = 3 };
    RegionalBin dist[DIST_BINS] = {{0}};
    RegionalBin grad[GRAD_BINS] = {{0}};
    RegionalBin curv[CURV_BINS] = {{0}};
    RegionalBin quad[QUAD_BINS] = {{0}};
    RegionalBin tmag[TMAG_BINS] = {{0}};

    const double lo = -2.5, hi = 2.5;
    const double step = (hi - lo) / (double)(REGIONAL_GRID - 1);
    double total_mse = 0.0;
    int total_n = 0;

    for (int ia = 0; ia < REGIONAL_GRID; ++ia) {
        const double a = lo + (double)ia * step;
        for (int ib = 0; ib < REGIONAL_GRID; ++ib) {
            const double b = lo + (double)ib * step;
            const double target = recurrence_next(a, b);
            const double input[2] = {a, b};
            const double output = genann_run_ln(ann, input)[0];
            const double abs_err = fabs(output - target);

            total_mse += (output - target) * (output - target);
            total_n++;

            /* distance from origin */
            {
                const double r = sqrt(a * a + b * b);
                const int idx = r < 1.0 ? 0 : r < 2.0 ? 1 : r < 3.0 ? 2 : 3;
                regional_bin_add(&dist[idx], abs_err);
            }

            /* gradient magnitude |df/da| = |2.5*cos(2.5*a)| */
            {
                const double g = fabs(cos(2.5 * a));
                const int idx = g < 0.33 ? 0 : g < 0.67 ? 1 : 2;
                regional_bin_add(&grad[idx], abs_err);
            }

            /* curvature |d²f/da²| = |6.25*sin(2.5*a)| */
            {
                const double c = fabs(sin(2.5 * a));
                const int idx = c < 0.33 ? 0 : c < 0.67 ? 1 : 2;
                regional_bin_add(&curv[idx], abs_err);
            }

            /* sign quadrant */
            {
                const int idx = (a >= 0.0 && b >= 0.0) ? 0
                              : (a <  0.0 && b >= 0.0) ? 1
                              : (a <  0.0 && b <  0.0) ? 2 : 3;
                regional_bin_add(&quad[idx], abs_err);
            }

            /* target magnitude */
            {
                const double at = fabs(target);
                const int idx = at < 0.8 ? 0 : at < 1.6 ? 1 : 2;
                regional_bin_add(&tmag[idx], abs_err);
            }
        }
    }

    printf("\n  grid %dx%d overall MSE %.8f  (random test comparison)\n",
           REGIONAL_GRID, REGIONAL_GRID, total_mse / total_n);

    {
        const char *labels[] = {
            "center r<1.0", "mid r 1.0-2.0",
            "edge r 2.0-3.0", "corner r>=3.0"};
        regional_print("L2 distance from origin", dist, DIST_BINS, labels);
    }
    {
        const char *labels[] = {
            "|cos(2.5a)|<0.33  (flat)",
            "|cos(2.5a)| 0.33-0.67",
            "|cos(2.5a)|>=0.67 (steep)"};
        regional_print("Gradient magnitude |df/da|", grad, GRAD_BINS, labels);
    }
    {
        const char *labels[] = {
            "|sin(2.5a)|<0.33  (linear)",
            "|sin(2.5a)| 0.33-0.67",
            "|sin(2.5a)|>=0.67 (curved)"};
        regional_print("Curvature |d^2f/da^2|", curv, CURV_BINS, labels);
    }
    {
        const char *labels[] = {
            "Q1 a>=0,b>=0", "Q2 a<0,b>=0",
            "Q3 a<0,b<0",   "Q4 a>=0,b<0"};
        regional_print("Sign quadrant", quad, QUAD_BINS, labels);
    }
    {
        const char *labels[] = {
            "|target|<0.8", "|target| 0.8-1.6", "|target|>=1.6"};
        regional_print("Target magnitude", tmag, TMAG_BINS, labels);
    }
}

static int double_compare(const void *a, const void *b) {
    const double da = *(const double *)a;
    const double db = *(const double *)b;
    return (da > db) - (da < db);
}

static void evaluate_abs_error_stats(genann *ann, const NnTask *task,
                                     const double *input, const double *target,
                                     int samples,
                                     double *p95_abs_out,
                                     double *max_abs_out) {
    double *errs = (double *)malloc((size_t)samples * sizeof(double));
    if (!errs) {
        *p95_abs_out = INFINITY;
        *max_abs_out = INFINITY;
        return;
    }

    double max_abs = 0.0;
    for (int i = 0; i < samples; ++i) {
        const double *xs = input + (size_t)i * task->inputs;
        const double *ts = target + (size_t)i * task->target_stride;
        const double e = fabs(genann_run_ln(ann, xs)[0] - ts[0]);
        errs[i] = e;
        if (e > max_abs)
            max_abs = e;
    }

    qsort(errs, (size_t)samples, sizeof(errs[0]), double_compare);
    int p95_index = (int)(0.95 * (double)samples);
    if (p95_index >= samples)
        p95_index = samples - 1;
    *p95_abs_out = errs[p95_index];
    *max_abs_out = max_abs;
    free(errs);
}

static void evaluate_recursive_two_step(genann *ann,
                                        const double *input,
                                        const double *target2,
                                        int samples,
                                        double *mse0_out,
                                        double *mae0_out,
                                        double *mse1_out,
                                        double *mae1_out) {
    double mse0 = 0.0, mae0 = 0.0;
    double mse1 = 0.0, mae1 = 0.0;
    for (int i = 0; i < samples; ++i) {
        const double *xs = input + (size_t)i * 2;
        const double *ts = target2 + (size_t)i * 2;

        const double y0 = genann_run_ln(ann, xs)[0];
        const double next_input[2] = { y0, xs[0] };
        const double y1 = genann_run_ln(ann, next_input)[0];

        const double e0 = y0 - ts[0];
        const double e1 = y1 - ts[1];
        mse0 += e0 * e0;
        mae0 += fabs(e0);
        mse1 += e1 * e1;
        mae1 += fabs(e1);
    }

    *mse0_out = mse0 / (double)samples;
    *mae0_out = mae0 / (double)samples;
    *mse1_out = mse1 / (double)samples;
    *mae1_out = mae1 / (double)samples;
}

static void print_by_output_metrics(const char *label, const NnTask *task,
                                    const double *mse, const double *mae) {
    for (int o = 0; o < task->outputs; ++o)
        printf("%s[%d] MSE %.8g  MAE %.8g\n", label, o, mse[o], mae[o]);
}

static long long de_trial_evaluation_budget(int initial_popsize,
                                            int min_popsize,
                                            int generations) {
    long long evals = initial_popsize;
    int active = initial_popsize;
    for (int g = 0; g < generations; ++g) {
        evals += active;
        active = de_lpsr_population_size(initial_popsize, min_popsize,
                                         generations, g);
    }
    return evals;
}

/* ── run one task ───────────────────────────────────────────────── */
static int run_task(const NnTask *task, int runs, int generations_override) {
    genann *template = genann_init(task->inputs, task->hidden_layers,
                                   task->hidden, task->outputs);
    if (!template) {
        fprintf(stderr, "genann_init failed\n");
        return 1;
    }
    configure_task_ann(template, task);

    const int n_reg = template->total_weights;
    const int total_params = task_total_params(task, n_reg);
    genann_free(template);
    current_task = task;

    const uint64_t base_seed = 0xD1FF3A9E2B1C4D77ULL;
    double best_loss = INFINITY;
    uint64_t best_seed = 0;
    long best_ms = 0;
    double *best_weights = (double *)malloc((size_t)total_params * sizeof(double));
    double *run_weights = (double *)malloc((size_t)total_params * sizeof(double));
    if (!best_weights || !run_weights) {
        fprintf(stderr, "allocation failed\n");
        free(best_weights);
        free(run_weights);
        return 1;
    }

    const int generations = generations_override > 0
        ? generations_override
        : task->generations;

    const long long eval_budget = de_trial_evaluation_budget(
        task->popsize, task->min_popsize, generations);

    printf("%s: inputs=%d hidden=%d outputs=%d weights=%d(+%dLN) pop=%d->%d gens=%d evals/run=%lld runs=%d\n",
           task->name, task->inputs, task->hidden, task->outputs,
           n_reg, total_params - n_reg,
           task->popsize, task->min_popsize, generations, eval_budget, runs);
    printf("%-5s %14s %10s\n", "run", "loss", "ms");

    for (int r = 0; r < runs; ++r) {
        const uint64_t seed = base_seed + (uint64_t)r * 0x9e3779b97f4a7c15ULL;
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        double loss;
        const int strategy = task_mutation_strategy(task);
        if (strategy == 5)
            loss = de_solve_lpsr_basepbest(total_params, task->popsize,
                                  task->min_popsize, generations,
                                  task->bounds_min, task->bounds_max,
                                  task_loss, seed, run_weights,
                                  task_pbest_percent(task));
        else if (strategy == 4)
            loss = de_solve_lpsr_pbest(total_params, task->popsize,
                                  task->min_popsize, generations,
                                  task->bounds_min, task->bounds_max,
                                  task_loss, seed, run_weights,
                                  task_pbest_percent(task));
        else if (strategy == 3)
            loss = de_solve_lpsr_anneal(total_params, task->popsize,
                                  task->min_popsize, generations,
                                  task->bounds_min, task->bounds_max,
                                  task_loss, seed, run_weights);
        else if (strategy == 2)
            loss = de_solve_lpsr_curtobest1(total_params, task->popsize,
                                  task->min_popsize, generations,
                                  task->bounds_min, task->bounds_max,
                                  task_loss, seed, run_weights);
        else if (strategy == 1)
            loss = de_solve_lpsr_best1(total_params, task->popsize,
                                  task->min_popsize, generations,
                                  task->bounds_min, task->bounds_max,
                                  task_loss, seed, run_weights);
        else
            loss = de_solve_lpsr_best(total_params, task->popsize,
                                 task->min_popsize, generations,
                                 task->bounds_min, task->bounds_max,
                                 task_loss, seed, run_weights);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        const long ms = elapsed_ms(t0, t1);
        if (loss < best_loss) {
            best_loss = loss;
            best_seed = seed;
            best_ms = ms;
            memcpy(best_weights, run_weights, (size_t)total_params * sizeof(double));
        }
        printf("%-5d %14.8g %10ld\n", r, loss, ms);
    }

    printf("best loss %.8g seed 0x%016llx ms %ld\n\n",
           best_loss, (unsigned long long)best_seed, best_ms);

    /* ── print sample predictions ──────────────────────────────── */
    genann *ann = genann_init(task->inputs, task->hidden_layers,
                              task->hidden, task->outputs);
    if (!ann) {
        fprintf(stderr, "genann_init failed\n");
        free(best_weights);
        free(run_weights);
        return 1;
    }
    configure_task_ann(ann, task);
    memcpy(ann->weight, best_weights, (size_t)n_reg * sizeof(double));
    if (task_uses_layernorm(task)) {
        const int n_ln = task->hidden * task->hidden_layers;
        double *ln_g = (double *)malloc((size_t)n_ln * sizeof(double));
        double *ln_b = (double *)malloc((size_t)n_ln * sizeof(double));
        if (!ln_g || !ln_b) {
            fprintf(stderr, "LN alloc failed\n");
            free(ln_g); free(ln_b);
            genann_free(ann);
            free(best_weights); free(run_weights);
            return 1;
        }
        clear_thread_layernorm();
        memcpy(ln_g, best_weights + n_reg, (size_t)n_ln * sizeof(double));
        memcpy(ln_b, best_weights + n_reg + n_ln, (size_t)n_ln * sizeof(double));
        thread_ln_gamma = ln_g;
        thread_ln_beta = ln_b;
        thread_ln_hidden = task->hidden_layers;
    } else {
        clear_thread_layernorm();
    }

    /* header */
    printf("%-14s", "input");
    for (int o = 0; o < task->outputs; ++o)
        printf(" target[%d]", o);
    printf(" ");
    for (int o = 0; o < task->outputs; ++o)
        printf(" output[%d]", o);
    printf("\n");

    const int rows = task->samples <= 16 ? task->samples : 16;
    for (int i = 0; i < rows; ++i) {
        const double *input = task->input + (size_t)i * task->inputs;
        const double *target = task->target + (size_t)i * task->target_stride;
        const double *output = genann_run_ln(ann, input);
        for (int j = 0; j < task->inputs; ++j)
            printf("% .6f%s", input[j], j == task->inputs - 1 ? "" : " ");
        for (int o = 0; o < task->outputs; ++o)
            printf(" % .6f ]", target[o]);
        printf(" ");
        for (int o = 0; o < task->outputs; ++o)
            printf(" % .6f ]", output[o]);
        printf("\n");
    }
    if (rows < task->samples)
        printf("... %d more train rows omitted\n", task->samples - rows);

    double train_mse, train_mae, test_mse, test_mae;
    double train_p95_abs, train_max_abs, test_p95_abs, test_max_abs;
    double train_mse_by_output[8], train_mae_by_output[8];
    double test_mse_by_output[8], test_mae_by_output[8];
    if (task->outputs > 8) {
        fprintf(stderr, "too many outputs for metric buffer\n");
        clear_thread_layernorm();
        genann_free(ann);
        free(best_weights);
        free(run_weights);
        return 1;
    }
    evaluate_task(ann, task, task->input, task->target, task->samples,
                  &train_mse, &train_mae);
    evaluate_task(ann, task, task->test_input, task->test_target, task->test_samples,
                  &test_mse, &test_mae);
    printf("train MSE %.8g  MAE %.8g\n", train_mse, train_mae);
    printf("test  MSE %.8g  MAE %.8g\n", test_mse, test_mae);
    {
        const double train_var = target_variance(task, task->target, task->samples);
        const double test_var = target_variance(task, task->test_target,
                                                task->test_samples);
        if (train_var > 0.0 && test_var > 0.0) {
            printf("train nMSE %.8g  target_var %.8g\n",
                   train_mse / train_var, train_var);
            printf("test  nMSE %.8g  target_var %.8g\n",
                   test_mse / test_var, test_var);
        }
    }
    evaluate_abs_error_stats(ann, task, task->input, task->target, task->samples,
                             &train_p95_abs, &train_max_abs);
    evaluate_abs_error_stats(ann, task, task->test_input, task->test_target,
                             task->test_samples, &test_p95_abs, &test_max_abs);
    printf("train abs p95 %.8g  max %.8g\n", train_p95_abs, train_max_abs);
    printf("test  abs p95 %.8g  max %.8g\n", test_p95_abs, test_max_abs);

    evaluate_task_by_output(ann, task, task->input, task->target, task->samples,
                            train_mse_by_output, train_mae_by_output);
    evaluate_task_by_output(ann, task, task->test_input, task->test_target, task->test_samples,
                            test_mse_by_output, test_mae_by_output);
    print_by_output_metrics("train output", task, train_mse_by_output, train_mae_by_output);
    print_by_output_metrics("test  output", task, test_mse_by_output, test_mae_by_output);

    if (task->outputs == 1 && task->inputs <= 2) {
        double train_r_mse0, train_r_mae0, train_r_mse1, train_r_mae1;
        double test_r_mse0, test_r_mae0, test_r_mse1, test_r_mae1;
        const double *roll_train_input = task->target_stride >= 2
            ? task->input
            : SEQ2_TRAIN_INPUT;
        const double *roll_train_target = task->target_stride >= 2
            ? task->target
            : SEQ2_TRAIN_TARGET;
        const double *roll_test_input = task->target_stride >= 2
            ? task->test_input
            : SEQ2_TEST_INPUT;
        const double *roll_test_target = task->target_stride >= 2
            ? task->test_target
            : SEQ2_TEST_TARGET;
        evaluate_recursive_two_step(ann, roll_train_input, roll_train_target,
                                    task->samples,
                                    &train_r_mse0, &train_r_mae0,
                                    &train_r_mse1, &train_r_mae1);
        evaluate_recursive_two_step(ann, roll_test_input, roll_test_target,
                                    task->test_samples,
                                    &test_r_mse0, &test_r_mae0,
                                    &test_r_mse1, &test_r_mae1);
        printf("recursive train step0 MSE %.8g  MAE %.8g\n",
               train_r_mse0, train_r_mae0);
        printf("recursive train step1 MSE %.8g  MAE %.8g\n",
               train_r_mse1, train_r_mae1);
        printf("recursive test  step0 MSE %.8g  MAE %.8g\n",
               test_r_mse0, test_r_mae0);
        printf("recursive test  step1 MSE %.8g  MAE %.8g\n",
               test_r_mse1, test_r_mae1);

        if (is_full_box_pair_task(task->name))
            evaluate_regional_errors(ann);
    }

    clear_thread_layernorm();
    genann_free(ann);
    free(best_weights);
    free(run_weights);
    printf("\n");
    return best_loss < task->success_loss ? 0 : 1;
}

static const NnTask *find_task(const char *name) {
    const int task_count = (int)(sizeof(TASKS) / sizeof(TASKS[0]));
    for (int i = 0; i < task_count; ++i) {
        if (strcmp(name, TASKS[i].name) == 0)
            return &TASKS[i];
    }
    return NULL;
}

static void report_one_output_model(const char *label, const NnTask *task,
                                    const double *weights, int weight_count) {
    genann *ann = genann_init(task->inputs, task->hidden_layers,
                              task->hidden, task->outputs);
    if (!ann) {
        fprintf(stderr, "genann_init failed\n");
        return;
    }
    configure_task_ann(ann, task);
    memcpy(ann->weight, weights, (size_t)weight_count * sizeof(double));

    double train_mse, train_mae, test_mse, test_mae;
    double train_r_mse0, train_r_mae0, train_r_mse1, train_r_mae1;
    double test_r_mse0, test_r_mae0, test_r_mse1, test_r_mae1;

    evaluate_task(ann, task, task->input, task->target, task->samples,
                  &train_mse, &train_mae);
    evaluate_task(ann, task, task->test_input, task->test_target,
                  task->test_samples, &test_mse, &test_mae);
    evaluate_recursive_two_step(ann, task->input, task->target, task->samples,
                                &train_r_mse0, &train_r_mae0,
                                &train_r_mse1, &train_r_mae1);
    evaluate_recursive_two_step(ann, task->test_input, task->test_target,
                                task->test_samples,
                                &test_r_mse0, &test_r_mae0,
                                &test_r_mse1, &test_r_mae1);

    printf("%s train MSE %.8g  MAE %.8g\n", label, train_mse, train_mae);
    printf("%s test  MSE %.8g  MAE %.8g\n", label, test_mse, test_mae);
    printf("%s recursive train step1 MSE %.8g  MAE %.8g\n",
           label, train_r_mse1, train_r_mae1);
    printf("%s recursive test  step1 MSE %.8g  MAE %.8g\n",
           label, test_r_mse1, test_r_mae1);
    genann_free(ann);
}

static int run_curriculum(int generations_per_stage, int first_stage) {
    const NnTask *stages[] = {
        find_task("pair1-mid"),
        find_task("pair1-2p0"),
        find_task("pair1"),
    };
    const int stage_count = (int)(sizeof(stages) / sizeof(stages[0]));
    for (int i = first_stage; i < stage_count; ++i) {
        if (!stages[i]) {
            fprintf(stderr, "missing curriculum task\n");
            return 1;
        }
    }

    const NnTask *shape = stages[first_stage];
    genann *template = genann_init(shape->inputs, shape->hidden_layers,
                                   shape->hidden, shape->outputs);
    if (!template) {
        fprintf(stderr, "genann_init failed\n");
        return 1;
    }
    configure_task_ann(template, shape);
    const int weights = template->total_weights;
    genann_free(template);

    DeState state;
    if (!de_state_alloc(&state, weights, shape->popsize)) {
        fprintf(stderr, "de_state_alloc failed\n");
        return 1;
    }

    const uint64_t seed = 0xC0111C0111C0111ULL;
    double *best_weights = (double *)malloc((size_t)weights * sizeof(double));
    if (!best_weights) {
        fprintf(stderr, "allocation failed\n");
        de_state_free(&state);
        return 1;
    }

    printf("curriculum: stages=%s weights=%d pop=%d gens/stage=%d\n",
           first_stage == 0 ? "pair1-mid->pair1-2p0->pair1" :
           first_stage == 1 ? "pair1-2p0->pair1" : "pair1",
           weights, shape->popsize, generations_per_stage);

    current_task = stages[first_stage];
    de_state_init_random(&state, stages[first_stage]->bounds_min,
                         stages[first_stage]->bounds_max,
                         task_loss, seed);
    for (int s = first_stage; s < stage_count; ++s) {
        current_task = stages[s];
        de_state_rescore(&state, task_loss);
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        const double loss = de_state_evolve(&state, state.active_popsize,
                                            generations_per_stage,
                                            stages[s]->bounds_min,
                                            stages[s]->bounds_max,
                                            task_loss,
                                            seed + (uint64_t)s * 0x9e3779b97f4a7c15ULL,
                                            best_weights, 0, 0);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        printf("stage %-10s loss %.8g ms %ld active_pop=%d\n",
               stages[s]->name, loss, elapsed_ms(t0, t1), state.active_popsize);
        report_one_output_model("stage full-box", find_task("pair1"),
                                best_weights, weights);
    }

    free(best_weights);
    de_state_free(&state);
    printf("\n");
    return 0;
}

/* ── main ───────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    init_generated_data();

    const int runs = argc > 2 ? atoi(argv[2]) : DEFAULT_RUNS;
    const int generations_override = argc > 3 ? atoi(argv[3]) : 0;
    if (argc > 1 && (strcmp(argv[1], "curriculum") == 0 ||
                     strcmp(argv[1], "curriculum2") == 0 ||
                     strcmp(argv[1], "fixedfull") == 0)) {
        const int generations = argc > 2 && atoi(argv[2]) > 0
            ? atoi(argv[2])
            : 6000;
        const int first_stage = strcmp(argv[1], "curriculum") == 0 ? 0 :
            strcmp(argv[1], "curriculum2") == 0 ? 1 : 2;
        return run_curriculum(generations, first_stage);
    }

    int failed = 0;
    int selected = 0;
    const int task_count = (int)(sizeof(TASKS) / sizeof(TASKS[0]));
    for (int i = 0; i < task_count; ++i) {
        if (argc > 1 && strcmp(argv[1], TASKS[i].name) != 0)
            continue;
        selected = 1;
        failed += run_task(&TASKS[i], runs > 0 ? runs : DEFAULT_RUNS,
                           generations_override);
    }
    if (!selected) {
        fprintf(stderr, "usage: %s [task|curriculum] [runs] [generations]\n",
                argv[0]);
        return 2;
    }
    return failed ? 1 : 0;
}

#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    MAX_PEOPLE = 20,
    MAX_PAIRS = 128,
    POPSIZE = 200,
    GENERATIONS = 20000,
    ATTEMPTS = 5
};

static const double MIN_KEY = 0.0;
static const double MINIMUM_TRANSACTION_AMOUNT = 0.01;
static const double BALANCE_TOLERANCE = 0.05;
static const double UNSETTLED_BALANCE_PENALTY = 1000000.0;
static const double REAL_TRANSACTION_PENALTY = 1000.0;
static const double SOFT_TRANSACTION_PENALTY = 10.0;
static const double SOFT_TRANSACTION_SCALE = 0.1;
static const double UNSETTLED_SOFT_TRANSACTION_PENALTY = 100.0;
static const double MONEY_MOVED_PENALTY = 0.001;

typedef struct {
    const char *name;
    int n_people;
    const char *people[MAX_PEOPLE];
    double balances[MAX_PEOPLE];
    int expected_transaction_count;
    int expected_payment_count;
    struct {
        const char *from;
        const char *to;
        double amount;
    } expected_payments[4];
} Scenario;

typedef struct {
    int n_people;
    int n_pairs;
    const char *people[MAX_PEOPLE];
    double balances[MAX_PEOPLE];
    int from[MAX_PAIRS];
    int to[MAX_PAIRS];
    double max_payment;
} Problem;

typedef struct {
    double xs[MAX_PAIRS];
    double score;
} Agent;

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

static int person_index(const Problem *problem, const char *name) {
    for (int i = 0; i < problem->n_people; ++i) {
        if (strcmp(problem->people[i], name) == 0) return i;
    }
    return -1;
}

static int pair_index(const Problem *problem, int from, int to) {
    for (int i = 0; i < problem->n_pairs; ++i) {
        if (problem->from[i] == from && problem->to[i] == to) return i;
    }
    return -1;
}

static void sort_indices_by_desc_key(int *order, const double *xs, int left, int right) {
    while (left < right) {
        int i = left;
        int j = right;
        double pivot = xs[order[left + (right - left) / 2]];

        while (i <= j) {
            while (xs[order[i]] > pivot) i++;
            while (xs[order[j]] < pivot) j--;
            if (i <= j) {
                int tmp = order[i];
                order[i] = order[j];
                order[j] = tmp;
                i++;
                j--;
            }
        }

        if (j - left < right - i) {
            if (left < j) sort_indices_by_desc_key(order, xs, left, j);
            left = i;
        } else {
            if (i < right) sort_indices_by_desc_key(order, xs, i, right);
            right = j;
        }
    }
}

static Problem make_problem(const Scenario *scenario) {
    Problem problem = {0};
    problem.n_people = scenario->n_people;
    memcpy(problem.people, scenario->people, (size_t)scenario->n_people * sizeof(const char *));
    memcpy(problem.balances, scenario->balances, (size_t)scenario->n_people * sizeof(double));

    for (int i = 0; i < problem.n_people; ++i) {
        if (problem.balances[i] > 0.0) problem.max_payment += problem.balances[i];
    }

    for (int from = 0; from < problem.n_people; ++from) {
        if (problem.balances[from] >= 0.0) continue;
        for (int to = 0; to < problem.n_people; ++to) {
            if (problem.balances[to] <= 0.0) continue;
            problem.from[problem.n_pairs] = from;
            problem.to[problem.n_pairs] = to;
            problem.n_pairs++;
        }
    }

    return problem;
}

static void decode_payment_priorities(const Problem *problem, const double *xs, double *payments, double *final_balances) {
    int order[MAX_PAIRS];
    for (int i = 0; i < problem->n_pairs; ++i) {
        order[i] = i;
        payments[i] = 0.0;
    }
    memcpy(final_balances, problem->balances, (size_t)problem->n_people * sizeof(double));

    sort_indices_by_desc_key(order, xs, 0, problem->n_pairs - 1);

    for (int k = 0; k < problem->n_pairs; ++k) {
        int pair = order[k];
        int from = problem->from[pair];
        int to = problem->to[pair];
        double debtor_remaining = -final_balances[from];
        double creditor_remaining = final_balances[to];
        double amount = debtor_remaining < creditor_remaining ? debtor_remaining : creditor_remaining;
        if (amount > MINIMUM_TRANSACTION_AMOUNT) {
            payments[pair] = amount;
            final_balances[from] += amount;
            final_balances[to] -= amount;
        }
    }
}

static double settlement_score(const Problem *problem, const double *xs) {
    double payments[MAX_PAIRS];
    double final_balances[MAX_PEOPLE];
    decode_payment_priorities(problem, xs, payments, final_balances);

    double total_unsettled = 0.0;
    for (int i = 0; i < problem->n_people; ++i) total_unsettled += fabs(final_balances[i]);

    int real_transactions = 0;
    double smooth_transactions = 0.0;
    double money_moved = 0.0;
    for (int i = 0; i < problem->n_pairs; ++i) {
        double amount = payments[i];
        if (amount > MINIMUM_TRANSACTION_AMOUNT) real_transactions++;
        smooth_transactions += amount / (amount + SOFT_TRANSACTION_SCALE);
        money_moved += amount;
    }

    if (total_unsettled > BALANCE_TOLERANCE) {
        return total_unsettled * UNSETTLED_BALANCE_PENALTY +
               smooth_transactions * UNSETTLED_SOFT_TRANSACTION_PENALTY;
    }

    return real_transactions * REAL_TRANSACTION_PENALTY +
           smooth_transactions * SOFT_TRANSACTION_PENALTY +
           money_moved * MONEY_MOVED_PENALTY +
           total_unsettled;
}

static Agent run_de(const Problem *problem, uint64_t seed) {
    Agent pop[POPSIZE];
    double trials[POPSIZE][MAX_PAIRS];
    double scores[POPSIZE];

    int threads = omp_get_max_threads();
    uint64_t rng_states[256];
    if (threads > 256) threads = 256;
    for (int t = 0; t < threads; ++t) {
        uint64_t s = seed + (uint64_t)t * 0x9e3779b97f4a7c15ULL;
        rng_states[t] = splitmix64(&s);
    }

    for (int i = 0; i < POPSIZE; ++i) {
        for (int j = 0; j < problem->n_pairs; ++j) {
            pop[i].xs[j] = rand_range(&rng_states[0], MIN_KEY, problem->max_payment);
        }
        scores[i] = settlement_score(problem, pop[i].xs);
        pop[i].score = scores[i];
    }

    double crossover = 0.0;
    double mutate = 0.0;
#pragma omp parallel default(none) shared(problem, pop, trials, scores, rng_states, crossover, mutate)
    {
        int tid = omp_get_thread_num();
        uint64_t *state = &rng_states[tid];

        for (int g = 0; g < GENERATIONS; ++g) {
#pragma omp single
            {
                crossover = rand_range(&rng_states[0], 0.1, 1.0);
                mutate = rand_range(&rng_states[0], 0.2, 0.95);
            }

#pragma omp for schedule(static)
            for (int i = 0; i < POPSIZE; ++i) {
                int i0 = rand_int(state, POPSIZE);
                int i1 = rand_int(state, POPSIZE);
                int i2 = rand_int(state, POPSIZE);

                for (int j = 0; j < problem->n_pairs; ++j) {
                    if (rand01(state) < crossover) {
                        trials[i][j] = clamp(pop[i0].xs[j] + (pop[i1].xs[j] - pop[i2].xs[j]) * mutate, MIN_KEY, problem->max_payment);
                    } else {
                        trials[i][j] = pop[i].xs[j];
                    }
                }

                double trial_score = settlement_score(problem, trials[i]);
                if (trial_score < scores[i]) {
                    memcpy(pop[i].xs, trials[i], (size_t)problem->n_pairs * sizeof(double));
                    scores[i] = trial_score;
                    pop[i].score = trial_score;
                }
            }
        }
    }

    int best = 0;
    for (int i = 1; i < POPSIZE; ++i) {
        if (scores[i] < scores[best]) best = i;
    }
    return pop[best];
}

static void greedy_largest_balance_settlement(const Problem *problem, double *payments) {
    double balances[MAX_PEOPLE];
    memcpy(balances, problem->balances, (size_t)problem->n_people * sizeof(double));
    for (int i = 0; i < problem->n_pairs; ++i) payments[i] = 0.0;

    while (1) {
        int debtor = -1;
        int creditor = -1;
        for (int i = 0; i < problem->n_people; ++i) {
            if (balances[i] < -MINIMUM_TRANSACTION_AMOUNT && (debtor < 0 || -balances[i] > -balances[debtor])) debtor = i;
            if (balances[i] > MINIMUM_TRANSACTION_AMOUNT && (creditor < 0 || balances[i] > balances[creditor])) creditor = i;
        }
        if (debtor < 0 || creditor < 0) break;

        double amount = -balances[debtor] < balances[creditor] ? -balances[debtor] : balances[creditor];
        int idx = pair_index(problem, debtor, creditor);
        payments[idx] += amount;
        balances[debtor] += amount;
        balances[creditor] -= amount;
    }
}

static int transaction_count(const Problem *problem, const double *payments) {
    int count = 0;
    for (int i = 0; i < problem->n_pairs; ++i) {
        if (payments[i] > MINIMUM_TRANSACTION_AMOUNT) count++;
    }
    return count;
}

static void final_balances_from_payments(const Problem *problem, const double *payments, double *balances) {
    memcpy(balances, problem->balances, (size_t)problem->n_people * sizeof(double));
    for (int i = 0; i < problem->n_pairs; ++i) {
        balances[problem->from[i]] += payments[i];
        balances[problem->to[i]] -= payments[i];
    }
}

static int exact_minimum_transaction_count(const Problem *problem) {
    double nonzero[MAX_PEOPLE];
    int n = 0;
    for (int i = 0; i < problem->n_people; ++i) {
        if (fabs(problem->balances[i]) > BALANCE_TOLERANCE) nonzero[n++] = problem->balances[i];
    }

    int subset_count = 1 << n;
    int *groups = malloc((size_t)subset_count * sizeof(int));
    if (!groups) {
        fprintf(stderr, "allocation failed\n");
        exit(1);
    }
    for (int i = 0; i < subset_count; ++i) groups[i] = -1000000;
    groups[0] = 0;

    for (int mask = 1; mask < subset_count; ++mask) {
        double sum = 0.0;
        for (int bit = 0; bit < n; ++bit) {
            if (mask & (1 << bit)) sum += nonzero[bit];
        }
        if (fabs(sum) < BALANCE_TOLERANCE) groups[mask] = 1;

        for (int submask = (mask - 1) & mask; submask > 0; submask = (submask - 1) & mask) {
            int remaining = mask ^ submask;
            if (groups[submask] > -1000000 && groups[remaining] > -1000000) {
                int candidate = groups[submask] + groups[remaining];
                if (candidate > groups[mask]) groups[mask] = candidate;
            }
        }
    }

    int result = n - groups[subset_count - 1];
    free(groups);
    return result;
}

static void require_balanced(const char *scenario_name, const double *balances, int n) {
    for (int i = 0; i < n; ++i) {
        if (fabs(balances[i]) >= BALANCE_TOLERANCE) {
            fprintf(stderr, "%s: final balance %d was %.6f\n", scenario_name, i, balances[i]);
            exit(1);
        }
    }
}

static void print_payments(const Problem *problem, const double *payments) {
    printf("Payments:\n");
    for (int i = 0; i < problem->n_pairs; ++i) {
        if (payments[i] > MINIMUM_TRANSACTION_AMOUNT) {
            printf("%s pays %s: %.2f\n", problem->people[problem->from[i]], problem->people[problem->to[i]], payments[i]);
        }
    }
}

static void run_scenario(const Scenario *scenario) {
    Problem problem = make_problem(scenario);
    uint64_t seed = (uint64_t)time(NULL) ^ 0x51771E4EULL;

    Agent best = run_de(&problem, seed);
    for (int attempt = 1; attempt < ATTEMPTS; ++attempt) {
        Agent candidate = run_de(&problem, seed + (uint64_t)attempt * 0x9e3779b97f4a7c15ULL);
        if (candidate.score < best.score) best = candidate;
    }

    double de_payments[MAX_PAIRS];
    double de_balances[MAX_PEOPLE];
    decode_payment_priorities(&problem, best.xs, de_payments, de_balances);

    double greedy_payments[MAX_PAIRS];
    double greedy_balances[MAX_PEOPLE];
    greedy_largest_balance_settlement(&problem, greedy_payments);
    final_balances_from_payments(&problem, greedy_payments, greedy_balances);

    int de_count = transaction_count(&problem, de_payments);
    int greedy_count = transaction_count(&problem, greedy_payments);
    int expected_count = scenario->expected_transaction_count > 0 ? scenario->expected_transaction_count : exact_minimum_transaction_count(&problem);

    require_balanced(scenario->name, de_balances, problem.n_people);
    require_balanced(scenario->name, greedy_balances, problem.n_people);

    if (de_count != expected_count) {
        fprintf(stderr, "%s: expected DE transaction count %d, got %d\n", scenario->name, expected_count, de_count);
        exit(1);
    }

    for (int i = 0; i < scenario->expected_payment_count; ++i) {
        int from = person_index(&problem, scenario->expected_payments[i].from);
        int to = person_index(&problem, scenario->expected_payments[i].to);
        int idx = pair_index(&problem, from, to);
        double actual = de_payments[idx];
        if (fabs(actual - scenario->expected_payments[i].amount) >= BALANCE_TOLERANCE) {
            fprintf(stderr, "%s: expected %s pays %s %.2f, got %.2f\n",
                    scenario->name,
                    scenario->expected_payments[i].from,
                    scenario->expected_payments[i].to,
                    scenario->expected_payments[i].amount,
                    actual);
            exit(1);
        }
    }

    printf("Settlement scenario passed: %s\n", scenario->name);
    printf("DE score: %.12f\n", best.score);
    printf("DE transaction count: %d\n", de_count);
    print_payments(&problem, de_payments);
    printf("Greedy transaction count: %d\n\n", greedy_count);
}

int main(void) {
    Scenario scenarios[] = {
        {
            .name = "three people, one creditor",
            .n_people = 3,
            .people = {"A", "B", "C"},
            .balances = {40.0, -10.0, -30.0},
            .expected_payment_count = 2,
            .expected_payments = {{"B", "A", 10.0}, {"C", "A", 30.0}},
        },
        {
            .name = "four people, two clean pairs",
            .n_people = 4,
            .people = {"A", "B", "C", "D"},
            .balances = {50.0, 25.0, -50.0, -25.0},
            .expected_payment_count = 2,
            .expected_payments = {{"C", "A", 50.0}, {"D", "B", 25.0}},
        },
        {
            .name = "ten people, four zero-sum groups",
            .n_people = 10,
            .people = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"},
            .balances = {70.0, 40.0, 25.0, 15.0, -50.0, -20.0, -40.0, -10.0, -15.0, -15.0},
        },
        {
            .name = "twenty people, mixed unknown optimum",
            .n_people = 20,
            .people = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"},
            .balances = {115.0, 90.0, 75.0, 65.0, 55.0, 45.0, 35.0, 30.0, 25.0, 40.0,
                         -100.0, -85.0, -80.0, -70.0, -60.0, -55.0, -40.0, -35.0, -25.0, -25.0},
            .expected_transaction_count = 13,
        },
    };

    int count = (int)(sizeof(scenarios) / sizeof(scenarios[0]));
    for (int i = 0; i < count; ++i) {
        run_scenario(&scenarios[i]);
    }
    return 0;
}

module main;
import std::io;
import std::math;
import std::time;
import std::time::clock;
import std::core::mem;

const double MIN = -10.0;
const double MAX = 10.0;
const usz ARG_SIZE = 1000;
const usz POP_SIZE = 200;
const int GENERATIONS = 20000;
const int PRINT_INTERVAL = 20000;

struct Agent {
    double[ARG_SIZE] xs;
    double score;
}

fn ulong splitmix64(ulong* x) @inline {
    ulong z = *x + 0x9e3779b97f4a7c15UL;
    *x = z;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL;
    return z ^ (z >> 31);
}

fn double rand01(ulong* state) @inline {
    return (double)(splitmix64(state) >> 11) * (1.0 / 9007199254740992.0);
}

fn usz rand_index(ulong* state) @inline {
    return (usz)(splitmix64(state) % (ulong)POP_SIZE);
}

fn double rand_range(ulong* state, double min, double max) @inline {
    return rand01(state) * (max - min) + min;
}

fn double clamp(double x) @inline {
    return x < MIN ? MIN : (x > MAX ? MAX : x);
}

fn double f1(double[ARG_SIZE]* xs) @inline {
    double sum = 0.0;
    for (usz i = 0; i < ARG_SIZE; i++) {
        double x = (*xs)[i];
        sum += x * x;
    }
    return sum;
}

fn void algorithm() {
    Agent[POP_SIZE] agents;
    Agent*[POP_SIZE] pool;
    double[ARG_SIZE][POP_SIZE] trials;
    ulong state = 0xD1FF3A9E2B1C4D77UL;
    state = splitmix64(&state);
    
    for (usz i = 0; i < POP_SIZE; i++) {
        pool[i] = &agents[i];
        for (usz j = 0; j < ARG_SIZE; j++) {
            pool[i].xs[j] = rand_range(&state, MIN, MAX);
        }
        pool[i].score = f1(&pool[i].xs);
    }

    for (int g = 0; g < GENERATIONS; g++) {
        if (g % PRINT_INTERVAL == 0) {
            double sum_score = 0.0;
            double min_score = pool[0].score;
            foreach (a : pool) {
                sum_score += a.score;
                if (a.score < min_score) min_score = a.score;
            }
            io::printfn("Generation %d", g);
            io::printfn("Mean %f", sum_score / (double)POP_SIZE);
            io::printfn("Min %f", min_score);
        }

        double crossover = rand_range(&state, 0.1, 1.0);
        double mutate = rand_range(&state, 0.2, 0.95);

        for (usz i = 0; i < POP_SIZE; i++) {
            usz i0 = rand_index(&state);
            usz i1 = rand_index(&state);
            usz i2 = rand_index(&state);

            Agent* x0 = pool[i0];
            Agent* x1 = pool[i1];
            Agent* x2 = pool[i2];
            Agent* xt = pool[i];
            double[ARG_SIZE]* trial = &trials[i];

            for (usz j = 0; j < ARG_SIZE; j++) {
                if (rand01(&state) < crossover) {
                    (*trial)[j] = clamp(x0.xs[j] + (x1.xs[j] - x2.xs[j]) * mutate);
                } else {
                    (*trial)[j] = xt.xs[j];
                }
            }

            double trial_score = f1(trial);
            if (trial_score < xt.score) {
                mem::copy(&xt.xs[0], &(*trial)[0], ARG_SIZE * double.sizeof);
                xt.score = trial_score;
            }
        }
    }

    double best_score = pool[0].score;
    Agent* best_agent = pool[0];
    foreach (a : pool) {
        if (a.score < best_score) {
            best_score = a.score;
            best_agent = a;
        }
    }
    
    io::print("Generation best [");
    // Print first few for brevity?
    for (usz i = 0; i < 5; i++) {
        io::printf("%f, ", best_agent.xs[i]);
    }
    io::printn("...]");
    io::printfn("score: %f", best_score);
}

fn void main() {
    const int RUNS = 1;
    long[RUNS] times;
    
    io::printfn("Starting %d benchmark runs...", RUNS);
    
    for (int i = 0; i < RUNS; i++) {
        io::printf("Run %d/%d... ", i + 1, RUNS);
        
        Clock start = clock::now();
        algorithm();
        long duration = start.mark().to_ms();
        times[i] = duration;
        
        io::printfn("Done in %dms", duration);
    }
    
    double avg = 0.0;
    long min_time = times[0];
    long max_time = times[0];
    foreach (t : times) {
        avg += (double)t;
        if (t < min_time) min_time = t;
        if (t > max_time) max_time = t;
    }
    avg /= (double)RUNS;
    
    io::printfn("\nBenchmark Results (%d runs):", RUNS);
    io::printfn("Average: %.2f ms", avg);
    io::printfn("Min: %d ms", min_time);
    io::printfn("Max: %d ms", max_time);
}

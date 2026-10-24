// Canonical differential evolution, DE/rand/1/bin (Storn & Price 1997).
// Single-threaded. Same parameters, RNG (splitmix64) and draw order as
// ../roc and ../c, so all three should print the same score.
using System;
using System.Diagnostics;

const int Dim = 200, Np = 100, Generations = 10000;
const double F = 0.5, Cr = 0.9, Lo = -10.0, Hi = 10.0;

var sw = Stopwatch.StartNew();
ulong rng = 42;

var pop = new double[Np * Dim];
var next = new double[Np * Dim];
var scores = new double[Np];
var nextScores = new double[Np];
var trial = new double[Dim];

for (int k = 0; k < Np * Dim; k++) pop[k] = Lo + ToUnit(NextU64(ref rng)) * (Hi - Lo);
for (int i = 0; i < Np; i++) scores[i] = Sphere(pop.AsSpan(i * Dim, Dim));

for (int g = 0; g < Generations; g++)
{
    for (ulong i = 0; i < Np; i++)
    {
        ulong r1, r2, r3;
        do r1 = NextU64(ref rng) % Np; while (r1 == i);
        do r2 = NextU64(ref rng) % Np; while (r2 == i || r2 == r1);
        do r3 = NextU64(ref rng) % Np; while (r3 == i || r3 == r1 || r3 == r2);
        ulong jrand = NextU64(ref rng) % Dim;

        int a = (int)r1 * Dim, b = (int)r2 * Dim, c = (int)r3 * Dim, x = (int)i * Dim;
        for (int j = 0; j < Dim; j++)
        {
            if (ToUnit(NextU64(ref rng)) < Cr || (ulong)j == jrand)
                trial[j] = Math.Clamp(pop[a + j] + F * (pop[b + j] - pop[c + j]), Lo, Hi);
            else
                trial[j] = pop[x + j];
        }

        double ts = Sphere(trial);
        if (ts <= scores[i])
        {
            trial.CopyTo(next, x);
            nextScores[i] = ts;
        }
        else
        {
            Array.Copy(pop, x, next, x, Dim);
            nextScores[i] = scores[i];
        }
    }
    (pop, next) = (next, pop);
    (scores, nextScores) = (nextScores, scores);
}

double best = scores[0];
for (int i = 1; i < Np; i++) if (scores[i] < best) best = scores[i];

sw.Stop();
Console.WriteLine($"score {best:R}");
Console.WriteLine($"execution time {sw.ElapsedMilliseconds} ms");

static ulong NextU64(ref ulong s)
{
    ulong z = s += 0x9E3779B97F4A7C15UL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    return z ^ (z >> 31);
}

static double ToUnit(ulong v) => (v >> 11) * 1.1102230246251565e-16;

static double Sphere(ReadOnlySpan<double> xs)
{
    double sum = 0.0;
    for (int i = 0; i < xs.Length; i++) sum += xs[i] * xs[i];
    return sum;
}

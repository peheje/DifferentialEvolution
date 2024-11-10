using System.Diagnostics;

var optimizer = F1;
const int argsize = 1000;
const double min = -10.0;
const double max = 10.0;
const int generations = 20000;
const int popsize = 200;
const int print = 1000;
ParallelOptions pOptions = new() { MaxDegreeOfParallelism = 64 };

var sw = Stopwatch.StartNew();

var pop = new double[popsize][];
var scores = new double[popsize];
var trials = new double[popsize][];
for (var i = 0; i < popsize; i++)
{
    pop[i] = Enumerable.Range(0, argsize).Select(_ => RandomFloatRange(min, max)).ToArray();
    trials[i] = new double[argsize];
    scores[i] = optimizer(pop[i]);
}

for (var g = 0; g <= generations; g++)
{
    var crossover = CrossoverOdds();
    var mutate = MutateOdds();

    Parallel.For(0, popsize, pOptions, i =>
    {
        var x0 = pop[Random.Shared.Next(popsize)];
        var x1 = pop[Random.Shared.Next(popsize)];
        var x2 = pop[Random.Shared.Next(popsize)];
        var xt = pop[i];

        for (var j = 0; j < argsize; j++)
        {
            if (Random.Shared.NextDouble() < crossover)
                trials[i][j] = Clamp(x0[j] + (x1[j] - x2[j]) * mutate);
            else
                trials[i][j] = xt[j];
        }

        var trialScore = optimizer(trials[i]);

        if (trialScore < scores[i])
        {
            Array.Copy(trials[i], pop[i], argsize);
            scores[i] = trialScore;
        }
    });

    if (g % print == 0)
    {
        Console.WriteLine($"generation {g}");
        Console.WriteLine($"mean {scores.Average():F6}");
        Console.WriteLine($"minimum {scores.Min():F6}");
    }
}

var bestIndex = Array.IndexOf(scores, scores.Min());
var best = pop[bestIndex];
Console.WriteLine($"best [{string.Join(", ", best)}]");
Console.WriteLine($"score {scores[bestIndex]:F6}");
Console.WriteLine($"execution time {sw.ElapsedMilliseconds} ms");
return;

double F1(double[] args) => args.Sum(x => x * x);

double CrossoverOdds() => RandomFloatRange(0.1, 1.0);

double MutateOdds() => RandomFloatRange(0.2, 0.95);

double Clamp(double x) => Math.Clamp(x, min, max);

double RandomFloatRange(double minn, double maxx) => Random.Shared.NextDouble() * (maxx - minn) + minn;
open Problems
open System.Threading.Tasks

let sw = System.Diagnostics.Stopwatch.StartNew()

let random = System.Random.Shared
let rand () = random.NextDouble()
let randomFloatRange min max = rand () * (max - min) + min

let print = 20000
let optimizer = f1
let argsize = 1000
let min, max = -10.0, 10.0
let generations = 20000
let popsize = 200
let crossoverOdds () = randomFloatRange 0.1 1.0
let mutateOdds () = randomFloatRange 0.2 0.95
let clamp x = System.Math.Clamp(x, min, max)
let pOptions = ParallelOptions(MaxDegreeOfParallelism = 64)

let pop = Array.init popsize (fun _ -> Array.init argsize (fun _ -> randomFloatRange min max))
let scores = pop |> Array.map optimizer
let trials = Array.init popsize (fun _ -> Array.zeroCreate argsize)

for g in 0..generations-1 do
    let crossover = crossoverOdds ()
    let mutate = mutateOdds ()

    Parallel.For(0, popsize, pOptions, fun i ->
        let x0 = pop |> Array.randomChoice
        let x1 = pop |> Array.randomChoice
        let x2 = pop |> Array.randomChoice
        let xt = pop[i]

        for j in 0..argsize-1 do
            if rand () < crossover then
                trials[i][j] <- x0[j] + (x1[j] - x2[j]) * mutate |> clamp
            else
                trials[i][j] <- xt[j]

        let trialScore = optimizer trials[i]

        if trialScore < scores[i] then
            System.Array.Copy(trials[i], pop[i], argsize)
            scores[i] <- trialScore
    ) |> ignore

    if g % print = 0 then
        printfn $"generation %i{g}"
        printfn $"mean %f{scores |> Array.average}"
        printfn $"minimum %f{scores |> Array.min}"

let bestIndex = scores |> Array.indexed |> Array.minBy snd |> fst
let best = pop[bestIndex]
printfn $"best %A{best}"
printfn $"score %A{scores[bestIndex]}"
printfn $"execution time %i{sw.ElapsedMilliseconds} ms"

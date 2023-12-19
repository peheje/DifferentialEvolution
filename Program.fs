open Problems

let sw = System.Diagnostics.Stopwatch.StartNew()

let random = System.Random.Shared
let rand () = random.NextDouble()
let randomFloatRange max min = rand () * (max - min) + min
let randomInt max = random.Next(max)

type Agent = { xs: float array; score: float }

let sample agents =
    let i = randomInt (agents |> Array.length)
    agents[i].xs

let print = 1000
let optimizer = f1
let argsize = 1000
let min, max = -10.0, 10.0
let generations = 20000
let popsize = 200
let crossoverOdds () = randomFloatRange 0.1 1.0
let mutateOdds () = randomFloatRange 0.2 0.95
let clamp x = System.Math.Clamp(x, min, max)

let createAgent () =
    let xs = Array.init argsize (fun _ -> randomFloatRange min max)
    { xs = xs; score = optimizer xs }

let pool = Array.init popsize (fun _ -> createAgent ())

let mate crossover mutate pool agent =
    let x0, x1, x2 = sample pool, sample pool, sample pool

    let trial =
        Array.init argsize (fun j ->
            if rand () < crossover then
                x0[j] + (x1[j] - x2[j]) * mutate |> clamp
            else
                agent.xs[j])

    let trialScore = optimizer trial

    if trialScore < agent.score then
        { xs = trial; score = trialScore }
    else
        agent

let rec loop generation pool =
    if generation % print = 0 then
        let scores = pool |> Array.map _.score
        printfn "generation %i" generation
        printfn "mean %f" (scores |> Array.average)
        printfn "minimum %f" (scores |> Array.min)

    if generation = generations then
        pool
    else
        let mate' = mate (crossoverOdds ()) (mutateOdds ()) pool
        let next = pool |> Array.Parallel.map mate'
        loop (generation + 1) next

let best = loop 0 pool |> Array.minBy _.score

printfn "generation best %A" best
printfn "execution time %i ms" (sw.ElapsedMilliseconds)

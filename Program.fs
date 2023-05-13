open Problems
open System.Threading

let sw = System.Diagnostics.Stopwatch.StartNew ()

let random = System.Random.Shared
let maxRandoms = 1000
let mutable randoms = Array.init maxRandoms (fun _ -> 0.0)
let mutable randomsIndex = 0

let rand () =
    Interlocked.Increment(&randomsIndex) |> ignore
    randoms[randomsIndex % maxRandoms]

let randomFloatRange max min = random.NextDouble() * (max - min) + min
let randomInt max = random.Next(max)

let backgroundTask() =
    printfn "Background task started"
    while true do
        Interlocked.Exchange(&randoms[randomsIndex % maxRandoms], random.NextDouble()) |> ignore
        //randoms[randomsIndex % maxRandoms] <- random.NextDouble()
    printfn "Background task completed"

let thread = new Thread(backgroundTask)
thread.Start()
Thread.Sleep(1000)

type Agent = { xs: float array; score: float }

let sample agents =
    let i = randomInt(agents |> Array.length)
    agents[i].xs

let print = 500
let optimizer = rastrigin
let argsize = 100
let min, max = -10.0, 10.0
let generations = 10_000
let popsize = 200
let crossoverOdds () = randomFloatRange 0.1 1.0
let mutateOdds () = randomFloatRange 0.2 0.95
let clamp x = System.Math.Clamp(x, min, max)

let createAgent () =
    let xs = Array.init argsize (fun _ -> randomFloatRange min max)
    { xs = xs; score = optimizer xs }

let pool = Array.init popsize (fun _ -> createAgent ())

let mate pool agent =
    let crossover = crossoverOdds ()
    let mutate = mutateOdds ()
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
        let scores = pool |> Array.map (fun agent -> agent.score)
        printfn "generation %i" generation
        printfn "mean %f" (scores |> Array.average)
        printfn "minimum %f" (scores |> Array.min)

    if generation = generations then
        pool
    else
        let next = pool |> Array.Parallel.map (mate pool)
        loop (generation + 1) next

let best =
    loop 0 pool
    |> Array.minBy (fun agent -> agent.score)

printfn "generation best %A" best
printfn "execution time %i ms" (sw.ElapsedMilliseconds)

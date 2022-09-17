module Problems

let square x = x * x

// Solve systems of equations like, x + y = 6 and -3x + y = 2, but with twists like please make x > y
let systemOfEquations (xs: float array) =
    let x = xs[0]
    let y = xs[1]
    abs (x + y - 6.0) + abs (-3.0 * x + y - 2.0)

// There are 36 heads and 100 legs, how many horses and jockeys are there? 14 and 22
let horsesAndJockeys (xs: float array) =
    let horses = xs[0]
    let jockeys = xs[1]
    let legs = horses * 4.0 + jockeys * 2.0
    let heads = horses + jockeys
    abs (36.0 - heads) + abs (100.0 - legs)

// booth(1.0, 3.0) = 0
let booth (xs: float array) =
    square (xs[0] + 2.0 * xs[1] - 7.0)
    + square (2.0 * xs[0] + xs[1] - 5.0)

let rastrigin (xs: float array) =
    let pi = System.Math.PI
    let n = xs |> Array.length |> float
    let a = 10.0

    let sum =
        xs
        |> Array.sumBy (fun x -> (square x) - (a * cos (2.0 * pi * x)))

    a * n + sum

// f1(0..) = 0
let f1 (xs: float array) = xs |> Array.sumBy (fun x -> x * x)
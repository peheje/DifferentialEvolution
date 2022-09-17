# Differential evolution algorithm
Inspired by [this paper](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.344.546&rep=rep1&type=pdf). (Jakob Vesterstram, Rene Thomsen, 2004)

## Usage

Implement a function with signature float array -> float, the closer to 0 the function returns, the better the parameters fit.

## Examples

Using these parameters:

```fsharp
let optimizer = systemOfEquations
let generations = 1000
let argsize = 2
let popsize = 200
let min, max = -100.0, 100.0
```

You can solve systems of equations:

```fsharp
// Solve systems of equations like, x + y = 6 and -3x + y = 2
let systemOfEquations (xs: float array) =
    let x = xs[0]
    let y = xs[1]
    abs (x + y - 6.0) + abs (-3.0 * x + y - 2.0)
```

Outputs:

```
xs = [|1.0; 5.0|]
score = 0.0
```

Also solve nonlinear system of equations with constraints, like [this example from Matlabs documentation](https://www.mathworks.com/help/optim/ug/systems-of-equations-with-constraints-problem-based.html):

```fsharp
let nonlinearSystemOfEquationsWithConstraints (xs: float array) =
    let x = xs[0]
    let y = xs[1]
    let t1 = (x + 1.0)*(10.0 - x)*((1.0 + y*y)/(1.0 + y*y + y))
    let t2 = (y + 2.0)*(20.0 - y)*((1.0 + x*x)/(1.0 + x*x + x))
    let c1 = x*x + y*y
    let e = if c1 > 10.0 then c1 else 0.0

    abs(t1) + abs(t2) + e
```

Which outputs:

```
xs = [|-1.0; -2.0|]
score = 0.0
```

Another example:

```fsharp
// There are 36 heads and 100 legs, how many horses and jockeys are there?
let horsesAndJockeys (xs: float array) =
    let horses = xs[0]
    let jockeys = xs[1]
    let legs = horses * 4.0 + jockeys * 2.0
    let heads = horses + jockeys
    abs (36.0 - heads) + abs (100.0 - legs)
```

Outputs:

```
xs = [|14.0; 22.0|] 
score = 0.0
```

Not guaranteed to find any solution or any good solution.

## Parameters

Beside setting the argsize and min/max parameters, which depends on the optimizer function, you can tune the parameters ```generations```,  ```popsize```, ```crossoverOdds``` and ```mutateOdds```.
package main

import (
	"fmt"
	"math/rand/v2"
	"time"
)

const (
	params      = 1000
	popsize     = 200
	generations = 20000
	printEvery  = 1000
	cpus        = 16
)

func f1(x *[params]float64) float64 {
	var sum float64
	for _, xi := range x {
		sum += xi * xi
	}
	return sum
}

func clamp(x, min, max float64) float64 {
	if x < min {
		return min
	}
	if x > max {
		return max
	}
	return x
}

func sample(pop *[popsize][params]float64) [params]float64 {
	idx := rand.IntN(len(pop))
	return pop[idx]
}

func minIndex(scores *[popsize]float64) int {
	minIdx := 0
	for i, score := range scores {
		if score < scores[minIdx] {
			minIdx = i
		}
	}
	return minIdx
}

func mean(scores *[popsize]float64) float64 {
	sum := 0.0
	for _, score := range scores {
		sum += score
	}
	return sum / float64(len(scores))
}

func randRange(min float64, max float64) float64 {
	return rand.Float64() * (max - min)
}

func main() {
	// f, err := os.Create("cpu.pprof")
	// if err != nil {
	// 	panic(err)
	// }
	// pprof.StartCPUProfile(f)
	// defer pprof.StopCPUProfile()

	var (
		optimizer      = f1
		bounds         = [2]float64{-10.0, 10.0}
		mutateRange    = [2]float64{0.2, 0.95}
		crossoverRange = [2]float64{0.1, 1.0}
	)

	start := time.Now()

	// Initialize population
	var pop [popsize][params]float64
	for i := range pop {
		for j := range pop[i] {
			pop[i][j] = bounds[0] + randRange(bounds[0], bounds[1])
		}
	}

	// Evaluate initial population
	var scores [popsize]float64
	for i, p := range pop {
		scores[i] = optimizer(&p)
	}

	// Optimization loop
	semaphore := make(chan struct{}, cpus)
	for g := 0; g < generations; g++ {
		crossover := crossoverRange[0] + randRange(crossoverRange[0], crossoverRange[1])
		mutate := mutateRange[0] + randRange(mutateRange[0], mutateRange[1])

		for i := range pop {
			semaphore <- struct{}{}
			go func() {
				defer func() { <-semaphore }()
				var trial [params]float64
				var x0 [params]float64
				var x1 [params]float64
				var x2 [params]float64
				x0 = sample(&pop)
				x1 = sample(&pop)
				x2 = sample(&pop)
				xt := pop[i]

				for j := range xt {
					if rand.Float64() < crossover {
						trial[j] = clamp(x0[j]+(x1[j]-x2[j])*mutate, bounds[0], bounds[1])
					} else {
						trial[j] = xt[j]
					}
				}

				scoreTrial := optimizer(&trial)
				if scoreTrial < scores[i] {
					pop[i] = trial
					scores[i] = scoreTrial
				}
			}()
		}

		if g%printEvery == 0 || g == generations-1 {
			fmt.Printf("Generation: %d\n", g)
			fmt.Printf("Mean score: %f\n", mean(&scores))
			fmt.Printf("Best score: %f\n", scores[minIndex(&scores)])
		}
	}

	best := pop[minIndex(&scores)]
	fmt.Println("Best solution:", best)
	fmt.Println("Score:", scores[minIndex(&scores)])
	fmt.Printf("Time taken: %s\n", time.Since(start))

	// pprof.StopCPUProfile()
	// f, err = os.Create("mem.pprof")
	// if err != nil {
	// 	panic(err)
	// }
	// pprof.WriteHeapProfile(f)
	// f.Close()
}

package main

import (
	"fmt"
	"math/rand/v2"
	"time"
)

const (
	params       = 1000
	popsize      = 2000
	generations  = 50000
	printEvery   = 1000
	cpus         = 32
	boundsMin    = -10.0
	boundsMax    = 10.0
	crossoverMin = 0.1
	crossoverMax = 1.0
	mutateMin    = 0.2
	mutateMax    = 0.95
)

type Agent [params]float64

func f1(xs *Agent) float64 {
	var sum float64
	for _, x := range xs {
		sum += x * x
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

func sample(pop *[popsize]Agent) *Agent {
	idx := rand.IntN(len(pop))
	return &pop[idx]
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

func randRange(min, max float64) float64 {
	return rand.Float64()*(max-min) + min
}

func main() {
	// f, err := os.Create("default.pgo")
	// if err != nil {
	// 	panic(err)
	// }
	// pprof.StartCPUProfile(f)
	// defer pprof.StopCPUProfile()

	optimizer := f1
	start := time.Now()

	var scores [popsize]float64
	var pop [popsize]Agent
	for i := range pop {
		for j := range pop[i] {
			pop[i][j] = randRange(boundsMin, boundsMax)
		}
		scores[i] = optimizer(&pop[i])
	}

	trials := [popsize]Agent{}
	semaphore := make(chan struct{}, cpus)
	for g := 0; g < generations; g++ {
		crossover := randRange(crossoverMin, crossoverMax)
		mutate := randRange(mutateMin, mutateMax)

		for i := range pop {
			semaphore <- struct{}{}
			go func() {
				defer func() { <-semaphore }()
				x0 := sample(&pop)
				x1 := sample(&pop)
				x2 := sample(&pop)
				xt := pop[i]

				for j := range xt {
					if rand.Float64() < crossover {
						trials[i][j] = clamp(x0[j]+(x1[j]-x2[j])*mutate, boundsMin, boundsMax)
					} else {
						trials[i][j] = xt[j]
					}
				}

				scoreTrial := optimizer(&trials[i])
				if scoreTrial < scores[i] {
					pop[i] = trials[i]
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

	bestIndex := minIndex(&scores)
	best := pop[bestIndex]
	fmt.Println("Best solution:", best)
	fmt.Println("Score:", scores[bestIndex])
	fmt.Printf("Time taken: %s\n", time.Since(start))

	// pprof.StopCPUProfile()
	// f, err = os.Create("mem.pprof")
	// if err != nil {
	// 	panic(err)
	// }
	// pprof.WriteHeapProfile(f)
	// f.Close()
}

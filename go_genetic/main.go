package main

import (
	"fmt"
	"math/rand/v2"
	"time"
)

const (
	params       = 1000
	popsize      = 200
	generations  = 20000
	printEvery   = 1000
	cpus         = 32
	boundsMin    = -10.0
	boundsMax    = 10.0
	crossoverMin = 0.1
	crossoverMax = 1.0
	mutateMin    = 0.2
	mutateMax    = 0.95
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

func sample(pop *[popsize][params]float64) *[params]float64 {
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

func randRange(min float64, max float64) float64 {
	return rand.Float64() * (max - min)
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
	var pop [popsize][params]float64
	for i := range pop {
		for j := range pop[i] {
			pop[i][j] = randRange(boundsMin, boundsMax)
		}
		scores[i] = optimizer(&pop[i])
	}

	semaphore := make(chan struct{}, cpus)
	for g := 0; g < generations; g++ {
		crossover := randRange(crossoverMin, crossoverMax)
		mutate := randRange(mutateMin, mutateMax)

		for i := range pop {
			semaphore <- struct{}{}
			go func() {
				defer func() { <-semaphore }()
				var trial [params]float64
				x0 := sample(&pop)
				x1 := sample(&pop)
				x2 := sample(&pop)
				xt := pop[i]

				for j := range xt {
					if rand.Float64() < crossover {
						trial[j] = clamp(x0[j]+(x1[j]-x2[j])*mutate, boundsMin, boundsMax)
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

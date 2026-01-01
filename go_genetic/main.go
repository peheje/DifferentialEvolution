package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"os"
	"sort"
	"strconv"
	"sync"
	"time"
)

const (
	// DE Parameters
	popsize      = 500
	generations  = 10000
	printEvery   = 500
	cpus         = 32
	boundsMin    = 0.0
	boundsMax    = 1.0
	crossoverMin = 0.1
	crossoverMax = 0.9
	mutateMin    = 0.2
	mutateMax    = 0.8

	// Team Parameters
	teamSize     = 6
	penaltyScore = 10000.0 // Penalty for every un-resisted type
)

// Pokemon struct adapted for CSV data
type Pokemon struct {
	ID        int
	Name      string
	BaseStat  float64
	ResistMap [18]float64 // Stores damage multiplier for 18 types
}

// Global storage for loaded Pokemon
var pokedex []Pokemon

// List of type columns to look for in the CSV (Order matters for consistency)
var typeColumns = []string{
	"against_bug", "against_dark", "against_dragon", "against_electric",
	"against_fairy", "against_fight", "against_fire", "against_flying",
	"against_ghost", "against_grass", "against_ground", "against_ice",
	"against_normal", "against_poison", "against_psychic", "against_rock",
	"against_steel", "against_water",
}

// Agent is now a slice (dynamic length)
type Agent []float64

// Helper for sorting
type PkmnPriority struct {
	Index int
	Value float64
}

// loadCSV reads the specific columns from your file
func loadCSV(filename string) {
	file, err := os.Open(filename)
	if err != nil {
		log.Fatalf("Could not open csv: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		log.Fatalf("Could not read csv: %v", err)
	}

	if len(records) < 2 {
		log.Fatal("CSV appears empty or missing header")
	}

	header := records[0]
	
	// Create a map of "Column Name" -> Index
	colMap := make(map[string]int)
	for i, h := range header {
		colMap[h] = i
	}

	// Validate required columns exist
	required := []string{"name", "base_total"}
	required = append(required, typeColumns...)
	for _, req := range required {
		if _, ok := colMap[req]; !ok {
			log.Fatalf("Missing required column in CSV: %s", req)
		}
	}

	// Parse rows
	for i, row := range records[1:] {
		p := Pokemon{
			ID:   i,
			Name: row[colMap["name"]],
		}

		// Parse Base Total
		bst, _ := strconv.ParseFloat(row[colMap["base_total"]], 64)
		p.BaseStat = bst

		// Parse Type Matchups
		// Note: The CSV contains damage multipliers (e.g., 0.5, 2.0).
		for tIdx, colName := range typeColumns {
			valStr := row[colMap[colName]]
			val, _ := strconv.ParseFloat(valStr, 64)
			p.ResistMap[tIdx] = val
		}

		pokedex = append(pokedex, p)
	}

	fmt.Printf("Successfully loaded %d Pokemon from CSV.\n", len(pokedex))
}

// optimizer function
func optimizer(xs *Agent) float64 {
	candidates := make([]PkmnPriority, len(*xs))
	for i, val := range *xs {
		candidates[i] = PkmnPriority{Index: i, Value: val}
	}

	// Sort descending
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Value > candidates[j].Value
	})

	// Pick top 6
	teamIndices := make([]int, teamSize)
	totalStats := 0.0
	for i := 0; i < teamSize; i++ {
		idx := candidates[i].Index
		teamIndices[i] = idx
		totalStats += pokedex[idx].BaseStat
	}

	// Check Coverage Constraints
	uncoveredTypes := 0
	for t := 0; t < 18; t++ {
		isResisted := false
		for _, pkmnIdx := range teamIndices {
			// In Pokemon, resistance is damage <= 0.5
			if pokedex[pkmnIdx].ResistMap[t] <= 0.5 {
				isResisted = true
				break
			}
		}
		if !isResisted {
			uncoveredTypes++
		}
	}

	// Minimize Score
	return (float64(uncoveredTypes) * penaltyScore) - totalStats
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

func sample(pop *[]Agent) *Agent {
	idx := rand.IntN(len(*pop))
	return &(*pop)[idx]
}

func minIndex(scores *[]float64) int {
	minIdx := 0
	for i, score := range *scores {
		if score < (*scores)[minIdx] {
			minIdx = i
		}
	}
	return minIdx
}

func mean(scores *[]float64) float64 {
	sum := 0.0
	for _, score := range *scores {
		sum += score
	}
	return sum / float64(len(*scores))
}

func randRange(min, max float64) float64 {
	return rand.Float64()*(max-min) + min
}

func main() {
	// 1. Load Data
	loadCSV("/home/peter/repos/DifferentialEvolution/go_genetic/pokemon_data.csv")

	if len(pokedex) < teamSize {
		log.Fatal("Not enough Pokemon in CSV to form a team")
	}

	start := time.Now()
	
	// Dynamic allocation based on CSV size
	numParams := len(pokedex)
	
	scores := make([]float64, popsize)
	pop := make([]Agent, popsize)

	// Initialize Population
	for i := range pop {
		pop[i] = make(Agent, numParams)
		for j := range pop[i] {
			pop[i][j] = randRange(boundsMin, boundsMax)
		}
		scores[i] = optimizer(&pop[i])
	}

	semaphore := make(chan struct{}, cpus)
	var wg sync.WaitGroup

	fmt.Println("Starting Evolution...")

	for g := 0; g < generations; g++ {
		crossover := randRange(crossoverMin, crossoverMax)
		mutate := randRange(mutateMin, mutateMax)

		// Create shallow copy of population for thread safety
		// Note: We copy the slice headers, but the underlying arrays 
		// are not modified during the read phase of the loop, so this is safe enough for DE
		parentPop := make([]Agent, len(pop))
		copy(parentPop, pop)

		wg.Add(len(pop))
		for i := range pop {
			semaphore <- struct{}{}
			go func(i int) {
				defer func() {
					<-semaphore
					wg.Done()
				}()

				x0 := sample(&parentPop)
				x1 := sample(&parentPop)
				x2 := sample(&parentPop)
				xt := parentPop[i]

				// Create Trial Agent
				trial := make(Agent, numParams)
				R := rand.IntN(numParams)

				for j := range xt {
					if rand.Float64() < crossover || j == R {
						trial[j] = clamp((*x0)[j]+((*x1)[j]-(*x2)[j])*mutate, boundsMin, boundsMax)
					} else {
						trial[j] = xt[j]
					}
				}

				scoreTrial := optimizer(&trial)

				if scoreTrial < scores[i] {
					pop[i] = trial
					scores[i] = scoreTrial
				}
			}(i)
		}
		wg.Wait()

		if g%printEvery == 0 || g == generations-1 {
			bestScore := scores[minIndex(&scores)]
			
			// Reverse engineer stats for display
			missingTypes := 0
			currentStats := -bestScore
			if bestScore > 0 {
				missingTypes = int(math.Floor(bestScore / penaltyScore))
				currentStats = -(bestScore - (float64(missingTypes) * penaltyScore))
			}

			fmt.Printf("Gen: %d | Mean: %.2f | Best: %.2f | Missing: %d | Stats: %.0f\n",
				g, mean(&scores), bestScore, missingTypes, currentStats)
		}
	}

	bestIndex := minIndex(&scores)
	bestAgent := pop[bestIndex]
	
	fmt.Printf("\n--- Optimization Complete ---\n")
	fmt.Printf("Time taken: %s\n", time.Since(start))

	// Decode best team
	candidates := make([]PkmnPriority, numParams)
	for i, val := range bestAgent {
		candidates[i] = PkmnPriority{Index: i, Value: val}
	}
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Value > candidates[j].Value
	})

	fmt.Println("\n--- Best Team Found ---")
	totalStats := 0.0
	
	// Track which types are actually covered
	coveredTypes := make(map[string]bool)

	for i := 0; i < teamSize; i++ {
		idx := candidates[i].Index
		pkmn := pokedex[idx]
		fmt.Printf("#%d - %s (BST: %.0f)\n", i+1, pkmn.Name, pkmn.BaseStat)
		totalStats += pkmn.BaseStat

		for tIdx, val := range pkmn.ResistMap {
			if val <= 0.5 {
				coveredTypes[typeColumns[tIdx]] = true
			}
		}
	}
	fmt.Printf("\nTotal Team Stats: %.0f\n", totalStats)
	
	fmt.Println("--- Coverage Report ---")
	if len(coveredTypes) == 18 {
		fmt.Println("All 18 Types Resisted! Perfect Coverage.")
	} else {
		fmt.Printf("Covered: %d/18 types.\n", len(coveredTypes))
		fmt.Print("Missing: ")
		for _, t := range typeColumns {
			if !coveredTypes[t] {
				fmt.Printf("[%s] ", t)
			}
		}
		fmt.Println()
	}
}
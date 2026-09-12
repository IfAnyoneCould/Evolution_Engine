package main

import (
	"Evolution_Engine/internal/simulation"
	"fmt"
)

func main() {
	sim, err := simulation.NewSimulation(0.99, 500, 100, "cmd/engine/main_test.json")
	if err != nil {
		fmt.Println(err)
		return
	}
	sim.Run()
	fmt.Printf("Best weights: %v", sim.GetBestWeights())
}

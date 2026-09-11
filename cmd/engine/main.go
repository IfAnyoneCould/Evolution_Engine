package main

import (
	"Evolution_Engine/internal/simulation"
	"fmt"
)

func main() {
	sim, err := simulation.NewSimulation(0.99, 1000, 10, "cmd/engine/main_test.json")
	if err != nil {
		fmt.Println(err)
	}
	sim.Run()
	fmt.Printf("Best weights: %v", sim.GetBestWeights())
}

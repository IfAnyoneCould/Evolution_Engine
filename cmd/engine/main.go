package main

import (
	"Evolution_Engine/internal/simulation"
	"fmt"
	"os"
)

func main() {
	wd, _ := os.Getwd()
	fmt.Println("working directory:", wd)
	sim, err := simulation.NewSimulation(0.9, 1000, 10, "cmd/engine/main_test.json")
	if err != nil {
		fmt.Println(err)
	}
	sim.Run()
}

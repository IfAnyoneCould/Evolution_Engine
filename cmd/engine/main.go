package main

import (
	"Evolution_Engine/internal/config"
	"Evolution_Engine/internal/simulation"
	"fmt"
)

// TODO make a md doc or something will all of the config options, so understanding the config is simple and fast. This is for development, the main readme will be done later
// TODO change all errors and communication to be compatible with a frontend app, not just spitting information out into the terminal

func main() {
	cfg, err := config.Load("cmd/engine/main_test.json")
	if err != nil {
		fmt.Println(err)
		return
	}
	sim, err := simulation.NewSimulation(cfg) // TODO add all of these to the config
	if err != nil {
		fmt.Println(err)
		return
	}
	sim.Run()
	fmt.Printf("Best weights: %v", sim.GetBestWeights()) // TODO save the best weights to an output file. At first should just overwrite current info, then add a history functionality including config settings, run time, cycles, etc
}

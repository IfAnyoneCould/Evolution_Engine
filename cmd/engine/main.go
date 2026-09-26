package main

import (
	"Evolution_Engine/internal/config"
	"Evolution_Engine/internal/simulation"
	"fmt"
	"os"
)

// TODO make a md doc or something will all of the config options, so understanding the config is simple and fast. This is for development, the main readme will be done later
// TODO change all errors and communication to be compatible with a frontend app, not just spitting information out into the terminal

func main() {

	if len(os.Args) < 2 {
		fmt.Println("error: include path to config in command args")
		return
	}

	cfg, err := config.Load(os.Args[1])
	if err != nil {
		fmt.Println(err)
		return
	}
	sim, err := simulation.NewSimulation(cfg)
	if err != nil {
		fmt.Println(err)
		return
	}
	if err = sim.Run(); err != nil {
		fmt.Println(err)
		return
	}
	if err = sim.WriteBestWeights(); err != nil {
		fmt.Println(err)
		return
	}

	fmt.Printf("Best weights: %v", sim.GetBestWeights()) // TODO save the best weights to an output file. At first should just overwrite current info, then add a history functionality including config settings, run time, cycles, etc
}

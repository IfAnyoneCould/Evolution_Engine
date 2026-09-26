package main

import (
	"Evolution_Engine/internal/config"
	"Evolution_Engine/internal/simulation"
	"fmt"
	"os"
)

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

	_, best := sim.GetBestWeights()
	fmt.Printf("Best weights: %v", best)
}

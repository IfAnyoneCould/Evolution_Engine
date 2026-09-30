package main

import (
	"Evolution_Engine/internal/api"
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

	var em api.Emitter
	if len(os.Args) == 2 {
		em = api.NewPrinter()
	} else {
		switch os.Args[2] {
		case "print":
			em = api.NewPrinter()
		//TODO add the sender types here, not sure exactly what io.Writer
		default:
			em = api.NewPrinter()
			e := api.Error{Message: "error: emitter type not recognized"}
			_ = em.Send(e)
		}
	}

	sim, err := simulation.NewSimulation(cfg, em)
	if err != nil {
		e := api.Error{Message: err.Error()}
		_ = em.Send(e)
		return
	}
	if err = sim.Run(); err != nil {
		e := api.Error{Message: err.Error()}
		_ = em.Send(e)
		return
	}
	if err = sim.WriteBestWeights(); err != nil {
		e := api.Error{Message: err.Error()}
		_ = em.Send(e)
		return
	}
}

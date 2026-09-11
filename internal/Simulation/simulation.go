package Simulation

import (
	"Evolution_Engine/internal/population"
	"context"
	"fmt"
)

type Simulation struct {
	pop                    *population.Population
	cycleMax, currentCycle int
	targetFitness          float64
}

func newSimulation(fitness float64, max int, count int, path string) (*Simulation, error) {
	pop, err := population.NewPopulation(count, path)
	if err != nil {
		return &Simulation{}, err
	}
	return &Simulation{pop, max, 0, fitness}, nil
}

func (s *Simulation) Run(workers int, log bool) {
	for range s.cycleMax {
		ctx := context.Background()
		err := s.pop.RunBatch(ctx, workers)
		if err != nil {
			fmt.Println(err)
			return
		}
		if s.pop.TopFitness >= s.targetFitness {
			fmt.Printf("Simulation reached or exceed fitness target %f in %d cycles", s.pop.TopFitness, s.currentCycle)
			return
		}
		err = s.pop.NewGen(&population.NewGenConfig{Pressure: 0.25, Elite: 2})
		if err != nil {
			fmt.Println(err)
		}
		fmt.Printf("Current cycle: %d -- Max fitness: %f -- Goal Fitness: %f", s.currentCycle, s.pop.TopFitness, s.targetFitness)
		s.currentCycle++
	}
	fmt.Printf("Reached a fitness of %f in %d cycles. Target fitness: %f\n", s.pop.TopFitness, s.currentCycle, s.targetFitness)
}

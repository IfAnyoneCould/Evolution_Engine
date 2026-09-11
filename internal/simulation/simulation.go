package simulation

import (
	"Evolution_Engine/internal/population"
	"context"
	"fmt"
	"time"
)

type Simulation struct {
	Pop                    *population.Population
	cycleMax, currentCycle int
	targetFitness          float64
}

func NewSimulation(fitness float64, max int, count int, path string) (*Simulation, error) {
	pop, err := population.NewPopulation(count, path, 10)
	if err != nil {
		return &Simulation{}, err
	}
	return &Simulation{pop, max, 0, fitness}, nil
}

func (s *Simulation) Run() {
	for range s.cycleMax {
		ctx := context.Background()
		start := time.Now()
		err := s.Pop.RunBatch(ctx)
		fmt.Printf("batch took %v\n", time.Since(start))
		if err != nil {
			fmt.Println(err)
			_ = s.Pop.CloseSims()
			return
		}
		if s.Pop.TopFitness >= s.targetFitness {
			_ = s.Pop.CloseSims()
			fmt.Printf("simulation reached or exceed fitness target %f in %d cycles", s.targetFitness, s.currentCycle)
			return
		}
		err = s.Pop.NewGen(&population.NewGenConfig{Pressure: 0.25, Elite: 2})
		if err != nil {
			fmt.Println(err)
			_ = s.Pop.CloseSims()
			return
		}
		fmt.Printf("Current cycle: %d -- Max fitness: %f -- Goal Fitness: %f\n", s.currentCycle, s.Pop.TopFitness, s.targetFitness)
		s.currentCycle++
	}
	fmt.Printf("Reached a fitness of %f in %d cycles. Target fitness: %f\n", s.Pop.TopFitness, s.currentCycle, s.targetFitness)
	_ = s.Pop.CloseSims()
}

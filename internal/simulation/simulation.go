package simulation

import (
	"Evolution_Engine/internal/genome"
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
	total := time.Now()
	lastTop := float64(0)
	for range s.cycleMax {

		ctx := context.Background()
		start := time.Now()
		err := s.Pop.RunBatch(ctx)
		fmt.Printf("batch took %v\n", time.Since(start))
		if s.currentCycle == 1 {
			s.Pop.StartFitness = s.Pop.TopFitness
			fmt.Println(s.Pop.StartFitness)
		}
		if err != nil {
			fmt.Println(err)
			fmt.Printf("Simulation took %v\n", time.Since(total))
			_ = s.Pop.CloseSims()
			return
		}
		if s.Pop.TopFitness >= s.targetFitness {
			fmt.Printf("simulation reached or exceed fitness target %f in %d cycles\n", s.targetFitness, s.currentCycle)
			fmt.Printf("Simulation took %v\n", time.Since(total))
			_ = s.Pop.CloseSims()
			return
		}
		err = s.Pop.NewGen(&population.NewGenConfig{Pressure: 0.45, Elite: 1}, s.targetFitness)
		if err != nil {
			fmt.Println(err)
			fmt.Printf("Simulation took %v\n", time.Since(total))
			_ = s.Pop.CloseSims()
			return
		}

		fmt.Printf("Current cycle: %d -- Max fitness: %f -- Goal Fitness: %f\n", s.currentCycle, s.Pop.TopFitness, s.targetFitness)
		s.currentCycle++

		if lastTop == s.Pop.TopFitness {
			fmt.Printf("Simulation stalled at %f fitness on cycle %d\n", s.Pop.TopFitness, s.currentCycle)
			_ = s.Pop.CloseSims()
			return
		}
		if s.currentCycle%10 == 0 {
			//	lastTop = s.Pop.TopFitness
		}
	}
	fmt.Printf("Reached a fitness of %f in %d cycles. Target fitness: %f\n", s.Pop.TopFitness, s.currentCycle, s.targetFitness)
	fmt.Printf("Simulation took %v\n", time.Since(total))
	_ = s.Pop.CloseSims()
}

func (s *Simulation) GetBestGenome() *genome.Genome {
	s.Pop.Rank()
	return s.Pop.Agents[0].Gene
}

func (s *Simulation) GetBestWeights() []float64 {
	s.Pop.Rank()
	return s.Pop.Agents[0].Gene.GetWeights()
}

func (s *Simulation) GetBestAgent() population.Agent {
	s.Pop.Rank()
	return s.Pop.Agents[0]
}

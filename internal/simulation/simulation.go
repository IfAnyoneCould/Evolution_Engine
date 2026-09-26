package simulation

import (
	"Evolution_Engine/internal/config"
	"Evolution_Engine/internal/genome"
	"Evolution_Engine/internal/population"
	"context"
	"fmt"
	"time"
)

type Simulation struct {
	Pop                    *population.Population
	cycleMax, currentCycle uint
	targetFitness          float64
	newGenConfig           config.Selection
	stagnation             config.StagnationDetect
}

func NewSimulation(cfg config.JsonParams) (*Simulation, error) {
	pop, err := population.NewPopulation(cfg)
	if err != nil {
		return &Simulation{}, err
	}
	return &Simulation{pop, cfg.RunSettings.MaxCycles, 0, cfg.RunSettings.TargetFitness, cfg.Selection, cfg.StagnationDetect}, nil
}

func (s *Simulation) Run() {
	total := time.Now()
	lastTop := float64(0)
	cycleSinceImprove := 0
	for range s.cycleMax {

		ctx := context.Background()
		start := time.Now()
		err := s.Pop.RunBatch(ctx)
		fmt.Printf("batch took %v\n", time.Since(start))
		if s.currentCycle == 0 {
			s.Pop.StartFitness = s.Pop.TopFitness
			lastTop = s.Pop.TopFitness
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
		err = s.Pop.NewGen(s.newGenConfig, s.targetFitness)
		if err != nil {
			fmt.Println(err)
			fmt.Printf("Simulation took %v\n", time.Since(total))
			_ = s.Pop.CloseSims()
			return
		}

		fmt.Printf("Current cycle: %d -- Max fitness: %f -- Goal Fitness: %f\n", s.currentCycle, s.Pop.TopFitness, s.targetFitness)
		s.currentCycle++

		if s.Pop.TopFitness-lastTop < s.stagnation.Epsilon {
			cycleSinceImprove++
		} else {
			cycleSinceImprove = 0
			lastTop = s.Pop.TopFitness
		}

		if cycleSinceImprove >= int(s.stagnation.Patience) {
			fmt.Printf("simulation exited, didn't see a fitness improvement greater than %f in %d cycles\n", s.stagnation.Epsilon, s.stagnation.Patience)
			_ = s.Pop.CloseSims()
			return
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

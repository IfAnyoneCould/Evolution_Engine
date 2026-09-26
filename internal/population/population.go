package population

import (
	"Evolution_Engine/internal/config"
	"Evolution_Engine/internal/genome"
	"math"
	"math/rand"
)

type Agent struct {
	Gene      *genome.Genome
	Fitness   float64
	Evaluated bool
}

func NewAgent(bounds [][2]float64, r *rand.Rand) (Agent, error) {
	g := genome.NewGenome(len(bounds))
	err := g.SetBounds(bounds...)
	if err != nil {
		return Agent{}, err
	}
	g.Init(r)
	return Agent{g, math.Inf(-1), false}, nil
}

func (a *Agent) Evaluate(s *config.SimProcess) error {
	fitness, err := s.Eval(a.Gene.GetWeights())
	if err != nil {
		return err
	}
	a.Fitness = fitness
	a.Evaluated = true
	return nil
}

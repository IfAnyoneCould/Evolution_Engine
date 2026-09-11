package population

import (
	"Evolution_Engine/internal/config"
	"Evolution_Engine/internal/genome"
)

type Agent struct {
	Gene      *genome.Genome
	Fitness   float64
	Evaluated bool
}

func NewAgent(bounds [][2]float64) (Agent, error) {
	g := genome.NewGenome(len(bounds))
	err := g.SetBounds(bounds...)
	if err != nil {
		return Agent{}, nil
	}
	g.Init()
	return Agent{g, -1, false}, nil
}

func (a Agent) Evaluate(p *config.ProgramBin) error {
	fitness, err := p.Run(a.Gene)
	if err != nil {
		return err
	}
	a.Fitness = fitness
	a.Evaluated = true
	return nil
}

type Population struct {
	Agents []Agent
	Prog   *config.ProgramBin
	Nudge  float64
}

func NewPopulation(count int, path string) (*Population, error) {
	bounds, prog, nudge, err := config.ParseInputFile(path)

	if err != nil {
		return &Population{}, err
	}

	var agents []Agent
	for range count {
		a, err := NewAgent(bounds)
		if err != nil {
			return &Population{}, err
		}
		agents = append(agents, a)
	}

	return &Population{agents, prog, nudge}, nil
}

package main

type Agent struct {
	Gene      *Genome
	Fitness   float64
	Evaluated bool
}

func NewAgent(bounds [][2]float64) (Agent, error) {
	g := NewGenome(len(bounds))
	err := g.SetBounds(bounds...)
	if err != nil {
		return Agent{}, nil
	}
	g.Init()
	return Agent{g, -1, false}, nil
}

func (a Agent) Evaluate(p *ProgramBin) error {
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
	Prog   *ProgramBin
	Nudge  float64
}

func NewPopulation(count int, path string) (*Population, error) {
	bounds, prog, nudge, err := ParseInputFile(path)

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

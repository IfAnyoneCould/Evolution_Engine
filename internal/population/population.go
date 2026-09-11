package population

import (
	"Evolution_Engine/internal/config"
	"Evolution_Engine/internal/genome"
	"cmp"
	"context"
	"errors"
	"fmt"
	"math/rand"
	"slices"
	"sync"
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
		return Agent{}, err
	}
	g.Init()
	return Agent{g, -1, false}, nil
}

func (a *Agent) Evaluate(p *config.ProgramBin) error {
	fitness, err := p.Run(a.Gene)
	if err != nil {
		return err
	}
	a.Fitness = fitness
	a.Evaluated = true
	return nil
}

type Population struct {
	Agents     []Agent
	Prog       *config.ProgramBin
	TopFitness float64
	Nudge      float64
	//GenConfig NewGenConfig
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

	return &Population{agents, prog, -1, nudge}, nil
}

func (p *Population) RunBatch(ctx context.Context, workers int) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	agents := make(chan *Agent, len(p.Agents))
	for i := range p.Agents {
		agents <- &p.Agents[i]
	}
	close(agents)

	errs := make(chan error)

	wg := sync.WaitGroup{}
	for range workers {
		wg.Go(func() {
			for a := range agents {
				err := a.Evaluate(p.Prog)
				if err != nil {
					select {
					case errs <- err:
					case <-ctx.Done():
					}
					cancel()
					return
				}
				if ctx.Err() != nil {
					return
				}
			}
		})
	}

	go func() {
		wg.Wait()
		close(errs)
	}()

	var errList []error
	for e := range errs {
		errList = append(errList, e)
	}

	return errors.Join(errList...)
}

func (p *Population) Rank() {
	slices.SortFunc(p.Agents, func(a, b Agent) int {
		return cmp.Compare(b.Fitness, a.Fitness)
	})
	p.TopFitness = p.Agents[0].Fitness
}

type NewGenConfig struct {
	Elite    int     // how many of the top should be copied over
	Pressure float64 // 0 to 1, how much of the current generation is included
}

func (p *Population) NewGen(config *NewGenConfig) error {
	p.Rank()

	n := len(p.Agents)
	cutoff := int(float64(n) * config.Pressure)
	if cutoff < 1 {
		return fmt.Errorf("pressure %f too low, so survivors from population of %d", config.Pressure, n)
	}

	newAgents := make([]Agent, 0, n)

	for i := 0; i < config.Elite && i < n; i++ {
		newAgents = append(newAgents, p.Agents[i])
	}

	for len(newAgents) < n {
		parent := p.Agents[rand.Intn(cutoff)]
		childGene := parent.Gene.Clone()
		childGene.Nudge(p.Nudge)
		newAgents = append(newAgents, Agent{Gene: childGene, Evaluated: false})
	}
	p.Agents = newAgents
	return nil
}

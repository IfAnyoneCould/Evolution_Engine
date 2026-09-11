package population

import (
	"Evolution_Engine/internal/config"
	"Evolution_Engine/internal/genome"
	"cmp"
	"context"
	"errors"
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
		wg.Add(1)
		go func() {
			defer wg.Done()
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
		}()
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
		return cmp.Compare(a.Fitness, b.Fitness)
	})
}

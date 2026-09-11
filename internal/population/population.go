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

func (a *Agent) Evaluate(s *config.SimProcess) error {
	fitness, err := s.Eval(a.Gene)
	if err != nil {
		return err
	}
	a.Fitness = fitness
	a.Evaluated = true
	return nil
}

type Population struct {
	Agents      []Agent
	ProcList    []*config.SimProcess
	WorkerCount int
	TopFitness  float64
	Nudge       float64
	//GenConfig NewGenConfig
}

func NewPopulation(gCount int, path string, wCount int) (*Population, error) {
	bounds, prog, nudge, err := config.ParseInputFile(path)

	if err != nil {
		return &Population{}, err
	}

	var agents []Agent
	for range gCount {
		a, err := NewAgent(bounds)
		if err != nil {
			return &Population{}, err
		}
		agents = append(agents, a)
	}

	var procList []*config.SimProcess
	for range wCount {
		sim, err := config.NewSimProcess(prog.Path, prog.Args)
		if err != nil {
			return nil, err
		}
		procList = append(procList, sim)
	}

	return &Population{agents, procList, wCount, -1, nudge}, nil
}

func (p *Population) CloseSims() error {
	for i := range p.ProcList {
		if err := p.ProcList[i].Close(); err != nil {
			return err
		}
	}
	return nil
}

func (p *Population) RunBatch(ctx context.Context) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	agents := make(chan *Agent, len(p.Agents))
	for i := range p.Agents {
		agents <- &p.Agents[i]
	}
	close(agents)

	errs := make(chan error, len(p.Agents))

	wg := sync.WaitGroup{}
	for i := range p.WorkerCount {
		wg.Go(func() {
			for a := range agents {
				err := a.Evaluate(p.ProcList[i])
				if err != nil {
					select {
					case errs <- err:
					case <-ctx.Done():
						return
					}
					continue
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

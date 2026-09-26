package population

import (
	"Evolution_Engine/internal/config"
	"Evolution_Engine/internal/genome"
	"Evolution_Engine/internal/nudge"
	"cmp"
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"strings"
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
	return Agent{g, math.Inf(-1), false}, nil
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
	Agents       []Agent
	ProcList     []*config.SimProcess
	WorkerCount  uint
	TopFitness   float64
	BaseFraction float64
	Fraction     float64
	NudgeFunc    nudge.Function
	StartFitness float64
}

func NewPopulation(params config.JsonParams) (*Population, error) {

	var agents []Agent
	for range params.RunSettings.PopulationSize {
		a, err := NewAgent(params.Bounds)
		if err != nil {
			return &Population{}, err
		}
		agents = append(agents, a)
	}

	var procList []*config.SimProcess
	for range params.Workers {
		sim, err := config.NewSimProcess(params.Prog.Path, params.Prog.Args, params.SimSettings.Timeout)
		if err != nil {
			return nil, err
		}
		procList = append(procList, sim)
	}

	var nudgeFunc nudge.Function
	switch strings.ToLower(params.NudgeFunc.Type) {
	case "constant":
		nudgeFunc = nudge.NewConstantFunction(params.NudgeFunc.Param[0])
	case "linear":
		nudgeFunc = nudge.NewLinearFunction(params.NudgeFunc.Param[0])
	case "quadratic":
		nudgeFunc = nudge.NewQuadraticFunction(params.NudgeFunc.Param[0])
	}

	return &Population{agents, procList, params.Workers, -1, params.Fraction, params.MinNudge, nudgeFunc, 0}, nil
}

func (p *Population) CalcNudge(fit float64, goal float64) float64 {
	progress := nudge.Clamp((fit-p.StartFitness)/(goal-p.StartFitness), 0, 1)
	if goal-p.StartFitness == 0 {
		progress = 1
	}
	raw := p.BaseFraction * nudge.Clamp(p.NudgeFunc.Get(progress), 0, 1)
	return max(raw, p.Fraction)
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
				if a.Evaluated {
					continue
				}
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

	p.Rank()
	return errors.Join(errList...)
}

func (p *Population) Rank() {
	slices.SortFunc(p.Agents, func(a, b Agent) int {
		if a.Evaluated != b.Evaluated {
			if a.Evaluated {
				return -1
			}
			return 1
		}
		return cmp.Compare(b.Fitness, a.Fitness)
	})
	p.TopFitness = p.Agents[0].Fitness
}

func (p *Population) NewGen(sel config.Selection, goal float64) error {
	p.Rank()

	n := len(p.Agents)
	cutoff := int(float64(n) * sel.Pressure)
	if cutoff < 1 {
		return fmt.Errorf("pressure %f too low, so survivors from population of %d", sel.Pressure, n)
	}

	newAgents := make([]Agent, 0, n)

	for i := 0; i < int(sel.Elite) && i < n; i++ {
		if p.Agents[i].Evaluated {
			newAgents = append(newAgents, p.Agents[i])
		}
	}

	for len(newAgents) < n {
		parent := p.Agents[rand.Intn(cutoff)]

		childGene := parent.Gene.Clone()
		childGene.Nudge(p.CalcNudge(parent.Fitness, goal))
		newAgents = append(newAgents, Agent{Gene: childGene, Fitness: math.Inf(-1), Evaluated: false})
	}
	p.Agents = newAgents
	return nil
}

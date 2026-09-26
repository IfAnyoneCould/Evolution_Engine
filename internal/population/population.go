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
	MinNudge     float64
	Fraction     float64
	NudgeFunc    nudge.Function
	hold         float64
	end          float64
	StartFitness float64
	Distribution nudge.Distribution
}

func NewPopulation(params config.JsonParams, r *rand.Rand) (*Population, error) {

	var agents []Agent
	for range params.RunSettings.PopulationSize {
		a, err := NewAgent(params.Bounds, r)
		if err != nil {
			return &Population{}, err
		}
		agents = append(agents, a)
	}

	var nudgeFunc nudge.Function

	switch strings.ToLower(*params.Mutation.Schedule.Type) {
	case "constant":
		nudgeFunc = nudge.NewConstantFunction()
	case "linear":
		nudgeFunc = nudge.NewLinearFunction()
	case "quadratic":
		nudgeFunc = nudge.NewQuadraticFunction()
	case "power":
		nudgeFunc = nudge.NewPowerFunction(*params.Mutation.Schedule.Exponent)
	case "exponential":
		nudgeFunc = nudge.NewExponentialFunction(*params.Mutation.Schedule.Rate)
	case "cosine":
		nudgeFunc = nudge.NewCosineFunction()
	case "step":
		nudgeFunc = nudge.NewStepFunction(*params.Mutation.Schedule.Steps)
	default:
		return &Population{}, errors.New("config error: mutation.schedule.type not recognized")
	}

	var d nudge.Distribution
	switch strings.ToLower(params.Mutation.Distribution) {
	case "uniform":
		d = nudge.NewUniformDistribution()
	case "gaussian":
		d = nudge.NewGaussianDistribution()
	default:
		return &Population{}, errors.New("config error: mutation.distribution not recognized")
	}

	var procList []*config.SimProcess
	for range params.Workers {
		sim, err := config.NewSimProcess(params.Prog.Path, params.Prog.Args, params.SimSettings.Timeout)
		if err != nil {
			return nil, err
		}
		procList = append(procList, sim)
	}

	return &Population{
		Agents:       agents,
		ProcList:     procList,
		WorkerCount:  params.Workers,
		TopFitness:   -1,
		MinNudge:     params.Mutation.MinNudge,
		Fraction:     params.Mutation.Fraction,
		NudgeFunc:    nudgeFunc,
		hold:         *params.Mutation.Schedule.Hold,
		end:          *params.Mutation.Schedule.End,
		StartFitness: 0,
		Distribution: d}, nil
}

func (p *Population) CalcNudge(fit float64, goal float64) float64 {
	progress := nudge.Clamp((fit-p.StartFitness)/(goal-p.StartFitness), 0, 1)
	if goal-p.StartFitness == 0 {
		progress = 1
	}

	u := nudge.Clamp((progress-p.hold)/(1-p.hold), 0, 1)
	f := p.end + (1-p.end)*p.NudgeFunc.Get(u)
	return max(f*p.Fraction, p.MinNudge)
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

func (p *Population) NewGen(sel config.Selection, goal float64, r *rand.Rand) error {
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
		parent := p.Agents[r.Intn(cutoff)]

		childGene := parent.Gene.Clone()
		childGene.Nudge(p.CalcNudge(parent.Fitness, goal), p.Distribution, r)
		newAgents = append(newAgents, Agent{Gene: childGene, Fitness: math.Inf(-1), Evaluated: false})
	}
	p.Agents = newAgents
	return nil
}

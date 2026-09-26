package optimizers

import (
	"Evolution_Engine/internal/config"
	"Evolution_Engine/internal/nudge"
	"Evolution_Engine/internal/population"
	"cmp"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"strings"
)

type Optimizer interface {
	Ask() ([][]float64, error)
	Tell(fit []float64) error
	Best() ([]float64, float64)
}

type GA struct {
	Agents       []population.Agent
	pending      []int
	sel          config.Selection
	schedule     config.Schedule
	scheduleFunc nudge.Function
	dist         nudge.Distribution
	goal         float64
	startFitness float64
	started      bool
	bestW        []float64
	bestF        float64
	Fraction     float64
	MinNudge     float64
	r            *rand.Rand
}

/* TODO implement this. hardest part is fining good place to put it in the config
func New(cfg config.JsonParams, r *rand.Rand) Optimizer{
	switch cfg.
} */

func NewGA(params config.JsonParams, r *rand.Rand) (*GA, error) {
	var agents []population.Agent
	for range params.RunSettings.PopulationSize {
		a, err := population.NewAgent(params.Bounds, r)
		if err != nil {
			return &GA{}, err
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
		return &GA{}, errors.New("config error: mutation.schedule.type not recognized")
	}

	var d nudge.Distribution
	switch strings.ToLower(params.Mutation.Distribution) {
	case "uniform":
		d = nudge.NewUniformDistribution()
	case "gaussian":
		d = nudge.NewGaussianDistribution()
	default:
		return &GA{}, errors.New("config error: mutation.distribution not recognized")
	}

	return &GA{
		Agents:       agents,
		pending:      []int{},
		sel:          params.Selection,
		schedule:     params.Mutation.Schedule,
		scheduleFunc: nudgeFunc,
		dist:         d,
		goal:         params.RunSettings.TargetFitness,
		startFitness: 0,
		started:      false,
		bestW:        []float64{},
		bestF:        math.Inf(-1),
		Fraction:     params.Mutation.Fraction,
		MinNudge:     params.Mutation.MinNudge,
		r:            r,
	}, nil
}

func (g *GA) Ask() ([][]float64, error) {
	if len(g.pending) > 0 || g.started {
		if err := g.newGen(); err != nil {
			return [][]float64{}, err
		}
	}
	g.pending = g.pending[:0]
	var weights [][]float64
	for i, a := range g.Agents {
		if !a.Evaluated {
			g.pending = append(g.pending, i)
			weights = append(weights, a.Gene.GetWeights())
		}
	}
	return weights, nil
}

func (g *GA) Tell(fit []float64) error {
	if len(fit) != len(g.pending) {
		return fmt.Errorf("told %d fitnesses for %d agents", len(fit), len(g.pending))
	}
	for k, i := range g.pending {
		g.Agents[i].Fitness, g.Agents[i].Evaluated = fit[k], true
	}
	g.Rank()
	if !g.started {
		g.startFitness, g.started = g.Agents[0].Fitness, true
	}
	if g.Agents[0].Fitness > g.bestF {
		g.bestF, g.bestW = g.Agents[0].Fitness, g.Agents[0].Gene.GetWeights()
	}
	return nil
}

func (g *GA) Best() ([]float64, float64) {
	return slices.Clone(g.bestW), g.bestF
}

func (g *GA) newGen() error {
	n := len(g.Agents)
	cutoff := int(float64(n) * g.sel.Pressure)
	if cutoff < 1 {
		return fmt.Errorf("pressure %f too low, so survivors from population of %d", g.sel.Pressure, n)
	}

	newAgents := make([]population.Agent, 0, n)

	for i := 0; i < int(g.sel.Elite) && i < n; i++ {
		if g.Agents[i].Evaluated {
			newAgents = append(newAgents, g.Agents[i])
		}
	}

	for len(newAgents) < n {
		parent := g.Agents[g.r.Intn(cutoff)]

		childGene := parent.Gene.Clone()
		childGene.Nudge(g.CalcNudge(parent.Fitness), g.dist, g.r)
		newAgents = append(newAgents, population.Agent{Gene: childGene, Fitness: math.Inf(-1), Evaluated: false})
	}
	g.Agents = newAgents
	return nil
}

func (g *GA) CalcNudge(fit float64) float64 {
	progress := nudge.Clamp((fit-g.startFitness)/(g.goal-g.startFitness), 0, 1)
	if g.goal-g.startFitness == 0 {
		progress = 1
	}

	u := nudge.Clamp((progress-*g.schedule.Hold)/(1-*g.schedule.Hold), 0, 1)
	f := *g.schedule.End + (1-*g.schedule.End)*g.scheduleFunc.Get(u)
	return max(f*g.Fraction, g.MinNudge)
}

func (g *GA) Rank() {
	slices.SortFunc(g.Agents, func(a, b population.Agent) int {
		if a.Evaluated != b.Evaluated {
			if a.Evaluated {
				return -1
			}
			return 1
		}
		return cmp.Compare(b.Fitness, a.Fitness)
	})
}

package population

import (
	"Evolution_Engine/internal/config"
	"Evolution_Engine/internal/nudge"
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

var simBin string

var testRand = rand.New(rand.NewSource(1))

func ptr[T any](v T) *T { return &v }

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "evoengine")
	if err != nil {
		fmt.Printf("could not make a temp dir: %v\n", err)
		os.Exit(1)
	}
	simBin = filepath.Join(dir, "sim.exe")
	if out, err := exec.Command("go", "build", "-o", simBin, "../../testdata/sim").CombinedOutput(); err != nil {
		fmt.Printf("could not build the test sim: %v\n%s", err, out)
		os.Exit(1)
	}
	code := m.Run()
	_ = os.RemoveAll(dir)
	os.Exit(code)
}

func testConfig(mode string, dims int, agents int, workers int) config.JsonParams {
	bounds := make([][2]float64, dims)
	for i := range bounds {
		bounds[i] = [2]float64{-5, 5}
	}
	return config.JsonParams{
		Prog:   config.Program{Path: simBin, Args: []string{mode}},
		Bounds: bounds,
		Mutation: config.Mutation{
			Distribution: "uniform",
			Fraction:     0.05,
			MinNudge:     0.0001,
			Schedule:     config.Schedule{Type: ptr("constant"), Hold: ptr(0.0), End: ptr(0.0)},
		},
		RunSettings:      config.RunSettings{TargetFitness: 1000, MaxCycles: 10, PopulationSize: uint(agents)},
		Workers:          uint(workers),
		Selection:        config.Selection{Pressure: 0.45, Elite: 1},
		StagnationDetect: config.StagnationDetect{Patience: 1000, Epsilon: 0.01},
		SimSettings:      config.SimSettings{Timeout: 5000, RandSeed: ptr(int64(1))},
	}
}

func newTestPop(t *testing.T, mode string, agents int, workers int) *Population {
	t.Helper()
	p, err := NewPopulation(testConfig(mode, 3, agents, workers), testRand)
	if err != nil {
		t.Fatalf("could not build the population: %v", err)
	}
	t.Cleanup(func() { _ = p.CloseSims() })
	return p
}

func sum(weights []float64) float64 {
	total := 0.0
	for _, w := range weights {
		total += w
	}
	return total
}

func TestNewAgent(t *testing.T) {
	tests := []struct {
		name    string
		bounds  [][2]float64
		wantErr bool
	}{
		{"standard", [][2]float64{{-1, 1}, {0, 10}, {-100, -50}}, false},
		{"fixed param", [][2]float64{{5, 5}}, false},
		{"lower above upper", [][2]float64{{1, -1}}, true},
		{"no bounds", [][2]float64{}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a, err := NewAgent(tt.bounds, testRand)
			if tt.wantErr {
				if err == nil {
					t.Errorf("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(a.Gene.Params) != len(tt.bounds) {
				t.Errorf("incorrect genome length: expected %d, got %d", len(tt.bounds), len(a.Gene.Params))
			}
			if a.Evaluated {
				t.Errorf("a brand new agent should not be marked evaluated")
			}
			if !math.IsInf(a.Fitness, -1) {
				t.Errorf("incorrect starting fitness: expected -Inf, got %f", a.Fitness)
			}
			for i, p := range a.Gene.Params {
				if p.Weight < p.Lower || p.Weight > p.Upper {
					t.Errorf("param %d: weight %f outside bounds [%f, %f]", i, p.Weight, p.Lower, p.Upper)
				}
			}
		})
	}
}

func TestNewPopulation(t *testing.T) {
	p := newTestPop(t, "sum", 12, 4)

	if len(p.Agents) != 12 {
		t.Errorf("incorrect agent count: expected %d, got %d", 12, len(p.Agents))
	}
	if len(p.ProcList) != 4 {
		t.Errorf("incorrect worker count: expected %d, got %d", 4, len(p.ProcList))
	}
	if p.WorkerCount != uint(len(p.ProcList)) {
		t.Errorf("WorkerCount %d does not match the %d processes started", p.WorkerCount, len(p.ProcList))
	}
	if p.Fraction != 0.05 {
		t.Errorf("incorrect fraction: expected %f, got %f", 0.05, p.Fraction)
	}
	if p.MinNudge != 0.0001 {
		t.Errorf("incorrect min nudge: expected %f, got %f", 0.0001, p.MinNudge)
	}
	if p.NudgeFunc == nil {
		t.Errorf("no nudge function set")
	}
	for i, a := range p.Agents {
		if len(a.Gene.Params) != 3 {
			t.Errorf("agent %d has %d params, config asked for %d", i, len(a.Gene.Params), 3)
		}
	}
}

func TestNewPopulationSchedule(t *testing.T) {
	tests := []struct {
		name     string
		function config.Schedule
		progress float64
		want     float64
	}{
		{"constant", config.Schedule{Type: ptr("constant")}, 0.9, 1},
		{"linear", config.Schedule{Type: ptr("linear")}, 0.25, 0.75},
		{"quadratic", config.Schedule{Type: ptr("quadratic")}, 0.5, 0.25},
		{"power", config.Schedule{Type: ptr("power"), Exponent: ptr(3.0)}, 0.5, 0.125},
		{"exponential", config.Schedule{Type: ptr("exponential"), Rate: ptr(4.0)}, 1, 0},
		{"cosine", config.Schedule{Type: ptr("cosine")}, 0.5, 0.5},
		{"step", config.Schedule{Type: ptr("step"), Steps: ptr(uint(4))}, 0.3, 0.75},
		{"type is case insensitive", config.Schedule{Type: ptr("LINEAR")}, 0.25, 0.75},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := testConfig("sum", 2, 4, 1)
			cfg.Mutation.Fraction = 0.3
			cfg.Mutation.MinNudge = 0.02
			tt.function.Hold, tt.function.End = ptr(0.1), ptr(0.2)
			cfg.Mutation.Schedule = tt.function
			p, err := NewPopulation(cfg, testRand)
			if err != nil {
				t.Fatalf("could not build the population: %v", err)
			}
			defer p.CloseSims()

			if p.Fraction != 0.3 {
				t.Errorf("incorrect fraction: expected %f, got %f", 0.3, p.Fraction)
			}
			if p.MinNudge != 0.02 {
				t.Errorf("incorrect min nudge: expected %f, got %f", 0.02, p.MinNudge)
			}
			if p.hold != 0.1 || p.end != 0.2 {
				t.Errorf("hold and end not carried over: expected 0.1 and 0.2, got %f and %f", p.hold, p.end)
			}
			if p.NudgeFunc == nil {
				t.Fatalf("no nudge function set")
			}
			if got := p.NudgeFunc.Get(tt.progress); math.Abs(got-tt.want) > 1e-9 {
				t.Errorf("wrong nudge function: expected %f, got %f", tt.want, got)
			}
		})
	}
}

func TestNewPopulationUnknownSchedule(t *testing.T) {
	cfg := testConfig("sum", 2, 4, 1)
	cfg.Mutation.Schedule.Type = ptr("sawtooth")
	if _, err := NewPopulation(cfg, testRand); err == nil {
		t.Errorf("expected error")
	}
}

func TestNewPopulationDistribution(t *testing.T) {
	tests := []struct {
		name     string
		dist     string
		gaussian bool
	}{
		{"uniform", "uniform", false},
		{"gaussian", "gaussian", true},
		{"case insensitive", "GAUSSIAN", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := testConfig("sum", 3, 10, 2)
			cfg.Mutation.Distribution = tt.dist
			p, err := NewPopulation(cfg, testRand)
			if err != nil {
				t.Fatalf("could not build the population: %v", err)
			}
			defer p.CloseSims()

			switch p.Distribution.(type) {
			case *nudge.Gaussian:
				if !tt.gaussian {
					t.Errorf("wanted a uniform distribution, got gaussian")
				}
			case *nudge.Uniform:
				if tt.gaussian {
					t.Errorf("wanted a gaussian distribution, got uniform")
				}
			default:
				t.Fatalf("no distribution set, got %T", p.Distribution)
			}

			if err := p.RunBatch(context.Background()); err != nil {
				t.Fatalf("could not run the first batch: %v", err)
			}
			if err := p.NewGen(config.Selection{Elite: 1, Pressure: 0.5}, 1000, testRand); err != nil {
				t.Fatalf("NewGen failed: %v", err)
			}
			for i, a := range p.Agents {
				for j, param := range a.Gene.Params {
					if param.Weight < param.Lower || param.Weight > param.Upper {
						t.Errorf("agent %d param %d: weight %f outside bounds", i, j, param.Weight)
					}
				}
			}
		})
	}
}

// a config built in code skips validate, so an unknown or missing distribution has to fail here instead of
// leaving a nil distribution for NewGen to panic on
func TestNewPopulationBadDistribution(t *testing.T) {
	for _, dist := range []string{"cauchy", ""} {
		cfg := testConfig("sum", 3, 10, 2)
		cfg.Mutation.Distribution = dist
		p, err := NewPopulation(cfg, testRand)
		if err == nil {
			_ = p.CloseSims()
			t.Errorf("distribution %q should be an error", dist)
		}
	}
}

func TestNewPopulationBadProgram(t *testing.T) {
	cfg := testConfig("sum", 3, 5, 2)
	cfg.Prog.Path = filepath.Join(t.TempDir(), "not_a_program.exe")
	if _, err := NewPopulation(cfg, testRand); err == nil {
		t.Errorf("expected error")
	}
}

func TestAgentEvaluate(t *testing.T) {
	p := newTestPop(t, "sum", 1, 1)

	a := &p.Agents[0]
	if err := a.Evaluate(p.ProcList[0]); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !a.Evaluated {
		t.Errorf("agent not marked evaluated")
	}
	if want := sum(a.Gene.GetWeights()); math.Abs(a.Fitness-want) > 1e-9 {
		t.Errorf("incorrect fitness: expected %f, got %f", want, a.Fitness)
	}
}

func TestAgentEvaluateError(t *testing.T) {
	p := newTestPop(t, "garbage", 1, 1)

	a := &p.Agents[0]
	if err := a.Evaluate(p.ProcList[0]); err == nil {
		t.Errorf("expected error")
	}
	if a.Evaluated {
		t.Errorf("agent marked evaluated after a failed run")
	}
}

func TestCalcNudge(t *testing.T) {
	tests := []struct {
		name        string
		function    nudge.Function
		hold, end   float64
		fraction    float64
		minNudge    float64
		start, goal float64
		fit         float64
		want        float64
	}{
		{"constant ignores progress", nudge.NewConstantFunction(), 0, 0, 0.1, 0.001, 0, 100, 50, 0.1},
		{"constant ignores hold and end", nudge.NewConstantFunction(), 0.5, 0.2, 0.1, 0.001, 0, 100, 90, 0.1},
		{"linear at the start", nudge.NewLinearFunction(), 0, 0, 0.1, 0.001, 0, 100, 0, 0.1},
		{"linear halfway", nudge.NewLinearFunction(), 0, 0, 0.1, 0.001, 0, 100, 50, 0.05},
		{"quadratic halfway", nudge.NewQuadraticFunction(), 0, 0, 0.1, 0.001, 0, 100, 50, 0.025},
		{"floor applies at the goal", nudge.NewLinearFunction(), 0, 0, 0.1, 0.001, 0, 100, 100, 0.001},
		{"progress clamps above the goal", nudge.NewLinearFunction(), 0, 0, 0.1, 0.001, 0, 100, 500, 0.001},
		{"progress clamps below the start", nudge.NewLinearFunction(), 0, 0, 0.1, 0.001, 0, 100, -500, 0.1},
		{"negative fitness range", nudge.NewLinearFunction(), 0, 0, 0.2, 0.001, -100, -50, -75, 0.1},
		{"full nudge before hold", nudge.NewLinearFunction(), 0.5, 0, 0.1, 0.001, 0, 100, 25, 0.1},
		{"decay starts at hold", nudge.NewLinearFunction(), 0.5, 0, 0.1, 0.001, 0, 100, 75, 0.05},
		{"end is the value at the goal", nudge.NewLinearFunction(), 0, 0.2, 0.1, 0.001, 0, 100, 100, 0.02},
		{"hold and end together", nudge.NewQuadraticFunction(), 0.5, 0.2, 0.1, 0.001, 0, 100, 75, 0.04},
		{"floor still applies under end", nudge.NewLinearFunction(), 0, 0.001, 0.1, 0.01, 0, 100, 100, 0.01},
		{"hold of 1 never decays", nudge.NewLinearFunction(), 1, 0, 0.1, 0.001, 0, 100, 90, 0.1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &Population{Fraction: tt.fraction, MinNudge: tt.minNudge, NudgeFunc: tt.function, hold: tt.hold, end: tt.end, StartFitness: tt.start}
			got := p.CalcNudge(tt.fit, tt.goal)
			if math.Abs(got-tt.want) > 1e-9 {
				t.Errorf("incorrect nudge: expected %f, got %f", tt.want, got)
			}
			if got < tt.minNudge {
				t.Errorf("nudge %f dropped below the floor %f", got, tt.minNudge)
			}
		})
	}
}

func TestRank(t *testing.T) {
	p := &Population{Agents: make([]Agent, 5)}
	for i, f := range []float64{-3, 10, 0.5, 100, 7} {
		p.Agents[i].Fitness = f
	}
	p.Rank()

	want := []float64{100, 10, 7, 0.5, -3}
	for i, a := range p.Agents {
		if a.Fitness != want[i] {
			t.Errorf("agent %d out of order: expected %f, got %f", i, want[i], a.Fitness)
		}
	}
	if p.TopFitness != 100 {
		t.Errorf("incorrect top fitness: expected %f, got %f", 100.0, p.TopFitness)
	}
}

func TestRankPutsUnevaluatedLast(t *testing.T) {
	p := &Population{Agents: make([]Agent, 4)}
	for i, a := range []Agent{
		{Fitness: 2, Evaluated: true},
		{Fitness: 100, Evaluated: false},
		{Fitness: -3, Evaluated: true},
		{Fitness: 50, Evaluated: false},
	} {
		p.Agents[i] = a
	}
	p.Rank()

	if p.TopFitness != 2 {
		t.Errorf("an unevaluated agent was treated as the best: expected %f, got %f", 2.0, p.TopFitness)
	}
	for i, a := range p.Agents {
		if i < 2 && !a.Evaluated {
			t.Errorf("agent %d is unevaluated but sorted above an evaluated one", i)
		}
		if i >= 2 && a.Evaluated {
			t.Errorf("agent %d is evaluated but sorted below an unevaluated one", i)
		}
	}
}

func TestRunBatch(t *testing.T) {
	tests := []struct {
		name    string
		agents  int
		workers int
	}{
		{"one worker", 6, 1},
		{"more agents than workers", 50, 4},
		{"a worker per agent", 8, 8},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := newTestPop(t, "sum", tt.agents, tt.workers)

			if err := p.RunBatch(context.Background()); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(p.Agents) != tt.agents {
				t.Errorf("batch changed the agent count: expected %d, got %d", tt.agents, len(p.Agents))
			}
			for i, a := range p.Agents {
				if !a.Evaluated {
					t.Errorf("agent %d never got evaluated", i)
				}
				if want := sum(a.Gene.GetWeights()); math.Abs(a.Fitness-want) > 1e-9 {
					t.Errorf("agent %d got the wrong fitness: expected %f, got %f", i, want, a.Fitness)
				}
			}
		})
	}
}

func TestRunBatchAfterNewGen(t *testing.T) {
	p := newTestPop(t, "sum", 10, 2)

	if err := p.RunBatch(context.Background()); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := p.NewGen(config.Selection{Elite: 1, Pressure: 0.5}, 1000, testRand); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if err := p.RunBatch(context.Background()); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for i, a := range p.Agents {
		if !a.Evaluated {
			t.Errorf("agent %d never got evaluated", i)
		}
		if want := sum(a.Gene.GetWeights()); math.Abs(a.Fitness-want) > 1e-9 {
			t.Errorf("agent %d got the wrong fitness: expected %f, got %f", i, want, a.Fitness)
		}
	}
}

func TestRunBatchError(t *testing.T) {
	p := newTestPop(t, "garbage", 6, 2)

	if err := p.RunBatch(context.Background()); err == nil {
		t.Errorf("expected error")
	}
}

func TestRunBatchSetsTopFitness(t *testing.T) {
	p := newTestPop(t, "sum", 10, 2)

	if err := p.RunBatch(context.Background()); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	best := math.Inf(-1)
	for _, a := range p.Agents {
		best = max(best, a.Fitness)
	}
	if p.TopFitness != best {
		t.Errorf("TopFitness not updated by the batch: expected %f, got %f", best, p.TopFitness)
	}
}

func TestNewGen(t *testing.T) {
	tests := []struct {
		name    string
		config  config.Selection
		agents  int
		wantErr bool
	}{
		{"standard", config.Selection{Elite: 1, Pressure: 0.45}, 20, false},
		{"no elites", config.Selection{Elite: 0, Pressure: 0.5}, 20, false},
		{"everything survives", config.Selection{Elite: 2, Pressure: 1}, 20, false},
		{"more elites than agents", config.Selection{Elite: 50, Pressure: 0.5}, 20, false},
		{"pressure too low", config.Selection{Elite: 1, Pressure: 0.01}, 20, true},
		{"no pressure", config.Selection{Elite: 1, Pressure: 0}, 20, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := newTestPop(t, "sum", tt.agents, 2)
			p.NudgeFunc = nudge.NewConstantFunction()
			if err := p.RunBatch(context.Background()); err != nil {
				t.Fatalf("could not run the first batch: %v", err)
			}
			p.Rank()
			best := p.Agents[0].Gene.GetWeights()
			bestFitness := p.Agents[0].Fitness

			err := p.NewGen(tt.config, 1000, testRand)
			if tt.wantErr {
				if err == nil {
					t.Errorf("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(p.Agents) != tt.agents {
				t.Errorf("population changed size: expected %d, got %d", tt.agents, len(p.Agents))
			}
			if tt.config.Elite > 0 {
				for i, w := range p.Agents[0].Gene.GetWeights() {
					if w != best[i] {
						t.Errorf("the elite agent was mutated at param %d: expected %f, got %f", i, best[i], w)
					}
				}
				if p.Agents[0].Fitness != bestFitness {
					t.Errorf("the elite agent lost its fitness: expected %f, got %f", bestFitness, p.Agents[0].Fitness)
				}
				if !p.Agents[0].Evaluated {
					t.Errorf("the elite agent should still count as evaluated")
				}
			}
			elites := min(int(tt.config.Elite), tt.agents)
			for i := elites; i < len(p.Agents); i++ {
				if p.Agents[i].Evaluated {
					t.Errorf("child %d is marked evaluated before being run", i)
				}
				for j, param := range p.Agents[i].Gene.Params {
					if param.Weight < param.Lower || param.Weight > param.Upper {
						t.Errorf("child %d param %d: weight %f outside bounds [%f, %f]", i, j, param.Weight, param.Lower, param.Upper)
					}
				}
			}
		})
	}
}

func TestNewGenUnevaluatedElite(t *testing.T) {
	p := newTestPop(t, "sum", 20, 2)

	if err := p.NewGen(config.Selection{Elite: 2, Pressure: 0.5}, 1000, testRand); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(p.Agents) != 20 {
		t.Errorf("population changed size: expected %d, got %d", 20, len(p.Agents))
	}
	for i, a := range p.Agents {
		if a.Evaluated {
			t.Errorf("agent %d came out evaluated, nothing has been run", i)
		}
		if !math.IsInf(a.Fitness, -1) {
			t.Errorf("agent %d has a fitness of %f before being run", i, a.Fitness)
		}
	}
}

func TestNewGenChildrenAreOwnGenomes(t *testing.T) {
	p := newTestPop(t, "sum", 10, 2)
	p.NudgeFunc = nudge.NewConstantFunction()
	if err := p.RunBatch(context.Background()); err != nil {
		t.Fatalf("could not run the first batch: %v", err)
	}
	if err := p.NewGen(config.Selection{Elite: 1, Pressure: 0.5}, 1000, testRand); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for i := range p.Agents {
		for j := range p.Agents {
			if i != j && p.Agents[i].Gene == p.Agents[j].Gene {
				t.Fatalf("agents %d and %d point at the same genome", i, j)
			}
		}
	}

	before := p.Agents[len(p.Agents)-1].Gene.GetWeights()
	p.Agents[0].Gene.Nudge(1.0, nudge.NewUniformDistribution(), testRand)
	after := p.Agents[len(p.Agents)-1].Gene.GetWeights()
	for i := range before {
		if before[i] != after[i] {
			t.Errorf("mutating one agent changed another at param %d: %f became %f", i, before[i], after[i])
		}
	}
}

func TestNewGenPullsFromTheTop(t *testing.T) {
	p := newTestPop(t, "sum", 40, 2)
	p.NudgeFunc = nudge.NewConstantFunction()
	p.Fraction = 0
	p.MinNudge = 0
	if err := p.RunBatch(context.Background()); err != nil {
		t.Fatalf("could not run the first batch: %v", err)
	}
	p.Rank()
	worst := p.Agents[len(p.Agents)-1].Gene.GetWeights()

	if err := p.NewGen(config.Selection{Elite: 1, Pressure: 0.25}, 1000, testRand); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for i, a := range p.Agents {
		if sum(a.Gene.GetWeights()) == sum(worst) {
			t.Errorf("agent %d was bred from an agent outside the cutoff", i)
		}
	}
}

func TestCloseSims(t *testing.T) {
	p, err := NewPopulation(testConfig("sum", 3, 4, 2), testRand)
	if err != nil {
		t.Fatalf("could not build the population: %v", err)
	}
	if err := p.CloseSims(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if err := p.RunBatch(context.Background()); err == nil {
		t.Errorf("expected an error running a batch against closed sims")
	}
}

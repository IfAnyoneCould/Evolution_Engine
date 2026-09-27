package optimizers

import (
	"Evolution_Engine/internal/config"
	"Evolution_Engine/internal/nudge"
	"math"
	"math/rand"
	"slices"
	"testing"
)

var _ Optimizer = (*GA)(nil)

func ptr[T any](v T) *T { return &v }

func testConfig(dims int, agents int) config.JsonParams {
	bounds := make([][2]float64, dims)
	for i := range bounds {
		bounds[i] = [2]float64{-5, 5}
	}
	return config.JsonParams{
		Bounds: bounds,
		Mutation: config.Mutation{
			Distribution: "uniform",
			Fraction:     0.05,
			MinNudge:     0.0001,
			Schedule:     config.Schedule{Type: ptr("constant"), Hold: ptr(0.0), End: ptr(0.0)},
		},
		RunSettings: config.RunSettings{TargetFitness: 1000, MaxCycles: 10, PopulationSize: uint(agents)},
		Selection:   config.Selection{Pressure: 0.45, Elite: 1},
	}
}

func newTestGA(t *testing.T, cfg config.JsonParams, seed int64) *GA {
	t.Helper()
	g, err := NewGA(cfg, rand.New(rand.NewSource(seed)))
	if err != nil {
		t.Fatalf("could not build the ga: %v", err)
	}
	return g
}

func sum(ws []float64) float64 {
	total := 0.0
	for _, w := range ws {
		total += w
	}
	return total
}

func sums(xs [][]float64) []float64 {
	fit := make([]float64, len(xs))
	for i, x := range xs {
		fit[i] = sum(x)
	}
	return fit
}

// one ask and tell, scoring every vector by the sum of its weights
func step(t *testing.T, g *GA) [][]float64 {
	t.Helper()
	xs, err := g.Ask()
	if err != nil {
		t.Fatalf("ask failed: %v", err)
	}
	if err := g.Tell(sums(xs)); err != nil {
		t.Fatalf("tell failed: %v", err)
	}
	return xs
}

func inBounds(t *testing.T, g *GA) {
	t.Helper()
	for i, a := range g.Agents {
		for j, p := range a.Gene.Params {
			if p.Weight < p.Lower || p.Weight > p.Upper {
				t.Fatalf("agent %d param %d: weight %f outside [%f, %f]", i, j, p.Weight, p.Lower, p.Upper)
			}
		}
	}
}

func TestNewGA(t *testing.T) {
	g := newTestGA(t, testConfig(3, 12), 1)
	if len(g.Agents) != 12 {
		t.Errorf("incorrect agent count: expected %d, got %d", 12, len(g.Agents))
	}
	for i, a := range g.Agents {
		if len(a.Gene.Params) != 3 {
			t.Errorf("agent %d has %d params, config asked for %d", i, len(a.Gene.Params), 3)
		}
		if a.Evaluated {
			t.Errorf("agent %d starts out evaluated", i)
		}
	}
	inBounds(t, g)
	if w, f := g.Best(); len(w) != 0 || !math.IsInf(f, -1) {
		t.Errorf("nothing has been told yet, best should be empty and -Inf, got %v and %f", w, f)
	}
}

func TestNewGASchedule(t *testing.T) {
	tests := []struct {
		name     string
		schedule config.Schedule
		u        float64
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
			cfg := testConfig(2, 4)
			tt.schedule.Hold, tt.schedule.End = ptr(0.1), ptr(0.2)
			cfg.Mutation.Schedule = tt.schedule
			g := newTestGA(t, cfg, 1)
			if *g.schedule.Hold != 0.1 || *g.schedule.End != 0.2 {
				t.Errorf("hold and end not carried over: got %f and %f", *g.schedule.Hold, *g.schedule.End)
			}
			if got := g.scheduleFunc.Get(tt.u); math.Abs(got-tt.want) > 1e-9 {
				t.Errorf("wrong schedule: expected %f, got %f", tt.want, got)
			}
		})
	}
}

func TestNewGADistribution(t *testing.T) {
	tests := []struct {
		dist     string
		gaussian bool
	}{
		{"uniform", false},
		{"gaussian", true},
		{"GAUSSIAN", true},
	}
	for _, tt := range tests {
		t.Run(tt.dist, func(t *testing.T) {
			cfg := testConfig(3, 10)
			cfg.Mutation.Distribution = tt.dist
			g := newTestGA(t, cfg, 1)
			_, isGaussian := g.dist.(*nudge.Gaussian)
			_, isUniform := g.dist.(*nudge.Uniform)
			if isGaussian != tt.gaussian || isUniform == tt.gaussian {
				t.Errorf("wrong distribution for %q: got %T", tt.dist, g.dist)
			}
			for range 5 {
				step(t, g)
				inBounds(t, g)
			}
		})
	}
}

// a config built in code skips validate, so bad values have to fail here instead of panicking later
func TestNewGABadConfig(t *testing.T) {
	tests := []struct {
		name   string
		change func(*config.JsonParams)
	}{
		{"unknown schedule", func(c *config.JsonParams) { c.Mutation.Schedule.Type = ptr("sawtooth") }},
		{"unknown distribution", func(c *config.JsonParams) { c.Mutation.Distribution = "cauchy" }},
		{"no distribution", func(c *config.JsonParams) { c.Mutation.Distribution = "" }},
		{"lower above upper", func(c *config.JsonParams) { c.Bounds[0] = [2]float64{1, -1} }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := testConfig(2, 4)
			tt.change(&cfg)
			if _, err := NewGA(cfg, rand.New(rand.NewSource(1))); err == nil {
				t.Errorf("expected error")
			}
		})
	}
}

func TestFirstAskReturnsEveryAgent(t *testing.T) {
	g := newTestGA(t, testConfig(3, 10), 1)
	xs, err := g.Ask()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(xs) != 10 {
		t.Fatalf("first ask should hand out every agent: expected %d, got %d", 10, len(xs))
	}
	for i, x := range xs {
		if !slices.Equal(x, g.Agents[g.pending[i]].Gene.GetWeights()) {
			t.Errorf("vector %d doesn't match the agent it was handed out for", i)
		}
	}
}

func TestAskHandsOutCopies(t *testing.T) {
	g := newTestGA(t, testConfig(3, 4), 1)
	xs, _ := g.Ask()
	before := g.Agents[g.pending[0]].Gene.GetWeights()
	xs[0][0] = 999
	if !slices.Equal(before, g.Agents[g.pending[0]].Gene.GetWeights()) {
		t.Errorf("changing an asked vector changed the agent's genome")
	}
}

func TestTellWrongCount(t *testing.T) {
	g := newTestGA(t, testConfig(3, 10), 1)
	if _, err := g.Ask(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := g.Tell(make([]float64, 9)); err == nil {
		t.Errorf("expected an error telling 9 fitnesses for 10 vectors")
	}
}

func TestLaterAsksOnlyReturnChildren(t *testing.T) {
	tests := []struct {
		elite uint
		want  int
	}{
		{0, 20},
		{1, 19},
		{3, 17},
	}
	for _, tt := range tests {
		t.Run("elite "+string(rune('0'+tt.elite)), func(t *testing.T) {
			cfg := testConfig(3, 20)
			cfg.Selection.Elite = tt.elite
			g := newTestGA(t, cfg, 1)
			step(t, g)
			xs, err := g.Ask()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(xs) != tt.want {
				t.Errorf("elites shouldn't be handed out again: expected %d vectors, got %d", tt.want, len(xs))
			}
			if len(g.Agents) != 20 {
				t.Errorf("population changed size: expected %d, got %d", 20, len(g.Agents))
			}
		})
	}
}

func TestTellRanksAndMarksEvaluated(t *testing.T) {
	g := newTestGA(t, testConfig(3, 10), 1)
	step(t, g)
	for i, a := range g.Agents {
		if !a.Evaluated {
			t.Errorf("agent %d not marked evaluated after tell", i)
		}
		if math.Abs(a.Fitness-sum(a.Gene.GetWeights())) > 1e-9 {
			t.Errorf("agent %d got someone else's fitness: %f for weights summing to %f", i, a.Fitness, sum(a.Gene.GetWeights()))
		}
		if i > 0 && a.Fitness > g.Agents[i-1].Fitness {
			t.Errorf("agents not ranked: %d has %f above %f", i, a.Fitness, g.Agents[i-1].Fitness)
		}
	}
}

func TestBestMatchesItsWeights(t *testing.T) {
	g := newTestGA(t, testConfig(3, 10), 1)
	xs := step(t, g)
	w, f := g.Best()
	if want := slices.Max(sums(xs)); f != want {
		t.Errorf("best fitness should be the top of the batch: expected %f, got %f", want, f)
	}
	if math.Abs(sum(w)-f) > 1e-9 {
		t.Errorf("best weights don't belong to the best fitness: they sum to %f, fitness %f", sum(w), f)
	}
}

// with no elites the best agent is gone after the next generation, Best has to remember it
func TestBestIsTheBestSoFar(t *testing.T) {
	cfg := testConfig(3, 10)
	cfg.Selection.Elite = 0
	g := newTestGA(t, cfg, 1)
	step(t, g)
	w1, f1 := g.Best()

	xs, err := g.Ask()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	low := make([]float64, len(xs))
	for i := range low {
		low[i] = f1 - 100
	}
	if err := g.Tell(low); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	w2, f2 := g.Best()
	if f2 != f1 || !slices.Equal(w1, w2) {
		t.Errorf("a worse generation replaced the best: %f became %f", f1, f2)
	}
}

func TestBestReturnsACopy(t *testing.T) {
	g := newTestGA(t, testConfig(3, 10), 1)
	step(t, g)
	w, _ := g.Best()
	w[0] = 999
	if w2, _ := g.Best(); w2[0] == 999 {
		t.Errorf("changing the returned best weights changed the ga's copy")
	}
}

func TestStartFitnessIsTheFirstGeneration(t *testing.T) {
	g := newTestGA(t, testConfig(3, 10), 1)
	xs := step(t, g)
	first := slices.Max(sums(xs))
	if g.startFitness != first {
		t.Errorf("start fitness should be the first generation's best: expected %f, got %f", first, g.startFitness)
	}
	step(t, g)
	if g.startFitness != first {
		t.Errorf("start fitness moved after the first generation: %f became %f", first, g.startFitness)
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
		{"start already at the goal", nudge.NewLinearFunction(), 0, 0, 0.1, 0.001, 100, 100, 100, 0.001},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := &GA{
				scheduleFunc: tt.function,
				schedule:     config.Schedule{Hold: ptr(tt.hold), End: ptr(tt.end)},
				Fraction:     tt.fraction,
				MinNudge:     tt.minNudge,
				startFitness: tt.start,
				goal:         tt.goal,
			}
			got := g.CalcNudge(tt.fit)
			if math.Abs(got-tt.want) > 1e-9 {
				t.Errorf("incorrect nudge: expected %f, got %f", tt.want, got)
			}
			if got < tt.minNudge {
				t.Errorf("nudge %f dropped below the floor %f", got, tt.minNudge)
			}
		})
	}
}

func TestNewGen(t *testing.T) {
	tests := []struct {
		name    string
		sel     config.Selection
		wantErr bool
	}{
		{"standard", config.Selection{Elite: 1, Pressure: 0.45}, false},
		{"no elites", config.Selection{Elite: 0, Pressure: 0.5}, false},
		{"everything survives", config.Selection{Elite: 2, Pressure: 1}, false},
		{"more elites than agents", config.Selection{Elite: 50, Pressure: 0.5}, false},
		{"pressure too low", config.Selection{Elite: 1, Pressure: 0.01}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := testConfig(3, 20)
			cfg.Selection = tt.sel
			g := newTestGA(t, cfg, 1)
			step(t, g)
			best := g.Agents[0].Gene.GetWeights()
			bestFitness := g.Agents[0].Fitness

			_, err := g.Ask()
			if tt.wantErr {
				if err == nil {
					t.Errorf("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(g.Agents) != 20 {
				t.Errorf("population changed size: expected %d, got %d", 20, len(g.Agents))
			}
			if tt.sel.Elite > 0 {
				if !slices.Equal(g.Agents[0].Gene.GetWeights(), best) {
					t.Errorf("the elite agent was mutated")
				}
				if g.Agents[0].Fitness != bestFitness || !g.Agents[0].Evaluated {
					t.Errorf("the elite agent lost its fitness or its evaluated flag")
				}
			}
			elites := min(int(tt.sel.Elite), 20)
			for i := elites; i < len(g.Agents); i++ {
				if g.Agents[i].Evaluated {
					t.Errorf("child %d is marked evaluated before being run", i)
				}
			}
			inBounds(t, g)
		})
	}
}

func TestChildrenAreOwnGenomes(t *testing.T) {
	g := newTestGA(t, testConfig(3, 10), 1)
	step(t, g)
	if _, err := g.Ask(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for i := range g.Agents {
		for j := range g.Agents {
			if i != j && g.Agents[i].Gene == g.Agents[j].Gene {
				t.Fatalf("agents %d and %d point at the same genome", i, j)
			}
		}
	}
}

func TestChildrenComeFromTheTop(t *testing.T) {
	cfg := testConfig(3, 40)
	cfg.Mutation.Fraction, cfg.Mutation.MinNudge = 0, 0
	cfg.Selection = config.Selection{Elite: 1, Pressure: 0.25}
	g := newTestGA(t, cfg, 1)
	step(t, g)
	var top [][]float64
	for _, a := range g.Agents[:10] {
		top = append(top, a.Gene.GetWeights())
	}
	if _, err := g.Ask(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for i, a := range g.Agents {
		w := a.Gene.GetWeights()
		if !slices.ContainsFunc(top, func(p []float64) bool { return slices.Equal(p, w) }) {
			t.Errorf("agent %d was bred from outside the top quarter", i)
		}
	}
}

func TestGAImproves(t *testing.T) {
	g := newTestGA(t, testConfig(3, 20), 1)
	step(t, g)
	_, first := g.Best()
	for range 60 {
		step(t, g)
	}
	if _, f := g.Best(); f < 13 || f <= first {
		t.Errorf("60 generations barely moved the best: started at %f, ended at %f, max is 15", first, f)
	}
}

func TestSameSeedSameRun(t *testing.T) {
	run := func(seed int64) []float64 {
		g := newTestGA(t, testConfig(3, 20), seed)
		for range 15 {
			step(t, g)
		}
		w, _ := g.Best()
		return w
	}
	a, b := run(7), run(7)
	if !slices.Equal(a, b) {
		t.Errorf("same seed gave different runs: %v and %v", a, b)
	}
	if c := run(8); slices.Equal(a, c) {
		t.Errorf("different seeds gave the same run: %v", a)
	}
}

var _ Optimizer = (*CMAES)(nil)

func cmaConfig(bounds ...[2]float64) config.JsonParams {
	return config.JsonParams{Bounds: bounds, Optimizer: "cmaes"}
}

func uniformBounds(n int, lo, hi float64) [][2]float64 {
	b := make([][2]float64, n)
	for i := range b {
		b[i] = [2]float64{lo, hi}
	}
	return b
}

func newTestCMA(t *testing.T, cfg config.JsonParams, seed int64) *CMAES {
	t.Helper()
	c, err := NewCMAES(cfg, rand.New(rand.NewSource(seed)))
	if err != nil {
		t.Fatalf("could not build cma-es: %v", err)
	}
	return c
}

// runs gens generations maximising f, returns how many generations it took to reach target (or gens)
func runCMA(t *testing.T, c *CMAES, f func([]float64) float64, gens int, target float64) int {
	t.Helper()
	for g := 0; g < gens; g++ {
		xs, err := c.Ask()
		if err != nil {
			t.Fatalf("ask failed: %v", err)
		}
		fit := make([]float64, len(xs))
		for i, x := range xs {
			fit[i] = f(x)
		}
		if err := c.Tell(fit); err != nil {
			t.Fatalf("tell failed: %v", err)
		}
		if _, best := c.Best(); best >= target {
			return g + 1
		}
	}
	return gens
}

func TestNewCMAESConstants(t *testing.T) {
	c := newTestCMA(t, cmaConfig(uniformBounds(10, -1, 1)...), 1)
	if c.n != 10 || c.y != 10 || c.u != 5 {
		t.Errorf("wrong sizes for n=10: expected n 10, lambda 10, mu 5, got %d, %d, %d", c.n, c.y, c.u)
	}
	sum := 0.0
	for i, w := range c.wI {
		sum += w
		if i > 0 && w >= c.wI[i-1] {
			t.Errorf("weights should decrease, %d is %f after %f", i, w, c.wI[i-1])
		}
	}
	if math.Abs(sum-1) > 1e-12 {
		t.Errorf("weights should sum to 1, got %f", sum)
	}
	if math.Abs(c.uEff-3.1672) > 1e-3 {
		t.Errorf("incorrect mu_eff: expected about 3.1672, got %f", c.uEff)
	}
	if c.o != 0.3 {
		t.Errorf("incorrect starting sigma: expected 0.3, got %f", c.o)
	}
	for j := range c.n {
		if c.m.AtVec(j) != 0.5 {
			t.Errorf("mean should start in the middle of every range, dim %d is %f", j, c.m.AtVec(j))
		}
	}
}

func TestNewCMAESSkipsFixedWeights(t *testing.T) {
	c := newTestCMA(t, cmaConfig([2]float64{-1, 1}, [2]float64{5, 5}, [2]float64{0, 10}), 1)
	if c.n != 2 || !slices.Equal(c.free, []int{0, 2}) {
		t.Errorf("only free weights should be dimensions: expected n 2 on [0 2], got n %d on %v", c.n, c.free)
	}
}

func TestNewCMAESAllFixed(t *testing.T) {
	if _, err := NewCMAES(cmaConfig([2]float64{5, 5}, [2]float64{1, 1}), rand.New(rand.NewSource(1))); err == nil {
		t.Errorf("expected an error when every weight is fixed")
	}
}

func TestCMAESAsk(t *testing.T) {
	bounds := [][2]float64{{-1, 1}, {0, 10}, {-100, -50}, {5, 5}}
	c := newTestCMA(t, cmaConfig(bounds...), 1)
	for g := range 20 {
		xs, err := c.Ask()
		if err != nil {
			t.Fatalf("ask failed: %v", err)
		}
		if len(xs) != c.y {
			t.Fatalf("ask should hand out lambda samples: expected %d, got %d", c.y, len(xs))
		}
		for _, x := range xs {
			if len(x) != len(bounds) {
				t.Fatalf("sample has %d weights, bounds have %d", len(x), len(bounds))
			}
			for i, w := range x {
				if w < bounds[i][0] || w > bounds[i][1] {
					t.Fatalf("generation %d: weight %d is %f, outside [%f, %f]", g, i, w, bounds[i][0], bounds[i][1])
				}
			}
			if x[3] != 5 {
				t.Fatalf("fixed weight moved to %f", x[3])
			}
		}
		fit := make([]float64, len(xs))
		for i, x := range xs {
			fit[i] = -x[0] * x[0]
		}
		if err := c.Tell(fit); err != nil {
			t.Fatalf("tell failed: %v", err)
		}
	}
}

func TestCMAESAskTwiceGivesTheSameSamples(t *testing.T) {
	c := newTestCMA(t, cmaConfig(uniformBounds(3, -1, 1)...), 1)
	a, _ := c.Ask()
	b, _ := c.Ask()
	for i := range a {
		if !slices.Equal(a[i], b[i]) {
			t.Fatalf("asking again before telling should give the same generation, sample %d changed", i)
		}
	}
}

func TestCMAESAskHandsOutCopies(t *testing.T) {
	c := newTestCMA(t, cmaConfig(uniformBounds(3, -1, 1)...), 1)
	a, _ := c.Ask()
	a[0][0] = 999
	if b, _ := c.Ask(); b[0][0] == 999 {
		t.Errorf("changing an asked sample changed cma-es's copy")
	}
}

func TestCMAESTellMisuse(t *testing.T) {
	c := newTestCMA(t, cmaConfig(uniformBounds(3, -1, 1)...), 1)
	if err := c.Tell(make([]float64, c.y)); err == nil {
		t.Errorf("tell before any ask should be an error")
	}
	xs, _ := c.Ask()
	if err := c.Tell(make([]float64, len(xs)-1)); err == nil {
		t.Errorf("telling the wrong number of fitnesses should be an error")
	}
	if err := c.Tell(make([]float64, len(xs))); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := c.Tell(make([]float64, len(xs))); err == nil {
		t.Errorf("telling twice for one ask should be an error")
	}
}

func TestCMAESBest(t *testing.T) {
	c := newTestCMA(t, cmaConfig(uniformBounds(3, -1, 1)...), 1)
	if w, f := c.Best(); len(w) != 0 || !math.IsInf(f, -1) {
		t.Errorf("nothing told yet, best should be empty and -Inf, got %v and %f", w, f)
	}

	xs, _ := c.Ask()
	fit := make([]float64, len(xs))
	fit[3] = 10
	if err := c.Tell(fit); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	w, f := c.Best()
	if f != 10 || !slices.Equal(w, xs[3]) {
		t.Errorf("best should be the sample told 10: got %f with %v, wanted %v", f, w, xs[3])
	}

	xs, _ = c.Ask()
	low := make([]float64, len(xs))
	for i := range low {
		low[i] = -5
	}
	if err := c.Tell(low); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if w2, f2 := c.Best(); f2 != 10 || !slices.Equal(w2, w) {
		t.Errorf("a worse generation replaced the best: got %f", f2)
	}

	w[0] = 999
	if w3, _ := c.Best(); w3[0] == 999 {
		t.Errorf("changing the returned best weights changed cma-es's copy")
	}
}

func TestReflect01(t *testing.T) {
	tests := []struct{ in, want float64 }{
		{0.3, 0.3}, {0, 0}, {1, 1}, {1.2, 0.8}, {-0.3, 0.3}, {2.4, 0.4}, {-1.7, 0.3}, {2, 0}, {3.5, 0.5}, {-2, 0},
	}
	for _, tt := range tests {
		if got := reflect01(tt.in); math.Abs(got-tt.want) > 1e-12 {
			t.Errorf("reflect01(%f): expected %f, got %f", tt.in, tt.want, got)
		}
	}
	for i := -300; i <= 300; i++ {
		if v := reflect01(float64(i) / 37); v < 0 || v > 1 {
			t.Fatalf("reflect01(%f) = %f, outside [0,1]", float64(i)/37, v)
		}
	}
}

func TestCMAESSphere(t *testing.T) {
	c := newTestCMA(t, cmaConfig(uniformBounds(10, -5, 5)...), 1)
	sphere := func(x []float64) float64 {
		s := 0.0
		for _, v := range x {
			s += (v - 1.3) * (v - 1.3)
		}
		return -s
	}
	runCMA(t, c, sphere, 400, math.Inf(1))
	if _, f := c.Best(); f < -1e-6 {
		t.Errorf("400 generations on a 10d sphere should get within 1e-6, got %g", f)
	}
}

// the case normalizing fixes: one sigma used to be 60 in real units, which kept the small weight on its edges
func TestCMAESMixedScales(t *testing.T) {
	c := newTestCMA(t, cmaConfig([2]float64{-1, 1}, [2]float64{-100, 100}), 1)
	f := func(x []float64) float64 {
		b := (x[1] - 50) / 100
		return -(x[0]*x[0] + b*b)
	}
	runCMA(t, c, f, 150, math.Inf(1))
	w, _ := c.Best()
	if math.Abs(w[0]) > 0.01 || math.Abs(w[1]-50) > 1 {
		t.Errorf("expected the best near (0, 50), got %v", w)
	}
}

// the case folding handles: the best value is right on the upper bound
func TestCMAESOptimumOnABound(t *testing.T) {
	c := newTestCMA(t, cmaConfig(uniformBounds(4, -5, 5)...), 1)
	f := func(x []float64) float64 {
		s := 0.0
		for _, v := range x {
			s += (v - 5) * (v - 5)
		}
		return -s
	}
	runCMA(t, c, f, 200, math.Inf(1))
	w, _ := c.Best()
	for i, v := range w {
		if v < 4.99 || v > 5 {
			t.Errorf("weight %d should end on the bound at 5, got %f", i, v)
		}
	}
}

func rotation(n int, r *rand.Rand) [][]float64 {
	rows := make([][]float64, n)
	for i := range rows {
		v := make([]float64, n)
		for j := range v {
			v[j] = r.NormFloat64()
		}
		for _, u := range rows[:i] {
			dot := 0.0
			for j := range v {
				dot += v[j] * u[j]
			}
			for j := range v {
				v[j] -= dot * u[j]
			}
		}
		norm := 0.0
		for _, x := range v {
			norm += x * x
		}
		norm = math.Sqrt(norm)
		for j := range v {
			v[j] /= norm
		}
		rows[i] = v
	}
	return rows
}

// learning C is the point of cma-es: a tilted valley should take about as long as an axis aligned one
func TestCMAESRotationInvariance(t *testing.T) {
	n := 5
	ellipsoid := func(rot [][]float64) func([]float64) float64 {
		return func(x []float64) float64 {
			s := 0.0
			for i := range n {
				z := x[i]
				if rot != nil {
					z = 0
					for j := range n {
						z += rot[i][j] * x[j]
					}
				}
				s += math.Pow(1e4, float64(i)/float64(n-1)) * z * z
			}
			return -s
		}
	}
	plain := runCMA(t, newTestCMA(t, cmaConfig(uniformBounds(n, -5, 5)...), 1), ellipsoid(nil), 2000, -1e-8)
	tilted := runCMA(t, newTestCMA(t, cmaConfig(uniformBounds(n, -5, 5)...), 1), ellipsoid(rotation(n, rand.New(rand.NewSource(3)))), 2000, -1e-8)
	if plain >= 2000 || tilted >= 2000 {
		t.Fatalf("never reached 1e-8: plain took %d, tilted %d generations", plain, tilted)
	}
	if float64(tilted) > 1.5*float64(plain) {
		t.Errorf("the tilted ellipsoid took %d generations against %d for the plain one, C isn't learning the rotation", tilted, plain)
	}
}

func TestCMAESSameSeedSameRun(t *testing.T) {
	run := func(seed int64) []float64 {
		c := newTestCMA(t, cmaConfig(uniformBounds(4, -5, 5)...), seed)
		runCMA(t, c, func(x []float64) float64 { return -x[0]*x[0] - x[1]*x[1] }, 30, math.Inf(1))
		w, _ := c.Best()
		return w
	}
	a, b := run(7), run(7)
	if !slices.Equal(a, b) {
		t.Errorf("same seed gave different runs: %v and %v", a, b)
	}
	if c := run(8); slices.Equal(a, c) {
		t.Errorf("different seeds gave the same run")
	}
}

func TestNewPicksTheOptimizer(t *testing.T) {
	ga := testConfig(3, 10)
	for _, name := range []string{"ga", "GA"} {
		ga.Optimizer = name
		o, err := New(ga, rand.New(rand.NewSource(1)))
		if _, ok := o.(*GA); err != nil || !ok {
			t.Errorf("optimizer %q should build the ga, got %T and %v", name, o, err)
		}
	}
	cma := cmaConfig(uniformBounds(3, -1, 1)...)
	for _, name := range []string{"cmaes", "CMAES"} {
		cma.Optimizer = name
		o, err := New(cma, rand.New(rand.NewSource(1)))
		if _, ok := o.(*CMAES); err != nil || !ok {
			t.Errorf("optimizer %q should build cma-es, got %T and %v", name, o, err)
		}
	}
	for _, name := range []string{"", "pso"} {
		cma.Optimizer = name
		if _, err := New(cma, rand.New(rand.NewSource(1))); err == nil {
			t.Errorf("optimizer %q should be an error", name)
		}
	}
}

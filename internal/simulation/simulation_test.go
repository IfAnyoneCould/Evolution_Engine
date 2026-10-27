package simulation

import (
	"Evolution_Engine/internal/api"
	"Evolution_Engine/internal/config"
	"Evolution_Engine/internal/optimizers"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"testing"
)

var simBin string

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

func testConfig(mode string, target float64, cycles int, agents int) config.Config {
	bounds := make([][2]float64, 3)
	for i := range bounds {
		bounds[i] = [2]float64{-5, 5}
	}
	return config.Config{
		Prog:   config.Program{Path: simBin, Args: []string{mode}},
		Bounds: bounds,
		Mutation: config.Mutation{
			Distribution: "uniform",
			Fraction:     0.05,
			MinNudge:     0.0001,
			Schedule:     config.Schedule{Type: "constant"},
		},
		RunSettings:      config.RunSettings{TargetFitness: target, MaxCycles: uint(cycles), PopulationSize: uint(agents)},
		Workers:          4,
		Selection:        config.Selection{Pressure: 0.45, Elite: 2},
		StagnationDetect: config.StagnationDetect{Patience: 1000, Epsilon: 0.01},
		SimSettings:      config.SimSettings{Timeout: 5000, RandSeed: 1},
		Optimizer:        "ga",
	}
}

type recorder struct {
	events []api.Event
}

func (r *recorder) Send(e api.Event) error {
	r.events = append(r.events, e)
	return nil
}

type failSender struct{}

func (failSender) Send(e api.Event) error {
	return errors.New("send failed")
}

func newTestSim(t *testing.T, cfg config.Config) *Simulation {
	t.Helper()
	s, _ := newRecordedSim(t, cfg)
	return s
}

func newRecordedSim(t *testing.T, cfg config.Config) (*Simulation, *recorder) {
	t.Helper()
	r := &recorder{}
	s, err := NewSimulation(cfg, r)
	if err != nil {
		t.Fatalf("could not build the simulation: %v", err)
	}
	return s, r
}

func runQuiet(t *testing.T, s *Simulation) error {
	t.Helper()
	return s.Run()
}

func sum(ws []float64) float64 {
	total := 0.0
	for _, w := range ws {
		total += w
	}
	return total
}

func best(s *Simulation) float64 {
	_, f := s.GetBestWeights()
	return f
}

func TestNewSimulation(t *testing.T) {
	cfg := testConfig("sum", 10, 25, 20)
	s := newTestSim(t, cfg)
	defer s.pool.CloseSims()

	ga, ok := s.opt.(*optimizers.GA)
	if !ok {
		t.Fatalf("expected the ga as the optimizer, got %T", s.opt)
	}
	if len(ga.Agents) != 20 {
		t.Errorf("incorrect agent count: expected %d, got %d", 20, len(ga.Agents))
	}
	if s.cycleMax != 25 {
		t.Errorf("incorrect cycle max: expected %d, got %d", 25, s.cycleMax)
	}
	if s.currentCycle != 0 {
		t.Errorf("incorrect starting cycle: expected %d, got %d", 0, s.currentCycle)
	}
	if s.targetFitness != 10 {
		t.Errorf("incorrect target fitness: expected %f, got %f", 10.0, s.targetFitness)
	}
	if s.stagnation != cfg.StagnationDetect {
		t.Errorf("incorrect stagnation detection: expected %+v, got %+v", cfg.StagnationDetect, s.stagnation)
	}
}

func TestNewSimulationBadProgram(t *testing.T) {
	cfg := testConfig("sum", 1, 10, 10)
	cfg.Prog.Path = filepath.Join(t.TempDir(), "not_a_program.exe")
	if _, err := NewSimulation(cfg, &recorder{}); err == nil {
		t.Errorf("expected error")
	}
}

func TestRunStopsAtTarget(t *testing.T) {
	s := newTestSim(t, testConfig("sum", 5, 200, 20))
	runQuiet(t, s)

	if best(s) < 5 {
		t.Errorf("run ended below its target: expected at least %f, got %f", 5.0, best(s))
	}
	if s.currentCycle >= 200 {
		t.Errorf("run used every cycle instead of stopping at the target: %d", s.currentCycle)
	}
}

func TestRunStopsAtCycleMax(t *testing.T) {
	s := newTestSim(t, testConfig("sum", 1000, 5, 20))
	runQuiet(t, s)

	if s.currentCycle != 5 {
		t.Errorf("incorrect cycle count: expected %d, got %d", 5, s.currentCycle)
	}
	if best(s) >= 1000 {
		t.Errorf("a target of %f should be unreachable, got %f", 1000.0, best(s))
	}
}

func TestRunImproves(t *testing.T) {
	s := newTestSim(t, testConfig("sum", 1000, 60, 20))
	runQuiet(t, s)

	if best(s) < 13 {
		t.Errorf("60 cycles barely moved the population: expected at least %f, got %f", 13.0, best(s))
	}
}

func TestRunWithFailingSim(t *testing.T) {
	s := newTestSim(t, testConfig("garbage", 5, 50, 20))
	if err := runQuiet(t, s); err == nil {
		t.Errorf("a failing sim should make Run return an error")
	}

	if s.currentCycle != 0 {
		t.Errorf("run kept going after the sim failed: got cycle %d", s.currentCycle)
	}
}

func TestGetBestWeights(t *testing.T) {
	s := newTestSim(t, testConfig("sum", 5, 200, 20))
	runQuiet(t, s)

	w, f := s.GetBestWeights()
	if len(w) != 3 {
		t.Errorf("incorrect weight count: expected %d, got %d", 3, len(w))
	}
	if math.Abs(sum(w)-f) > 1e-9 {
		t.Errorf("best weights don't belong to the best fitness: they sum to %f, fitness %f", sum(w), f)
	}
}

func TestWriteBestWeights(t *testing.T) {
	cfg := testConfig("sum", 5, 200, 20)
	cfg.Output.WeightPath = filepath.Join(t.TempDir(), "best.json")
	s := newTestSim(t, cfg)
	runQuiet(t, s)
	if err := s.WriteBestWeights(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	b, err := os.ReadFile(cfg.Output.WeightPath)
	if err != nil {
		t.Fatalf("no weights file written: %v", err)
	}
	var got []float64
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("weights file isn't a json array: %v", err)
	}
	if want, _ := s.GetBestWeights(); !slices.Equal(got, want) {
		t.Errorf("wrote %v, best weights are %v", got, want)
	}
}

func TestWriteBestWeightsNoPath(t *testing.T) {
	s := newTestSim(t, testConfig("sum", 5, 10, 10))
	runQuiet(t, s)
	if err := s.WriteBestWeights(); err != nil {
		t.Errorf("no weight path should write nothing and not error, got %v", err)
	}
}

func TestRunStopsOnStall(t *testing.T) {
	cfg := testConfig("len", 1000, 200, 20)
	cfg.StagnationDetect.Patience = 5
	s := newTestSim(t, cfg)
	runQuiet(t, s)

	if s.currentCycle < 5 || s.currentCycle > 6 {
		t.Errorf("a flat run should stop after about %d cycles, got %d", 5, s.currentCycle)
	}
}

func TestRunKeepsGoingWhileImproving(t *testing.T) {
	cfg := testConfig("count", 1e9, 20, 20)
	cfg.StagnationDetect.Patience = 2
	s := newTestSim(t, cfg)
	runQuiet(t, s)

	if s.currentCycle != 20 {
		t.Errorf("run stopped as stalled while fitness was still climbing: got cycle %d", s.currentCycle)
	}
}

func TestSameSeedSameRun(t *testing.T) {
	run := func(seed int64) []float64 {
		cfg := testConfig("sum", 1000, 15, 20)
		cfg.SimSettings.RandSeed = seed
		s := newTestSim(t, cfg)
		runQuiet(t, s)
		w, _ := s.GetBestWeights()
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

func TestRunWithCMAES(t *testing.T) {
	cfg := testConfig("sum", 14.5, 200, 20)
	cfg.Optimizer = "cmaes"
	cfg.CMAES.Sigma = 0.3
	s := newTestSim(t, cfg)
	if _, ok := s.opt.(*optimizers.CMAES); !ok {
		t.Fatalf("expected cma-es as the optimizer, got %T", s.opt)
	}
	if err := runQuiet(t, s); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	w, f := s.GetBestWeights()
	if f < 14.5 {
		t.Errorf("cma-es should reach the target of 14.5 out of 15, got %f", f)
	}
	if s.currentCycle >= 200 {
		t.Errorf("run used every cycle instead of stopping at the target")
	}
	if math.Abs(sum(w)-f) > 1e-9 {
		t.Errorf("best weights don't belong to the best fitness: they sum to %f, fitness %f", sum(w), f)
	}
}

func generations(r *recorder) []api.Generation {
	var gens []api.Generation
	for _, e := range r.events {
		if g, ok := e.(api.Generation); ok {
			gens = append(gens, g)
		}
	}
	return gens
}

func lastDone(t *testing.T, r *recorder) api.Done {
	t.Helper()
	if len(r.events) == 0 {
		t.Fatalf("no events sent")
	}
	d, ok := r.events[len(r.events)-1].(api.Done)
	if !ok {
		t.Fatalf("last event should be done, got %T", r.events[len(r.events)-1])
	}
	return d
}

func TestNewSimulationSendsStart(t *testing.T) {
	cfg := testConfig("sum", 5, 10, 10)
	cfg.SimSettings.RandSeed = 42
	s, r := newRecordedSim(t, cfg)
	defer s.pool.CloseSims()

	if len(r.events) != 1 {
		t.Fatalf("expected only the start event before Run, got %d events", len(r.events))
	}
	st, ok := r.events[0].(api.Start)
	if !ok {
		t.Fatalf("first event should be start, got %T", r.events[0])
	}
	if st.Seed != 42 || st.Optimizer != "ga" {
		t.Errorf("incorrect start event: got %+v", st)
	}
}

func TestRunEventOrder(t *testing.T) {
	s, r := newRecordedSim(t, testConfig("sum", 1000, 5, 20))
	if err := s.Run(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if _, ok := r.events[0].(api.Start); !ok {
		t.Errorf("first event should be start, got %T", r.events[0])
	}
	lastDone(t, r)
	gens := generations(r)
	if len(gens) != 5 {
		t.Fatalf("expected one generation event per cycle: expected %d, got %d", 5, len(gens))
	}
	if len(r.events) != len(gens)+2 {
		t.Errorf("expected only start, generations and done, got %d events", len(r.events))
	}
	for i, g := range gens {
		if g.Cycle != i {
			t.Errorf("generation %d has cycle %d", i, g.Cycle)
		}
	}
}

func TestRunGenerationEvents(t *testing.T) {
	s, r := newRecordedSim(t, testConfig("sum", 1000, 10, 20))
	if err := s.Run(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	prev := math.Inf(-1)
	for _, g := range generations(r) {
		if g.Min > g.Mean || g.Mean > g.Best {
			t.Errorf("cycle %d: expected min <= mean <= best, got %f %f %f", g.Cycle, g.Min, g.Mean, g.Best)
		}
		if g.Spread < 0 {
			t.Errorf("cycle %d: negative spread %f", g.Cycle, g.Spread)
		}
		if g.Best < prev {
			t.Errorf("cycle %d: best went down from %f to %f", g.Cycle, prev, g.Best)
		}
		prev = g.Best
		if len(g.BestGenome) != 3 {
			t.Errorf("cycle %d: expected %d genes, got %d", g.Cycle, 3, len(g.BestGenome))
		}
		if math.Abs(sum(g.BestGenome)-g.Best) > 1e-9 {
			t.Errorf("cycle %d: best genome doesn't belong to best, sums to %f, best %f", g.Cycle, sum(g.BestGenome), g.Best)
		}
		if g.BatchTime < 0 {
			t.Errorf("cycle %d: negative batch time %f", g.Cycle, g.BatchTime)
		}
	}
}

func TestRunDoneReason(t *testing.T) {
	stalled := testConfig("len", 1000, 200, 20)
	stalled.StagnationDetect.Patience = 5
	tests := []struct {
		name   string
		cfg    config.Config
		reason string
	}{
		{"target", testConfig("sum", 5, 200, 20), "target"},
		{"max cycles", testConfig("sum", 1000, 5, 20), "max_cycles"},
		{"stalled", stalled, "stalled"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, r := newRecordedSim(t, tt.cfg)
			if err := s.Run(); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			d := lastDone(t, r)
			if d.Reason != tt.reason {
				t.Errorf("incorrect reason: expected %q, got %q", tt.reason, d.Reason)
			}
		})
	}
}

func TestRunDoneMatchesRun(t *testing.T) {
	stalled := testConfig("len", 1000, 200, 20)
	stalled.StagnationDetect.Patience = 5
	tests := []struct {
		name string
		cfg  config.Config
	}{
		{"target", testConfig("sum", 5, 200, 20)},
		{"max cycles", testConfig("sum", 1000, 5, 20)},
		{"stalled", stalled},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, r := newRecordedSim(t, tt.cfg)
			if err := s.Run(); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			d := lastDone(t, r)
			w, f := s.GetBestWeights()
			if d.Best != f || !slices.Equal(d.BestGenome, w) {
				t.Errorf("done doesn't match the best: got %f %v, best is %f %v", d.Best, d.BestGenome, f, w)
			}
			gens := generations(r)
			if d.Cycles != len(gens) {
				t.Errorf("done cycles should match generations sent: expected %d, got %d", len(gens), d.Cycles)
			}
			if d.Best != gens[len(gens)-1].Best {
				t.Errorf("done best %f doesn't match the last generation %f", d.Best, gens[len(gens)-1].Best)
			}
			if d.Total < 0 {
				t.Errorf("negative total time %f", d.Total)
			}
		})
	}
}

func TestRunDoneStall(t *testing.T) {
	cfg := testConfig("len", 1000, 200, 20)
	cfg.StagnationDetect.Patience = 5
	s, r := newRecordedSim(t, cfg)
	if err := s.Run(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if d := lastDone(t, r); d.Stall != 5 {
		t.Errorf("incorrect stall count: expected %d, got %d", 5, d.Stall)
	}
}

func TestRunFailingSimNoDone(t *testing.T) {
	s, r := newRecordedSim(t, testConfig("garbage", 5, 50, 20))
	if err := s.Run(); err == nil {
		t.Fatalf("a failing sim should make Run return an error")
	}
	for _, e := range r.events {
		if _, ok := e.(api.Done); ok {
			t.Errorf("a failed run shouldn't send done")
		}
	}
}

func TestNewSimulationSendFails(t *testing.T) {
	if _, err := NewSimulation(testConfig("sum", 5, 10, 10), failSender{}); err == nil {
		t.Errorf("expected error when the start event can't be sent")
	}
}

func TestRunSendFails(t *testing.T) {
	s := newTestSim(t, testConfig("sum", 1000, 10, 20))
	s.sender = failSender{}
	if err := s.Run(); err == nil {
		t.Errorf("expected error when a generation can't be sent")
	}
	if s.currentCycle != 0 {
		t.Errorf("run kept going after a send failed: got cycle %d", s.currentCycle)
	}
}

func positiveConfig(mode string) config.Config {
	cfg := testConfig(mode, 1000, 10, 20)
	for i := range cfg.Bounds {
		cfg.Bounds[i] = [2]float64{0, 5}
	}
	return cfg
}

func checkSurvived(t *testing.T, s *Simulation, r *recorder) {
	t.Helper()
	if err := s.Run(); err != nil {
		t.Fatalf("failed evals shouldn't end the run: %v", err)
	}
	if d := lastDone(t, r); d.Reason != "max_cycles" {
		t.Errorf("incorrect done reason: expected %q, got %q", "max_cycles", d.Reason)
	}
	errs := 0
	for _, e := range r.events {
		if _, ok := e.(api.Error); ok {
			errs++
		}
	}
	if errs == 0 {
		t.Errorf("failed evals should send an error event")
	}
	for _, g := range generations(r) {
		for _, v := range []float64{g.Best, g.Mean, g.Min, g.Spread} {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("cycle %d has a non finite stat: %+v", g.Cycle, g)
			}
		}
	}
}

func TestRunSurvivesFailedEvals(t *testing.T) {
	tests := []struct {
		name string
		mode string
		opt  string
	}{
		{"ga with a sim that sometimes answers garbage", "flaky", "ga"},
		{"ga with a sim that keeps crashing", "crash", "ga"},
		{"cmaes with a sim that sometimes answers garbage", "flaky", "cmaes"},
		{"cmaes with a sim that keeps crashing", "crash", "cmaes"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := positiveConfig(tt.mode)
			cfg.Optimizer = tt.opt
			s, r := newRecordedSim(t, cfg)
			checkSurvived(t, s, r)
		})
	}
}

func TestRunFailedEvalNeverBest(t *testing.T) {
	s, r := newRecordedSim(t, positiveConfig("flaky"))
	if err := s.Run(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	d := lastDone(t, r)
	if math.Abs(d.Best-sum(d.BestGenome)) > 1e-9 {
		t.Errorf("best fitness didn't come from a real eval: genome scores %f, done says %f", sum(d.BestGenome), d.Best)
	}
}

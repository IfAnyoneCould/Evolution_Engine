package simulation

import (
	"Evolution_Engine/internal/config"
	"Evolution_Engine/internal/optimizers"
	"encoding/json"
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

func newTestSim(t *testing.T, cfg config.Config) *Simulation {
	t.Helper()
	s, err := NewSimulation(cfg)
	if err != nil {
		t.Fatalf("could not build the simulation: %v", err)
	}
	return s
}

func runQuiet(t *testing.T, s *Simulation) error {
	t.Helper()
	devnull, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		t.Fatalf("could not open %s: %v", os.DevNull, err)
	}
	old := os.Stdout
	os.Stdout = devnull
	defer func() {
		os.Stdout = old
		_ = devnull.Close()
	}()
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
	if _, err := NewSimulation(cfg); err == nil {
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

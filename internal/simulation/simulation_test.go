package simulation

import (
	"Evolution_Engine/internal/config"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"testing"
)

var simBin string

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

func testConfig(mode string, target float64, cycles int, agents int) config.JsonParams {
	bounds := make([][2]float64, 3)
	for i := range bounds {
		bounds[i] = [2]float64{-5, 5}
	}
	return config.JsonParams{
		Prog:             config.Program{Path: simBin, Args: []string{mode}},
		Bounds:           bounds,
		Fraction:         0.05,
		MinNudge:         0.0001,
		NudgeFunc:        config.NudgeFunc{Type: ptr("constant"), Hold: ptr(0.0), End: ptr(0.0)},
		RunSettings:      config.RunSettings{TargetFitness: target, MaxCycles: uint(cycles), PopulationSize: uint(agents)},
		Workers:          4,
		Selection:        config.Selection{Pressure: 0.45, Elite: 2},
		StagnationDetect: config.StagnationDetect{Patience: 1000, Epsilon: 0.01},
		SimSettings:      config.SimSettings{Timeout: 5000, RandSeed: ptr(int64(1))},
	}
}

func newTestSim(t *testing.T, cfg config.JsonParams) *Simulation {
	t.Helper()
	s, err := NewSimulation(cfg)
	if err != nil {
		t.Fatalf("could not build the simulation: %v", err)
	}
	return s
}

func runQuiet(t *testing.T, s *Simulation) {
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
	s.Run()
}

func TestNewSimulation(t *testing.T) {
	cfg := testConfig("sum", 10, 25, 20)
	s := newTestSim(t, cfg)
	defer s.Pop.CloseSims()

	if len(s.Pop.Agents) != 20 {
		t.Errorf("incorrect agent count: expected %d, got %d", 20, len(s.Pop.Agents))
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
	if s.newGenConfig != cfg.Selection {
		t.Errorf("incorrect selection: expected %+v, got %+v", cfg.Selection, s.newGenConfig)
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

	if s.Pop.TopFitness < 5 {
		t.Errorf("run ended below its target: expected at least %f, got %f", 5.0, s.Pop.TopFitness)
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
	if s.Pop.TopFitness >= 1000 {
		t.Errorf("a target of %f should be unreachable, got %f", 1000.0, s.Pop.TopFitness)
	}
}

func TestRunImproves(t *testing.T) {
	s := newTestSim(t, testConfig("sum", 1000, 60, 20))
	runQuiet(t, s)

	if s.Pop.TopFitness < 13 {
		t.Errorf("60 cycles barely moved the population: expected at least %f, got %f", 13.0, s.Pop.TopFitness)
	}
}

func TestRunWithFailingSim(t *testing.T) {
	s := newTestSim(t, testConfig("garbage", 5, 50, 20))
	runQuiet(t, s)

	if s.currentCycle != 0 {
		t.Errorf("run kept going after the sim failed: got cycle %d", s.currentCycle)
	}
}

func TestGetBest(t *testing.T) {
	s := newTestSim(t, testConfig("sum", 5, 200, 20))
	runQuiet(t, s)

	best := s.GetBestAgent()
	if best.Fitness != s.Pop.TopFitness {
		t.Errorf("best agent is not the top of the population: expected %f, got %f", s.Pop.TopFitness, best.Fitness)
	}

	weights := s.GetBestWeights()
	gene := s.GetBestGenome()
	if len(weights) != 3 {
		t.Errorf("incorrect weight count: expected %d, got %d", 3, len(weights))
	}
	for i, w := range gene.GetWeights() {
		if w != weights[i] {
			t.Errorf("genome and weights disagree at param %d: %f and %f", i, w, weights[i])
		}
	}
	for i, w := range best.Gene.GetWeights() {
		if w != weights[i] {
			t.Errorf("best agent and best weights disagree at param %d: %f and %f", i, w, weights[i])
		}
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

func TestRunSetsStartFitness(t *testing.T) {
	cfg := testConfig("len", 1000, 3, 10)
	s := newTestSim(t, cfg)
	runQuiet(t, s)

	if s.Pop.StartFitness != 3 {
		t.Errorf("incorrect start fitness: expected the first generation's best of %f, got %f", 3.0, s.Pop.StartFitness)
	}
}

func TestSameSeedSameRun(t *testing.T) {
	run := func(seed int64) []float64 {
		cfg := testConfig("sum", 1000, 15, 20)
		cfg.SimSettings.RandSeed = ptr(seed)
		s := newTestSim(t, cfg)
		runQuiet(t, s)
		return s.GetBestWeights()
	}
	a, b := run(7), run(7)
	if !slices.Equal(a, b) {
		t.Errorf("same seed gave different runs: %v and %v", a, b)
	}
	if c := run(8); slices.Equal(a, c) {
		t.Errorf("different seeds gave the same run: %v", a)
	}
}

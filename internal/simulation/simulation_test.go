package simulation

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
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

func simConfig(t *testing.T, mode string, dims int) string {
	t.Helper()
	bounds := make([]string, 0, dims)
	for range dims {
		bounds = append(bounds, "[-5,5]")
	}
	body := fmt.Sprintf(`{"program":{"path":%q,"args":[%q]},"bounds":[%s]}`, simBin, mode, strings.Join(bounds, ","))
	path := filepath.Join(t.TempDir(), "sim.json")
	if err := os.WriteFile(path, []byte(body), 0644); err != nil {
		t.Fatalf("could not write the test config: %v", err)
	}
	return path
}

func newTestSim(t *testing.T, mode string, target float64, cycles int, agents int) *Simulation {
	t.Helper()
	s, err := NewSimulation(target, cycles, agents, simConfig(t, mode, 3))
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
	s := newTestSim(t, "sum", 10, 25, 20)
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
}

func TestNewSimulationBadConfig(t *testing.T) {
	if _, err := NewSimulation(1, 10, 10, filepath.Join(t.TempDir(), "missing.json")); err == nil {
		t.Errorf("expected error")
	}
}

func TestRunStopsAtTarget(t *testing.T) {
	s := newTestSim(t, "sum", 5, 200, 20)
	runQuiet(t, s)

	if s.Pop.TopFitness < 5 {
		t.Errorf("run ended below its target: expected at least %f, got %f", 5.0, s.Pop.TopFitness)
	}
	if s.currentCycle >= 200 {
		t.Errorf("run used every cycle instead of stopping at the target: %d", s.currentCycle)
	}
}

func TestRunStopsAtCycleMax(t *testing.T) {
	s := newTestSim(t, "sum", 1000, 5, 20)
	runQuiet(t, s)

	if s.currentCycle != 5 {
		t.Errorf("incorrect cycle count: expected %d, got %d", 5, s.currentCycle)
	}
	if s.Pop.TopFitness >= 1000 {
		t.Errorf("a target of %f should be unreachable, got %f", 1000.0, s.Pop.TopFitness)
	}
}

func TestRunImproves(t *testing.T) {
	s := newTestSim(t, "sum", 1000, 60, 20)
	runQuiet(t, s)

	if s.Pop.TopFitness < 13 {
		t.Errorf("60 cycles barely moved the population: expected at least %f, got %f", 13.0, s.Pop.TopFitness)
	}
}

func TestRunWithFailingSim(t *testing.T) {
	s := newTestSim(t, "garbage", 5, 50, 20)
	runQuiet(t, s)

	if s.currentCycle != 0 {
		t.Errorf("run kept going after the sim failed: got cycle %d", s.currentCycle)
	}
}

func TestGetBest(t *testing.T) {
	s := newTestSim(t, "sum", 5, 200, 20)
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

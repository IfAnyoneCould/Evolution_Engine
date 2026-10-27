package pool

import (
	"Evolution_Engine/internal/config"
	"context"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"testing"
	"time"
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

func testConfig(mode string, workers int) config.Config {
	return config.Config{
		Prog:        config.Program{Path: simBin, Args: []string{mode}},
		Workers:     uint(workers),
		SimSettings: config.SimSettings{Timeout: 5000},
	}
}

func newTestPool(t *testing.T, mode string, workers int) *Pool {
	t.Helper()
	p, err := NewPool(testConfig(mode, workers))
	if err != nil {
		t.Fatalf("could not start the pool: %v", err)
	}
	t.Cleanup(func() { _ = p.CloseSims() })
	return p
}

func vectors(n, dims int) [][]float64 {
	xs := make([][]float64, n)
	for i := range xs {
		xs[i] = make([]float64, dims)
		for j := range xs[i] {
			xs[i][j] = float64(i) + float64(j)/10
		}
	}
	return xs
}

func sum(ws []float64) float64 {
	total := 0.0
	for _, w := range ws {
		total += w
	}
	return total
}

func TestNewPool(t *testing.T) {
	p := newTestPool(t, "sum", 4)
	if len(p.procs) != 4 {
		t.Errorf("incorrect worker count: expected %d, got %d", 4, len(p.procs))
	}
	for i, proc := range p.procs {
		if proc == nil {
			t.Errorf("worker %d is nil", i)
		}
	}
}

func TestNewPoolBadProgram(t *testing.T) {
	cfg := testConfig("sum", 3)
	cfg.Prog.Path = filepath.Join(t.TempDir(), "not_a_program.exe")
	if _, err := NewPool(cfg); err == nil {
		t.Errorf("expected error")
	}
}

func TestEvaluate(t *testing.T) {
	tests := []struct {
		name    string
		vectors int
		workers int
	}{
		{"one worker", 6, 1},
		{"more vectors than workers", 50, 4},
		{"a worker per vector", 8, 8},
		{"more workers than vectors", 3, 8},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := newTestPool(t, "sum", tt.workers)
			xs := vectors(tt.vectors, 3)
			fit, err := p.Evaluate(context.Background(), xs)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(fit) != len(xs) {
				t.Fatalf("wrong number of fitnesses: expected %d, got %d", len(xs), len(fit))
			}
			for i := range xs {
				if math.Abs(fit[i]-sum(xs[i])) > 1e-9 {
					t.Errorf("fitness %d out of order or wrong: expected %f, got %f", i, sum(xs[i]), fit[i])
				}
			}
		})
	}
}

func TestEvaluateNothing(t *testing.T) {
	p := newTestPool(t, "sum", 2)
	fit, err := p.Evaluate(context.Background(), nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(fit) != 0 {
		t.Errorf("expected no fitnesses, got %v", fit)
	}
}

func TestEvaluateKeepsProcessesAlive(t *testing.T) {
	p := newTestPool(t, "count", 1)
	for i := 1; i <= 3; i++ {
		fit, err := p.Evaluate(context.Background(), vectors(1, 2))
		if err != nil {
			t.Fatalf("evaluate %d failed: %v", i, err)
		}
		if fit[0] != float64(i) {
			t.Fatalf("process restarted between evaluates: expected %d, got %f", i, fit[0])
		}
	}
}

func TestEvaluateError(t *testing.T) {
	p := newTestPool(t, "garbage", 2)
	if _, err := p.Evaluate(context.Background(), vectors(6, 2)); err == nil {
		t.Errorf("expected error")
	}
}

func TestCloseSims(t *testing.T) {
	p, err := NewPool(testConfig("sum", 2))
	if err != nil {
		t.Fatalf("could not start the pool: %v", err)
	}
	if err := p.CloseSims(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if _, err := p.Evaluate(context.Background(), vectors(4, 2)); err == nil {
		t.Errorf("expected an error evaluating against closed sims")
	}
}

func checkFailed(t *testing.T, xs [][]float64, fit []float64, failed []int) {
	t.Helper()
	low := math.Inf(1)
	for i := range xs {
		if !slices.Contains(failed, i) {
			if math.Abs(fit[i]-sum(xs[i])) > 1e-9 {
				t.Errorf("fitness %d should have evaluated: expected %f, got %f", i, sum(xs[i]), fit[i])
			}
			low = math.Min(low, fit[i])
		}
	}
	for _, i := range failed {
		if math.IsNaN(fit[i]) || math.IsInf(fit[i], 0) {
			t.Errorf("failed eval %d should score a finite worst, got %f", i, fit[i])
		}
		if fit[i] > low {
			t.Errorf("failed eval %d scored %f, better than a real eval at %f", i, fit[i], low)
		}
	}
}

func TestEvaluateFailedScoresWorst(t *testing.T) {
	p := newTestPool(t, "flaky", 1)
	xs := vectors(9, 2)
	fit, err := p.Evaluate(context.Background(), xs)
	if err == nil {
		t.Errorf("failed evals should still be reported")
	}
	if len(fit) != len(xs) {
		t.Fatalf("wrong number of fitnesses: expected %d, got %d", len(xs), len(fit))
	}
	checkFailed(t, xs, fit, []int{2, 5, 8})
}

func TestEvaluateRestartsCrashedWorker(t *testing.T) {
	p := newTestPool(t, "crash", 1)
	xs := vectors(6, 2)
	fit, _ := p.Evaluate(context.Background(), xs)
	if len(fit) != len(xs) {
		t.Fatalf("wrong number of fitnesses: expected %d, got %d", len(xs), len(fit))
	}
	checkFailed(t, xs, fit, []int{2, 5})

	xs = vectors(2, 2)
	fit, err := p.Evaluate(context.Background(), xs)
	if err != nil {
		t.Fatalf("worker wasn't restarted after the batch: %v", err)
	}
	checkFailed(t, xs, fit, nil)
}

func TestEvaluateRestartsHungWorker(t *testing.T) {
	cfg := testConfig("hang", 1)
	cfg.SimSettings.Timeout = 200
	p, err := NewPool(cfg)
	if err != nil {
		t.Fatalf("could not start the pool: %v", err)
	}
	t.Cleanup(func() { _ = p.CloseSims() })

	xs := vectors(3, 2)
	start := time.Now()
	fit, _ := p.Evaluate(context.Background(), xs)
	if time.Since(start) > 2*time.Second {
		t.Errorf("evaluate waited on the hung sim instead of killing it: took %v", time.Since(start))
	}
	if len(fit) != len(xs) {
		t.Fatalf("wrong number of fitnesses: expected %d, got %d", len(xs), len(fit))
	}
	checkFailed(t, xs, fit, []int{1})
}

func TestEvaluateWorkerAlwaysDies(t *testing.T) {
	p := newTestPool(t, "die", 2)
	xs := vectors(4, 2)
	fit, err := p.Evaluate(context.Background(), xs)
	if err == nil {
		t.Errorf("expected error")
	}
	for i, f := range fit {
		if math.IsNaN(f) || math.IsInf(f, 0) {
			t.Errorf("failed eval %d should score a finite worst, got %f", i, f)
		}
	}
}

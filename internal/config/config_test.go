package config

import (
	"Evolution_Engine/internal/genome"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func genBoundsArray(bounds ...float64) [][2]float64 {
	var result [][2]float64
	for i := 0; i < len(bounds); i += 2 {
		result = append(result, [2]float64{bounds[i], bounds[i+1]})
	}
	return result
}

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

func startSim(t *testing.T, mode string) *SimProcess {
	t.Helper()
	sim, err := NewSimProcess(simBin, []string{mode})
	if err != nil {
		t.Fatalf("could not start the test sim: %v", err)
	}
	return sim
}

func genomeWith(t *testing.T, weights ...float64) *genome.Genome {
	t.Helper()
	g := genome.NewGenome(len(weights))
	if err := g.SetBoundsUniform(-1000, 1000); err != nil {
		t.Fatalf("could not set bounds: %v", err)
	}
	for i, w := range weights {
		if err := g.SetWeight(i, w); err != nil {
			t.Fatalf("could not set weight %d: %v", i, err)
		}
	}
	return g
}

func TestParseInputFile(t *testing.T) {
	tests := []struct {
		Name         string
		Path         string
		WantBound    [][2]float64
		WantProg     *ProgramBin
		WantNudge    float64
		WantMinNudge float64
		WantErr      bool
	}{
		{
			"correct input file",
			"config_test1.json",
			genBoundsArray(-1, 1, -1, 1, -1, 1, 0, 0, 2, 2, -10, 100),
			NewProgram("test_program.exe", []string{}),
			0.1,
			0.000001,
			false,
		},
		{
			"no bounds",
			"config_test2.json",
			genBoundsArray(-1, 1),
			NewProgram("test_program.exe", []string{}),
			0.1,
			0.000001,
			true,
		},
		{
			"no nudge",
			"config_test3.json",
			genBoundsArray(-1, 1, -1, 1, -1, 1, 0, 0, 2, 2, -10, 100),
			NewProgram("test_program.exe", []string{}),
			0.05,
			0.000001,
			false,
		},
		{
			"program with Args",
			"config_test4.json",
			genBoundsArray(-1, 1, 0, 5),
			NewProgram("test_program.exe", []string{"-v", "--seed", "42"}),
			0.1,
			0.000001,
			false,
		},
		{
			"nonexistent file",
			"does_not_exist.json",
			nil,
			nil,
			-1,
			-1,
			true,
		},
		{
			"malformed json",
			"config_test5.json",
			nil,
			nil,
			-1,
			-1,
			true,
		},
		{
			"empty bounds array",
			"config_test6.json",
			nil,
			nil,
			-1,
			-1,
			true,
		},
		{
			"min nudge set",
			"config_test8.json",
			genBoundsArray(-1, 1, -1, 1),
			NewProgram("test_program.exe", []string{}),
			0.2,
			0.01,
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			bounds, prog, nudge, minNudge, nudgeFunc, err := ParseInputFile(tt.Path)
			if tt.WantErr {
				if err == nil {
					t.Errorf("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !slices.Equal(bounds, tt.WantBound) {
				t.Errorf("incorrect bounds")
			}
			if !slices.Equal(prog.Args, tt.WantProg.Args) {
				t.Errorf("program incorrect Args")
			}
			if prog.Path != tt.WantProg.Path {
				t.Errorf("program incorrect Path: expected %s, got %s", tt.WantProg.Path, prog.Path)
			}
			if nudge != tt.WantNudge {
				t.Errorf("incorrect nudge: wanted %f, got %f", tt.WantNudge, nudge)
			}
			if minNudge != tt.WantMinNudge {
				t.Errorf("incorrect min nudge: wanted %f, got %f", tt.WantMinNudge, minNudge)
			}
			if nudgeFunc == nil {
				t.Errorf("no nudge function returned")
			}
		})
	}
}

func TestParseNudgeFunc(t *testing.T) {
	tests := []struct {
		Name     string
		Path     string
		Progress float64
		WantVal  float64
		WantErr  bool
	}{
		{"no nudge func defaults to constant 1", "config_test1.json", 0.5, 1, false},
		{"constant", "config_test8.json", 0.9, 0.5, false},
		{"linear, type is case insensitive", "config_test9.json", 0.25, 0.75, false},
		{"quadratic ignores extra params", "config_test10.json", 1, 1, false},
		{"unknown type", "config_test11.json", 0, 0, true},
		{"no params", "config_test12.json", 0, 0, true},
		{"no type", "config_test13.json", 0, 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			_, _, _, _, nudgeFunc, err := ParseInputFile(tt.Path)
			if tt.WantErr {
				if err == nil {
					t.Errorf("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			got := nudgeFunc.Get(tt.Progress)
			if math.Abs(got-tt.WantVal) > 1e-9 {
				t.Errorf("incorrect value at progress %f: wanted %f, got %f", tt.Progress, tt.WantVal, got)
			}
		})
	}
}

func TestParseInputFileArgCount(t *testing.T) {
	if _, _, _, _, _, err := ParseInputFile(); err == nil {
		t.Errorf("expected an error, the default path does not exist from here")
	}
	if _, _, _, _, _, err := ParseInputFile("config_test1.json", "config_test3.json"); err == nil {
		t.Errorf("expected an error for two paths")
	}
}

func TestSimProcessEval(t *testing.T) {
	tests := []struct {
		Name    string
		Weights []float64
		Want    float64
	}{
		{"positive", []float64{1, 2, 3}, 6},
		{"mixed signs", []float64{-5, 2.5, 2.5}, 0},
		{"single weight", []float64{42}, 42},
	}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			sim := startSim(t, "sum")
			defer sim.Close()

			got, err := sim.Eval(genomeWith(t, tt.Weights...))
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if math.Abs(got-tt.Want) > 1e-9 {
				t.Errorf("incorrect fitness: wanted %f, got %f", tt.Want, got)
			}
		})
	}
}

func TestSimProcessStaysAlive(t *testing.T) {
	sim := startSim(t, "count")
	defer sim.Close()

	g := genomeWith(t, 1, 2)
	for i := 1; i <= 5; i++ {
		got, err := sim.Eval(g)
		if err != nil {
			t.Fatalf("eval %d failed: %v", i, err)
		}
		if got != float64(i) {
			t.Fatalf("process restarted between evals: wanted %d, got %f", i, got)
		}
	}
}

func TestSimProcessSendsEveryWeight(t *testing.T) {
	sim := startSim(t, "len")
	defer sim.Close()

	got, err := sim.Eval(genomeWith(t, 1, 2, 3, 4, 5, 6, 7))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != 7 {
		t.Errorf("wrong number of weights reached the sim: wanted %d, got %f", 7, got)
	}
}

func TestNewSimProcessBadPath(t *testing.T) {
	if _, err := NewSimProcess(filepath.Join(t.TempDir(), "not_a_program.exe"), []string{}); err == nil {
		t.Errorf("expected error")
	}
}

func TestSimProcessEvalErrors(t *testing.T) {
	tests := []struct {
		Name string
		Mode string
	}{
		{"sim prints something that is not a number", "garbage"},
		{"sim exits mid run", "die"},
	}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			sim := startSim(t, tt.Mode)
			defer sim.Close()

			if _, err := sim.Eval(genomeWith(t, 1, 2)); err == nil {
				t.Errorf("expected error")
			}
		})
	}
}

func TestSimProcessClose(t *testing.T) {
	sim := startSim(t, "sum")
	if _, err := sim.Eval(genomeWith(t, 1, 1)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := sim.Close(); err != nil {
		t.Fatalf("close failed: %v", err)
	}
	if _, err := sim.Eval(genomeWith(t, 1, 1)); err == nil {
		t.Errorf("expected an error evaluating against a closed sim")
	}
}

func TestSimProcessCloseReportsExit(t *testing.T) {
	sim := startSim(t, "startfail")
	err := sim.Close()
	if err == nil {
		t.Fatalf("expected an error from a sim that exited nonzero")
	}
	if !strings.Contains(err.Error(), "exit status") && !strings.Contains(err.Error(), "file already closed") {
		t.Errorf("unexpected error: %v", err)
	}
}

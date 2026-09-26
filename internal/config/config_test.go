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
	sim, err := NewSimProcess(simBin, []string{mode}, 5000)
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

func writeConfig(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "cfg.json")
	if err := os.WriteFile(path, []byte(body), 0644); err != nil {
		t.Fatalf("could not write the test config: %v", err)
	}
	return path
}

func TestLoad(t *testing.T) {
	tests := []struct {
		Name         string
		Path         string
		WantBound    [][2]float64
		WantProg     Program
		WantFraction float64
		WantMinNudge float64
		WantErr      bool
	}{
		{
			"correct input file",
			"config_test1.json",
			genBoundsArray(-1, 1, -1, 1, -1, 1, 0, 0, 2, 2, -10, 100),
			Program{"test_program.exe", []string{}},
			0.1,
			0.0001,
			false,
		},
		{
			"no bounds",
			"config_test2.json",
			nil,
			Program{},
			-1,
			-1,
			true,
		},
		{
			"no fraction",
			"config_test3.json",
			genBoundsArray(-1, 1, -1, 1, -1, 1, 0, 0, 2, 2, -10, 100),
			Program{"test_program.exe", []string{}},
			0.05,
			0.0001,
			false,
		},
		{
			"program with Args",
			"config_test4.json",
			genBoundsArray(-1, 1, 0, 5),
			Program{"test_program.exe", []string{"-v", "--seed", "42"}},
			0.1,
			0.0001,
			false,
		},
		{
			"nonexistent file",
			"does_not_exist.json",
			nil,
			Program{},
			-1,
			-1,
			true,
		},
		{
			"malformed json",
			"config_test5.json",
			nil,
			Program{},
			-1,
			-1,
			true,
		},
		{
			"empty bounds array",
			"config_test6.json",
			nil,
			Program{},
			-1,
			-1,
			true,
		},
		{
			"min nudge set",
			"config_test8.json",
			genBoundsArray(-1, 1, -1, 1),
			Program{"test_program.exe", []string{}},
			0.2,
			0.01,
			false,
		},
		{
			"explicit zeros are kept, not replaced by defaults",
			"config_test14.json",
			genBoundsArray(-1, 1),
			Program{"test_program.exe", []string{}},
			0,
			0,
			false,
		},
		{
			"no program path",
			"config_test15.json",
			nil,
			Program{},
			-1,
			-1,
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			cfg, err := Load(tt.Path)
			if tt.WantErr {
				if err == nil {
					t.Errorf("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !slices.Equal(cfg.Bounds, tt.WantBound) {
				t.Errorf("incorrect bounds")
			}
			if !slices.Equal(cfg.Prog.Args, tt.WantProg.Args) {
				t.Errorf("program incorrect Args")
			}
			if cfg.Prog.Path != tt.WantProg.Path {
				t.Errorf("program incorrect Path: expected %s, got %s", tt.WantProg.Path, cfg.Prog.Path)
			}
			if cfg.Fraction != tt.WantFraction {
				t.Errorf("incorrect fraction: wanted %f, got %f", tt.WantFraction, cfg.Fraction)
			}
			if cfg.MinNudge != tt.WantMinNudge {
				t.Errorf("incorrect min nudge: wanted %f, got %f", tt.WantMinNudge, cfg.MinNudge)
			}
		})
	}
}

func TestLoadNudgeFunc(t *testing.T) {
	tests := []struct {
		Name      string
		Path      string
		WantType  string
		WantParam []float64
		WantErr   bool
	}{
		{"no nudge func defaults to constant 1", "config_test1.json", "constant", []float64{1}, false},
		{"constant", "config_test8.json", "constant", []float64{0.5}, false},
		{"type is case insensitive", "config_test9.json", "linear", []float64{1}, false},
		{"extra params are kept", "config_test10.json", "quadratic", []float64{2, 9}, false},
		{"unknown type", "config_test11.json", "", nil, true},
		{"no params", "config_test12.json", "", nil, true},
		{"no type defaults to constant", "config_test13.json", "constant", []float64{1}, false},
	}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			cfg, err := Load(tt.Path)
			if tt.WantErr {
				if err == nil {
					t.Errorf("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !strings.EqualFold(cfg.NudgeFunc.Type, tt.WantType) {
				t.Errorf("incorrect type: wanted %s, got %s", tt.WantType, cfg.NudgeFunc.Type)
			}
			if !slices.Equal(cfg.NudgeFunc.Param, tt.WantParam) {
				t.Errorf("incorrect params: wanted %v, got %v", tt.WantParam, cfg.NudgeFunc.Param)
			}
		})
	}
}

func TestLoadValidation(t *testing.T) {
	base := `{"program":{"path":"test_program.exe"},"bounds":[[-1,1]]`
	tests := []struct {
		Name    string
		Extra   string
		WantErr bool
	}{
		{"defaults are valid", ``, false},
		{"no workers", `,"workers":0`, true},
		{"negative workers", `,"workers":-1`, true},
		{"no population", `,"run_settings":{"population_size":0}`, true},
		{"elite equal to population", `,"run_settings":{"population_size":5},"selection":{"elite":5}`, true},
		{"elite under population", `,"run_settings":{"population_size":5},"selection":{"elite":4}`, false},
		{"no pressure", `,"selection":{"pressure":0}`, true},
		{"everything survives", `,"selection":{"pressure":1}`, false},
		{"pressure above 1", `,"selection":{"pressure":1.5}`, true},
		{"target of 0", `,"run_settings":{"target_fitness":0}`, true},
		{"target of 1", `,"run_settings":{"target_fitness":1}`, false},
		{"target above 1", `,"run_settings":{"target_fitness":1.5}`, true},
		{"no patience", `,"stagnation_detection":{"patience":0}`, true},
		{"no epsilon", `,"stagnation_detection":{"epsilon":0}`, true},
		{"negative epsilon", `,"stagnation_detection":{"epsilon":-0.1}`, true},
		{"no timeout", `,"sim_settings":{"timeout_ms":0}`, true},
		{"nudge type in caps", `,"nudge_func":{"type":"QUADRATIC","params":[2]}`, false},
	}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			_, err := Load(writeConfig(t, base+tt.Extra+`}`))
			if tt.WantErr && err == nil {
				t.Errorf("expected error")
			}
			if !tt.WantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestLoadPartialSection(t *testing.T) {
	cfg, err := Load(writeConfig(t, `{"program":{"path":"test_program.exe"},"bounds":[[-1,1]],"run_settings":{"max_cycles":10}}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.RunSettings.MaxCycles != 10 {
		t.Errorf("incorrect max cycles: wanted %d, got %d", 10, cfg.RunSettings.MaxCycles)
	}
	if cfg.RunSettings.PopulationSize != 100 {
		t.Errorf("setting one field wiped the default population size: got %d", cfg.RunSettings.PopulationSize)
	}
	if cfg.RunSettings.TargetFitness != 0.99 {
		t.Errorf("setting one field wiped the default target: got %f", cfg.RunSettings.TargetFitness)
	}
}

func TestLoadAllSettings(t *testing.T) {
	cfg, err := Load(writeConfig(t, `{
		"program":{"path":"test_program.exe"},
		"bounds":[[-1,1]],
		"workers":3,
		"run_settings":{"target_fitness":0.5,"max_cycles":7,"population_size":30},
		"selection":{"pressure":0.3,"elite":4},
		"stagnation_detection":{"patience":12,"epsilon":0.002},
		"sim_settings":{"timeout_ms":250},
		"output":{"weight_path":"out.json"}
	}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := struct {
		Workers    uint
		Run        RunSettings
		Selection  Selection
		Stagnation StagnationDetect
		Timeout    float64
		WeightPath string
	}{3, RunSettings{0.5, 7, 30}, Selection{0.3, 4}, StagnationDetect{12, 0.002}, 250, "out.json"}

	if cfg.Workers != want.Workers {
		t.Errorf("incorrect workers: wanted %d, got %d", want.Workers, cfg.Workers)
	}
	if cfg.RunSettings != want.Run {
		t.Errorf("incorrect run settings: wanted %+v, got %+v", want.Run, cfg.RunSettings)
	}
	if cfg.Selection != want.Selection {
		t.Errorf("incorrect selection: wanted %+v, got %+v", want.Selection, cfg.Selection)
	}
	if cfg.StagnationDetect != want.Stagnation {
		t.Errorf("incorrect stagnation detection: wanted %+v, got %+v", want.Stagnation, cfg.StagnationDetect)
	}
	if cfg.SimSettings.Timeout != want.Timeout {
		t.Errorf("incorrect timeout: wanted %f, got %f", want.Timeout, cfg.SimSettings.Timeout)
	}
	if cfg.Output.WeightPath != want.WeightPath {
		t.Errorf("incorrect weight path: wanted %s, got %s", want.WeightPath, cfg.Output.WeightPath)
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
	if _, err := NewSimProcess(filepath.Join(t.TempDir(), "not_a_program.exe"), []string{}, 5000); err == nil {
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

func TestSimProcessTimeout(t *testing.T) {
	tests := []struct {
		Name    string
		Timeout float64
		WantErr bool
	}{
		{"sim answers inside the timeout", 5000, false},
		{"sim is slower than the timeout", 50, true},
	}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			sim, err := NewSimProcess(simBin, []string{"slow"}, tt.Timeout)
			if err != nil {
				t.Fatalf("could not start the test sim: %v", err)
			}
			defer sim.Close()

			_, err = sim.Eval(genomeWith(t, 1, 2))
			if tt.WantErr && err == nil {
				t.Errorf("expected a timeout error")
			}
			if !tt.WantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
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

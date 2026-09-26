package config

import (
	"Evolution_Engine/internal/genome"
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"slices"
	"strconv"
	"strings"
	"time"
)

type ProgramBin struct {
	Path string
	Args []string
}

func NewProgram(path string, args []string) *ProgramBin {
	return &ProgramBin{path, args}
}

type SimProcess struct {
	Proc    *exec.Cmd
	stdin   io.WriteCloser
	stdout  *bufio.Reader
	timeout time.Duration
}

func NewSimProcess(path string, args []string, timeout float64) (*SimProcess, error) {
	cmd := exec.Command(path, args...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		stdin.Close()
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		stdin.Close()
		return nil, err
	}
	if err = cmd.Start(); err != nil {
		stdin.Close()
		return nil, err
	}
	return &SimProcess{cmd, stdin, bufio.NewReader(stdout), time.Duration(timeout * float64(time.Millisecond))}, nil
}

func (s *SimProcess) Eval(g *genome.Genome) (float64, error) {
	weights := g.GetWeights()
	jsonBytes, _ := json.Marshal(weights)

	if _, err := s.stdin.Write(append(jsonBytes, '\n')); err != nil {
		return -1, err
	}

	type result struct {
		line string
		err  error
	}

	ch := make(chan result, 1)

	go func() {
		resp, err := s.stdout.ReadString('\n')
		ch <- result{resp, err}
	}()

	select {
	case r := <-ch:
		resp, err := r.line, r.err
		if err != nil {
			return math.Inf(-1), err
		}
		return strconv.ParseFloat(strings.TrimSpace(resp), 64)
	case <-time.After(s.timeout):
		s.Close() //TODO think about making simulations more robust, so they don't die on error. That robustness level should be handled in the config too.
		return math.Inf(-1), fmt.Errorf("reading from simulation timed out, simulation may have hanged")
	}
}

func (s *SimProcess) Close() error {
	if err := s.stdin.Close(); err != nil {
		return err
	}
	return s.Proc.Wait()
}

type Program struct {
	Path string   `json:"path"`
	Args []string `json:"args"`
}

type NudgeFunc struct {
	Type     *string  `json:"type"`
	Hold     *float64 `json:"hold"`
	End      *float64 `json:"end"`
	Exponent *float64 `json:"exponent"`
	Rate     *float64 `json:"rate"`
	Steps    *uint    `json:"steps"`
}

type RunSettings struct {
	TargetFitness  float64 `json:"target_fitness"`
	MaxCycles      uint    `json:"max_cycles"`
	PopulationSize uint    `json:"population_size"`
}

type Selection struct {
	Pressure float64 `json:"pressure"`
	Elite    uint    `json:"elite"`
}

type StagnationDetect struct {
	Patience uint    `json:"patience"`
	Epsilon  float64 `json:"epsilon"`
}

type SimSettings struct {
	Timeout  float64 `json:"timeout_ms"`
	RandSeed *int64  `json:"rand_seed"`
	//TODO add robustness settings later, such as retry count, etc.
}

type Output struct {
	WeightPath string `json:"weight_path"`
	//TODO add history control and pathing
}

type JsonParams struct {
	Prog             Program          `json:"program"`
	Bounds           [][2]float64     `json:"bounds"`
	Fraction         float64          `json:"fraction"`
	MinNudge         float64          `json:"min_nudge"`
	NudgeFunc        NudgeFunc        `json:"nudge_func"`
	RunSettings      RunSettings      `json:"run_settings"`
	Workers          uint             `json:"workers"`
	Selection        Selection        `json:"selection"`
	StagnationDetect StagnationDetect `json:"stagnation_detection"`
	SimSettings      SimSettings      `json:"sim_settings"`
	Output           Output           `json:"output"`
}

func defaults() JsonParams {
	prog := Program{
		Path: "",
		Args: []string{},
	}
	nudgeFunc := NudgeFunc{}
	runSettings := RunSettings{
		TargetFitness:  0.99,
		MaxCycles:      500,
		PopulationSize: 60,
	}
	selection := Selection{
		Pressure: 0.45,
		Elite:    1,
	}
	stagnant := StagnationDetect{
		Patience: 200,
		Epsilon:  0.01,
	}
	simSettings := SimSettings{
		Timeout: 5000,
	}
	output := Output{
		WeightPath: "",
	}

	return JsonParams{
		Prog:             prog,
		Bounds:           nil,
		Fraction:         0.05,
		MinNudge:         0.0005,
		NudgeFunc:        nudgeFunc,
		RunSettings:      runSettings,
		Workers:          10,
		Selection:        selection,
		StagnationDetect: stagnant,
		SimSettings:      simSettings,
		Output:           output,
	}

}

func (p JsonParams) validate() error {
	if p.Prog.Path == "" {
		return errors.New("config error: program.path required")
	}
	if len(p.Bounds) == 0 {
		return errors.New("config error: bounds is required")
	}
	if p.Workers <= 0 {
		return errors.New("config error: worker count needs to be above 0")
	}
	if p.RunSettings.PopulationSize <= 0 {
		return errors.New("config error: run_settings.population_size needs to be greater than 0")
	}
	if p.RunSettings.PopulationSize <= p.Selection.Elite {
		return errors.New("config error: selection.elite cannot be greater than or equal to run_settings.population_size")
	}
	if p.Selection.Pressure <= 0 || p.Selection.Pressure > 1 {
		return errors.New("config error: invalid range for selection.pressure, must be within (0,1]")
	}
	if p.RunSettings.TargetFitness <= 0 || p.RunSettings.TargetFitness > 1 {
		return errors.New("config error: run_settings.target_fitness outside of bounds (0,1]")
	}
	if p.StagnationDetect.Patience <= 0 {
		return errors.New("config error: stagnation_detection.patience must be greater than 0")
	}
	if p.StagnationDetect.Epsilon <= 0 {
		return errors.New("config error: stagnation_detection.epsilon must be greater than 0")
	}
	if p.SimSettings.Timeout <= 0 {
		return errors.New("config error: sim_settings.timeout must be greater than 0")
	}
	if float64(p.RunSettings.PopulationSize)*p.Selection.Pressure < 1 {
		return errors.New("config error: run_settings.population_size * selection.pressure cannot be less than 1")
	}
	if p.NudgeFunc.Type != nil {
		if !slices.Contains([]string{"constant", "quadratic", "linear", "power", "exponential", "cosine", "step"}, strings.ToLower(*p.NudgeFunc.Type)) {
			return errors.New("config error: nudge_func.type not recognized")
		}
		switch strings.ToLower(*p.NudgeFunc.Type) {
		case "exponential":
			{
				if p.NudgeFunc.Rate == nil {
					return errors.New("config error: if nudge_func.type is 'exponential', then nudge_func.rate is required")
				}
				if *p.NudgeFunc.Rate <= 0 {
					return errors.New("config error: nudge_func.rate must be greater than 0")
				}
			}
		case "step":
			{
				if p.NudgeFunc.Steps == nil {
					return errors.New("config error: if nudge_func.type is 'step', then nudge_func.steps is required")
				}
				if *p.NudgeFunc.Steps < 1 {
					return errors.New("config error: nudge_func.steps must be greater than 0")
				}
			}
		case "power":
			{
				if p.NudgeFunc.Exponent == nil {
					return errors.New("config error: if nudge_func.type is 'power', then nudge_func.exponent is required")
				}
				if *p.NudgeFunc.Exponent <= 0 {
					return errors.New("config error: nudge_func.exponent must be greater than 0")
				}
			}
		}
	}
	if p.NudgeFunc.End != nil && (*p.NudgeFunc.End < 0 || *p.NudgeFunc.End > 1) {
		return errors.New("config error: nudge_func.end must be in range [0,1]")
	}
	if p.NudgeFunc.Hold != nil && (*p.NudgeFunc.Hold < 0 || *p.NudgeFunc.Hold > 1) {
		return errors.New("config error: nudge_func.hold must be in range [0,1]")
	}
	return nil
}

func Load(path string) (JsonParams, error) {

	file, err := os.ReadFile(path)
	if err != nil {
		return JsonParams{}, err
	}

	cfg := defaults()

	if err = json.Unmarshal(file, &cfg); err != nil {
		return JsonParams{}, err
	}

	if cfg.NudgeFunc.Type == nil {
		t := "quadratic"
		h := 0.45
		e := 0.2
		cfg.NudgeFunc.Type = &t
		cfg.NudgeFunc.Hold = &h
		cfg.NudgeFunc.End = &e
	} else {
		if cfg.NudgeFunc.Hold == nil {
			h := float64(0)
			cfg.NudgeFunc.Hold = &h
		}
		if cfg.NudgeFunc.End == nil {
			e := float64(0)
			cfg.NudgeFunc.End = &e
		}
	}

	if err = cfg.validate(); err != nil {
		return JsonParams{}, err
	}

	if cfg.SimSettings.RandSeed == nil {
		s := time.Now().UnixMicro()
		cfg.SimSettings.RandSeed = &s
	}

	return cfg, nil
}

package config

import (
	"Evolution_Engine/internal/genome"
	"Evolution_Engine/internal/nudge"
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
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
	Proc   *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader
}

func NewSimProcess(path string, args []string) (*SimProcess, error) {
	cmd := exec.Command(path, args...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err = cmd.Start(); err != nil {
		return nil, err
	}
	return &SimProcess{cmd, stdin, bufio.NewReader(stdout)}, nil
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
	case <-time.After(5 * time.Second): // TODO make the duration customizable in the config
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

func ParseInputFile(p ...string) ([][2]float64, *ProgramBin, float64, float64, nudge.Function, error) { // TODO make this return a struct, dealing with this is annoying. Alternatively, just have it return the population, as that's the only place this function is ever used
	defaultPath := "data/simInfo.json"
	var path string
	switch len(p) {
	case 0:
		path = defaultPath
	case 1:
		path = p[0]
	default:
		err := fmt.Errorf("illegal number of arguments: expected 0 or 1, got %d", len(p))
		return nil, nil, -1, -1, nil, err
	}

	file, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, -1, -1, nil, err
	}

	var cfg struct { //TODO pull this out and make it a separate struct, though don't export it
		Prog struct {
			Path string   `json:"path"`
			Args []string `json:"args"`
		} `json:"program"`
		Bounds    [][2]float64 `json:"bounds"`
		Nudge     float64      `json:"nudge"`
		MinNudge  float64      `json:"min_nudge"`
		NudgeFunc *struct {
			Type  *string   `json:"type"`
			Param []float64 `json:"params"`
		} `json:"nudge_func"`
	}

	if err := json.Unmarshal(file, &cfg); err != nil {
		return nil, nil, -1, -1, nil, err
	}

	if len(cfg.Bounds) == 0 {
		return nil, nil, -1, -1, nil, fmt.Errorf("no bounds data, check config")
	}

	args := cfg.Prog.Args
	if args == nil {
		args = []string{}
	}

	if cfg.Nudge == 0 {
		cfg.Nudge = 0.05 // TODO make all defaults present in a default struct, so that i don't have to do this everytime I add a new parameter
	}

	if cfg.MinNudge == 0 {
		cfg.MinNudge = 0.000001 // TODO add this to config
	}

	var nudgeFunc nudge.Function
	nudgeFunc = nudge.NewConstantFunction(1)
	if cfg.NudgeFunc != nil {
		if cfg.NudgeFunc.Type == nil || len(cfg.NudgeFunc.Param) < 1 {
			return nil, nil, -1, -1, nil, fmt.Errorf("missing nudgeFunc data, check config")
		}
		switch strings.ToLower(*cfg.NudgeFunc.Type) {
		case "constant":
			nudgeFunc = nudge.NewConstantFunction(cfg.NudgeFunc.Param[0])
		case "linear":
			nudgeFunc = nudge.NewLinearFunction(cfg.NudgeFunc.Param[0])
		case "quadratic":
			nudgeFunc = nudge.NewQuadraticFunction(cfg.NudgeFunc.Param[0])
		default:
			return nil, nil, -1, -1, nil, fmt.Errorf("nudgeFunc type not recognize, check config")
		}
	}

	return cfg.Bounds, NewProgram(cfg.Prog.Path, args), cfg.Nudge, cfg.MinNudge, nudgeFunc, nil
}

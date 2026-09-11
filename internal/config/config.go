package config

import (
	"Evolution_Engine/internal/genome"
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"
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
		return nil, nil
	}
	return &SimProcess{cmd, stdin, bufio.NewReader(stdout)}, nil
}

func (s *SimProcess) Eval(g *genome.Genome) (float64, error) {
	weights := g.GetWeights()
	jsonBytes, _ := json.Marshal(weights)

	if _, err := s.stdin.Write(append(jsonBytes, '\n')); err != nil {
		return -1, err
	}

	resp, err := s.stdout.ReadString('\n')
	if err != nil {
		return -1, err
	}
	return strconv.ParseFloat(strings.TrimSpace(resp), 64)
}

func (s *SimProcess) Close() error {
	if err := s.stdin.Close(); err != nil {
		return err
	}
	return s.Proc.Wait()
}

func ParseInputFile(p ...string) ([][2]float64, *ProgramBin, float64, error) {
	defaultPath := "data/simInfo.json"
	var path string
	switch len(p) {
	case 0:
		path = defaultPath
	case 1:
		path = p[0]
	default:
		err := fmt.Errorf("illegal number of arguments: expected 0 or 1, got %d", len(p))
		return nil, nil, -1, err
	}

	file, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, -1, err
	}

	var cfg struct {
		Prog struct {
			Path string   `json:"Path"`
			Args []string `json:"Args"`
		} `json:"program"`
		Bounds [][2]float64 `json:"bounds"`
		Nudge  float64      `json:"nudge"`
	}

	if err := json.Unmarshal(file, &cfg); err != nil {
		return nil, nil, -1, err
	}

	if len(cfg.Bounds) == 0 {
		return nil, nil, -1, fmt.Errorf("no bounds data, check config")
	}

	if cfg.Nudge == 0 {
		return nil, nil, -1, fmt.Errorf("no nudge data, check config")
	}

	args := cfg.Prog.Args
	if args == nil {
		args = []string{}
	}

	return cfg.Bounds, NewProgram(cfg.Prog.Path, args), cfg.Nudge, nil
}

package config

import (
	"Evolution_Engine/internal/genome"
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

type ProgramBin struct {
	path string
	args []string
}

func NewProgram(path string, args []string) *ProgramBin {
	return &ProgramBin{path, args}
}

func (p *ProgramBin) Run(g *genome.Genome) (float64, error) {
	weights := g.GetWeights()
	jsonBytes, err := json.Marshal(weights)
	if err != nil {
		return -1, err
	}

	allArgs := []string{string(jsonBytes)}
	allArgs = append(allArgs, p.args...)

	cmd := exec.Command("./"+p.path, allArgs...)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err = cmd.Run()
	if err != nil {
		return -1, err
	}

	output := strings.TrimSpace(stdout.String())
	result, err := strconv.ParseFloat(output, 64)
	if err != nil {
		return -1, err
	}

	return result, nil
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
			Path string   `json:"path"`
			Args []string `json:"args"`
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

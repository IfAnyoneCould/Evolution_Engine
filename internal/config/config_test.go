package config

import (
	"Evolution_Engine/internal/genome"
	"bytes"
	"fmt"
	"os/exec"
	"slices"
	"testing"
)

// helper function for writing the tests
func genBoundsArray(bounds ...float64) [][2]float64 {
	var result [][2]float64
	for i := 0; i < len(bounds); i += 2 {
		result = append(result, [2]float64{bounds[i], bounds[i+1]})
	}
	return result
}
func TestParseInputFile(t *testing.T) {
	tests := []struct {
		Name      string
		Path      string
		WantBound [][2]float64
		WantProg  *ProgramBin
		WantNudge float64
		WantErr   bool
	}{
		{
			"correct input file",
			"config_test1.json",
			genBoundsArray(-1, 1, -1, 1, -1, 1, 0, 0, 2, 2, -10, 100),
			NewProgram("test_program.exe", []string{}),
			0.1,
			false,
		},
		{
			"no bounds",
			"config_test2.json",
			genBoundsArray(-1, 1),
			NewProgram("test_program.exe", []string{}),
			0.1,
			true,
		},
		{
			"no nudge",
			"config_test3.json",
			genBoundsArray(-1, 1, -1, 1, -1, 1, 0, 0, 2, 2, -10, 100),
			NewProgram("test_program.exe", []string{}),
			0.1,
			true,
		},
		{
			"program with args",
			"config_test4.json",
			genBoundsArray(-1, 1, 0, 5),
			NewProgram("test_program.exe", []string{"-v", "--seed", "42"}),
			0.1,
			false,
		},
		{
			"nonexistent file",
			"does_not_exist.json",
			nil,
			nil,
			-1,
			true,
		},
		{
			"malformed json",
			"config_test5.json",
			nil,
			nil,
			-1,
			true,
		},
		{
			"empty bounds array",
			"config_test6.json",
			nil,
			nil,
			-1,
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			bounds, prog, nudge, err := ParseInputFile(tt.Path)
			if tt.WantErr {
				if err == nil {
					t.Errorf("expected error")
				}
				return
			}
			if !slices.Equal(bounds, tt.WantBound) {
				t.Errorf("incorrect bounds")
			}
			if !slices.Equal(prog.args, tt.WantProg.args) {
				t.Errorf("program incorrect args")
			}
			if prog.path != tt.WantProg.path {
				t.Errorf("program incorrect path: expected %s, got %s", tt.WantProg.path, prog.path)
			}
			if nudge != tt.WantNudge {
				t.Errorf("incorrect nudge")
			}
		})
	}
}

func TestRun(t *testing.T) {
	want := 0.1799
	bounds, prog, _, _ := ParseInputFile("config_test7.json")
	g := genome.NewGenome(2)
	g.SetBounds(bounds...)
	result, err := prog.Run(g)
	if err != nil {
		t.Errorf("%s", err)
	}
	if result != want {
		t.Errorf("incorrect result: expected %f, got %f", want, result)
	}
}

func TestEXE(t *testing.T) {
	cmd := exec.Command("./TestRun.exe", "[0,0]")

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()

	if err != nil {
		t.Errorf("%s", err)
	}
	fmt.Println(stdout.String())
	fmt.Println(stderr.String())
}

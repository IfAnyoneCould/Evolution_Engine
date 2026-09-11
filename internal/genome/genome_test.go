package genome

import (
	"slices"
	"testing"
)

func TestBoundsUniform(t *testing.T) {
	tests := []struct {
		name    string
		a, b    float64
		wantErr bool
	}{
		{"standard", -1, 1, false},
		{"equal", 0, 0, false},
		{"incorrect", 1, -1, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := NewGenome(50)
			err := g.SetBoundsUniform(tt.a, tt.b)
			if tt.wantErr {
				if err == nil {
					t.Errorf("expected error")
				}
				return
			}
			for i, v := range g.Params {
				if v.Upper != tt.b {
					t.Errorf("upper bound at index %d incorrect. expected %f got %f", i, tt.b, v.Upper)
				}
				if v.Lower != tt.a {
					t.Errorf("lower bound at index %d incorrect. expected %f got %f", i, tt.a, v.Lower)
				}
			}
		})
	}
}

// helper function for writing the tests
func genBoundsArray(bounds ...float64) [][2]float64 {
	var result [][2]float64
	for i := 0; i < len(bounds); i += 2 {
		result = append(result, [2]float64{bounds[i], bounds[i+1]})
	}
	return result
}

func TestBounds(t *testing.T) {
	tests := []struct {
		name    string
		bounds  [][2]float64
		wantErr bool
	}{
		{"standard", genBoundsArray(-1, 1, -1, 1, -2, 2, 0, 2, 3, 3), false},
		{"wrong length", genBoundsArray(-1, 1, 1, 1, 2, 2), true},
		{"lower above upper", genBoundsArray(-1, 1, 2, -1, 0, 5, 1, 2, 3, 4), true},
		{"constant param allowed", genBoundsArray(-1, 1, 5, 5, 0, 2, 3, 3, -1, 1), false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := NewGenome(5)
			err := g.SetBounds(tt.bounds...)
			if tt.wantErr {
				if err == nil {
					t.Errorf("expected error")
				}
				return
			}
			for i, v := range g.Params {
				if v.Upper != tt.bounds[i][1] {
					t.Errorf("upper bound at index %d incorrect. expected %f got %f", i, tt.bounds[i][1], v.Upper)
				}
				if v.Lower != tt.bounds[i][0] {
					t.Errorf("lower bound at index %d incorrect. expected %f got %f", i, tt.bounds[i][0], v.Lower)
				}
			}
		})
	}
}

func TestInit(t *testing.T) {
	g := NewGenome(50)
	g.SetBoundsUniform(-2, 5)
	g.Init()
	for i, p := range g.Params {
		if p.Weight < p.Lower || p.Weight > p.Upper {
			t.Errorf("param %d: weight %f outside bounds [%f, %f]", i, p.Weight, p.Lower, p.Upper)
		}
	}
}

func TestNudgeChangesWeights(t *testing.T) {
	g := NewGenome(20)
	g.SetBoundsUniform(-10, 10)
	g.Init()
	before := g.GetWeights()
	g.Nudge(1.0)
	after := g.GetWeights()
	if slices.Equal(before, after) {
		t.Errorf("Nudge changed nothing; weights identical before and after")
	}
}

func TestGetWeights(t *testing.T) {
	g := NewGenome(50)
	w := g.GetWeights()
	w[0] = 99999
	if g.Params[0].Weight == 99999 {
		t.Errorf("GetWeights returned a reference, not a copy — genome got mutated")
	}
}

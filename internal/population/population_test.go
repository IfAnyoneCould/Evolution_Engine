package population

import (
	"math"
	"math/rand"
	"testing"
)

var testRand = rand.New(rand.NewSource(1))

func TestNewAgent(t *testing.T) {
	tests := []struct {
		name    string
		bounds  [][2]float64
		wantErr bool
	}{
		{"standard", [][2]float64{{-1, 1}, {0, 10}, {-100, -50}}, false},
		{"fixed param", [][2]float64{{5, 5}}, false},
		{"lower above upper", [][2]float64{{1, -1}}, true},
		{"no bounds", [][2]float64{}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a, err := NewAgent(tt.bounds, testRand)
			if tt.wantErr {
				if err == nil {
					t.Errorf("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(a.Gene.Params) != len(tt.bounds) {
				t.Errorf("incorrect genome length: expected %d, got %d", len(tt.bounds), len(a.Gene.Params))
			}
			if a.Evaluated {
				t.Errorf("a brand new agent should not be marked evaluated")
			}
			if !math.IsInf(a.Fitness, -1) {
				t.Errorf("incorrect starting fitness: expected -Inf, got %f", a.Fitness)
			}
			for i, p := range a.Gene.Params {
				if p.Weight < p.Lower || p.Weight > p.Upper {
					t.Errorf("param %d: weight %f outside bounds [%f, %f]", i, p.Weight, p.Lower, p.Upper)
				}
			}
		})
	}
}

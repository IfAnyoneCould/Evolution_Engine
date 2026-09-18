package nudge

import (
	"math"
	"testing"
)

func closeEnough(a, b float64) bool {
	return math.Abs(a-b) < 1e-9
}

func TestClampFloats(t *testing.T) {
	tests := []struct {
		name        string
		val, lo, hi float64
		want        float64
	}{
		{"inside", 0.5, 0, 1, 0.5},
		{"below", -3, 0, 1, 0},
		{"above", 7, 0, 1, 1},
		{"on lower", 0, 0, 1, 0},
		{"on upper", 1, 0, 1, 1},
		{"negative range", -5, -10, -1, -5},
		{"empty range", 4, 2, 2, 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Clamp(tt.val, tt.lo, tt.hi)
			if got != tt.want {
				t.Errorf("incorrect clamp: expected %f, got %f", tt.want, got)
			}
		})
	}
}

func TestClampInts(t *testing.T) {
	if got := Clamp(15, 0, 10); got != 10 {
		t.Errorf("incorrect clamp: expected %d, got %d", 10, got)
	}
	if got := Clamp(-15, 0, 10); got != 0 {
		t.Errorf("incorrect clamp: expected %d, got %d", 0, got)
	}
}

func TestConstantFunction(t *testing.T) {
	c := NewConstantFunction(0.75)
	for _, progress := range []float64{0, 0.25, 0.5, 1} {
		if got := c.Get(progress); !closeEnough(got, 0.75) {
			t.Errorf("constant changed with progress %f: expected %f, got %f", progress, 0.75, got)
		}
	}
}

func TestLinearFunction(t *testing.T) {
	tests := []struct {
		name     string
		slope    float64
		progress float64
		want     float64
	}{
		{"start", 1, 0, 1},
		{"halfway", 1, 0.5, 0.5},
		{"finished", 1, 1, 0},
		{"slope above one", 2, 0.5, 1.5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := NewLinearFunction(tt.slope)
			if got := l.Get(tt.progress); !closeEnough(got, tt.want) {
				t.Errorf("incorrect value: expected %f, got %f", tt.want, got)
			}
		})
	}
}

func TestQuadraticFunction(t *testing.T) {
	tests := []struct {
		name     string
		constant float64
		progress float64
		want     float64
	}{
		{"start", 1, 0, 1},
		{"halfway", 1, 0.5, 0.25},
		{"finished", 1, 1, 0},
		{"constant above one", 2, 0.5, 2.25},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			q := NewQuadraticFunction(tt.constant)
			if got := q.Get(tt.progress); !closeEnough(got, tt.want) {
				t.Errorf("incorrect value: expected %f, got %f", tt.want, got)
			}
		})
	}
}

func TestDecayingFunctionsNeverRise(t *testing.T) {
	funcs := []struct {
		name string
		f    Function
	}{
		{"constant", NewConstantFunction(1)},
		{"linear", NewLinearFunction(1)},
		{"quadratic", NewQuadraticFunction(1)},
	}
	for _, fn := range funcs {
		t.Run(fn.name, func(t *testing.T) {
			last := fn.f.Get(0)
			for i := 1; i <= 100; i++ {
				progress := float64(i) / 100
				got := fn.f.Get(progress)
				if got > last {
					t.Errorf("value rose at progress %f: %f then %f", progress, last, got)
				}
				last = got
			}
		})
	}
}

func TestFunctionsNeverNegativeInRange(t *testing.T) {
	funcs := []struct {
		name string
		f    Function
	}{
		{"constant", NewConstantFunction(1)},
		{"linear", NewLinearFunction(1)},
		{"quadratic", NewQuadraticFunction(1)},
	}
	for _, fn := range funcs {
		t.Run(fn.name, func(t *testing.T) {
			for i := 0; i <= 100; i++ {
				progress := float64(i) / 100
				if got := fn.f.Get(progress); got < 0 {
					t.Errorf("negative scale at progress %f: got %f", progress, got)
				}
			}
		})
	}
}

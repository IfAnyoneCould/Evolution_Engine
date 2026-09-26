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

func TestShapes(t *testing.T) {
	tests := []struct {
		name string
		f    Function
		u    float64
		want float64
	}{
		{"constant at the start", NewConstantFunction(), 0, 1},
		{"constant at the end", NewConstantFunction(), 1, 1},
		{"linear quarter", NewLinearFunction(), 0.25, 0.75},
		{"linear halfway", NewLinearFunction(), 0.5, 0.5},
		{"quadratic halfway", NewQuadraticFunction(), 0.5, 0.25},
		{"quadratic quarter", NewQuadraticFunction(), 0.25, 0.5625},
		{"power 3 halfway", NewPowerFunction(3), 0.5, 0.125},
		{"power 0.5 halfway", NewPowerFunction(0.5), 0.5, math.Sqrt(0.5)},
		{"exponential 4 halfway", NewExponentialFunction(4), 0.5, (math.Exp(-2) - math.Exp(-4)) / (1 - math.Exp(-4))},
		{"cosine halfway", NewCosineFunction(), 0.5, 0.5},
		{"cosine quarter", NewCosineFunction(), 0.25, (1 + math.Sqrt(0.5)) / 2},
		{"step 4 inside the first step", NewStepFunction(4), 0.2, 1},
		{"step 4 second step", NewStepFunction(4), 0.3, 0.75},
		{"step 4 last step", NewStepFunction(4), 0.99, 0.25},
		{"step 1 holds until the end", NewStepFunction(1), 0.99, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.f.Get(tt.u); !closeEnough(got, tt.want) {
				t.Errorf("incorrect value at %f: expected %f, got %f", tt.u, tt.want, got)
			}
		})
	}
}

func decaying() []struct {
	name string
	f    Function
} {
	return []struct {
		name string
		f    Function
	}{
		{"linear", NewLinearFunction()},
		{"quadratic", NewQuadraticFunction()},
		{"power 0.5", NewPowerFunction(0.5)},
		{"power 1", NewPowerFunction(1)},
		{"power 4", NewPowerFunction(4)},
		{"exponential 0.5", NewExponentialFunction(0.5)},
		{"exponential 4", NewExponentialFunction(4)},
		{"exponential 20", NewExponentialFunction(20)},
		{"cosine", NewCosineFunction()},
		{"step 1", NewStepFunction(1)},
		{"step 5", NewStepFunction(5)},
	}
}

func TestDecayingShapesGoFromOneToZero(t *testing.T) {
	for _, fn := range decaying() {
		t.Run(fn.name, func(t *testing.T) {
			if got := fn.f.Get(0); !closeEnough(got, 1) {
				t.Errorf("should start at 1, got %f", got)
			}
			if got := fn.f.Get(1); !closeEnough(got, 0) {
				t.Errorf("should end at 0, got %f", got)
			}
		})
	}
}

func TestShapesNeverRise(t *testing.T) {
	all := append(decaying(), struct {
		name string
		f    Function
	}{"constant", NewConstantFunction()})
	for _, fn := range all {
		t.Run(fn.name, func(t *testing.T) {
			last := fn.f.Get(0)
			for i := 1; i <= 100; i++ {
				u := float64(i) / 100
				got := fn.f.Get(u)
				if got > last+1e-12 {
					t.Errorf("value rose at %f: %f then %f", u, last, got)
				}
				last = got
			}
		})
	}
}

func TestShapesStayInZeroToOne(t *testing.T) {
	for _, fn := range decaying() {
		t.Run(fn.name, func(t *testing.T) {
			for i := 0; i <= 100; i++ {
				u := float64(i) / 100
				if got := fn.f.Get(u); got < -1e-12 || got > 1+1e-12 || math.IsNaN(got) {
					t.Errorf("value outside [0,1] at %f: got %f", u, got)
				}
			}
		})
	}
}

func TestPowerMatchesLinearAndQuadratic(t *testing.T) {
	for i := 0; i <= 10; i++ {
		u := float64(i) / 10
		if a, b := NewPowerFunction(1).Get(u), NewLinearFunction().Get(u); !closeEnough(a, b) {
			t.Errorf("power 1 and linear disagree at %f: %f and %f", u, a, b)
		}
		if a, b := NewPowerFunction(2).Get(u), NewQuadraticFunction().Get(u); !closeEnough(a, b) {
			t.Errorf("power 2 and quadratic disagree at %f: %f and %f", u, a, b)
		}
	}
}

func TestExponentialFlattensToLinear(t *testing.T) {
	e := NewExponentialFunction(1e-6)
	for i := 0; i <= 10; i++ {
		u := float64(i) / 10
		if got, want := e.Get(u), 1-u; math.Abs(got-want) > 1e-4 {
			t.Errorf("tiny rate should be close to linear at %f: expected %f, got %f", u, want, got)
		}
	}
}

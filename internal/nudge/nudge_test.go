package nudge

import (
	"math"
	"math/rand"
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

func distributions() []struct {
	name string
	d    Distribution
} {
	return []struct {
		name string
		d    Distribution
	}{
		{"uniform", NewUniformDistribution()},
		{"gaussian", NewGaussianDistribution()},
	}
}

func TestMutateStaysInBounds(t *testing.T) {
	tests := []struct {
		name      string
		w, lo, hi float64
		size      float64
	}{
		{"middle, small", 0, -1, 1, 0.05},
		{"on the upper bound", 1, -1, 1, 0.3},
		{"on the lower bound", -1, -1, 1, 0.3},
		{"asymmetric range", 2, 0, 10, 0.2},
		{"negative range", -75, -100, -50, 0.5},
		{"size bigger than the range", 0.9, -1, 1, 5},
	}
	for _, dist := range distributions() {
		for _, tt := range tests {
			t.Run(dist.name+" "+tt.name, func(t *testing.T) {
				r := rand.New(rand.NewSource(1))
				for range 2000 {
					if x := dist.d.Mutate(tt.w, tt.lo, tt.hi, tt.size, r); x < tt.lo || x > tt.hi || math.IsNaN(x) {
						t.Fatalf("mutated weight %f outside [%f, %f]", x, tt.lo, tt.hi)
					}
				}
			})
		}
	}
}

func TestMutateFixedParamAndZeroSize(t *testing.T) {
	for _, dist := range distributions() {
		t.Run(dist.name, func(t *testing.T) {
			r := rand.New(rand.NewSource(1))
			if x := dist.d.Mutate(5, 5, 5, 1, r); x != 5 {
				t.Errorf("fixed param moved: expected %f, got %f", 5.0, x)
			}
			if x := dist.d.Mutate(0.3, -1, 1, 0, r); x != 0.3 {
				t.Errorf("a size of 0 moved the weight: expected %f, got %f", 0.3, x)
			}
		})
	}
}

// the result has to be near w, not somewhere tied to the range. a range that isn't centred on 0 catches that
func TestMutateStartsFromTheWeight(t *testing.T) {
	tests := []struct {
		name    string
		d       Distribution
		maxDist float64
	}{
		{"uniform never moves past size times range", NewUniformDistribution(), 0.1},
		{"gaussian stays within 6 sigma", NewGaussianDistribution(), 0.6},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := rand.New(rand.NewSource(1))
			for range 2000 {
				if x := tt.d.Mutate(2, 0, 10, 0.01, r); math.Abs(x-2) > tt.maxDist {
					t.Fatalf("weight 2 on [0,10] with size 0.01 moved to %f", x)
				}
			}
		})
	}
}

func stats(d Distribution, n int) (mean, std float64, xs []float64) {
	r := rand.New(rand.NewSource(1))
	xs = make([]float64, n)
	for i := range xs {
		xs[i] = d.Mutate(0, -1000, 1000, 0.001, r)
		mean += xs[i]
	}
	mean /= float64(n)
	for _, x := range xs {
		std += (x - mean) * (x - mean)
	}
	return mean, math.Sqrt(std / float64(n)), xs
}

// size 0.001 on a range of 2000 is a step of 2, far from the bounds so nothing gets clamped or reflected
func TestUniformStepShape(t *testing.T) {
	mean, std, xs := stats(NewUniformDistribution(), 20000)
	for _, x := range xs {
		if math.Abs(x) > 2 {
			t.Fatalf("uniform step of %f is past the cap of 2", x)
		}
	}
	if math.Abs(mean) > 0.05 {
		t.Errorf("uniform steps should average 0, got %f", mean)
	}
	if want := 2 / math.Sqrt(3); math.Abs(std-want) > 0.03*want {
		t.Errorf("uniform spread should be %f, got %f", want, std)
	}
}

func TestGaussianStepShape(t *testing.T) {
	mean, std, xs := stats(NewGaussianDistribution(), 20000)
	if math.Abs(mean) > 0.06 {
		t.Errorf("gaussian steps should average 0, got %f", mean)
	}
	if math.Abs(std-2) > 0.06 {
		t.Errorf("gaussian sigma should be %f, got %f", 2.0, std)
	}
	within := 0
	for _, x := range xs {
		if math.Abs(x) <= 2 {
			within++
		}
	}
	if frac := float64(within) / float64(len(xs)); math.Abs(frac-0.6827) > 0.02 {
		t.Errorf("about 68%% of gaussian steps should be within one sigma, got %.1f%%", frac*100)
	}
}

// clamping piles every overshoot onto the bound, reflecting shouldn't leave anything sitting exactly on it
func TestGaussianDoesNotPileUpOnTheBound(t *testing.T) {
	onBound := func(d Distribution) float64 {
		r := rand.New(rand.NewSource(1))
		n := 0
		for range 2000 {
			if d.Mutate(1, -1, 1, 0.3, r) == 1 {
				n++
			}
		}
		return float64(n) / 2000
	}
	if frac := onBound(NewGaussianDistribution()); frac > 0.001 {
		t.Errorf("gaussian left %.1f%% of weights exactly on the bound, it should reflect them back in", frac*100)
	}
	if frac := onBound(NewUniformDistribution()); frac < 0.3 {
		t.Errorf("uniform should clamp about half its steps onto the bound from there, got %.1f%%", frac*100)
	}
}

func TestMutateSameSeed(t *testing.T) {
	for _, dist := range distributions() {
		t.Run(dist.name, func(t *testing.T) {
			a, b := rand.New(rand.NewSource(7)), rand.New(rand.NewSource(7))
			for range 100 {
				if x, y := dist.d.Mutate(0, -1, 1, 0.1, a), dist.d.Mutate(0, -1, 1, 0.1, b); x != y {
					t.Fatalf("same seed gave %f and %f", x, y)
				}
			}
		})
	}
}

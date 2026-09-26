package genome

import (
	"math/rand"
	"slices"
	"testing"
)

var testRand = rand.New(rand.NewSource(1))

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
	g.Init(testRand)
	for i, p := range g.Params {
		if p.Weight < p.Lower || p.Weight > p.Upper {
			t.Errorf("param %d: weight %f outside bounds [%f, %f]", i, p.Weight, p.Lower, p.Upper)
		}
	}
}

func TestNudgeChangesWeights(t *testing.T) {
	g := NewGenome(20)
	g.SetBoundsUniform(-10, 10)
	g.Init(testRand)
	before := g.GetWeights()
	g.Nudge(1.0, testRand)
	after := g.GetWeights()
	if slices.Equal(before, after) {
		t.Errorf("nudge changed nothing; weights identical before and after")
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

func TestNewGenome(t *testing.T) {
	tests := []struct {
		name   string
		length int
	}{
		{"standard", 10},
		{"single param", 1},
		{"empty", 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := NewGenome(tt.length)
			if len(g.Params) != tt.length {
				t.Errorf("incorrect length: expected %d, got %d", tt.length, len(g.Params))
			}
			for i, p := range g.Params {
				if p.Weight != 0 || p.Lower != 0 || p.Upper != 0 {
					t.Errorf("param %d not zeroed: %+v", i, p)
				}
			}
		})
	}
}

func TestSetWeight(t *testing.T) {
	tests := []struct {
		name    string
		val     float64
		wantErr bool
	}{
		{"middle", 0, false},
		{"on lower", -1, false},
		{"on upper", 1, false},
		{"below lower", -1.5, true},
		{"above upper", 1.5, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := NewGenome(3)
			if err := g.SetBoundsUniform(-1, 1); err != nil {
				t.Fatalf("could not set bounds: %v", err)
			}
			err := g.SetWeight(1, tt.val)
			if tt.wantErr {
				if err == nil {
					t.Errorf("expected error")
				}
				if g.Params[1].Weight != 0 {
					t.Errorf("weight was written anyway: got %f", g.Params[1].Weight)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if g.Params[1].Weight != tt.val {
				t.Errorf("incorrect weight: expected %f, got %f", tt.val, g.Params[1].Weight)
			}
			if g.Params[0].Weight != 0 || g.Params[2].Weight != 0 {
				t.Errorf("SetWeight touched the wrong params: %v", g.GetWeights())
			}
		})
	}
}

func TestClone(t *testing.T) {
	g := NewGenome(10)
	if err := g.SetBoundsUniform(-5, 5); err != nil {
		t.Fatalf("could not set bounds: %v", err)
	}
	g.Init(testRand)

	c := g.Clone()
	if !slices.Equal(g.GetWeights(), c.GetWeights()) {
		t.Errorf("clone does not match original: %v vs %v", g.GetWeights(), c.GetWeights())
	}

	before := g.GetWeights()
	c.Nudge(1.0, testRand)
	if !slices.Equal(g.GetWeights(), before) {
		t.Errorf("nudging the clone changed the original: %v vs %v", before, g.GetWeights())
	}

	for i, p := range c.Params {
		if p.Lower != g.Params[i].Lower || p.Upper != g.Params[i].Upper {
			t.Errorf("param %d bounds not copied: expected [%f,%f], got [%f,%f]", i, g.Params[i].Lower, g.Params[i].Upper, p.Lower, p.Upper)
		}
	}
}

func TestNudgeStaysInBounds(t *testing.T) {
	tests := []struct {
		name   string
		bounds float64
	}{
		{"small", 0.01},
		{"whole range", 1},
		{"larger than range", 50},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := NewGenome(5)
			if err := g.SetBounds(genBoundsArray(-1, 1, 0, 10, -100, -50, 5, 5, -0.5, 0.5)...); err != nil {
				t.Fatalf("could not set bounds: %v", err)
			}
			g.Init(testRand)
			for range 100 {
				g.Nudge(tt.bounds, testRand)
				for i, p := range g.Params {
					if p.Weight < p.Lower || p.Weight > p.Upper {
						t.Fatalf("param %d: weight %f outside bounds [%f, %f]", i, p.Weight, p.Lower, p.Upper)
					}
				}
			}
		})
	}
}

func TestNudgeZeroDoesNothing(t *testing.T) {
	g := NewGenome(10)
	if err := g.SetBoundsUniform(-3, 3); err != nil {
		t.Fatalf("could not set bounds: %v", err)
	}
	g.Init(testRand)
	before := g.GetWeights()
	g.Nudge(0, testRand)
	if !slices.Equal(before, g.GetWeights()) {
		t.Errorf("a nudge of 0 moved the weights: %v became %v", before, g.GetWeights())
	}
}

func TestNudgeFixedParamStays(t *testing.T) {
	g := NewGenome(2)
	if err := g.SetBounds(genBoundsArray(5, 5, -1, 1)...); err != nil {
		t.Fatalf("could not set bounds: %v", err)
	}
	g.Init(testRand)
	g.Nudge(1.0, testRand)
	if g.Params[0].Weight != 5 {
		t.Errorf("fixed param moved off its bound: expected %f, got %f", 5.0, g.Params[0].Weight)
	}
}

func TestInitVaries(t *testing.T) {
	a := NewGenome(30)
	b := NewGenome(30)
	if err := a.SetBoundsUniform(-10, 10); err != nil {
		t.Fatalf("could not set bounds: %v", err)
	}
	if err := b.SetBoundsUniform(-10, 10); err != nil {
		t.Fatalf("could not set bounds: %v", err)
	}
	a.Init(testRand)
	b.Init(testRand)
	if slices.Equal(a.GetWeights(), b.GetWeights()) {
		t.Errorf("two initialised genomes came out identical: %v", a.GetWeights())
	}
}

func TestSameSeedSameGenome(t *testing.T) {
	a := NewGenome(10)
	b := NewGenome(10)
	if err := a.SetBoundsUniform(-10, 10); err != nil {
		t.Fatalf("could not set bounds: %v", err)
	}
	if err := b.SetBoundsUniform(-10, 10); err != nil {
		t.Fatalf("could not set bounds: %v", err)
	}
	ra, rb := rand.New(rand.NewSource(42)), rand.New(rand.NewSource(42))
	a.Init(ra)
	b.Init(rb)
	a.Nudge(0.1, ra)
	b.Nudge(0.1, rb)
	if !slices.Equal(a.GetWeights(), b.GetWeights()) {
		t.Errorf("same seed gave different genomes: %v and %v", a.GetWeights(), b.GetWeights())
	}
}

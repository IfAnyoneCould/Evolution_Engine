package genome

import (
	"Evolution_Engine/internal/nudge"
	"fmt"
	"math/rand"
)

type Param struct {
	Weight, Lower, Upper float64
}
type Genome struct {
	Params []Param
}

func NewGenome(l int) *Genome {
	g := &Genome{}
	for range l {
		g.Params = append(g.Params, Param{0, 0, 0})
	}
	return g
}

func (g *Genome) SetBounds(b ...[2]float64) error {
	if len(b) != len(g.Params) {
		err := fmt.Errorf("incorrect number of bound elements. expected %d, got %d", len(g.Params), len(b))
		return err
	}
	for i, v := range b {
		if v[0] > v[1] {
			err := fmt.Errorf("illegal arguments for bounds at index %d.\nupper must be greater than lower, and lower cannot equal upper", i)
			return err
		}
		g.Params[i].Lower = v[0]
		g.Params[i].Upper = v[1]
	}
	return nil
}

func (g *Genome) Clone() *Genome {
	params := make([]Param, len(g.Params))
	copy(params, g.Params)
	return &Genome{Params: params}
}

func (g *Genome) SetBoundsUniform(lower float64, upper float64) error {
	if lower > upper {
		err := fmt.Errorf("illegal arguments for bounds. upper must be greater than lower, and lower cannot equal upper")
		return err
	}

	for i := range len(g.Params) {
		g.Params[i].Lower = lower
		g.Params[i].Upper = upper
	}
	return nil
}

func (g *Genome) SetWeight(i int, val float64) error {
	if val < g.Params[i].Lower || val > g.Params[i].Upper {
		err := fmt.Errorf("weight %f not in bounds [%f,%f]", val, g.Params[i].Lower, g.Params[i].Upper)
		return err
	}
	g.Params[i].Weight = val
	return nil
}

func (g *Genome) GetWeights() []float64 {
	slice := make([]float64, len(g.Params))
	for i, v := range g.Params {
		slice[i] = v.Weight
	}
	return slice
}

func (g *Genome) Init(r *rand.Rand) {
	for i, v := range g.Params {
		val := v.Lower + r.Float64()*(v.Upper-v.Lower)
		_ = g.SetWeight(i, val) // don't check error, weight it always between bounds
	}
}

func (g *Genome) Nudge(size float64, d nudge.Distribution, r *rand.Rand) {
	for i := range len(g.Params) {
		p := &g.Params[i]
		p.Weight = d.Mutate(p.Weight, p.Lower, p.Upper, size, r)
	}
}

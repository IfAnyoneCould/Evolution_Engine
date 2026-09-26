package nudge

import (
	"cmp"
	"math"
)

func Clamp[T cmp.Ordered](val, lo, hi T) T {
	return min(max(lo, val), hi)
}

type Function interface {
	Get(progress float64) float64
}

type ConstantFunction struct{}

func NewConstantFunction() *ConstantFunction {
	return &ConstantFunction{}
}
func (c *ConstantFunction) Get(progress float64) float64 {
	return 1
}

type LinearFunction struct{}

func NewLinearFunction() *LinearFunction {
	return &LinearFunction{}
}
func (l *LinearFunction) Get(progress float64) float64 {
	return 1 - progress
}

type QuadraticFunction struct{}

func NewQuadraticFunction() *QuadraticFunction {
	return &QuadraticFunction{}
}
func (q *QuadraticFunction) Get(progress float64) float64 {
	dist := 1 - progress
	return dist * dist
}

type PowerFunction struct {
	exponent float64
}

func NewPowerFunction(ex float64) *PowerFunction {
	return &PowerFunction{ex}
}
func (p *PowerFunction) Get(progress float64) float64 {
	return math.Pow(1-progress, p.exponent)
}

type ExponentialFunction struct {
	rate float64
}

func NewExponentialFunction(r float64) *ExponentialFunction {
	return &ExponentialFunction{r}
}
func (e *ExponentialFunction) Get(progress float64) float64 {
	return (math.Exp(-e.rate*progress) - math.Exp(-e.rate)) / (1 - math.Exp(-e.rate))
}

type CosineFunction struct{}

func NewCosineFunction() *CosineFunction {
	return &CosineFunction{}
}
func (c *CosineFunction) Get(progress float64) float64 {
	return (1 + math.Cos(math.Pi*progress)) / 2
}

type StepFunction struct {
	steps uint
}

func NewStepFunction(s uint) *StepFunction {
	return &StepFunction{s}
}
func (s *StepFunction) Get(progress float64) float64 {
	return 1 - math.Floor(progress*float64(s.steps))/float64(s.steps)
}

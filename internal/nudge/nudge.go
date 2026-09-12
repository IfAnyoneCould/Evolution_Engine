package nudge

import (
	"cmp"
)

func Clamp[T cmp.Ordered](val, lo, hi T) T {
	return min(max(lo, val), hi)
}

type Function interface {
	Get(progress float64) float64
}

type ConstantFunction struct {
	val float64
}

func NewConstantFunction(val float64) *ConstantFunction {
	return &ConstantFunction{val}
}

func (c *ConstantFunction) Get(progress float64) float64 {
	return c.val
}

type LinearFunction struct {
	slope float64
}

func NewLinearFunction(slope float64) *LinearFunction {
	return &LinearFunction{slope}
}

func (l *LinearFunction) Get(progress float64) float64 {
	return l.slope - progress
}

type QuadraticFunction struct {
	constant float64
}

func NewQuadraticFunction(constant float64) *QuadraticFunction {
	return &QuadraticFunction{constant}
}

func (q *QuadraticFunction) Get(progress float64) float64 {
	dist := q.constant - progress
	return dist * dist
}

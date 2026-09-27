package optimizers

import (
	"Evolution_Engine/internal/config"
	"Evolution_Engine/internal/nudge"
	"Evolution_Engine/internal/population"
	"cmp"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"sort"
	"strings"

	"gonum.org/v1/gonum/mat"
)

type Optimizer interface {
	Ask() ([][]float64, error)
	Tell(fit []float64) error
	Best() ([]float64, float64)
}

type GA struct {
	Agents       []population.Agent
	pending      []int
	sel          config.Selection
	schedule     config.Schedule
	scheduleFunc nudge.Function
	dist         nudge.Distribution
	goal         float64
	startFitness float64
	started      bool
	bestW        []float64
	bestF        float64
	Fraction     float64
	MinNudge     float64
	r            *rand.Rand
}

func New(cfg config.Config, r *rand.Rand) (Optimizer, error) {
	switch strings.ToLower(cfg.Optimizer) {
	case "ga":
		return NewGA(cfg, r)
	case "cmaes":
		return NewCMAES(cfg, r)
	default:
		return nil, errors.New("config error: optimizer not recognized test")
	}
}

func NewGA(params config.Config, r *rand.Rand) (*GA, error) {
	var agents []population.Agent
	for range params.RunSettings.PopulationSize {
		a, err := population.NewAgent(params.Bounds, r)
		if err != nil {
			return &GA{}, err
		}
		agents = append(agents, a)
	}

	var nudgeFunc nudge.Function
	switch strings.ToLower(params.Mutation.Schedule.Type) {
	case "constant":
		nudgeFunc = nudge.NewConstantFunction()
	case "linear":
		nudgeFunc = nudge.NewLinearFunction()
	case "quadratic":
		nudgeFunc = nudge.NewQuadraticFunction()
	case "power":
		nudgeFunc = nudge.NewPowerFunction(params.Mutation.Schedule.Exponent)
	case "exponential":
		nudgeFunc = nudge.NewExponentialFunction(params.Mutation.Schedule.Rate)
	case "cosine":
		nudgeFunc = nudge.NewCosineFunction()
	case "step":
		nudgeFunc = nudge.NewStepFunction(params.Mutation.Schedule.Steps)
	default:
		return &GA{}, errors.New("config error: mutation.schedule.type not recognized")
	}

	var d nudge.Distribution
	switch strings.ToLower(params.Mutation.Distribution) {
	case "uniform":
		d = nudge.NewUniformDistribution()
	case "gaussian":
		d = nudge.NewGaussianDistribution()
	default:
		return &GA{}, errors.New("config error: mutation.distribution not recognized")
	}

	return &GA{
		Agents:       agents,
		pending:      []int{},
		sel:          params.Selection,
		schedule:     params.Mutation.Schedule,
		scheduleFunc: nudgeFunc,
		dist:         d,
		goal:         params.RunSettings.TargetFitness,
		startFitness: 0,
		started:      false,
		bestW:        []float64{},
		bestF:        math.Inf(-1),
		Fraction:     params.Mutation.Fraction,
		MinNudge:     params.Mutation.MinNudge,
		r:            r,
	}, nil
}

func (g *GA) Ask() ([][]float64, error) {
	if len(g.pending) > 0 || g.started {
		if err := g.newGen(); err != nil {
			return [][]float64{}, err
		}
	}
	g.pending = g.pending[:0]
	var weights [][]float64
	for i, a := range g.Agents {
		if !a.Evaluated {
			g.pending = append(g.pending, i)
			weights = append(weights, a.Gene.GetWeights())
		}
	}
	return weights, nil
}

func (g *GA) Tell(fit []float64) error {
	if len(fit) != len(g.pending) {
		return fmt.Errorf("told %d fitnesses for %d agents", len(fit), len(g.pending))
	}
	for k, i := range g.pending {
		g.Agents[i].Fitness, g.Agents[i].Evaluated = fit[k], true
	}
	g.Rank()
	if !g.started {
		g.startFitness, g.started = g.Agents[0].Fitness, true
	}
	if g.Agents[0].Fitness > g.bestF {
		g.bestF, g.bestW = g.Agents[0].Fitness, g.Agents[0].Gene.GetWeights()
	}
	return nil
}

func (g *GA) Best() ([]float64, float64) {
	return slices.Clone(g.bestW), g.bestF
}

func (g *GA) newGen() error {
	n := len(g.Agents)
	cutoff := int(float64(n) * g.sel.Pressure)
	if cutoff < 1 {
		return fmt.Errorf("pressure %f too low, so survivors from population of %d", g.sel.Pressure, n)
	}

	newAgents := make([]population.Agent, 0, n)

	for i := 0; i < int(g.sel.Elite) && i < n; i++ {
		if g.Agents[i].Evaluated {
			newAgents = append(newAgents, g.Agents[i])
		}
	}

	for len(newAgents) < n {
		parent := g.Agents[g.r.Intn(cutoff)]

		childGene := parent.Gene.Clone()
		childGene.Nudge(g.CalcNudge(parent.Fitness), g.dist, g.r)
		newAgents = append(newAgents, population.Agent{Gene: childGene, Fitness: math.Inf(-1), Evaluated: false})
	}
	g.Agents = newAgents
	return nil
}

func (g *GA) CalcNudge(fit float64) float64 {
	progress := nudge.Clamp((fit-g.startFitness)/(g.goal-g.startFitness), 0, 1)
	if g.goal-g.startFitness == 0 {
		progress = 1
	}

	u := nudge.Clamp((progress-g.schedule.Hold)/(1-g.schedule.Hold), 0, 1)
	f := g.schedule.End + (1-g.schedule.End)*g.scheduleFunc.Get(u)
	return max(f*g.Fraction, g.MinNudge)
}

func (g *GA) Rank() {
	slices.SortFunc(g.Agents, func(a, b population.Agent) int {
		if a.Evaluated != b.Evaluated {
			if a.Evaluated {
				return -1
			}
			return 1
		}
		return cmp.Compare(b.Fitness, a.Fitness)
	})
}

// im gonna be real, i have no idea how the cma-es algorithm works, just followed the equations blindly. maybe when i learn
// linear algebra ill come back to understand it

type CMAES struct {
	bounds [][2]float64
	pop    []offspring
	asked  bool
	BestF  float64
	BestW  []float64
	free   []int
	m      *mat.VecDense
	o      float64
	C      *mat.SymDense
	B      *mat.Dense
	D      []float64 //holds eigenvalues, not their square roots
	pO     *mat.VecDense
	pC     *mat.VecDense
	n      int
	y      int
	u      int
	wI     []float64
	uEff   float64
	cO     float64
	dO     float64
	cC     float64
	c1     float64
	cU     float64
	xN     float64
	g      uint
	r      *rand.Rand
}

func NewCMAES(cfg config.Config, r *rand.Rand) (*CMAES, error) {
	var free []int
	for i, b := range cfg.Bounds {
		if b[1] > b[0] {
			free = append(free, i)
		}
	}
	if len(free) == 0 {
		return nil, errors.New("config error: every weight is fixed, nothing to optimize")
	}
	n := len(free)

	m := mat.NewVecDense(n, nil)
	for j := range n {
		m.SetVec(j, 0.5)
	}
	sigma := 0.3 // TODO add to cmaes config setting

	y := int(4 + math.Floor(3*math.Log(float64(n))))
	u := int(math.Floor(float64(y) / 2))
	var w_i []float64
	for i := range int(u) {
		w_i = append(w_i, math.Log((float64(y)+1)/2)-math.Log(float64(i+1)))
	}
	sum := 0.0
	for _, s := range w_i {
		sum += s
	}
	for i := range w_i {
		w_i[i] /= sum
	}
	sumSq := 0.0
	for _, w := range w_i {
		sumSq += w * w
	}
	u_eff := 1 / sumSq
	c_o := (u_eff + 2) / (float64(n) + u_eff + 5)
	d_o := 1 + 2*math.Max(0, math.Sqrt((u_eff-1)/(float64(n)+1))-1) + c_o
	c_c := (4 + u_eff/float64(n)) / (float64(n) + 4 + 2*u_eff/float64(n))
	c_1 := 2 / (math.Pow(float64(n)+1.3, 2) + u_eff)
	c_u := math.Min(1-c_1, 2*(u_eff-2+1/u_eff)/(math.Pow(float64(n)+2, 2)+u_eff))
	x_n := math.Sqrt(float64(n)) * (1 - 1/(4*float64(n)) + 1/(21*math.Pow(float64(n), 2)))

	c := mat.NewSymDense(n, nil)
	b := mat.NewDense(n, n, nil)
	d := make([]float64, n)
	for i := range n {
		c.SetSym(i, i, 1)
		b.Set(i, i, 1)
		d[i] = 1
	}

	return &CMAES{
		bounds: cfg.Bounds,
		pop:    []offspring{},
		asked:  false,
		BestF:  math.Inf(-1),
		BestW:  []float64{},
		free:   free,
		m:      m,
		o:      sigma,
		C:      c,
		B:      b,
		D:      d,
		pO:     mat.NewVecDense(int(n), nil),
		pC:     mat.NewVecDense(int(n), nil),
		n:      n,
		y:      y,
		u:      u,
		wI:     w_i,
		uEff:   u_eff,
		cO:     c_o,
		dO:     d_o,
		cC:     c_c,
		c1:     c_1,
		cU:     c_u,
		xN:     x_n,
		g:      0,
		r:      r,
	}, nil
}

type offspring struct {
	x   *mat.VecDense
	yk  *mat.VecDense
	w   []float64
	fit float64
}

func (c *CMAES) GenPop() {

	sqrtD := make([]float64, c.n)
	for i, v := range c.D {
		sqrtD[i] = math.Sqrt(v)
	}

	var BD mat.Dense
	BD.Mul(c.B, mat.NewDiagDense(c.n, sqrtD))

	pop := make([]offspring, 0, int(c.y))
	for range int(c.y) {
		z := sampleStandardNormal(c.n, c.r)

		yk := mat.NewVecDense(c.n, nil)
		yk.MulVec(&BD, z)

		xk := mat.NewVecDense(c.n, nil)
		xk.AddScaledVec(c.m, c.o, yk)

		w := make([]float64, len(c.bounds))
		for i, b := range c.bounds {
			w[i] = b[0]
		}
		for j, i := range c.free {
			lo, hi := c.bounds[i][0], c.bounds[i][1]
			w[i] = lo + reflect01(xk.AtVec(j))*(hi-lo)
		}

		pop = append(pop, offspring{x: xk, yk: yk, w: w})
	}
	c.pop = pop
}

func (c *CMAES) Ask() ([][]float64, error) {
	if !c.asked {
		c.GenPop()
		c.asked = true
	}
	result := make([][]float64, len(c.pop))
	for i, o := range c.pop {
		result[i] = slices.Clone(o.w)
	}
	return result, nil
}

func (c *CMAES) Tell(fit []float64) error {

	if !c.asked {
		return errors.New("tell called without an ask")
	}
	if len(fit) != len(c.pop) {
		return fmt.Errorf("told %d fitnesses, got %d samples", len(fit), len(c.pop))
	}

	for i := range len(c.pop) {
		c.pop[i].fit = fit[i]
	}

	c.asked = false

	sort.Slice(c.pop, func(i, j int) bool { return c.pop[i].fit > c.pop[j].fit })

	yw := mat.NewVecDense(c.n, nil)
	for i := range int(c.u) {
		yw.AddScaledVec(yw, c.wI[i], c.pop[i].yk)
	}

	c.m.AddScaledVec(c.m, c.o, yw)

	cInvSqrtYw := c.invSqrtMulVec(yw)
	c.pO.ScaleVec(1-c.cO, c.pO)
	c.pO.AddScaledVec(c.pO, math.Sqrt(c.cO*(2-c.cO)*c.uEff), cInvSqrtYw)

	pOnorm := mat.Norm(c.pO, 2)
	c.o *= math.Exp((c.cO / c.dO) * (pOnorm/c.xN - 1))

	c.g++
	denom := math.Sqrt(1 - math.Pow(1-c.cO, float64(2*c.g)))
	hSig := 0.0
	if pOnorm/denom < ((1.4 + 2/(float64(c.n)+1)) * c.xN) {
		hSig = 1.0
	}

	c.pC.ScaleVec(1-c.cC, c.pC)
	c.pC.AddScaledVec(c.pC, hSig*math.Sqrt(c.cC*(2-c.cC)*c.uEff), yw)

	delta := (1 - hSig) * c.cC * (2 - c.cC)
	c.C.ScaleSym(1-c.c1-c.cU+c.c1*delta, c.C)
	c.C.SymRankOne(c.C, c.c1, c.pC)
	for i := range int(c.u) {
		c.C.SymRankOne(c.C, c.cU*c.wI[i], c.pop[i].yk)
	}

	if err := c.EigenDecomp(); err != nil {
		return err
	}

	if c.pop[0].fit > c.BestF {
		c.BestF = c.pop[0].fit
		c.BestW = c.pop[0].w
	}

	return nil
}

func (c *CMAES) Best() ([]float64, float64) {
	return slices.Clone(c.BestW), c.BestF
}

func (c *CMAES) invSqrtMulVec(v *mat.VecDense) *mat.VecDense {
	ni := int(c.n)
	tmp := mat.NewVecDense(ni, nil)
	tmp.MulVec(c.B.T(), v)
	for i := range ni {
		tmp.SetVec(i, tmp.AtVec(i)/math.Sqrt(c.D[i]))
	}
	out := mat.NewVecDense(ni, nil)
	out.MulVec(c.B, tmp)
	return out
}

func (c *CMAES) EigenDecomp() error {
	var eig mat.EigenSym
	if ok := eig.Factorize(c.C, true); !ok {
		return errors.New("eigendecomposition failed")
	}

	c.D = eig.Values(c.D)
	for i := range c.D {
		c.D[i] = math.Max(c.D[i], 1e-14)
	}

	eig.VectorsTo(c.B)

	return nil
}

func sampleStandardNormal(n int, r *rand.Rand) *mat.VecDense {
	v := mat.NewVecDense(n, nil)
	for i := range n {
		v.SetVec(i, r.NormFloat64())
	}
	return v
}

func toSlice(v mat.Vector) []float64 {
	n := v.Len()
	out := make([]float64, n)
	for i := range n {
		out[i] = v.AtVec(i)
	}
	return out
}

func reflect01(v float64) float64 {
	v = math.Mod(v, 2)
	if v < 0 {
		v += 2
	}
	if v > 1 {
		v = 2 - v
	}
	return v
}

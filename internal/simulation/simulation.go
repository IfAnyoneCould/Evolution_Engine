package simulation

import (
	"Evolution_Engine/internal/api"
	"Evolution_Engine/internal/config"
	"Evolution_Engine/internal/optimizers"
	"Evolution_Engine/internal/pool"
	"context"
	"encoding/json"
	"errors"
	"math/rand"
	"os"
	"time"
)

type Simulation struct {
	opt                    optimizers.Optimizer
	pool                   *pool.Pool
	cycleMax, currentCycle uint
	targetFitness          float64
	stagnation             config.StagnationDetect
	output                 config.Output
	sender                 api.Emitter
}

func NewSimulation(cfg config.Config, emitter api.Emitter) (*Simulation, error) {
	r := rand.New(rand.NewSource(cfg.SimSettings.RandSeed))
	opt, err := optimizers.New(cfg, r)
	if err != nil {
		return &Simulation{}, err
	}
	p, err := pool.NewPool(cfg)
	if err != nil {
		return &Simulation{}, err
	}

	e := api.Start{
		Seed:      cfg.SimSettings.RandSeed,
		Optimizer: cfg.Optimizer,
	}
	if err := emitter.Send(e); err != nil {
		return &Simulation{}, err
	}

	return &Simulation{
		opt:           opt,
		pool:          p,
		cycleMax:      cfg.RunSettings.MaxCycles,
		currentCycle:  0,
		targetFitness: cfg.RunSettings.TargetFitness,
		stagnation:    cfg.StagnationDetect,
		output:        cfg.Output,
		sender:        emitter}, nil
}

func (s *Simulation) Run() error {
	lastTop := float64(0)
	total := time.Now()
	cycleSinceImprove := 0
	bestF := 0.0

	for s.currentCycle = 0; s.currentCycle < s.cycleMax; s.currentCycle++ {

		ctx := context.Background()
		start := time.Now()

		weights, err := s.opt.Ask()
		if err != nil {
			if nerr := s.pool.CloseSims(); nerr != nil {
				return errors.Join(err, nerr)
			}
			return err
		}
		fit, err := s.pool.Evaluate(ctx, weights)
		if err != nil {
			if nerr := s.pool.CloseSims(); nerr != nil {
				return errors.Join(err, nerr)
			}
			return err
		}
		if err := s.opt.Tell(fit); err != nil {
			if nerr := s.pool.CloseSims(); nerr != nil {
				return errors.Join(err, nerr)
			}
			return err
		}

		_, bestF = s.opt.Best()

		worstF := pool.Min(fit)
		w, _ := s.opt.Best()
		e := api.Generation{
			Cycle:      int(s.currentCycle),
			Best:       bestF,
			Mean:       pool.Mean(fit),
			Min:        worstF,
			Spread:     pool.Spread(fit),
			BestGenome: w,
			BatchTime:  float64(time.Since(start)) / float64(time.Millisecond),
		}
		if err := s.sender.Send(e); err != nil {
			return errors.Join(err, s.pool.CloseSims())
		}

		if bestF-lastTop < s.stagnation.Epsilon {
			cycleSinceImprove++
		} else {
			cycleSinceImprove = 0
			lastTop = bestF
		}
		if cycleSinceImprove >= int(s.stagnation.Patience) {
			w, _ := s.GetBestWeights()
			e := api.Done{
				Reason:     "stalled",
				Best:       bestF,
				BestGenome: w,
				Cycles:     int(s.currentCycle),
				Total:      float64(time.Since(total)) / float64(time.Millisecond),
			}
			if err := s.pool.CloseSims(); err != nil {
				return errors.Join(err, s.sender.Send(e))
			}
			if err := s.sender.Send(e); err != nil {
				return err
			}
			return nil
		}

		if s.currentCycle == 0 {
			lastTop = bestF
		}
		if bestF >= s.targetFitness {
			break
		}
	}
	r := "target"
	if s.currentCycle+1 == s.cycleMax {
		r = "max_cycles"
	}
	w, _ := s.GetBestWeights()
	e := api.Done{
		Reason:     r,
		Best:       bestF,
		BestGenome: w,
		Cycles:     int(s.currentCycle),
		Total:      float64(time.Since(total)) / float64(time.Millisecond),
	}
	if err := s.pool.CloseSims(); err != nil {
		return errors.Join(err, s.sender.Send(e))
	}
	if err := s.sender.Send(e); err != nil {
		return err
	}
	return nil
}

func (s *Simulation) WriteBestWeights() error {
	if s.output.WeightPath != "" {
		weights, _ := s.opt.Best()
		j, err := json.Marshal(weights)
		if err != nil {
			return err
		}
		if err = os.WriteFile(s.output.WeightPath, j, 0644); err != nil {
			return err
		}
	}
	return nil
}

func (s *Simulation) GetBestWeights() ([]float64, float64) {
	return s.opt.Best()
}

package simulation

import (
	"Evolution_Engine/internal/config"
	"Evolution_Engine/internal/optimizers"
	"Evolution_Engine/internal/pool"
	"context"
	"encoding/json"
	"errors"
	"fmt"
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
}

func NewSimulation(cfg config.JsonParams) (*Simulation, error) {
	r := rand.New(rand.NewSource(*cfg.SimSettings.RandSeed))
	opt, err := optimizers.NewGA(cfg, r)
	if err != nil {
		return &Simulation{}, err
	}
	pool, err := pool.NewPool(cfg)
	if err != nil {
		return &Simulation{}, err
	}
	fmt.Printf("using random seed: %d\n", *cfg.SimSettings.RandSeed)
	return &Simulation{
		opt:           opt,
		pool:          pool,
		cycleMax:      cfg.RunSettings.MaxCycles,
		currentCycle:  0,
		targetFitness: cfg.RunSettings.TargetFitness,
		stagnation:    cfg.StagnationDetect,
		output:        cfg.Output}, nil
}

func (s *Simulation) Run() error {
	lastTop := float64(0)
	total := time.Now()
	cycleSinceImprove := 0

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
		fmt.Printf("batch took %v\n", time.Since(start))
		if err := s.opt.Tell(fit); err != nil {
			if nerr := s.pool.CloseSims(); nerr != nil {
				return errors.Join(err, nerr)
			}
			return err
		}

		_, bestF := s.opt.Best()

		if bestF >= s.targetFitness {
			fmt.Printf("simulation reached a target fitness %f in %d cycles\n", bestF, s.currentCycle)
			break
		}

		if bestF-lastTop < s.stagnation.Epsilon {
			cycleSinceImprove++
		} else {
			cycleSinceImprove = 0
			lastTop = bestF
		}
		if cycleSinceImprove >= int(s.stagnation.Patience) {
			fmt.Printf("simulation exited, didn't see a fitness improvement greater than %f in %d cycles\n", s.stagnation.Epsilon, s.stagnation.Patience)
			if err := s.pool.CloseSims(); err != nil {
				return err
			}
			return nil
		}

		if s.currentCycle == 0 {
			lastTop = bestF
		}
		fmt.Printf("Current cycle: %d -- Max fitness: %f -- Goal Fitness: %f\n", s.currentCycle, bestF, s.targetFitness)
	}
	fmt.Printf("Simulation took %v\n", time.Since(total))
	_ = s.pool.CloseSims()
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

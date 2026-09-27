package pool

import (
	"Evolution_Engine/internal/config"
	"context"
	"errors"
	"sync"
)

type Pool struct {
	procs []*config.SimProcess
}

func NewPool(cfg config.Config) (*Pool, error) {
	procs := make([]*config.SimProcess, 0, cfg.Workers)
	for range cfg.Workers {
		sim, err := config.NewSimProcess(cfg.Prog.Path, cfg.Prog.Args, cfg.SimSettings.Timeout)
		if err != nil {
			for i := range procs {
				procs[i].Close()
			}
			return &Pool{}, err
		}
		procs = append(procs, sim)
	}
	return &Pool{procs}, nil
}

func (p *Pool) Evaluate(ctx context.Context, weights [][]float64) ([]float64, error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	fit := make([]float64, len(weights))
	jobs := make(chan int, len(weights))
	for i := range weights {
		jobs <- i
	}
	close(jobs)

	errs := make(chan error, len(weights))

	wg := sync.WaitGroup{}
	for w := range p.procs {
		wg.Go(func() {
			for j := range jobs {
				var err error
				fit[j], err = p.procs[w].Eval(weights[j])
				if err != nil {
					select {
					case errs <- err:
					case <-ctx.Done():
						return
					}
					continue
				}
			}
		})
	}

	go func() {
		wg.Wait()
		close(errs)
	}()

	var errList []error
	for e := range errs {
		errList = append(errList, e)
	}

	return fit, errors.Join(errList...)
}

func (p *Pool) CloseSims() error {
	var errs []error
	for i := range p.procs {
		if err := p.procs[i].Close(); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

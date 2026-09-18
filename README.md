# Evolution_Engine

A genetic algorithm optimiser in Go. You give it a program and a set of bounds,
and it evolves a vector of weights that maximises whatever score your program
prints back.

The point is that it doesn't care what your program is. Communication is a
newline-delimited pipe — the engine writes a JSON array of weights to the
process's stdin, the process writes a single fitness number to stdout. Anything
that can read stdin and write stdout can be optimised, in any language.

## How it works

```
                 weights (JSON) ->
  Population  ---------------------  Worker process (your program)
   (agents)    <- fitness (float)
```

1. A population of agents is created, each holding a genome of `n` floats
   inside per-dimension bounds.
2. `RunBatch` distributes agents over a pool of persistent worker processes via
   a buffered channel and a `sync.WaitGroup`.
3. Agents are ranked, and `NewGen` builds the next generation with elitism and
   a configurable selection pressure.
4. This repeats until the fitness target is hit, the cycle cap is reached, or
   the population stalls.

**One persistent process per worker.** The first working version spawned a
fresh process for every single evaluation, which meant process startup
dominated everything — batches took around 900 ms. Workers now start once and
stay alive for the whole run, reading and writing on the same pipe. That
dropped cycle times to **under 1 ms**, roughly a 900x improvement, and it's the
single change that made the engine usable.

**Adaptive mutation ("nudge").** The mutation magnitude scales with how far the
population has progressed toward the goal, so early generations explore widely
and later ones refine instead of overshooting a good answer. The scaling curve
sits behind a `nudge.Function` interface with constant, linear and quadratic
implementations, selectable from config, and `min_nudge` sets a floor so the
population can never fully stop moving.

## Configuring a run

```json
{
  "program": { "path": "cmd/engine/TestRun.exe", "args": ["8", "0", "5.12"] },
  "bounds": [[-5.12, 5.12], [-5.12, 5.12]],
  "nudge": 0.1,
  "min_nudge": 0.01,
  "nudge_func": { "type": "quadratic", "params": [2.0] }
}
```

`bounds` has one entry per dimension and sets both the genome length and the
legal range of each weight. The bundled example optimises the Rastrigin
function in 8 dimensions — a standard benchmark chosen because it's covered in
local minima and will expose an optimiser that converges too early.

## Running

```sh
go run ./cmd/engine
go test ./...
```

The tests drive real worker processes. `testdata/sim` is a small stand-in
program they build once per package and talk to over the same pipe a real
simulation would use, so nothing in the suite needs Python installed.

## Writing a program to optimise

Loop forever: read one line, parse it as a JSON array of floats, print your
fitness, flush. Higher is better.

```python
import sys, json
for line in sys.stdin:
    weights = json.loads(line)
    print(my_fitness(weights), flush=True)
```

## Examples

`examples/` has three working programs, standard library only.

| File | What it's for |
| --- | --- |
| `distance_sim.py` | The simplest possible worker — a 2D point scored on how close it is to a goal. One smooth basin, no local minima. Use it to check the engine is wired up correctly before pointing it at anything hard. |
| `rastrigin_sim.py` | The real benchmark. N-dimensional Rastrigin, covered in local optima but with a single global optimum in a wide bowl, so it actually rewards explore-then-refine and distinguishes the decaying nudge functions from a constant one. Per-evaluation cost is tunable so the worker pool has something to parallelise. |
| `reference_optimizer.py` | Not a worker — a plain Python GA over the same function, built to the same shape as the engine. It's the control: if it solves Rastrigin and the engine doesn't, the engine is the problem; if both plateau in the same place, that's just the population and mutation settings. |

```sh
# wire the sim into a config, then
go run ./cmd/engine

# or run the baseline on its own
python examples/reference_optimizer.py
```

## Layout

| Package | Responsibility |
| --- | --- |
| `internal/config` | JSON parsing, and `SimProcess` — the worker pipe |
| `internal/genome` | weight vectors, bounds, mutation |
| `internal/population` | agents, worker pool, ranking, generation building |
| `internal/nudge` | adaptive mutation scaling functions |
| `internal/simulation` | the outer loop and stopping conditions |

## Current state

Working end to end, with tests across every package. Still to do: the
stall-detection branch in `simulation.go` is partly commented out and needs
deciding on, and tuning parameters are hardcoded at the call site rather than
read from config.

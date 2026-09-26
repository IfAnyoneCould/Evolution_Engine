# Config Settings

List of JSON fields and what they do, along with their defaults. Only `program.path` and `bounds` are required,
everything else falls back to its default. The config path is the first argument:

```sh
go run ./cmd/engine path/to/config.json
```

Smallest working config:

```json
{
  "program": { "path": "cmd/engine/TestRun.exe", "args": ["8", "0", "5.12"] },
  "bounds": [[-5.12, 5.12], [-5.12, 5.12]]
}
```

Paths (`program.path`, `output.weight_path`) are relative to the directory the engine is run from, not to the config
file. Keys are case sensitive and unknown keys are ignored, so a misspelled key silently falls back to its default.

___

### program (required)

- path (string) &rarr; the executable to optimise. Required. One copy is started per worker and kept running for the
  whole run
- args ([]string) &rarr; arguments passed to the executable on startup. Default `[]`

---

### bounds ([][2]float64, required)

One `[lower, upper]` pair per weight. The number of pairs sets how many weights the sim gets, in the same order.
E.g. `[[-1,1],[0,10]]` is a sim with two weights, the first in [-1,1] and the second in [0,10]. Lower can equal upper
to hold a weight fixed. Lower above upper errors when the run starts.

---

### Mutation

Every child is a copy of a parent with each weight moved by a random amount, uniform in ±(nudge size × that weight's
range), then clamped to its bounds. The nudge size follows a schedule that shrinks as the parent gets closer to the
target:

```
u          = clamp((progress - hold) / (1 - hold), 0, 1)
f          = end + (1 - end) × shape(u)
nudge size = max(fraction × f, min_nudge)
```

progress runs from 0 at the first generation's best fitness to 1 at `run_settings.target_fitness`. Until progress
reaches hold the nudge stays at full size, then the shape takes it down to end by the target.

- fraction (float64) &rarr; the largest nudge, as a fraction of each weight's range. 0.05 on [-1,1] moves a weight by
  at most ±0.1. Default `0.05`
- min_nudge (float64) &rarr; floor on the nudge size, same units as fraction. With end above 0 the schedule already
  has a floor of fraction × end, so this only matters when that's smaller. Above fraction it just becomes the nudge
  size for the whole run. Default `0.0005`
- nudge_func &rarr; the schedule. Only give the fields the type uses
  - type (string) &rarr; one of the types below, case insensitive
  - hold (float64) &rarr; progress where the decay starts, 0 to 1. 1 never decays
  - end (float64) &rarr; the nudge at the target, as a fraction of the full nudge, 0 to 1
  - exponent (float64) &rarr; power only, above 0
  - rate (float64) &rarr; exponential only, above 0
  - steps (uint) &rarr; step only, 1 or more

| type | shape(u) | needs | how it behaves |
| --- | --- | --- | --- |
| constant | 1 | nothing, ignores hold and end | full nudge the whole run |
| linear | 1 − u | | steady decrease |
| quadratic | (1 − u)² | | drops fast right after hold, flattens near the end |
| power | (1 − u)^exponent | exponent | 1 is linear, 2 is quadratic, below 1 stays big longer then drops late |
| exponential | (e^(−rate·u) − e^(−rate)) / (1 − e^(−rate)) | rate | like quadratic with adjustable steepness, higher drops faster. Around 3 to 5 is typical |
| cosine | (1 + cos(π·u)) / 2 | | slow at the start, fastest in the middle, slow at the end |
| step | 1 − floor(u·steps) / steps | steps | drops in equal jumps and stays flat between them. 1 step is full nudge until the target |

Defaults: leave out nudge_func and it's quadratic with hold 0.45 and end 0.2, which keeps the full nudge until 45%
progress and ends at 20% of it. Give a type and hold and end default to 0 instead, so the schedule decays from the
first generation all the way down, unless you set them.

---

### run_settings

- target_fitness (float64) &rarr; the run stops as soon as the best fitness reaches this. Also where nudge progress
  ends. Must be in (0,1], sims are expected to score in that range with higher being better. Default `0.99`
- max_cycles (uint) &rarr; the most generations a run can go. Default `500`
- population_size (uint) &rarr; agents per generation. Must be greater than `selection.elite`. Default `60`

---

### workers (uint)

How many copies of the sim run at once. Each one is a process that stays alive for the whole run and gets fed one
genome at a time. More workers than CPU cores, or than population_size, doesn't speed anything up. Default `10`

---

### selection

- pressure (float64) &rarr; the top fraction of each generation that can be a parent. Each child picks its parent at
  random from the top population_size × pressure agents, so lower is greedier. Must be in (0,1], and
  population_size × pressure has to round down to at least 1 or the first generation errors. Default `0.45`
- elite (uint) &rarr; how many of the best agents are copied into the next generation unchanged. They aren't
  re-evaluated. Default `1`

---

### stagnation_detection

Ends a run that has stopped improving: after patience generations in a row where the best fitness hasn't beaten the
last real improvement by more than epsilon.

- patience (uint) &rarr; generations allowed without improvement. Counted in generations, not evals, so a smaller
  population needs a higher patience for the same number of evals. Must be above 0, to turn it off set it higher
  than max_cycles. Default `200`
- epsilon (float64) &rarr; the smallest gain in best fitness that counts as an improvement, in fitness units. Must be
  above 0. Default `0.01`

---

### sim_settings

- timeout_ms (float64) &rarr; how long to wait for one eval before giving up and ending the run. The first eval of
  each worker also includes the sim starting up, which can take a few hundred ms when every worker starts at once.
  Must be above 0. Default `5000`
- rand_seed (int64) &rarr; seed for the engine's randomness. Leave it out and one is picked from the clock and printed
  at the start of the run, so any run can be repeated by putting that seed here. Only covers the engine's side, a sim
  with its own randomness will still vary. Default: picked per run

---

### output

- weight_path (string) &rarr; file the best weights are written to when the run ends, as a JSON array in the same
  order as bounds. Overwritten every run. Empty means nothing is written. Default `""`

---

### Every field at its default

```json
{
  "program": { "path": "cmd/engine/TestRun.exe", "args": [] },
  "bounds": [[-5.12, 5.12], [-5.12, 5.12]],
  "fraction": 0.05,
  "min_nudge": 0.0005,
  "nudge_func": { "type": "quadratic", "hold": 0.45, "end": 0.2 },
  "run_settings": { "target_fitness": 0.99, "max_cycles": 500, "population_size": 60 },
  "workers": 10,
  "selection": { "pressure": 0.45, "elite": 1 },
  "stagnation_detection": { "patience": 200, "epsilon": 0.01 },
  "sim_settings": { "timeout_ms": 5000 },
  "output": { "weight_path": "" }
}
```

`program.path` and `bounds` have no defaults, the values above are just an example.

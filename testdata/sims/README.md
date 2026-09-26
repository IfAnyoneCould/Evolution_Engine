# Test sims

Simulations for testing the engine as a black box optimiser. Same protocol as
`examples/`, each one reads a JSON array of weights per line and prints a
fitness. Every sim scores in [0,1], and 0.99 only happens at or right next to
the optimum. Wrong weight counts score 0.

Each sim has a sel in `configs/` with its args and bounds filled in. Paths
are relative to the repo root.

The python ones are stdlib only. The go ones need building first

```sh
go build -o testdata/sims/go/bin/<name>.exe ./testdata/sims/go/<name>
```

Cost is per eval at the default args, measured on the pipe.

## Benchmark functions

| Sim | Dims | What it tests | Cost |
| --- | --- | --- | --- |
| `sphere` | 8 | convex baseline | <0.1ms |
| `rosenbrock` | 8 | curved narrow valley | <0.1ms |
| `ackley` | 8 | flat plateau with one funnel | <0.1ms |
| `schwefel` | 8 | deceptive, optimum near the bounds edge | <0.1ms |
| `griewank` | 8 | bowl with ripples, product coupling | <0.1ms |
| `levy` | 8 | wavy multimodal | <0.1ms |
| `styblinski_tang` | 8 | 2^n basins | <0.1ms |
| `michalewicz` | 8 | steep ridges and flat plateaus | <0.1ms |
| `ellipsoid` | 8 | ill conditioned, condition number arg | <0.1ms |
| `rotated_ellipsoid` | 8 | ellipsoid made non separable by a seeded rotation | <0.1ms |
| `rotated_rastrigin` | 8 | non separable multimodal | <0.1ms |
| `shifted_sphere` | 8 | optimum off the origin, catches origin bias | <0.1ms |
| `shifted_rastrigin` | 8 | same on rastrigin | <0.1ms |
| `step` | 8 | plateaus, no gradient | <0.1ms |
| `needle` | 8 | flat everywhere except one small ball | <0.1ms |
| `bent_cigar` | 8 | one loose dim, the rest 1e6 stiffer | <0.1ms |
| `zakharov` | 8 | coupled terms up to 4th power | <0.1ms |
| `dixon_price` | 8 | chained valley | <0.1ms |
| `eggholder` | 2 | rugged, optimum on the bound | <0.1ms |
| `himmelblau` | 2 | four equal optima | <0.1ms |
| `easom` | 2 | flat with one small hole | <0.1ms |
| `alpine` | 8 | v shaped kinks | <0.1ms |
| `trid` | 8 | tilted diagonal trough | <0.1ms |
| `sum_of_different_powers` | 8 | uneven sensitivity per dim | <0.1ms |
| `high_conditioning_elliptic` | 8 | CEC style, shifted, cond 1e6 | <0.1ms |
| `cliff` | 8 | optimum right on the edge of a drop to 0 | <0.1ms |

## Physics and control

| Sim | Dims | What it tests | Cost |
| --- | --- | --- | --- |
| `projectile` | 3 | hit a distance with drag and spin, sanity check | 0.4ms |
| `cannon_wall` | 2 | clear a wall into a zone, plateaus and cliffs | 0.3ms |
| `cartpole` | 4 | linear policy balancing a pole | <0.1ms |
| `pendulum_swingup` | 21 | tiny net, pump then catch, deceptive | 2ms |
| `acrobot` | 7 | chaotic two link, rough for its size | 5ms |
| `lunar_lander` | 20 | open loop thrust schedule, soft landing | 2ms |
| `orbit_insertion` | 20 | burn schedule to a circular orbit, hard | 2ms |
| `robot_arm_ik` | 6 | reach a point around an obstacle, redundant | <0.1ms |
| `pid_tuning` | 3 | pid gains on a second order plant | 4ms |
| `mass_spring_damper_fit` | 3 | system id, long thin valley | 3ms |
| `lotka_volterra_fit` | 4 | fit predator prey rates, fake basins | 3ms |
| `heat_source_inverse` | 6 | find heat sources on a rod, symmetric optima | 230ms |
| `racing_line` | 40 | open loop lap, very ill conditioned | 2.5ms |
| `boids` | 5 | flocking gains that fight each other | 50ms |
| `walker` | 13 | spring creature gait, chaotic | 60ms |
| `traffic_lights` | 6 | signal timing, optional noisy fitness | 5ms |

## Discrete, ML and odd ones

| Sim | Dims | What it tests | Cost |
| --- | --- | --- | --- |
| `xor_net` | 9 | tiny net on xor, symmetries | <0.1ms |
| `sine_regression` | 25 | small mlp fit to sin | 0.1ms |
| `classifier_spirals` | 105 | two spirals, hard | 0.8ms |
| `poly_fit` | 6 | badly conditioned polynomial fit | <0.1ms |
| `tsp_random_keys` | 20 | permutation through argsort | <0.1ms |
| `knapsack` | 30 | binary through a threshold, capacity penalty | <0.1ms |
| `onemax_trap` | 40 | deceptive trap blocks | <0.1ms |
| `nk_landscape` | 20 | tunable ruggedness with k | <0.1ms |
| `pressure_vessel` | 4 | constrained design, stepped dims | <0.1ms |
| `portfolio` | 10 | normalized allocation, many genomes per answer | <0.1ms |
| `integer_lattice` | 10 | rounded to ints, plateaus everywhere | <0.1ms |
| `noisy_sphere` | 5 | fresh noise on every eval | <0.1ms |
| `moving_target` | 3 | optimum drifts with eval count, per worker | <0.1ms |
| `deceptive_gradient` | 5 | slope leads away from the optimum | <0.1ms |
| `multi_objective_scalarized` | 4 | worse of two objectives, thin knee | <0.1ms |
| `modular` | 32 | block products, one bad block tanks it | <0.1ms |
| `image_circles` | 80 | rasterize circles to match an image | 50ms |
| `game_of_life` | 256 | match a target grid after t conway steps, chaotic | 0.8ms |
| `high_dim_sphere_rastrigin` | 200 | genome size and json throughput | 0.3ms |
| `sleepy` | 5 | sleeps per eval plus jitter and startup delay, for the pool and timeouts | 100ms |

## Go

| Sim | Dims | What it tests | Cost |
| --- | --- | --- | --- |
| `go/sphere` | 10 | fastest possible eval | <0.1ms |
| `go/rastrigin` | 10 | same, multimodal | <0.1ms |
| `go/nk_landscape` | 20 | port of nk_landscape, different landscape for the same seed | <0.1ms |
| `go/cpu_burn` | 10 | rastrigin plus real work, dial it up to ~1s | 70ms |

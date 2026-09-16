"""
a plain python GA over the same rastrigin function, kept as a baseline to check
the engine against.

the point of it is telling apart "the engine is underperforming" from "this
problem is just hard". if this reaches 0.99 in 200 generations and the engine
plateaus at 0.91, the gap is the engine's fault. if this plateaus around 0.91
too then 0.91 is simply what this population size and mutation regime gets you
and the engine is fine.

built it the same shape as the engine on purpose so the comparison is fair -
same fitness function, same genome of n floats in [-5.12, 5.12], and truncation
selection + elitism + gaussian mutation, which is the same family as the
nudge-based reproduction.

standard library only, so it runs with no setup:  python reference_optimizer.py

set the config block to match whatever the engine is running first, then move
it around to see what rastrigin can actually be solved with.
"""

import random
import math

# ---------------------------------------------------------------------------
# config. match the engine first, then experiment
# ---------------------------------------------------------------------------
N_DIMS          = 8          # genome length, match the bounds count
BOUND_LO        = -5.12
BOUND_HI        = 5.12
POP_SIZE        = 100        # population size, try 100-300
GENERATIONS     = 500        # max generations
ELITE_COUNT     = 2          # top n carried over untouched
SURVIVOR_FRAC   = 0.5        # truncation, the top fraction allowed to parent
MUTATION_STD    = 0.30       # gaussian mutation std dev, in absolute units.
                             #   0.30 over a ~10 wide range is about the same
                             #   as a 0.05 fraction, but gaussian, so it still
                             #   throws the occasional big jump that escapes a
                             #   local trap
MUTATION_RATE   = 0.5        # chance each gene mutates per reproduction
TARGET_FITNESS  = 0.99
SEED            = 42
# ---------------------------------------------------------------------------

random.seed(SEED)
A = 10.0

def rastrigin(x):
    return A * len(x) + sum(v*v - A*math.cos(2*math.pi*v) for v in x)

def fitness(genome):
    raw = rastrigin(genome)                       # 0 at the optimum, bigger is worse
    span = BOUND_HI                               # same normalization as the sim
    worst = A*N_DIMS + N_DIMS*(span*span + A)
    return max(0.0, 1.0 - raw/worst)

def random_genome():
    return [random.uniform(BOUND_LO, BOUND_HI) for _ in range(N_DIMS)]

def clamp(v, lo, hi):
    return max(lo, min(v, hi))

def mutate(genome):
    child = genome[:]                             # copy so the child is independent
    for i in range(len(child)):
        if random.random() < MUTATION_RATE:
            child[i] = clamp(child[i] + random.gauss(0, MUTATION_STD),
                             BOUND_LO, BOUND_HI)
    return child

def main():
    pop = [random_genome() for _ in range(POP_SIZE)]

    cutoff = max(1, int(POP_SIZE * SURVIVOR_FRAC))

    # rank the starting population up front, so the summary at the bottom still
    # has something to print if GENERATIONS is set to 0
    scored = sorted(((fitness(g), g) for g in pop),
                    key=lambda t: t[0], reverse=True)
    best_fit = scored[0][0]

    for gen in range(GENERATIONS):
        # evaluate and rank, best first
        scored = sorted(((fitness(g), g) for g in pop),
                        key=lambda t: t[0], reverse=True)

        best_fit = scored[0][0]
        worst_fit = scored[-1][0]
        mean_fit  = sum(f for f, _ in scored) / len(scored)

        # spread is the diversity signal. when it collapses the population has
        # converged and nothing more is coming
        print(f"gen {gen:4d}: best={best_fit:.4f} mean={mean_fit:.4f} "
              f"worst={worst_fit:.4f} spread={best_fit-worst_fit:.4f}")

        if best_fit >= TARGET_FITNESS:
            print(f"\nreached target {TARGET_FITNESS} at generation {gen}")
            print(f"best genome: {[round(v,3) for v in scored[0][1]]}")
            return

        survivors = [g for _, g in scored[:cutoff]]
        next_pop = [g[:] for _, g in scored[:ELITE_COUNT]]   # elites, untouched
        while len(next_pop) < POP_SIZE:
            parent = random.choice(survivors)
            next_pop.append(mutate(parent))
        pop = next_pop

    print(f"\nstopped at generation {GENERATIONS}, best fitness {best_fit:.4f}")
    print(f"best genome: {[round(v,3) for v in scored[0][1]]}")

if __name__ == "__main__":
    main()

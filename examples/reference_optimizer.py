import random
import math

# a plain python GA over the same rastrigin function, as a control. if this
# hits 0.99 and the engine plateaus at 0.91 then the engine is the problem. if
# this plateaus in the same place then 0.91 is just what these settings get you
#
# same shape as the engine on purpose so the comparison is fair. same fitness,
# same genome of n floats in the bounds, truncation + elitism + gaussian
# mutation, which is the same family as the nudge. stdlib only
#   python reference_optimizer.py

# match the engine first, then move these around
N_DIMS          = 8
BOUND_LO        = -5.12
BOUND_HI        = 5.12
POP_SIZE        = 100
GENERATIONS     = 500
ELITE_COUNT     = 2     # top n carried over untouched
SURVIVOR_FRAC   = 0.5   # top fraction allowed to parent
MUTATION_STD    = 0.30  # absolute, not a fraction. 0.30 over a ~10 wide range is
                        # about the same as 0.05, but gaussian, so it still throws
                        # the odd big jump that gets out of a local trap
MUTATION_RATE   = 0.5   # chance per gene per reproduction
TARGET_FITNESS  = 0.99
SEED            = 42

random.seed(SEED)
A = 10.0

def rastrigin(x):
    return A * len(x) + sum(v*v - A*math.cos(2*math.pi*v) for v in x)

def fitness(genome):
    raw = rastrigin(genome)
    worst = A*N_DIMS + N_DIMS*(BOUND_HI*BOUND_HI + A) # same normalization as the sim
    return max(0.0, 1.0 - raw/worst)

def random_genome():
    return [random.uniform(BOUND_LO,BOUND_HI) for _ in range(N_DIMS)]

def clamp(v,lo,hi):
    return max(lo, min(v, hi))

def mutate(genome):
    child = genome[:]
    for i in range(len(child)):
        if random.random() < MUTATION_RATE:
            child[i] = clamp(child[i] + random.gauss(0,MUTATION_STD),BOUND_LO,BOUND_HI)
    return child

def rank(pop):
    return sorted(((fitness(g),g) for g in pop), key=lambda t: t[0], reverse=True)

def main():
    pop = [random_genome() for _ in range(POP_SIZE)]
    cutoff = max(1, int(POP_SIZE * SURVIVOR_FRAC))

    scored = rank(pop) # ranked up front so the summary has something at 0 generations
    best_fit = scored[0][0]

    for gen in range(GENERATIONS):
        scored = rank(pop)

        best_fit = scored[0][0]
        worst_fit = scored[-1][0]
        mean_fit = sum(f for f,_ in scored) / len(scored)

        # spread is the diversity signal, once it collapses nothing more is coming
        print(f"gen {gen:4d}: best={best_fit:.4f} mean={mean_fit:.4f} "
              f"worst={worst_fit:.4f} spread={best_fit-worst_fit:.4f}")

        if best_fit >= TARGET_FITNESS:
            print(f"\nreached target {TARGET_FITNESS} at generation {gen}")
            print(f"best genome: {[round(v,3) for v in scored[0][1]]}")
            return

        survivors = [g for _,g in scored[:cutoff]]
        next_pop = [g[:] for _,g in scored[:ELITE_COUNT]]
        while len(next_pop) < POP_SIZE:
            next_pop.append(mutate(random.choice(survivors)))
        pop = next_pop

    print(f"\nstopped at generation {GENERATIONS}, best fitness {best_fit:.4f}")
    print(f"best genome: {[round(v,3) for v in scored[0][1]]}")


if __name__ == "__main__":
    main()

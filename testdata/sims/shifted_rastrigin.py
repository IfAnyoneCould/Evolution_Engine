import sys
import json
import random
from math import cos
from math import pi
from math import log1p
from math import trunc

# rastrigin with the global optimum moved off the origin by a seeded offset.
# plain rastrigin rewards anything that drifts to zero, this one doesnt
# args: n dims, seed (for the offset), span (half width of each range)

A = 10.0
SCALE = 20.0 # log squashed, 0.99 is about raw < 0.2. nearest local optima are ~1

def offset(n,seed,span):
    rng = random.Random(seed)
    return [rng.uniform(-0.8 * span, 0.8 * span) for _ in range(n)]

def rastrigin(x):
    total = A * len(x)
    for v in x:
        total += v * v - A * cos(2.0 * pi * v)
    return total

def compute(weights,n,goal):
    if len(weights) != n:
        return 0.0 # bad weight count, score it worst instead of dying on it

    raw = max(0.0, rastrigin([v - g for v, g in zip(weights, goal)]))
    return trunc(1 / (1 + log1p(raw / SCALE)) * 10000) / 10000

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    span = float(sys.argv[3]) if len(sys.argv) > 3 else 5.12
    goal = offset(n,seed,span)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights,n,goal)
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

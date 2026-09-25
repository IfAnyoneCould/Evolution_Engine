import sys
import json
import random
from math import sqrt
from math import trunc

# needle in a haystack. totally flat except one small ball somewhere seeded,
# and inside the ball its a cone up to the center. theres nothing to follow
# until something lands in it, so its pure exploration
# args: n dims, radius (of the ball), seed (for where it is), span (half width of each range)

FLAT = 0.1 # not 0, a flat 0 trips the stall check on the first cycle

def center(n,seed,span):
    rng = random.Random(seed)
    return [rng.uniform(-0.5 * span, 0.5 * span) for _ in range(n)]

def compute(weights,n,radius,goal):
    if len(weights) != n:
        return 0.0 # bad weight count, score it worst instead of dying on it

    dist = sqrt(sum((v - g) ** 2 for v, g in zip(weights, goal)))
    if dist >= radius:
        return FLAT
    return trunc((1 - (1 - FLAT) * dist / radius) * 10000) / 10000

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    radius = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    span = float(sys.argv[4]) if len(sys.argv) > 4 else 5.12
    goal = center(n,seed,span)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights,n,radius,goal)
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

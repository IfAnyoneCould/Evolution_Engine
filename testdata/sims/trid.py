import sys
import json
from math import log1p
from math import trunc

# a convex bowl, but the neighbor products tilt it into a long diagonal trough
# and the optimum is at x_i = i*(n+1-i), nowhere near the origin or the middle.
# bounds are +-n^2
# args: n dims

SCALE = 100.0 # log squashed, 0.99 is about raw < 1

def trid(x):
    total = sum((v - 1.0) ** 2 for v in x)
    for i in range(1, len(x)):
        total -= x[i] * x[i - 1]
    return total

def compute(weights,n):
    if len(weights) != n:
        return 0.0 # bad weight count, score it worst instead of dying on it

    f_min = -n * (n + 4) * (n - 1) / 6.0
    raw = max(0.0, trid(weights) - f_min)
    return trunc(1 / (1 + log1p(raw / SCALE)) * 10000) / 10000

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights,n)
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

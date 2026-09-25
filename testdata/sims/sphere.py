import sys
import json
from math import log1p
from math import trunc

# the baseline. one convex bowl at the origin, nothing to get stuck in. if the
# optimizer cant do this one nothing else matters
# args: n dims

SCALE = 2.0 # raw spans a lot so it gets log squashed, 0.99 is about raw < SCALE/100

def sphere(x):
    return sum(v * v for v in x)

def compute(weights,n):
    if len(weights) != n:
        return 0.0 # bad weight count, score it worst instead of dying on it

    raw = sphere(weights)
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

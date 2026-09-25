import sys
import json
from math import floor
from math import log1p
from math import trunc

# a sphere rounded into flat terraces. every nudge that stays on the same step
# scores exactly the same, so theres no gradient inside a step, only at the
# edges. tests plateau handling and the stall check
# args: n dims

SCALE = 1.0 # log squashed. the optimum is the whole unit cube around the origin

def step(x):
    return sum(floor(v + 0.5) ** 2 for v in x)

def compute(weights,n):
    if len(weights) != n:
        return 0.0 # bad weight count, score it worst instead of dying on it

    raw = step(weights)
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

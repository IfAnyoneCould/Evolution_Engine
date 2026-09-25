import sys
import json
from math import sin
from math import sqrt
from math import log1p
from math import trunc

# the deceptive one. the best basin is out at ~420.97 near the edge of the box,
# and the second best is way over on the other side, so following the local
# trend walks away from it
# args: n dims

SCALE = 100.0 # log squashed, 0.99 is about raw < 1. one dim in the wrong basin costs 100+

def schwefel(x):
    return 418.9828872724338 * len(x) - sum(v * sin(sqrt(abs(v))) for v in x)

def compute(weights,n):
    if len(weights) != n:
        return 0.0 # bad weight count, score it worst instead of dying on it

    raw = max(0.0, schwefel(weights))
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

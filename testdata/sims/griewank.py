import sys
import json
from math import cos
from math import sqrt
from math import log1p
from math import trunc

# a big shallow bowl with fine ripples on top. zoomed out its a sphere, zoomed
# in its a grid of local optima, and the ones next to the origin are almost as
# good as it
# args: n dims

SCALE = 0.5 # log squashed, 0.99 is about raw < 0.005. nearest local optima sit around 0.0099

def griewank(x):
    total = 0.0
    prod = 1.0
    for i, v in enumerate(x):
        total += v * v / 4000.0
        prod *= cos(v / sqrt(i + 1))
    return 1.0 + total - prod

def compute(weights,n):
    if len(weights) != n:
        return 0.0 # bad weight count, score it worst instead of dying on it

    raw = max(0.0, griewank(weights))
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

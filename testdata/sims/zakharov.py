import sys
import json
from math import log1p
from math import trunc

# a single basin at the origin, but the weighted sum term couples every dim
# together and blows up to the 4th power, so its steep and non-separable far
# out and nearly flat close in
# args: n dims

SCALE = 20.0 # log squashed, 0.99 is about raw < 0.2

def zakharov(x):
    sq = sum(v * v for v in x)
    s = sum(0.5 * (i + 1) * v for i, v in enumerate(x))
    return sq + s ** 2 + s ** 4

def compute(weights,n):
    if len(weights) != n:
        return 0.0 # bad weight count, score it worst instead of dying on it

    raw = zakharov(weights)
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

import sys
import json
from math import log1p
from math import trunc

# sum of |x_i|^(i+1). the first dim is a normal bowl, later ones get flatter
# and flatter near 0, so the late dims barely register until theyre way off.
# unequal sensitivity without being a straight ellipsoid
# args: n dims

SCALE = 0.01 # log squashed, 0.99 is about raw < 1e-4

def sum_of_different_powers(x):
    return sum(abs(v) ** (i + 2) for i, v in enumerate(x))

def compute(weights,n):
    if len(weights) != n:
        return 0.0 # bad weight count, score it worst instead of dying on it

    raw = sum_of_different_powers(weights)
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

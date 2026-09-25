import sys
import json
from math import log1p
from math import trunc

# one loose dim and every other dim a million times stiffer. a long thin
# cigar, the optimizer has to nail n-1 dims tight while the first one barely
# matters. the extreme version of ellipsoid.py
# args: n dims

SCALE = 1e4 # log squashed, 0.99 is about raw < 100

def bent_cigar(x):
    return x[0] * x[0] + 1e6 * sum(v * v for v in x[1:])

def compute(weights,n):
    if len(weights) != n:
        return 0.0 # bad weight count, score it worst instead of dying on it

    raw = bent_cigar(weights)
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

import sys
import json
from math import log1p
from math import trunc

# each dim has two dips, the better one at -2.9035 and a worse one at 2.7468,
# so theres 2^n basins and only one has every dim on the right side
# args: n dims

SCALE = 20.0 # log squashed, 0.99 is about raw < 0.2. one dim in the wrong dip costs ~14
DIM_MIN = -39.16616570377142

def styblinski_tang(x):
    return 0.5 * sum(v ** 4 - 16.0 * v * v + 5.0 * v for v in x)

def compute(weights,n):
    if len(weights) != n:
        return 0.0 # bad weight count, score it worst instead of dying on it

    raw = max(0.0, styblinski_tang(weights) - DIM_MIN * n)
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

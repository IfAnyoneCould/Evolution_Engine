import sys
import json
from math import log1p
from math import trunc

# a curved valley like rosenbrock but each dim is chained to the one before it,
# and the optimum is at an odd spot, x_i = 2^-((2^i - 2) / 2^i). the last dim
# can be either sign so there are two equal optima
# args: n dims

SCALE = 20.0 # log squashed, 0.99 is about raw < 0.2

def dixon_price(x):
    total = (x[0] - 1.0) ** 2
    for i in range(1, len(x)):
        total += (i + 1) * (2.0 * x[i] * x[i] - x[i - 1]) ** 2
    return total

def compute(weights,n):
    if len(weights) != n:
        return 0.0 # bad weight count, score it worst instead of dying on it

    raw = dixon_price(weights)
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

import sys
import json
from math import log1p
from math import trunc

# a bowl thats been squashed flat. each dim gets weighted harder than the last,
# the stiffest one by cond times the loosest. same step size for every dim, so
# one nudge is either too big for the stiff dims or too small for the loose ones
# args: n dims, cond (condition number, 1e6 is the classic)

SCALE = 100.0 # log squashed, 0.99 is about raw < 1

def ellipsoid(x,cond):
    n = len(x)
    if n == 1:
        return x[0] * x[0]
    return sum(cond ** (i / (n - 1)) * v * v for i, v in enumerate(x))

def compute(weights,n,cond):
    if len(weights) != n:
        return 0.0 # bad weight count, score it worst instead of dying on it

    raw = ellipsoid(weights,cond)
    return trunc(1 / (1 + log1p(raw / SCALE)) * 10000) / 10000

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    cond = float(sys.argv[2]) if len(sys.argv) > 2 else 1e6

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights,n,cond)
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

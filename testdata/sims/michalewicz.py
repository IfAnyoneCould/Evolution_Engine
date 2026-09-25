import sys
import json
from math import sin
from math import pi
from math import trunc

# flat almost everywhere with steep narrow ridges, and the ridges get thinner
# the higher the dim index. bigger m means steeper. tests if the optimizer can
# find something that thin and then stay on it
# args: n dims, m (steepness, 10 is the classic)

def term(v,i,m):
    return sin(v) * sin((i + 1) * v * v / pi) ** (2 * m)

def best_term(i,m):
    # separable, so the optimum is just the best spot in each dim. grid then refine
    steps = 20000
    best_v = 0.0
    for k in range(steps + 1):
        v = pi * k / steps
        if term(v,i,m) > term(best_v,i,m):
            best_v = v
    lo = max(0.0, best_v - pi / steps)
    hi = min(pi, best_v + pi / steps)
    for _ in range(60):
        a = lo + (hi - lo) / 3
        b = hi - (hi - lo) / 3
        if term(a,i,m) < term(b,i,m):
            lo = a
        else:
            hi = b
    return max(term(best_v,i,m), term((lo + hi) / 2,i,m))

def compute(weights,n,m,best):
    if len(weights) != n:
        return 0.0 # bad weight count, score it worst instead of dying on it

    total = sum(term(v,i,m) for i, v in enumerate(weights))
    return trunc(max(0.0, min(1.0, total / best)) * 10000) / 10000

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    best = sum(best_term(i,m) for i in range(n))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights,n,m,best)
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

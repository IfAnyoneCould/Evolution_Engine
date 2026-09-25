import sys
import json
from math import sin
from math import pi
from math import log1p
from math import trunc

# lots of local optima on a wavy surface, global one at (1,...,1). the waves
# are tall enough that small steps get stuck in whichever one they start in
# args: n dims

SCALE = 10.0 # log squashed, 0.99 is about raw < 0.1

def levy(x):
    w = [1.0 + (v - 1.0) / 4.0 for v in x]
    total = sin(pi * w[0]) ** 2
    for i in range(len(w) - 1):
        total += (w[i] - 1.0) ** 2 * (1.0 + 10.0 * sin(pi * w[i] + 1.0) ** 2)
    total += (w[-1] - 1.0) ** 2 * (1.0 + sin(2.0 * pi * w[-1]) ** 2)
    return total

def compute(weights,n):
    if len(weights) != n:
        return 0.0 # bad weight count, score it worst instead of dying on it

    raw = levy(weights)
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

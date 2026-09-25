import sys
import json
from math import cos
from math import exp
from math import sqrt
from math import e
from math import pi
from math import trunc

# mostly a flat bumpy plateau with one deep funnel at the origin. far out theres
# barely any signal, so it tests whether big early nudges find the funnel
# before the small ones matter
# args: n dims

def ackley(x):
    n = len(x)
    sq = sum(v * v for v in x) / n
    cs = sum(cos(2.0 * pi * v) for v in x) / n
    return -20.0 * exp(-0.2 * sqrt(sq)) - exp(cs) + 20.0 + e

def compute(weights,n):
    if len(weights) != n:
        return 0.0 # bad weight count, score it worst instead of dying on it

    raw = ackley(weights) # 0 at the optimum, tops out around 20+e
    return trunc(max(0.0, 1 - raw / (20.0 + e)) * 10000) / 10000

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

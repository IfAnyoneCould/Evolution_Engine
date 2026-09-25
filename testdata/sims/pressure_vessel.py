import sys
import json
from math import pi
from math import trunc

# the classic pressure vessel design problem. minimize material cost of a
# cylindrical vessel with hemispherical heads subject to 4 constraints. the two
# thicknesses only come in multiples of 0.0625in so those dims are plateaus.
# score is best known cost over penalized cost, the feasible corner is narrow
# args: none. genome is [shell thickness, head thickness, inner radius, length]

BEST = 6059.714 # best known with the discrete thicknesses
STEP = 0.0625

def plate(v):
    return max(1, round(v / STEP)) * STEP

def cost(ts,th,r,l):
    return 0.6224*ts*r*l + 1.7781*th*r*r + 3.1661*ts*ts*l + 19.84*ts*ts*r

def violation(ts,th,r,l):
    g = [
        0.0193 * r - ts,
        0.00954 * r - th,
        (1296000 - pi*r*r*l - 4/3*pi*r**3) / 1296000,
        (l - 240) / 240,
    ]
    return sum(max(0.0, v) for v in g)

def compute(weights):
    if len(weights) != 4:
        return 0.0

    ts = plate(weights[0])
    th = plate(weights[1])
    r = max(1e-6, weights[2])
    l = max(1e-6, weights[3])
    total = cost(ts,th,r,l) + 1e6 * violation(ts,th,r,l)
    return trunc(min(1.0, BEST / total) * 10000) / 10000

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights)
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

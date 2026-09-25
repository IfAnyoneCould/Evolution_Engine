import sys
import json
import random
from math import cos
from math import pi
from math import sqrt
from math import log1p
from math import trunc

# rastrigin spun by a fixed random rotation. the grid of local optima no longer
# lines up with the weight axes, so fixing one dim at a time doesnt work. still
# at the origin, just non-separable now
# args: n dims, seed (for the rotation)

A = 10.0
SCALE = 20.0 # log squashed, 0.99 is about raw < 0.2. nearest local optima are ~1

def rotation(n,seed):
    # random gaussian rows, gram-schmidt them into an orthonormal basis
    rng = random.Random(seed)
    rows = []
    while len(rows) < n:
        v = [rng.gauss(0.0, 1.0) for _ in range(n)]
        for r in rows:
            d = sum(a * b for a, b in zip(v, r))
            v = [a - d * b for a, b in zip(v, r)]
        norm = sqrt(sum(a * a for a in v))
        if norm > 1e-9:
            rows.append([a / norm for a in v])
    return rows

def rastrigin(x):
    total = A * len(x)
    for v in x:
        total += v * v - A * cos(2.0 * pi * v)
    return total

def compute(weights,n,rot):
    if len(weights) != n:
        return 0.0 # bad weight count, score it worst instead of dying on it

    z = [sum(a * b for a, b in zip(r, weights)) for r in rot]
    raw = max(0.0, rastrigin(z))
    return trunc(1 / (1 + log1p(raw / SCALE)) * 10000) / 10000

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    rot = rotation(n,seed)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights,n,rot)
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

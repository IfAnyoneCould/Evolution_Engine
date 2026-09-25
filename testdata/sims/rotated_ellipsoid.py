import sys
import json
import random
from math import sqrt
from math import log1p
from math import trunc

# ellipsoid.py but spun by a fixed random rotation, so the stiff direction
# isnt lined up with any one weight anymore. nudging dims one at a time or
# per dim step sizes stop helping, it tests non-separable ill conditioning
# args: n dims, cond (condition number), seed (for the rotation)

SCALE = 100.0 # log squashed, 0.99 is about raw < 1

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

def ellipsoid(x,cond):
    n = len(x)
    if n == 1:
        return x[0] * x[0]
    return sum(cond ** (i / (n - 1)) * v * v for i, v in enumerate(x))

def compute(weights,n,cond,rot):
    if len(weights) != n:
        return 0.0 # bad weight count, score it worst instead of dying on it

    z = [sum(a * b for a, b in zip(r, weights)) for r in rot]
    raw = ellipsoid(z,cond)
    return trunc(1 / (1 + log1p(raw / SCALE)) * 10000) / 10000

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    cond = float(sys.argv[2]) if len(sys.argv) > 2 else 1e6
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    rot = rotation(n,seed)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights,n,cond,rot)
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

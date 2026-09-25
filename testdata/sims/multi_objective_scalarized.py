import sys
import json
import random
from math import trunc

# two conflicting objectives, be close to point a and be close to point b. the
# pareto front is the segment between them. blended tchebycheff style (score is
# the worse of the two) so the surface has a sharp crease along the balance line
# and only a thin knee around the midpoint reaches 0.99
# args: n dims, sharpness (power on the score), span, seed

def compute(weights,a,b,d,sharp):
    if len(weights) != len(a):
        return 0.0

    ua = sum((x - y) ** 2 for x,y in zip(weights,a)) / d
    ub = sum((x - y) ** 2 for x,y in zip(weights,b)) / d
    worse = 1 / (1 + max(ua,ub)) / 0.8 # 0.8 is what the midpoint gets
    return trunc(min(1.0, worse) ** sharp * 10000) / 10000

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    sharp = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    span = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    rng = random.Random(seed)
    a = [rng.uniform(-span * 0.8,span * 0.8) for _ in range(n)]
    b = [rng.uniform(-span * 0.8,span * 0.8) for _ in range(n)]
    d = sum((x - y) ** 2 for x,y in zip(a,b))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights,a,b,d,sharp)
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

import sys
import json
import random
from math import sin
from math import sqrt
from math import pi
from math import trunc

# non-stationary. the goal point drifts along a lissajous-ish path as evals go
# by, so an elite that was great a few generations ago slowly goes stale. the
# clock is this process's eval count, so each worker process has its own clock
# args: n dims, drift speed (radians of path per eval), span, seed

def goal(t,phase,rate,span):
    return [0.6 * span * sin(rate[i] * t + phase[i]) for i in range(len(phase))]

def compute(weights,n,t,phase,rate,span):
    if len(weights) != n:
        return 0.0

    g = goal(t,phase,rate,span)
    dist = sqrt(sum((x - y) ** 2 for x,y in zip(weights,g)))
    return trunc(max(0.0, 1 - dist / (2 * span * sqrt(n))) * 10000) / 10000

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    speed = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0005
    span = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    rng = random.Random(seed)
    phase = [rng.uniform(0,2 * pi) for _ in range(n)]
    rate = [speed * rng.uniform(0.5,1.5) for _ in range(n)]

    t = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights,n,t,phase,rate,span)
        t += 1
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

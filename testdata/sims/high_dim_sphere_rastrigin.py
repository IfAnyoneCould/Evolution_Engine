import sys
import json
from math import cos
from math import pi
from math import sqrt
from math import trunc

# big genome stress test. sphere with a rastrigin ripple mixed in, 200 dims by
# default so the json lines are a few kb each and the genome ops get exercised.
# sqrt on the normalized value so 0.99 means basically every dim in the center
# basin, not just most of them
# args: n dims, mix (0 is pure sphere, 1 is full rastrigin), span

A = 10.0

def compute(weights,n,mix,span):
    if len(weights) != n:
        return 0.0

    raw = 0.0
    for v in weights:
        raw += v * v + mix * A * (1 - cos(2 * pi * v))
    worst = n * (span * span + 2 * mix * A)
    return trunc(max(0.0, 1 - sqrt(raw / worst)) * 10000) / 10000

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    mix = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    span = float(sys.argv[3]) if len(sys.argv) > 3 else 5.12

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights,n,mix,span)
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

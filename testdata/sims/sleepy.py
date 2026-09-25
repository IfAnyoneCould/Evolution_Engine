import sys
import json
import random
import time
from math import sqrt
from math import trunc

# stand in for an expensive black box. the fitness is just a sphere, the cost is
# a sleep per eval plus optional random jitter, and an optional delay before it
# starts reading. for poking at the worker pool and the per-eval timeout
# (jitter past the timeout should get the eval killed)
# args: n dims, sleep ms, jitter ms (uniform extra 0..jitter), startup ms, span

def compute(weights,n,span):
    if len(weights) != n:
        return 0.0

    dist = sqrt(sum(x * x for x in weights) / (n * span * span))
    return trunc(max(0.0, 1 - dist) * 10000) / 10000

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    sleep_ms = float(sys.argv[2]) if len(sys.argv) > 2 else 100
    jitter_ms = float(sys.argv[3]) if len(sys.argv) > 3 else 0
    startup_ms = float(sys.argv[4]) if len(sys.argv) > 4 else 0
    span = float(sys.argv[5]) if len(sys.argv) > 5 else 5.0

    time.sleep(startup_ms / 1000)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        time.sleep((sleep_ms + random.uniform(0,jitter_ms)) / 1000)
        score = compute(weights,n,span)
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

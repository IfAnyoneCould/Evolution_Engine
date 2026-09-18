import sys
import json
from math import cos
from math import pi
from math import trunc

# the real benchmark. rastrigin is covered in local optima but the global one
# sits at the origin in a wide bowl, so big early steps find the bowl and small
# late ones settle into it. thats the case a decaying nudge is supposed to win
# args: n dims, work iters, span (half width of each range, 5.12 is the classic)

A = 10.0

def rastrigin(x):
    total = A * len(x)
    for v in x:
        total += v * v - A * cos(2.0 * pi * v)
    return total

def busy_work(iters,seed):
    # filler, not part of the fitness. just makes an eval cost something so the
    # worker pool has work to spread out and batch timings mean anything
    acc = 0.0
    v = seed
    for _ in range(iters):
        v = (v * 1.000001) + 0.5
        acc += cos(v) * cos(v) + (v - int(v))
    return acc

def compute(weights,n,work_iters,span):
    if len(weights) != n:
        return 0.0 # bad weight count, score it worst instead of dying on it

    if work_iters > 0:
        busy_work(work_iters,weights[0])

    raw = rastrigin(weights) # 0 at the optimum, bigger is worse
    worst = A * n + n * (span * span + A) # rough upper bound, doesnt need to be tight
    return trunc(max(0.0, 1 - raw / worst) * 10000) / 10000

def main():
    n = int(sys.argv[1])
    work_iters = int(sys.argv[2])
    span = float(sys.argv[3])

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights,n,work_iters,span)
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

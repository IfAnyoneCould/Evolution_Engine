import sys
import json
from math import cos, pi

# a rastrigin worker for the engine. rastrigin is a standard optimization
# benchmark and its a good fit here for two reasons.
#
# first, the landscape rewards explore-then-refine. the cosine terms cover it in
# local optima, but there is a single global optimum at the origin sitting in a
# wide bowl. so big early steps are what find the right bowl and escape the
# traps, and small late steps are what settle onto the bottom of it. that is
# exactly the regime where a decaying nudge should beat a constant one, unlike
# the distance sim whose single smooth basin favors constant.
#
# second, its n-dimensional and the per-eval cost is tunable, so the worker pool
# actually has something to chew on and batch timings mean something.
#
# protocol is the same as any other worker: read one json array of weights per
# line on stdin, print one fitness per line, flush. higher is better, roughly
# in [0,1].
#
# args:
#   argv[1]  n           number of dimensions. has to match the bounds count
#   argv[2]  work_iters  extra trig per eval to fake a heavier simulation.
#                        0 for a light run, crank it to stress the pool
#   argv[3]  span        half width of each parameter range, only used to
#                        normalize fitness. 5.12 for the classic domain
#
# so bounds should be n entries of [-5.12, 5.12] and span 5.12.

A = 10.0  # rastrigin constant

def rastrigin(x):
    # global minimum of 0 at the origin, grows and gets bumpy away from it
    n = len(x)
    total = A * n
    for xi in range(n):
        v = x[xi]
        total += v * v - A * cos(2.0 * pi * v)
    return total

def busy_work(iters, seed):
    # pure filler. not part of the fitness, its here so each evaluation costs
    # something and parallelism across the worker pool is actually visible
    acc = 0.0
    v = seed
    for _ in range(iters):
        v = (v * 1.000001) + 0.5
        acc += cos(v) * cos(v) + (v - int(v))
    return acc

def compute(weights, n, work_iters, span):
    # the engine should always send exactly n weights. if it doesnt, something
    # is wrong upstream, so score it as the worst possible rather than crashing
    if len(weights) != n:
        return 0.0

    if work_iters > 0:
        busy_work(work_iters, weights[0] if weights else 1.0)

    raw = rastrigin(weights)  # 0 at the optimum, bigger is worse

    # flip it into a fitness in roughly [0,1] so the target-fitness stopping
    # logic has friendly numbers to work with. the denominator is a rough upper
    # bound on rastrigin over the domain, it doesnt have to be tight
    worst = A * n + n * (span * span + A)
    fitness = max(0.0, 1.0 - raw / worst)

    # truncate to 4 places, same as the distance sim, just to keep the pipe tidy
    return int(fitness * 10000) / 10000

def main():
    n = int(sys.argv[1])
    work_iters = int(sys.argv[2])
    span = float(sys.argv[3])

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights, n, work_iters, span)
        print(score, flush=True)

if __name__ == "__main__":
    main()
    sys.exit(0)

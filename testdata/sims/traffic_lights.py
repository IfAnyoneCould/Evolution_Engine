import sys
import json
from random import Random
from math import isnan
from math import isinf
from math import trunc

# three signals in a row on a main road, each with a side street. genome is
# (main green fraction, offset) per signal on a shared cycle. cars show up
# at random, main road cars that clear one signal arrive at the next after a
# fixed travel time, so offsets matter for green waves. fitness is a target
# wait over the average wait. with noisy on, every eval gets fresh arrivals so
# the same genome scores differently each time, the engine has to cope with
# a noisy fitness
# args: noisy (0 or 1), seed, target wait (seconds)

CYCLE = 60
SIM_TIME = 3600
TRAVEL = 20 # seconds between signals
MAIN_RATE = 0.3 # arrivals per second
SIDE_RATE = [0.12, 0.16, 0.1]
SERVICE = 0.5 # cars per second while green
N = 3

def finish(x):
    if isnan(x) or isinf(x):
        return 0.0
    return trunc(min(1.0, max(0.0, x)) * 10000) / 10000

def compute(weights,rng,target):
    if len(weights) != 2 * N:
        return 0.0
    split = [max(0.05, min(0.95, weights[2*i])) for i in range(N)]
    offset = [weights[2*i+1] % 1.0 for i in range(N)]

    main_q = [0] * N
    side_q = [0] * N
    main_credit = [0.0] * N
    side_credit = [0.0] * N
    in_transit = [[0] * TRAVEL for _ in range(N)] # ring buffer of cars on the road into each signal
    waited = 0
    served = 0
    for t in range(SIM_TIME):
        slot = t % TRAVEL
        if rng.random() < MAIN_RATE:
            main_q[0] += 1
        for i in range(1, N):
            main_q[i] += in_transit[i][slot]
            in_transit[i][slot] = 0
        for i in range(N):
            if rng.random() < SIDE_RATE[i]:
                side_q[i] += 1

            green = ((t + offset[i] * CYCLE) % CYCLE) < split[i] * CYCLE
            if green:
                main_credit[i] = min(1.0, main_credit[i] + SERVICE)
                side_credit[i] = 0.0
                if main_q[i] and main_credit[i] >= 1.0:
                    main_q[i] -= 1
                    main_credit[i] -= 1.0
                    served += 1
                    if i + 1 < N:
                        in_transit[i+1][slot] += 1 # shows up TRAVEL seconds from now
            else:
                side_credit[i] = min(1.0, side_credit[i] + SERVICE)
                main_credit[i] = 0.0
                if side_q[i] and side_credit[i] >= 1.0:
                    side_q[i] -= 1
                    side_credit[i] -= 1.0
                    served += 1
            waited += main_q[i] + side_q[i]

    # cars still stuck at the end count too
    stuck = sum(main_q) + sum(side_q)
    avg = waited / max(1, served + stuck)
    return finish(target / max(avg, 1e-9))

def arg(i,default):
    return sys.argv[i] if len(sys.argv) > i else default

def main():
    noisy = int(arg(1,0))
    seed = int(arg(2,1))
    target = float(arg(3,10.0))

    evals = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        rng = Random(seed + evals if noisy else seed)
        evals += 1
        try:
            weights = json.loads(line)
            score = compute(weights,rng,target)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            score = 0.0
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

import sys
import json
from math import log
from math import sqrt
from math import isnan
from math import isinf
from math import trunc
from random import Random

# fit the 4 predator prey rates (alpha, beta, delta, gamma) to a hidden
# trajectory. error is on log populations so the crashes count as much as the
# booms. long run over several cycles, so a wrong period lines up with the
# data every few cycles and gives fake basins. some genomes blow up or die
# out, populations get clamped instead of crashing
# args: sim time, noise std (on log pops), noise seed

TRUE = (1.1, 0.4, 0.1, 0.4)
START = (10.0, 10.0)
DT = 0.01
SAMPLE = 20
LO = 1e-6
HI = 1e6

def finish(x):
    if isnan(x) or isinf(x):
        return 0.0
    return trunc(min(1.0, max(0.0, x)) * 10000) / 10000

def simulate(p,sim_time):
    a, b, d, g = p
    def f(x,y):
        return a*x - b*x*y, d*x*y - g*y
    x, y = START
    out = []
    for i in range(int(sim_time / DT)):
        if i % SAMPLE == 0:
            out.append((log(x), log(y)))
        k1x, k1y = f(x,y)
        k2x, k2y = f(x + DT/2*k1x,y + DT/2*k1y)
        k3x, k3y = f(x + DT/2*k2x,y + DT/2*k2y)
        k4x, k4y = f(x + DT*k3x,y + DT*k3y)
        x += DT/6 * (k1x + 2*k2x + 2*k3x + k4x)
        y += DT/6 * (k1y + 2*k2y + 2*k3y + k4y)
        # clamp instead of letting it overflow or go negative
        x = min(HI, max(LO, x)) if x == x else LO
        y = min(HI, max(LO, y)) if y == y else LO
    return out

def compute(weights,measured,sim_time,noise):
    if len(weights) != 4:
        return 0.0
    if min(weights) < 0:
        return 0.0

    guess = simulate(weights,sim_time)
    mse = 0.0
    for (gx, gy), (mx, my) in zip(guess, measured):
        mse += (gx - mx)**2 + (gy - my)**2
    mse /= 2 * len(measured)
    err = sqrt(max(0.0, mse - noise * noise))
    return finish(1 / (1 + err))

def arg(i,default):
    return sys.argv[i] if len(sys.argv) > i else default

def main():
    sim_time = float(arg(1,30.0))
    noise = float(arg(2,0.0))
    rng = Random(int(arg(3,1)))

    measured = [(x + rng.gauss(0, noise), y + rng.gauss(0, noise)) for x, y in simulate(TRUE,sim_time)]

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            weights = json.loads(line)
            score = compute(weights,measured,sim_time,noise)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            score = 0.0
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

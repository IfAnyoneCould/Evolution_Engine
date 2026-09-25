import sys
import json
from math import exp
from math import sqrt
from math import isnan
from math import isinf
from math import trunc

# inverse problem on a 1d rod, ends held at 0. there are hidden gaussian heat
# sources, and a handful of sensors along the rod record the temperature over
# time. genome is (position, strength, width) per source, the forward model is
# an explicit finite difference solve, fitness is how well the guessed sources
# reproduce the sensor readings. sources can swap places so theres symmetric
# optima, and the forward solve is the expensive part
# args: n sources, grid points, time steps

ALPHA = 0.01
T_END = 2.0
TRUE = [(0.3, 1.0, 0.05), (0.72, 0.6, 0.08), (0.5, 0.8, 0.03)] # pos, strength, width
SENSORS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
READS = 10 # sample times

def finish(x):
    if isnan(x) or isinf(x):
        return 0.0
    return trunc(min(1.0, max(0.0, x)) * 10000) / 10000

def simulate(sources,grid,steps):
    dx = 1.0 / (grid - 1)
    need = int(T_END * ALPHA * 2.5 / (dx * dx)) + 1 # keep explicit scheme stable
    steps = max(steps, need)
    dt = T_END / steps
    lam = ALPHA * dt / (dx * dx)

    q = [0.0] * grid
    for p, s, w in sources:
        w = max(w, 1e-3)
        for i in range(grid):
            q[i] += dt * s * exp(-((i * dx - p)**2) / (2 * w * w))
    q_in = q[1:-1]

    idx = [min(grid - 2, int(p / dx)) for p in SENSORS]
    frac = [p / dx - i for p, i in zip(SENSORS, idx)]
    every = steps // READS

    u = [0.0] * grid
    out = []
    for step in range(1, steps + 1):
        u = [0.0] + [b + lam * (a - 2*b + c) + qi for a, b, c, qi in zip(u, u[1:], u[2:], q_in)] + [0.0]
        if step % every == 0:
            out.extend(u[i] + f * (u[i+1] - u[i]) for i, f in zip(idx, frac))
    return out

def compute(weights,n,grid,steps,measured,scale):
    if len(weights) != 3 * n:
        return 0.0

    sources = [(weights[3*i], weights[3*i+1], weights[3*i+2]) for i in range(n)]
    guess = simulate(sources,grid,steps)
    mse = sum((g - m)**2 for g, m in zip(guess, measured)) / len(measured)
    return finish(1 / (1 + sqrt(mse) / scale))

def arg(i,default):
    return sys.argv[i] if len(sys.argv) > i else default

def main():
    n = int(arg(1,2))
    grid = int(arg(2,250))
    steps = int(arg(3,6000))

    measured = simulate(TRUE[:n],grid,steps)
    scale = sqrt(sum(m * m for m in measured) / len(measured))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            weights = json.loads(line)
            score = compute(weights,n,grid,steps,measured,scale)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            score = 0.0
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

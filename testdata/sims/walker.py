import sys
import json
from math import sin
from math import sqrt
from math import tanh
from math import pi
from math import isnan
from math import isinf
from math import trunc

# soft body crawler. a square of 4 masses with 6 springs (sides plus both
# diagonals) sitting on the ground with friction. each spring's rest length
# wobbles as a sine, genome is one shared frequency then (amplitude, phase)
# per spring. fitness is how far the middle moves in +x against a target
# distance. gaits are chaotic and a lot of them just shuffle in place or go
# backwards, rough landscape
# args: sim seconds, dt, target distance

MASS = 1.0
K = 500.0
C = 5.0
GRAVITY = 9.8
K_GROUND = 5000.0
C_GROUND = 40.0
MU = 1.0
START = [(0.0, 0.01), (1.0, 0.01), (1.0, 1.01), (0.0, 1.01)]
SPRINGS = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]

def finish(x):
    if isnan(x) or isinf(x):
        return 0.0
    return trunc(min(1.0, max(0.0, x)) * 10000) / 10000

def compute(weights,sim_time,dt,target):
    if len(weights) != 1 + 2 * len(SPRINGS):
        return 0.0
    freq = weights[0]
    amp = [max(0.0, min(0.25, weights[1 + 2*i])) for i in range(len(SPRINGS))]
    phase = [weights[2 + 2*i] for i in range(len(SPRINGS))]

    px = [p[0] for p in START]
    py = [p[1] for p in START]
    vx = [0.0] * 4
    vy = [0.0] * 4
    rest = []
    for a, b in SPRINGS:
        rest.append(sqrt((px[a] - px[b])**2 + (py[a] - py[b])**2))
    x0 = sum(px) / 4

    t = 0.0
    w = 2 * pi * freq
    for _ in range(int(sim_time / dt)):
        fx = [0.0] * 4
        fy = [-GRAVITY * MASS] * 4
        for s, (a, b) in enumerate(SPRINGS):
            dx = px[b] - px[a]
            dy = py[b] - py[a]
            d = max(sqrt(dx*dx + dy*dy), 1e-6)
            ux = dx / d
            uy = dy / d
            want = rest[s] * (1 + amp[s] * sin(w * t + phase[s]))
            rel = (vx[b] - vx[a]) * ux + (vy[b] - vy[a]) * uy
            f = K * (d - want) + C * rel
            fx[a] += f * ux
            fy[a] += f * uy
            fx[b] -= f * ux
            fy[b] -= f * uy
        for i in range(4):
            if py[i] < 0:
                normal = max(0.0, -K_GROUND * py[i] - C_GROUND * vy[i])
                fy[i] += normal
                fx[i] -= MU * normal * tanh(vx[i] / 0.05)
            vx[i] += fx[i] / MASS * dt
            vy[i] += fy[i] / MASS * dt
            px[i] += vx[i] * dt
            py[i] += vy[i] * dt
        t += dt
        if abs(px[0]) > 1e3 or abs(py[0]) > 1e3:
            return 0.0

    moved = sum(px) / 4 - x0
    return finish(moved / target)

def arg(i,default):
    return sys.argv[i] if len(sys.argv) > i else default

def main():
    sim_time = float(arg(1,10.0))
    dt = float(arg(2,0.001))
    target = float(arg(3,25.0))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            weights = json.loads(line)
            score = compute(weights,sim_time,dt,target)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            score = 0.0
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

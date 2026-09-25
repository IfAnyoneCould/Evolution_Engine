import sys
import json
from math import sin
from math import cos
from math import sqrt
from math import pi
from math import isnan
from math import isinf
from math import trunc
from random import Random

# boids. genome is the flocking rule gains, cohesion, alignment, separation,
# plus separation radius and view radius. the flock starts scattered with
# random headings (seeded) and gets jostled by seeded noise every step.
# fitness over the back half of the run is how tight the flock is times how
# aligned it is times how few near collisions there were, so pushing any one
# rule too hard wrecks another. n^2 per step so agent count sets the cost
# args: agents, steps, seed

DT = 0.1
V_MIN = 0.5
V_MAX = 2.0
A_MAX = 2.0
NOISE = 0.6
HIT = 0.4 # closer than this is a collision

def finish(x):
    if isnan(x) or isinf(x):
        return 0.0
    return trunc(min(1.0, max(0.0, x)) * 10000) / 10000

def compute(weights,n,steps,seed):
    if len(weights) != 5:
        return 0.0
    wc, wa, ws, sep_r, view_r = weights

    rng = Random(seed)
    px = [rng.uniform(-5, 5) for _ in range(n)]
    py = [rng.uniform(-5, 5) for _ in range(n)]
    vx = []
    vy = []
    for _ in range(n):
        a = rng.uniform(0, 2 * pi)
        vx.append(cos(a))
        vy.append(sin(a))

    ok_r = 0.35 * sqrt(n)
    view2 = view_r * view_r
    sep2 = sep_r * sep_r
    tight = 0.0
    aligned = 0.0
    hits = 0
    scored = 0
    for step in range(steps):
        ax = [0.0] * n
        ay = [0.0] * n
        for i in range(n):
            cx = cy = avx = avy = sx = sy = 0.0
            seen = 0
            xi = px[i]
            yi = py[i]
            for j in range(n):
                if i == j:
                    continue
                dx = px[j] - xi
                dy = py[j] - yi
                d2 = dx*dx + dy*dy
                if d2 < view2:
                    seen += 1
                    cx += dx
                    cy += dy
                    avx += vx[j]
                    avy += vy[j]
                if d2 < sep2:
                    d2 = max(d2, 1e-4)
                    sx -= dx / d2
                    sy -= dy / d2
            if seen:
                ax[i] = wc * cx / seen + wa * (avx / seen - vx[i]) + ws * sx
                ay[i] = wc * cy / seen + wa * (avy / seen - vy[i]) + ws * sy
            else:
                ax[i] = ws * sx
                ay[i] = ws * sy

        for i in range(n):
            a = sqrt(ax[i]**2 + ay[i]**2)
            if a > A_MAX:
                ax[i] *= A_MAX / a
                ay[i] *= A_MAX / a
            vx[i] += (ax[i] + rng.gauss(0, NOISE)) * DT
            vy[i] += (ay[i] + rng.gauss(0, NOISE)) * DT
            v = sqrt(vx[i]**2 + vy[i]**2)
            if v > V_MAX:
                vx[i] *= V_MAX / v
                vy[i] *= V_MAX / v
            elif v < V_MIN:
                v = max(v, 1e-9)
                vx[i] *= V_MIN / v
                vy[i] *= V_MIN / v
            px[i] += vx[i] * DT
            py[i] += vy[i] * DT

        if step >= steps // 2:
            mx = sum(px) / n
            my = sum(py) / n
            spread = sum(sqrt((x - mx)**2 + (y - my)**2) for x, y in zip(px, py)) / n
            tight += min(1.0, ok_r / max(spread, 1e-9))
            hx = sum(x / max(sqrt(x*x + y*y), 1e-9) for x, y in zip(vx, vy)) / n
            hy = sum(y / max(sqrt(x*x + y*y), 1e-9) for x, y in zip(vx, vy)) / n
            aligned += sqrt(hx*hx + hy*hy)
            for i in range(n):
                for j in range(i + 1, n):
                    if (px[i] - px[j])**2 + (py[i] - py[j])**2 < HIT * HIT:
                        hits += 1
            scored += 1

    col = max(0.0, 1 - hits / scored / (0.05 * n))
    return finish(tight / scored * aligned / scored * col)

def arg(i,default):
    return sys.argv[i] if len(sys.argv) > i else default

def main():
    n = int(arg(1,30))
    steps = int(arg(2,200))
    seed = int(arg(3,1))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            weights = json.loads(line)
            score = compute(weights,n,steps,seed)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            score = 0.0
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

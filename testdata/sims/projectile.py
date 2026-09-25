import sys
import json
from math import sin
from math import cos
from math import sqrt
from math import radians
from math import isnan
from math import isinf
from math import trunc

# ball with quadratic drag and a magnus lift term from spin. genome is launch
# angle (deg), speed, spin. fitness is how close it lands to the target
# distance. cheap and low dim, a whole curve of answers hit it so its more of
# a sanity check that drag doesnt confuse anything
# args: target distance, drag coef, dt

G = 9.81
LIFT = 0.0015 # magnus coef per unit spin

def finish(x):
    if isnan(x) or isinf(x):
        return 0.0
    return trunc(min(1.0, max(0.0, x)) * 10000) / 10000

def fly(angle,speed,spin,drag,dt):
    x = 0.0
    y = 0.0
    vx = speed * cos(radians(angle))
    vy = speed * sin(radians(angle))
    for _ in range(100000):
        v = sqrt(vx*vx + vy*vy)
        ax = -drag * v * vx - LIFT * spin * v * vy
        ay = -G - drag * v * vy + LIFT * spin * v * vx
        nx = x + vx * dt
        ny = y + vy * dt
        vx += ax * dt
        vy += ay * dt
        if ny < 0:
            return x + (nx - x) * y / (y - ny) # lerp to where it crossed the ground
        x = nx
        y = ny
    return x

def compute(weights,target,drag,dt):
    if len(weights) != 3:
        return 0.0
    angle = max(0.0, min(90.0, weights[0]))
    speed = max(0.0, weights[1])
    spin = max(-1.0, min(1.0, weights[2]))

    land = fly(angle,speed,spin,drag,dt)
    return finish(1 - abs(land - target) / target)

def arg(i,default):
    return sys.argv[i] if len(sys.argv) > i else default

def main():
    target = float(arg(1,150.0))
    drag = float(arg(2,0.002))
    dt = float(arg(3,0.005))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            weights = json.loads(line)
            score = compute(weights,target,drag,dt)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            score = 0.0
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

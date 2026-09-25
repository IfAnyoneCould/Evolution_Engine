import sys
import json
from math import sin
from math import cos
from math import isnan
from math import isinf
from math import trunc

# classic cart pole. genome is a linear state feedback policy, force is
# w . (x, x dot, theta, theta dot) clipped to the motor limit. starts off
# center and tilted so it has to recover. fitness is half steps survived, half
# how centered and upright it stayed. low dim control, lots of dead genomes
# args: max steps, dt

G = 9.8
CART_M = 1.0
POLE_M = 0.1
HALF_LEN = 0.5
FORCE_MAX = 10.0
X_LIMIT = 2.4
TH_LIMIT = 0.21 # ~12 degrees
START = (0.5, 0.0, 0.1, 0.0)

def finish(x):
    if isnan(x) or isinf(x):
        return 0.0
    return trunc(min(1.0, max(0.0, x)) * 10000) / 10000

def compute(weights,max_steps,dt):
    if len(weights) != 4:
        return 0.0

    x, xd, th, thd = START
    total_m = CART_M + POLE_M
    survived = 0
    centered = 0.0
    for _ in range(max_steps):
        f = weights[0]*x + weights[1]*xd + weights[2]*th + weights[3]*thd
        f = max(-FORCE_MAX, min(FORCE_MAX, f))

        s = sin(th)
        c = cos(th)
        temp = (f + POLE_M * HALF_LEN * thd * thd * s) / total_m
        thdd = (G * s - c * temp) / (HALF_LEN * (4.0/3.0 - POLE_M * c * c / total_m))
        xdd = temp - POLE_M * HALF_LEN * thdd * c / total_m

        x += dt * xd
        xd += dt * xdd
        th += dt * thd
        thd += dt * thdd

        if abs(x) > X_LIMIT or abs(th) > TH_LIMIT:
            break
        survived += 1
        centered += 1 - ((x / X_LIMIT)**2 + (th / TH_LIMIT)**2) / 2

    return finish(0.5 * survived / max_steps + 0.5 * centered / max_steps)

def arg(i,default):
    return sys.argv[i] if len(sys.argv) > i else default

def main():
    max_steps = int(arg(1,1000))
    dt = float(arg(2,0.02))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            weights = json.loads(line)
            score = compute(weights,max_steps,dt)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            score = 0.0
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

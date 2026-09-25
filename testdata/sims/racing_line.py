import sys
import json
from math import sin
from math import cos
from math import tan
from math import atan2
from math import sqrt
from math import pi
from math import isnan
from math import isinf
from math import trunc

# open loop lap of a stadium shaped track. genome is k knots of (steer,
# throttle), linearly blended over the time limit. car is a kinematic bicycle
# with drag and a grip limit on yaw rate, so it has to brake for the bends
# and use the width. fitness is fraction of a lap done in the time limit,
# leaving the track ends the run and knocks 10% off. errors early in the
# schedule wreck everything after, so its very ill conditioned
# args: k knots, time limit, dt

STRAIGHT = 100.0
RADIUS = 30.0
HALF_W = 6.0
LAP = 2 * STRAIGHT + 2 * pi * RADIUS
WHEELBASE = 2.5
ACCEL = 6.0
BRAKE = 10.0
DRAG = 0.004
GRIP = 9.0
STEER_MAX = 0.5

def finish(x):
    if isnan(x) or isinf(x):
        return 0.0
    return trunc(min(1.0, max(0.0, x)) * 10000) / 10000

def locate(x,y):
    # arc length along the centerline and distance off it, counterclockwise
    # starting from the left end of the bottom straight
    h = STRAIGHT / 2
    if -h <= x <= h:
        if y < 0:
            return x + h, abs(y + RADIUS)
        return STRAIGHT + pi * RADIUS + (h - x), abs(y - RADIUS)
    if x > h:
        phi = atan2(y, x - h)
        return STRAIGHT + (phi + pi/2) * RADIUS, abs(sqrt((x - h)**2 + y*y) - RADIUS)
    phi = atan2(y, x + h)
    psi = phi - pi/2 if phi > 0 else phi + 3*pi/2
    return 2 * STRAIGHT + pi * RADIUS + psi * RADIUS, abs(sqrt((x + h)**2 + y*y) - RADIUS)

def compute(weights,k,limit,dt):
    if len(weights) != 2 * k:
        return 0.0

    x = 0.0
    y = -RADIUS
    heading = 0.0
    v = 0.0
    last, _ = locate(x,y)
    progress = 0.0
    t = 0.0
    span = limit / (k - 1)
    while t < limit:
        i = min(k - 2, int(t / span))
        f = t / span - i
        steer = weights[2*i] * (1 - f) + weights[2*i+2] * f
        throttle = weights[2*i+1] * (1 - f) + weights[2*i+3] * f
        steer = max(-STEER_MAX, min(STEER_MAX, steer))
        throttle = max(-1.0, min(1.0, throttle))

        acc = throttle * (ACCEL if throttle > 0 else BRAKE) - DRAG * v * v
        v = max(0.0, v + acc * dt)
        yaw = v * tan(steer) / WHEELBASE
        cap = GRIP / max(v, 1.0)
        yaw = max(-cap, min(cap, yaw)) # understeer past the grip limit
        heading += yaw * dt
        x += v * cos(heading) * dt
        y += v * sin(heading) * dt
        t += dt

        s, off = locate(x,y)
        ds = s - last
        if ds < -LAP / 2:
            ds += LAP
        elif ds > LAP / 2:
            ds -= LAP
        progress += ds
        last = s
        if off > HALF_W:
            return finish(0.9 * progress / LAP)
        if progress >= LAP:
            return 1.0

    return finish(progress / LAP)

def arg(i,default):
    return sys.argv[i] if len(sys.argv) > i else default

def main():
    k = int(arg(1,20))
    limit = float(arg(2,24.0))
    dt = float(arg(3,0.005))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            weights = json.loads(line)
            score = compute(weights,k,limit,dt)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            score = 0.0
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

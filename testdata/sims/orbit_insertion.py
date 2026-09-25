import sys
import json
from math import sin
from math import cos
from math import sqrt
from math import isnan
from math import isinf
from math import trunc

# launch from the surface of a 2d planet into a circular orbit. mu = 1 and
# planet radius = 1. genome is k burn segments of (throttle, pitch off the
# local horizon), rk4 inside each. after the schedule, fitness comes from the
# osculating orbit, semi major axis vs target radius plus eccentricity, so
# it has to gravity turn then circularize near apoapsis. crashing or
# escaping gets a little partial credit
# args: k segments, segment time, rk4 steps per segment, target radius

MU = 1.0
THRUST = 3.0 # max accel, surface gravity is 1

def finish(x):
    if isnan(x) or isinf(x):
        return 0.0
    return trunc(min(1.0, max(0.0, x)) * 10000) / 10000

def accel(s,thr,pitch):
    x, y, vx, vy = s
    r = sqrt(x*x + y*y)
    ux = x / r
    uy = y / r
    # horizon is the counterclockwise tangent
    dx = cos(pitch) * -uy + sin(pitch) * ux
    dy = cos(pitch) * ux + sin(pitch) * uy
    g = MU / (r * r)
    return (vx, vy, -g * ux + thr * dx, -g * uy + thr * dy)

def rk4(s,thr,pitch,h):
    k1 = accel(s,thr,pitch)
    k2 = accel([s[i] + h/2*k1[i] for i in range(4)],thr,pitch)
    k3 = accel([s[i] + h/2*k2[i] for i in range(4)],thr,pitch)
    k4 = accel([s[i] + h*k3[i] for i in range(4)],thr,pitch)
    return [s[i] + h/6*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i in range(4)]

def compute(weights,k,seg_time,steps,target):
    if len(weights) != 2 * k:
        return 0.0

    s = [0.0, 1.0, 0.0, 0.0]
    h = seg_time / steps
    lifted = False
    max_r = 1.0
    for seg in range(k):
        thr = THRUST * max(0.0, min(1.0, weights[2*seg]))
        pitch = weights[2*seg+1]
        for _ in range(steps):
            n = rk4(s,thr,pitch,h)
            r = sqrt(n[0]*n[0] + n[1]*n[1])
            if r < 1.0:
                if lifted:
                    return finish(0.2 * min(1.0, (max_r - 1) / (target - 1))) # crashed
                continue # still sitting on the pad
            lifted = True
            max_r = max(max_r, r)
            s = n

    if not lifted:
        return 0.0
    x, y, vx, vy = s
    r = sqrt(x*x + y*y)
    energy = (vx*vx + vy*vy) / 2 - MU / r
    if energy >= 0:
        return finish(0.2) # escaped
    a = -MU / (2 * energy)
    hm = x * vy - y * vx
    e = sqrt(max(0.0, 1 + 2 * energy * hm * hm / (MU * MU)))
    score = max(0.0, 1 - 2 * abs(a - target) / target - e)
    if a * (1 - e) < 1.0:
        score *= 0.5 # periapsis is underground, will crash
    return finish(0.2 + 0.8 * score)

def arg(i,default):
    return sys.argv[i] if len(sys.argv) > i else default

def main():
    k = int(arg(1,10))
    seg_time = float(arg(2,0.2))
    steps = int(arg(3,60))
    target = float(arg(4,1.5))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            weights = json.loads(line)
            score = compute(weights,k,seg_time,steps,target)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            score = 0.0
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

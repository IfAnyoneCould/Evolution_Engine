import sys
import json
from math import sin
from math import cos
from math import sqrt
from math import isnan
from math import isinf
from math import trunc

# point mass lander, open loop. genome is k segments of (throttle, gimbal
# angle off vertical), each held for a few seconds, then it coasts. starts
# high, off to the side and drifting. fitness is landing on the pad times how
# soft the touchdown was, minus a small fuel cost. tank can run dry
# args: k segments, segment seconds, dt

G = 1.62
THRUST = 4.0 # max accel
TANK = 60.0 # total delta v in the tank
START = (-40.0, 100.0, 4.0, -2.0) # x, y, vx, vy
PAD_HALF = 10.0
SAFE_SPEED = 1.5

def finish(x):
    if isnan(x) or isinf(x):
        return 0.0
    return trunc(min(1.0, max(0.0, x)) * 10000) / 10000

def compute(weights,k,seg_time,dt):
    if len(weights) != 2 * k:
        return 0.0

    x, y, vx, vy = START
    used = 0.0
    t = 0.0
    end = k * seg_time
    while True:
        seg = int(t / seg_time)
        if seg < k and used < TANK:
            throttle = max(0.0, min(1.0, weights[2*seg]))
            ang = max(-1.0, min(1.0, weights[2*seg+1]))
        else:
            throttle = 0.0
            ang = 0.0
        a = throttle * THRUST
        used += a * dt

        vx += a * sin(ang) * dt
        vy += (a * cos(ang) - G) * dt
        x += vx * dt
        y += vy * dt
        t += dt

        if y <= 0:
            break
        if t > 3 * end:
            return finish(0.05) # never came down

    speed = sqrt(vx*vx + vy*vy)
    if abs(x) <= PAD_HALF:
        pos = 1.0
    else:
        pos = max(0.0, 1 - (abs(x) - PAD_HALF) / 100)
    soft = 1.0 if speed <= SAFE_SPEED else SAFE_SPEED / speed
    return finish(pos * soft - 0.02 * min(used, TANK) / TANK)

def arg(i,default):
    return sys.argv[i] if len(sys.argv) > i else default

def main():
    k = int(arg(1,10))
    seg_time = float(arg(2,3.0))
    dt = float(arg(3,0.01))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            weights = json.loads(line)
            score = compute(weights,k,seg_time,dt)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            score = 0.0
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

import sys
import json
from math import sin
from math import cos
from math import sqrt
from math import exp
from math import radians
from math import isnan
from math import isinf
from math import trunc

# cannon has to lob over a tall wall and drop into a landing zone just past
# it. genome is angle (deg) and speed, with drag. fitness jumps between
# regions, short of the wall, hit the wall, cleared but missed, in the zone,
# so the landscape is a bunch of plateaus with cliffs between them. only
# lobs close to the zone center get 0.99
# args: wall x, wall height, zone start, zone end

G = 9.81
DRAG = 0.002
DT = 0.005

def finish(x):
    if isnan(x) or isinf(x):
        return 0.0
    return trunc(min(1.0, max(0.0, x)) * 10000) / 10000

def compute(weights,wall_x,wall_h,zone_a,zone_b):
    if len(weights) != 2:
        return 0.0
    angle = max(0.0, min(90.0, weights[0]))
    speed = max(0.0, weights[1])

    x = 0.0
    y = 0.0
    vx = speed * cos(radians(angle))
    vy = speed * sin(radians(angle))
    land = None
    for _ in range(100000):
        v = sqrt(vx*vx + vy*vy)
        nx = x + vx * DT
        ny = y + vy * DT
        vx += -DRAG * v * vx * DT
        vy += (-G - DRAG * v * vy) * DT

        if x < wall_x <= nx:
            wy = y + (ny - y) * (wall_x - x) / (nx - x)
            if wy < wall_h:
                return finish(0.15 + 0.15 * max(0.0, wy) / wall_h) # smacked the wall
        if ny < 0:
            land = x + (nx - x) * y / (y - ny)
            break
        x = nx
        y = ny
    if land is None:
        return 0.0

    if land < wall_x:
        return finish(0.15 * land / wall_x) # short
    mid = (zone_a + zone_b) / 2
    half = (zone_b - zone_a) / 2
    if abs(land - mid) <= half:
        return finish(0.9 + 0.1 * (1 - abs(land - mid) / half))
    return finish(0.35 + 0.45 * exp(-(abs(land - mid) - half) / 20)) # cleared, missed

def arg(i,default):
    return sys.argv[i] if len(sys.argv) > i else default

def main():
    wall_x = float(arg(1,60.0))
    wall_h = float(arg(2,40.0))
    zone_a = float(arg(3,68.0))
    zone_b = float(arg(4,80.0))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            weights = json.loads(line)
            score = compute(weights,wall_x,wall_h,zone_a,zone_b)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            score = 0.0
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

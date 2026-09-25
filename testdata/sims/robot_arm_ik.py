import sys
import json
from math import sin
from math import cos
from math import sqrt
from math import isnan
from math import isinf
from math import trunc

# planar n link arm, unit links, base at the origin. genome is the relative
# joint angles. fitness is how close the tip gets to the target minus a
# penalty for any link cutting into a circle sitting right on the straight
# line to the target, so the arm has to snake around it. redundant, lots of
# answers, but the obstacle splits them into a left and right family
# args: n links, target x, target y

OBST_AT = 0.5 # obstacle center as a fraction of the way to the target
OBST_R = 0.25 # radius as a fraction of the target distance

def finish(x):
    if isnan(x) or isinf(x):
        return 0.0
    return trunc(min(1.0, max(0.0, x)) * 10000) / 10000

def seg_dist(ax,ay,bx,by,cx,cy):
    dx = bx - ax
    dy = by - ay
    t = ((cx - ax) * dx + (cy - ay) * dy) / (dx*dx + dy*dy)
    t = max(0.0, min(1.0, t))
    px = ax + t * dx - cx
    py = ay + t * dy - cy
    return sqrt(px*px + py*py)

def compute(weights,n,target):
    if len(weights) != n:
        return 0.0

    tx, ty = target
    cx = tx * OBST_AT
    cy = ty * OBST_AT
    r = OBST_R * sqrt(tx*tx + ty*ty)

    x = 0.0
    y = 0.0
    ang = 0.0
    pen = 0.0
    for w in weights:
        ang += w
        nx = x + cos(ang)
        ny = y + sin(ang)
        pen += max(0.0, r - seg_dist(x,y,nx,ny,cx,cy))
        x = nx
        y = ny

    dist = sqrt((x - tx)**2 + (y - ty)**2)
    return finish(1 - dist / n - pen / r)

def arg(i,default):
    return sys.argv[i] if len(sys.argv) > i else default

def main():
    n = int(arg(1,6))
    target = (float(arg(2,3.5)),float(arg(3,2.5)))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            weights = json.loads(line)
            score = compute(weights,n,target)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            score = 0.0
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

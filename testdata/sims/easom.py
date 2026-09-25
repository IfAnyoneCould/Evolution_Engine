import sys
import json
from math import cos
from math import exp
from math import pi
from math import trunc

# 2d, flat everywhere except one small hole at (pi,pi). the hole is a tiny
# fraction of the box, so its the 2d needle in a haystack but with a smooth
# slope once youre in
# args: none, always 2d

FLAT = 0.1 # not 0, a flat 0 trips the stall check on the first cycle

def easom(x,y):
    return -cos(x) * cos(y) * exp(-((x - pi) ** 2 + (y - pi) ** 2))

def compute(weights):
    if len(weights) != 2:
        return 0.0 # bad weight count, score it worst instead of dying on it

    raw = easom(weights[0],weights[1]) + 1.0 # 0 in the hole, ~1 everywhere else
    return trunc(max(0.0, 1 - (1 - FLAT) * raw) * 10000) / 10000

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights)
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

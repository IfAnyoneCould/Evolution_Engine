import sys
import json
from math import sin
from math import sqrt
from math import log1p
from math import trunc

# 2d and brutally rugged, deep local optima everywhere with no overall trend.
# the global one is at (512, 404.23), right on the edge of the box, so it also
# checks that clamping to the bound doesnt get in the way
# args: none, always 2d

F_MIN = -959.640662720851
SCALE = 100.0 # log squashed, 0.99 is about raw < 1. the runner up is ~65 worse

def eggholder(x,y):
    return -(y + 47.0) * sin(sqrt(abs(x / 2.0 + y + 47.0))) - x * sin(sqrt(abs(x - (y + 47.0))))

def compute(weights):
    if len(weights) != 2:
        return 0.0 # bad weight count, score it worst instead of dying on it

    raw = max(0.0, eggholder(weights[0],weights[1]) - F_MIN)
    return trunc(1 / (1 + log1p(raw / SCALE)) * 10000) / 10000

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

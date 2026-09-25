import sys
import json
from math import log1p
from math import trunc

# 2d with four equally good optima, (3,2), (-2.805,3.131), (-3.779,-3.283) and
# (3.584,-1.848). any of them is a win. mostly a check that the population
# settles on one instead of bouncing between them
# args: none, always 2d

SCALE = 2.0 # log squashed, 0.99 is about raw < 0.02

def himmelblau(x,y):
    return (x * x + y - 11.0) ** 2 + (x + y * y - 7.0) ** 2

def compute(weights):
    if len(weights) != 2:
        return 0.0 # bad weight count, score it worst instead of dying on it

    raw = himmelblau(weights[0],weights[1])
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

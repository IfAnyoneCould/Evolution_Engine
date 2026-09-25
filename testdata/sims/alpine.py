import sys
import json
from math import sin
from math import log1p
from math import trunc

# alpine n.1. sharp v shaped kinks from the abs, not smooth anywhere near an
# optimum. every dim hits 0 wherever x = 0 or sin(x) = -0.1, so theres a bunch
# of equally good spots, but theyre all needle thin
# args: n dims

SCALE = 2.0 # log squashed, 0.99 is about raw < 0.02

def alpine(x):
    return sum(abs(v * sin(v) + 0.1 * v) for v in x)

def compute(weights,n):
    if len(weights) != n:
        return 0.0 # bad weight count, score it worst instead of dying on it

    raw = alpine(weights)
    return trunc(1 / (1 + log1p(raw / SCALE)) * 10000) / 10000

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights,n)
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

import sys
import json
from math import log1p
from math import trunc

# a smooth bowl centered at (edge,...,edge), except anything with x_0 past the
# edge scores 0. so the best spot is right on the lip of a drop, and about half
# the nudges around it fall off. tests that the elite holds and the population
# doesnt get dragged off the cliff
# args: n dims, edge (where the cliff is on dim 0)

SCALE = 2.0 # log squashed, 0.99 is about raw < 0.02

def compute(weights,n,edge):
    if len(weights) != n:
        return 0.0 # bad weight count, score it worst instead of dying on it

    if weights[0] > edge:
        return 0.0 # off the cliff
    raw = sum((v - edge) ** 2 for v in weights)
    return trunc(1 / (1 + log1p(raw / SCALE)) * 10000) / 10000

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    edge = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights,n,edge)
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

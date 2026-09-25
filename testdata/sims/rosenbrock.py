import sys
import json
from math import log1p
from math import trunc

# the banana. the valley is easy to fall into but its long, narrow and curved,
# so getting down it to (1,...,1) needs steps that shrink and follow the bend
# args: n dims

SCALE = 50.0 # log squashed, 0.99 is about raw < 0.5. keeps the n>=4 local min (~3.98) out

def rosenbrock(x):
    total = 0.0
    for i in range(len(x) - 1):
        total += 100.0 * (x[i + 1] - x[i] * x[i]) ** 2 + (1.0 - x[i]) ** 2
    return total

def compute(weights,n):
    if len(weights) != n:
        return 0.0 # bad weight count, score it worst instead of dying on it

    raw = rosenbrock(weights)
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

import sys
import json
import random
from math import sqrt
from math import trunc

# sphere with gaussian noise on the score, fresh noise every eval. the same
# genome scores differently each time, so an elite can just be one that got
# lucky, and something at 0.97 can roll a 0.99. tests if the loop holds up when
# it cant trust the numbers. clean score is 1 - normalized distance from origin
# args: n dims, noise std, span (half width of each range)

def compute(weights,n,noise,span,rng):
    if len(weights) != n:
        return 0.0

    clean = 1 - sqrt(sum(x * x for x in weights) / (n * span * span))
    score = clean + rng.gauss(0,noise)
    return trunc(max(0.0, min(1.0, score)) * 10000) / 10000

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    noise = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    span = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights,n,noise,span,random.Random()) # reseeded from the os every eval
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

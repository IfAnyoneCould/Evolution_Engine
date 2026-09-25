import sys
import json
import random
from math import sqrt
from math import trunc

# the slope lies. over most of the box the score goes up the further you get
# from a hidden point and tops out around 0.8 in the far corners. the real
# optimum is a narrow cone at the hidden point, so hill climbing from a random
# start walks the wrong way. only a lucky sample or a big jump finds the cone
# args: n dims, cone width (fraction of the max distance), span, seed

def compute(weights,goal,width,span):
    if len(weights) != len(goal):
        return 0.0

    n = len(goal)
    r = sqrt(sum((x - g) ** 2 for x,g in zip(weights,goal))) / (2 * span * sqrt(n))
    score = 0.8 * r # the decoy, better the further out
    if r < width:
        score = max(score, 1 - r / width * (1 - 0.8 * width)) # meets the decoy at the rim
    return trunc(max(0.0, min(1.0, score)) * 10000) / 10000

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    width = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    span = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    rng = random.Random(seed)
    goal = [rng.uniform(-span / 2,span / 2) for _ in range(n)]

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights,goal,width,span)
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

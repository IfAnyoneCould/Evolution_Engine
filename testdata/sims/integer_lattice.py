import sys
import json
import random
from math import trunc

# weights get rounded to ints and only the ints matter, so the whole space is
# flat plateaus one unit wide. a nudge smaller than half a unit usually does
# nothing at all. target is a seeded int point, score is squared closeness in l1
# so 0.99 means every coordinate exact (or one off by 1 somewhere)
# args: n dims, span (targets are ints in [-span, span]), seed

def compute(weights,target,span):
    if len(weights) != len(target):
        return 0.0

    off = sum(abs(round(x) - t) for x,t in zip(weights,target))
    worst = len(target) * 2 * span
    return trunc(max(0.0, 1 - off / worst) ** 2 * 10000) / 10000

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    span = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    rng = random.Random(seed)
    target = [rng.randint(-span,span) for _ in range(n)]

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights,target,span)
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

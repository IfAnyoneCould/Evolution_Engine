import sys
import json
import random
from math import exp
from math import trunc

# hierarchical royal road on a continuous genome. the genome is split into
# blocks, each block scores a bump for how close it is to its hidden target.
# then pairs of neighboring blocks score the product of theirs, then groups of 4,
# and so on up to all of them. each level is weighted the same, so the top levels
# only pay out once every block under them is good at the same time
# args: blocks (power of 2 works best), block size (genome is blocks*size), bump width, seed

def levels(q):
    # mean product score at each level, group size doubling each time
    out = []
    size = 1
    while size <= len(q):
        groups = []
        for i in range(0,len(q) - size + 1,size):
            p = 1.0
            for v in q[i:i+size]:
                p *= v
            groups.append(p)
        out.append(sum(groups) / len(groups))
        size *= 2
    return out

def compute(weights,blocks,size,width,target):
    if len(weights) != blocks * size:
        return 0.0

    q = []
    for b in range(blocks):
        d = sum((weights[i] - target[i]) ** 2 for i in range(b*size,(b+1)*size))
        q.append(exp(-d / (width * width)))
    scores = levels(q)
    return trunc(sum(scores) / len(scores) * 10000) / 10000

def main():
    blocks = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    width = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    rng = random.Random(seed)
    target = [rng.uniform(-0.8,0.8) for _ in range(blocks * size)]

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights,blocks,size,width,target)
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

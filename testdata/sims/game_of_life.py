import sys
import json
import random
from math import trunc

# reverse conway. the genome thresholded (weight > 0.5 is alive) is the starting
# grid on a torus, run it t steps and compare the alive cells to a hidden target
# (a seeded random start run the same t steps). score is overlap of alive cells
# (intersection over union). tiny changes to the start blow up after a few steps,
# so this is discrete and chaotic, pretty much no gradient to follow
# args: grid size n (genome is n*n), steps t, seed

def step(cells,n):
    out = [0] * (n * n)
    for y in range(n):
        up = ((y - 1) % n) * n
        mid = y * n
        down = ((y + 1) % n) * n
        for x in range(n):
            l = (x - 1) % n
            r = (x + 1) % n
            alive = (cells[up+l] + cells[up+x] + cells[up+r] + cells[mid+l] + cells[mid+r]
                     + cells[down+l] + cells[down+x] + cells[down+r])
            if alive == 3 or (alive == 2 and cells[mid+x]):
                out[mid+x] = 1
    return out

def run(cells,n,t):
    for _ in range(t):
        cells = step(cells,n)
    return cells

def compute(weights,n,t,target):
    if len(weights) != n * n:
        return 0.0

    cells = run([1 if x > 0.5 else 0 for x in weights],n,t)
    both = sum(1 for a,b in zip(cells,target) if a and b)
    either = sum(1 for a,b in zip(cells,target) if a or b)
    if either == 0:
        return 1.0 # both empty, only happens if the target died out
    return trunc(both / either * 10000) / 10000

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    t = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    rng = random.Random(seed)
    target = run([1 if rng.random() < 0.35 else 0 for _ in range(n * n)],n,t)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights,n,t,target)
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

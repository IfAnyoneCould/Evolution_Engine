import sys
import json
import random
from math import sqrt
from math import trunc

# paint k soft edged circles onto a small grayscale grid and compare to a hidden
# target painted the same way from seeded circles. the expensive one, rasterizing
# is most of the cost. circles are interchangeable and overlap, so lots of
# symmetry and some genes do nothing when a circle is off the canvas
# args: grid size, k circles (genome is 4k: x, y, radius, intensity), seed

def render(weights,grid):
    img = [0.0] * (grid * grid)
    for i in range(0,len(weights),4):
        x,y,r,c = weights[i:i+4]
        cx,cy,cr = x * grid, y * grid, max(0.0, r) * grid
        # every pixel for every circle on purpose, no bounding box. the cost is the point
        for py in range(grid):
            dy = py + 0.5 - cy
            row = py * grid
            for px in range(grid):
                dx = px + 0.5 - cx
                cover = cr - sqrt(dx * dx + dy * dy) + 0.5
                if cover > 0:
                    img[row + px] += c * min(1.0, cover)
    return [min(1.0, max(0.0, v)) for v in img]

def compute(weights,grid,k,target):
    if len(weights) != 4 * k:
        return 0.0

    img = render(weights,grid)
    err = sum((a - b) ** 2 for a,b in zip(img,target)) / len(img)
    return trunc(max(0.0, 1 - sqrt(err)) * 10000) / 10000

def main():
    grid = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    rng = random.Random(seed)
    hidden = []
    for _ in range(k):
        hidden += [rng.random(),rng.random(),rng.uniform(0.05,0.25),rng.uniform(0.2,1.0)]
    target = render(hidden,grid)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights,grid,k,target)
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

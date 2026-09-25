import sys
import json
from math import trunc

# concatenated deceptive traps over thresholded bits (weight > 0.5 is a 1). in
# each block every extra 1 lowers the score until the block is all 1s, so the
# local slope leads to all 0s (scores (k-1)/k) and the real optimum is all 1s
# args: block size k, blocks (genome is k*blocks)

def trap(ones,k):
    if ones == k:
        return k
    return k - 1 - ones

def compute(weights,k,blocks):
    if len(weights) != k * blocks:
        return 0.0

    total = 0
    for b in range(blocks):
        ones = sum(1 for x in weights[b*k:(b+1)*k] if x > 0.5)
        total += trap(ones,k)
    return trunc(total / (k * blocks) * 10000) / 10000

def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    blocks = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights,k,blocks)
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

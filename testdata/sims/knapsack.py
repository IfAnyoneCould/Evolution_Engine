import sys
import json
import random
from math import trunc

# 0/1 knapsack through a continuous genome, weight > 0.5 means the item is in.
# values track weights loosely so theres no obvious greedy pick, and going over
# capacity costs more than the extra items are worth. the real best is found by
# dp up front so 1 is the true optimum. capacity is half the total weight
# args: items (genome is one per item), seed

def best_value(items,cap):
    table = [0] * (cap + 1)
    for w,v in items:
        for c in range(cap,w - 1,-1):
            table[c] = max(table[c], table[c - w] + v)
    return table[cap]

def compute(weights,items,cap,best,ratio):
    if len(weights) != len(items):
        return 0.0

    total_w = 0
    total_v = 0
    for x,(w,v) in zip(weights,items):
        if x > 0.5:
            total_w += w
            total_v += v
    over = max(0, total_w - cap)
    score = (total_v - 3 * ratio * over) / best
    if over > 0:
        score -= 0.05 # flat hit so barely over cant sneak past 0.99
    return trunc(max(0.0, min(1.0, score)) * 10000) / 10000

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    rng = random.Random(seed)
    items = []
    for _ in range(n):
        w = rng.randint(5,40)
        items.append((w, max(1, w + rng.randint(-8,8))))
    cap = sum(w for w,_ in items) // 2
    best = best_value(items,cap)
    ratio = max(v / w for w,v in items)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights,items,cap,best,ratio)
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

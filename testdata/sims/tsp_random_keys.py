import sys
import json
import random
from math import sqrt
from math import trunc

# travelling salesman through random keys. each weight is a key for a city and
# sorting the keys gives the tour, so a permutation problem gets pushed through a
# continuous genome. most nudges dont change the order at all, then one swaps two
# cities. score is a 2-opt reference tour length over the genome's tour length
# args: cities (genome is one key per city), seed

def dist(a,b):
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def tour_len(tour,cities):
    return sum(dist(cities[tour[i-1]],cities[tour[i]]) for i in range(len(tour)))

def two_opt(tour,cities):
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2,n):
                a,b = cities[tour[i]],cities[tour[i+1]]
                c,d = cities[tour[j]],cities[tour[(j+1)%n]]
                if dist(a,c) + dist(b,d) < dist(a,b) + dist(c,d) - 1e-12:
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    improved = True
    return tour

def reference(cities,rng):
    # not guaranteed optimal, just best of a bunch of 2-opt restarts
    best = None
    for _ in range(20):
        tour = list(range(len(cities)))
        rng.shuffle(tour)
        length = tour_len(two_opt(tour,cities),cities)
        if best is None or length < best:
            best = length
    return best

def compute(weights,cities,ref):
    if len(weights) != len(cities):
        return 0.0

    tour = sorted(range(len(weights)), key=lambda i: weights[i])
    return trunc(min(1.0, ref / tour_len(tour,cities)) * 10000) / 10000

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    rng = random.Random(seed)
    cities = [(rng.random(),rng.random()) for _ in range(n)]
    ref = reference(cities,rng)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights,cities,ref)
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

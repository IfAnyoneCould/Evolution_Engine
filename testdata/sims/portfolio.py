import sys
import json
import random
from math import sqrt
from math import trunc

# mean-variance portfolio on seeded assets. weights are clipped at 0 and
# normalized to sum to 1, so theres a whole ray of genomes for every portfolio.
# utility is return - risk * variance, the covariance comes from a couple of
# shared factors so assets are correlated. the best allocation is found up front
# by projected gradient. score is 1 - sqrt of the regret (scaled by the gap to
# the worst single asset), so a lazy equal split lands around 0.75
# args: assets (genome is one per asset), risk aversion, seed

def utility(w,mu,cov,risk):
    n = len(w)
    ret = sum(w[i] * mu[i] for i in range(n))
    var = 0.0
    for i in range(n):
        for j in range(n):
            var += w[i] * cov[i][j] * w[j]
    return ret - risk * var

def project(v):
    # euclidean projection onto the simplex
    u = sorted(v,reverse=True)
    total = 0.0
    theta = 0.0
    for i,x in enumerate(u):
        total += x
        t = (total - 1) / (i + 1)
        if x - t > 0:
            theta = t
    return [max(0.0, x - theta) for x in v]

def best_utility(mu,cov,risk):
    n = len(mu)
    w = [1 / n] * n
    step = 1 / (2 * risk * sum(cov[i][i] for i in range(n)))
    for _ in range(3000):
        grad = [mu[i] - 2 * risk * sum(cov[i][j] * w[j] for j in range(n)) for i in range(n)]
        w = project([w[i] + step * grad[i] for i in range(n)])
    return utility(w,mu,cov,risk)

def compute(weights,mu,cov,risk,lo,hi):
    if len(weights) != len(mu):
        return 0.0

    w = [max(0.0, x) for x in weights]
    total = sum(w)
    if total <= 0:
        return 0.0
    w = [x / total for x in w]
    regret = max(0.0, hi - utility(w,mu,cov,risk)) / (hi - lo)
    return trunc(max(0.0, 1 - sqrt(regret)) * 10000) / 10000

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    risk = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    rng = random.Random(seed)
    mu = [rng.uniform(0.02,0.18) for _ in range(n)]
    loads = [[rng.gauss(0,0.15) for _ in range(3)] for _ in range(n)]
    own = [rng.uniform(0.05,0.3) ** 2 for _ in range(n)]
    cov = [[sum(loads[i][f] * loads[j][f] for f in range(3)) + (own[i] if i == j else 0.0) for j in range(n)] for i in range(n)]

    lo = min(utility([1.0 if j == i else 0.0 for j in range(n)],mu,cov,risk) for i in range(n))
    hi = best_utility(mu,cov,risk)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights,mu,cov,risk,lo,hi)
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

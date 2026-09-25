import sys
import json
import random
from math import trunc

# polynomial coefficients fit to hidden noisy data on [-1, 1]. the best fit is
# found up front by least squares so the score is how much worse than that it is.
# smooth and convex but badly conditioned, the high terms barely move anything
# args: degree (genome is degree+1, constant term first), points, noise std, seed

def poly(coefs,x):
    total = 0.0
    for c in reversed(coefs):
        total = total * x + c
    return total

def mse(coefs,xs,ys):
    return sum((poly(coefs,x) - y) ** 2 for x,y in zip(xs,ys)) / len(xs)

def least_squares(xs,ys,m):
    # normal equations + gaussian elimination, fine for small degrees
    a = [[float(sum(x ** (i + j) for x in xs)) for j in range(m)] for i in range(m)]
    b = [float(sum(y * x ** i for x,y in zip(xs,ys))) for i in range(m)]
    for col in range(m):
        piv = max(range(col,m), key=lambda r: abs(a[r][col]))
        a[col],a[piv] = a[piv],a[col]
        b[col],b[piv] = b[piv],b[col]
        for r in range(col + 1,m):
            f = a[r][col] / a[col][col]
            for c in range(col,m):
                a[r][c] -= f * a[col][c]
            b[r] -= f * b[col]
    out = [0.0] * m
    for r in range(m - 1,-1,-1):
        out[r] = (b[r] - sum(a[r][c] * out[c] for c in range(r + 1,m))) / a[r][r]
    return out

def compute(weights,degree,xs,ys,floor,var):
    if len(weights) != degree + 1:
        return 0.0

    extra = max(0.0, mse(weights,xs,ys) - floor)
    return trunc(1 / (1 + 10 * extra / var) * 10000) / 10000

def main():
    degree = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    points = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    noise = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    rng = random.Random(seed)
    hidden = [rng.uniform(-1,1) for _ in range(degree + 1)]
    xs = [-1 + 2 * i / (points - 1) for i in range(points)]
    ys = [poly(hidden,x) + rng.gauss(0,noise) for x in xs]

    best = least_squares(xs,ys,degree + 1)
    floor = mse(best,xs,ys)
    mean = sum(ys) / len(ys)
    var = sum((y - mean) ** 2 for y in ys) / len(ys)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights,degree,xs,ys,floor,var)
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

import sys
import json
from math import sin
from math import pi
from math import tanh
from math import trunc

# fit sin over [-pi, pi] with a 1-h-1 tanh net. smooth-ish but higher dim
# (1-8-1 is 25 weights) and full of symmetries, hidden nodes can swap or flip sign
# fitness is 1/(1+10*mse), so 0.99 needs mse under 0.001
# args: hidden size (genome is hidden*3+1), grid points

def compute(weights,hidden,xs,ys):
    if len(weights) != hidden * 3 + 1:
        return 0.0

    # layout: [w,bias] per hidden node, then one out weight per hidden, then out bias
    err = 0.0
    for x,y in zip(xs,ys):
        out = weights[-1]
        for j in range(hidden):
            out += weights[hidden*2+j] * tanh(weights[j*2]*x + weights[j*2+1])
        err += (out - y) ** 2
    mse = err / len(xs)
    return trunc(1 / (1 + 10 * mse) * 10000) / 10000

def main():
    hidden = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    points = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    xs = [-pi + 2 * pi * i / (points - 1) for i in range(points)]
    ys = [sin(x) for x in xs]

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights,hidden,xs,ys)
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

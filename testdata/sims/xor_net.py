import sys
import json
from math import exp
from math import tanh
from math import trunc

# a tiny net has to learn xor. genome is a 2-h-1 net (tanh hidden, sigmoid out),
# fitness is 1 - mean squared error over the 4 cases. lots of equivalent
# solutions, flat saturated spots, and a constant 0.5 output already scores 0.75
# args: hidden size (2 or 3, genome is hidden*4+1)

CASES = [((0,0),0),((0,1),1),((1,0),1),((1,1),0)]

def sigmoid(v):
    if v < -60:
        return 0.0
    return 1 / (1 + exp(-v))

def compute(weights,hidden):
    if len(weights) != hidden * 4 + 1:
        return 0.0

    # layout: [w1,w2,bias] per hidden node, then one out weight per hidden, then out bias
    err = 0.0
    for (a,b),want in CASES:
        out = weights[-1]
        for j in range(hidden):
            w = weights[j*3:j*3+3]
            out += weights[hidden*3+j] * tanh(w[0]*a + w[1]*b + w[2])
        err += (sigmoid(out) - want) ** 2
    return trunc(max(0.0, 1 - err / 4) * 10000) / 10000

def main():
    hidden = int(sys.argv[1]) if len(sys.argv) > 1 else 2

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights,hidden)
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

import sys
import json
from math import cos
from math import sin
from math import pi
from math import exp
from math import tanh
from math import trunc

# the two spirals problem, a classic for being mean to small nets. genome is a
# 2-h-h-1 tanh net, fitness is 1 - mean squared error of the sigmoid output vs
# the 0/1 labels. guessing 0.5 everywhere gets 0.75, 0.99 needs basically every
# point right and confident
# args: hidden size (genome is 2h + h + h*h + h + h + 1), points per arm, turns

def sigmoid(v):
    if v < -60:
        return 0.0
    return 1 / (1 + exp(-v))

def make_spirals(points,turns):
    data = []
    for i in range(points):
        t = (i + 1) / points
        a = t * turns * 2 * pi
        x = t * cos(a)
        y = t * sin(a)
        data.append((x,y,0))
        data.append((-x,-y,1))
    return data

def size(hidden):
    return hidden * 2 + hidden + hidden * hidden + hidden + hidden + 1

def compute(weights,hidden,data):
    if len(weights) != size(hidden):
        return 0.0

    # layout: layer 1 [w1,w2] per node then biases, layer 2 [h weights] per node
    # then biases, then out weights and out bias
    w1 = weights[0:hidden*2]
    b1 = weights[hidden*2:hidden*3]
    at = hidden * 3
    w2 = weights[at:at+hidden*hidden]
    b2 = weights[at+hidden*hidden:at+hidden*hidden+hidden]
    at += hidden * hidden + hidden
    w3 = weights[at:at+hidden]
    b3 = weights[-1]

    err = 0.0
    for x,y,want in data:
        h1 = [tanh(w1[j*2]*x + w1[j*2+1]*y + b1[j]) for j in range(hidden)]
        h2 = []
        for j in range(hidden):
            v = b2[j]
            row = w2[j*hidden:(j+1)*hidden]
            for m in range(hidden):
                v += row[m] * h1[m]
            h2.append(tanh(v))
        out = b3
        for j in range(hidden):
            out += w3[j] * h2[j]
        err += (sigmoid(out) - want) ** 2
    return trunc(max(0.0, 1 - err / len(data)) * 10000) / 10000

def main():
    hidden = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    points = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    turns = float(sys.argv[3]) if len(sys.argv) > 3 else 2.5

    data = make_spirals(points,turns)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights,hidden,data)
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

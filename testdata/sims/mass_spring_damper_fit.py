import sys
import json
from math import sin
from math import sqrt
from math import isnan
from math import isinf
from math import trunc
from random import Random

# system id. theres a hidden mass spring damper with fixed true m, c, k
# being pushed by a known forcing, and its sampled trajectory (plus optional
# seeded noise) is the measurement. genome is a guess at m, c, k, fitness is
# how well the guess reproduces the measurement. noise is taken out of the
# error so the true params still land near 1. smooth but m and k trade off
# against each other, long thin valley
# args: noise std, noise seed

TRUE = (1.5, 0.4, 6.0) # m, c, k
X0 = 1.0
T_END = 20.0
DT = 0.01
SAMPLE = 10 # steps between samples

def finish(x):
    if isnan(x) or isinf(x):
        return 0.0
    return trunc(min(1.0, max(0.0, x)) * 10000) / 10000

def force(t):
    return sin(1.3 * t) + 0.5 * sin(3.1 * t)

def simulate(m,c,k):
    def f(t,x,v):
        return v, (force(t) - c * v - k * x) / m
    x = X0
    v = 0.0
    t = 0.0
    out = []
    for i in range(int(T_END / DT)):
        if i % SAMPLE == 0:
            out.append(x)
        a1, b1 = f(t,x,v)
        a2, b2 = f(t + DT/2,x + DT/2*a1,v + DT/2*b1)
        a3, b3 = f(t + DT/2,x + DT/2*a2,v + DT/2*b2)
        a4, b4 = f(t + DT,x + DT*a3,v + DT*b3)
        x += DT/6 * (a1 + 2*a2 + 2*a3 + a4)
        v += DT/6 * (b1 + 2*b2 + 2*b3 + b4)
        t += DT
        if abs(x) > 1e6:
            return None
    return out

def compute(weights,measured,noise,scale):
    if len(weights) != 3:
        return 0.0
    m, c, k = weights
    if m <= 0.01:
        return 0.0

    guess = simulate(m,c,k)
    if guess is None:
        return 0.0
    mse = sum((g - y)**2 for g, y in zip(guess, measured)) / len(measured)
    err = sqrt(max(0.0, mse - noise * noise)) / scale
    return finish(1 / (1 + err))

def arg(i,default):
    return sys.argv[i] if len(sys.argv) > i else default

def main():
    noise = float(arg(1,0.0))
    rng = Random(int(arg(2,1)))

    measured = [y + rng.gauss(0, noise) for y in simulate(*TRUE)]
    mean = sum(measured) / len(measured)
    scale = sqrt(sum((y - mean)**2 for y in measured) / len(measured))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            weights = json.loads(line)
            score = compute(weights,measured,noise,scale)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            score = 0.0
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

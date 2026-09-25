import sys
import json
from math import sin
from math import cos
from math import tanh
from math import exp
from math import pi
from math import isnan
from math import isinf
from math import trunc

# torque limited pendulum starting hanging down. the motor cant lift it
# straight up so it has to pump energy then catch and hold it at the top.
# genome is a tiny tanh net, (cos, sin, theta dot) -> hidden -> torque.
# fitness is mostly how upright it held over the back half, a bit for max
# height so failed swings still get something. harder and more deceptive
# than cartpole, a lot of genomes just spin it forever
# args: hidden units, sim seconds, dt

G = 9.81
DAMP = 0.1
U_MAX = 2.0

def finish(x):
    if isnan(x) or isinf(x):
        return 0.0
    return trunc(min(1.0, max(0.0, x)) * 10000) / 10000

def policy(weights,hidden,c,s,thd):
    # weights laid out as [w1 (3 per unit), b1, w2, b2]
    out = weights[-1]
    for h in range(hidden):
        z = weights[3*h]*c + weights[3*h+1]*s + weights[3*h+2]*thd/8.0 + weights[3*hidden+h]
        out += weights[4*hidden+h] * tanh(z)
    return U_MAX * tanh(out)

def compute(weights,hidden,sim_time,dt):
    if len(weights) != 5 * hidden + 1:
        return 0.0

    th = pi # 0 is straight up
    thd = 0.0
    steps = int(sim_time / dt)
    best_up = 0.0
    hold = 0.0
    held_steps = 0
    for i in range(steps):
        u = policy(weights,hidden,cos(th),sin(th),thd)
        thd += dt * (G * sin(th) - DAMP * thd + u)
        th += dt * thd

        up = (1 + cos(th)) / 2
        best_up = max(best_up, up)
        if i >= steps // 2:
            hold += up * exp(-0.02 * thd * thd)
            held_steps += 1

    return finish(0.1 * best_up + 0.9 * hold / max(1, held_steps))

def arg(i,default):
    return sys.argv[i] if len(sys.argv) > i else default

def main():
    hidden = int(arg(1,4))
    sim_time = float(arg(2,15.0))
    dt = float(arg(3,0.01))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            weights = json.loads(line)
            score = compute(weights,hidden,sim_time,dt)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            score = 0.0
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

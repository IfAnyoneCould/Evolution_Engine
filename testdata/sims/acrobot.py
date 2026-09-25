import sys
import json
from math import sin
from math import cos
from math import pi
from math import isnan
from math import isinf
from math import trunc

# acrobot, two link pendulum with only the elbow powered. chaotic so tiny
# weight changes send it somewhere totally different. genome is a linear
# policy on (sin, cos of both joints, both joint speeds, bias) -> elbow torque.
# fitness is how high the tip got, and if it reached the goal height a bonus
# for how fast. rough landscape for such a small genome
# args: max steps, dt, goal height (tip height, 2 is fully straight up)

G = 9.8
M1 = 1.0
M2 = 1.0
L1 = 1.0
LC1 = 0.5
LC2 = 0.5
I1 = 1.0
I2 = 1.0
TORQUE_MAX = 2.0

def finish(x):
    if isnan(x) or isinf(x):
        return 0.0
    return trunc(min(1.0, max(0.0, x)) * 10000) / 10000

def deriv(s,tau):
    th1, th2, d1, d2 = s
    c2 = cos(th2)
    m11 = M1*LC1*LC1 + M2*(L1*L1 + LC2*LC2 + 2*L1*LC2*c2) + I1 + I2
    m12 = M2*(LC2*LC2 + L1*LC2*c2) + I2
    m22 = M2*LC2*LC2 + I2
    phi2 = M2*LC2*G*cos(th1 + th2 - pi/2)
    phi1 = -M2*L1*LC2*d2*d2*sin(th2) - 2*M2*L1*LC2*d2*d1*sin(th2) + (M1*LC1 + M2*L1)*G*cos(th1 - pi/2) + phi2
    dd2 = (tau + m12/m11*phi1 - M2*L1*LC2*d1*d1*sin(th2) - phi2) / (m22 - m12*m12/m11)
    dd1 = -(m12*dd2 + phi1) / m11
    return (d1, d2, dd1, dd2)

def rk4(s,tau,dt):
    k1 = deriv(s,tau)
    k2 = deriv([s[i] + dt/2*k1[i] for i in range(4)],tau)
    k3 = deriv([s[i] + dt/2*k2[i] for i in range(4)],tau)
    k4 = deriv([s[i] + dt*k3[i] for i in range(4)],tau)
    return [s[i] + dt/6*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i in range(4)]

def compute(weights,max_steps,dt,goal):
    if len(weights) != 7:
        return 0.0

    s = [0.0, 0.0, 0.0, 0.0] # hanging straight down, still
    best_h = -2.0
    for i in range(max_steps):
        th1, th2, d1, d2 = s
        f = (sin(th1), cos(th1), sin(th2), cos(th2), d1 / 4.0, d2 / 9.0, 1.0)
        tau = sum(weights[j] * f[j] for j in range(7))
        tau = max(-TORQUE_MAX, min(TORQUE_MAX, tau))
        s = rk4(s,tau,dt)
        if abs(s[2]) > 100 or abs(s[3]) > 100:
            break

        h = -cos(s[0]) - cos(s[0] + s[1])
        best_h = max(best_h, h)
        if h >= goal:
            return finish(0.95 + 0.05 * (1 - i / max_steps))

    return finish(0.9 * (best_h + 2) / (goal + 2))

def arg(i,default):
    return sys.argv[i] if len(sys.argv) > i else default

def main():
    max_steps = int(arg(1,600))
    dt = float(arg(2,0.05))
    goal = float(arg(3,1.9))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            weights = json.loads(line)
            score = compute(weights,max_steps,dt,goal)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            score = 0.0
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

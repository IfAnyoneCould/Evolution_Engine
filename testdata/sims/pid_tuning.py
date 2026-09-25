import sys
import json
from math import isnan
from math import isinf
from math import trunc

# tune kp, ki, kd for a unit step on an underdamped second order plant with a
# lagging, saturating actuator. d term is on the measurement so no kick.
# fitness takes off for overshoot past 2%, settling (2% band) slower than the
# target time, and steady state error. high gains ring or go unstable, low
# gains are slow, so the good region is a narrow ridge
# args: target settle time, sim seconds, dt

ZETA = 0.2
WN = 1.0
LAG = 0.1 # actuator time constant
U_MAX = 3.0
BAND = 0.02

def finish(x):
    if isnan(x) or isinf(x):
        return 0.0
    return trunc(min(1.0, max(0.0, x)) * 10000) / 10000

def compute(weights,ts_target,sim_time,dt):
    if len(weights) != 3:
        return 0.0
    kp, ki, kd = weights

    x = 0.0
    xd = 0.0
    act = 0.0
    integ = 0.0
    steps = int(sim_time / dt)
    peak = 0.0
    settle = 0.0
    tail = 0.0
    tail_n = 0
    for i in range(steps):
        e = 1.0 - x
        integ += e * dt
        u = kp * e + ki * integ - kd * xd
        u = max(-U_MAX, min(U_MAX, u))
        act += (u - act) / LAG * dt
        xdd = act - 2 * ZETA * WN * xd - WN * WN * x
        xd += xdd * dt
        x += xd * dt
        if abs(x) > 1e6:
            return 0.0 # went unstable

        peak = max(peak, x)
        if abs(x - 1.0) > BAND:
            settle = (i + 1) * dt
        if i >= steps * 0.9:
            tail += abs(x - 1.0)
            tail_n += 1

    over = max(0.0, peak - 1.0 - BAND)
    os_pen = min(1.0, over / 0.3)
    ts_pen = min(1.0, max(0.0, settle - ts_target) / (sim_time - ts_target))
    ess_pen = min(1.0, tail / tail_n / 0.05)
    return finish(1 - 0.3 * os_pen - 0.4 * ts_pen - 0.3 * ess_pen)

def arg(i,default):
    return sys.argv[i] if len(sys.argv) > i else default

def main():
    ts_target = float(arg(1,1.5))
    sim_time = float(arg(2,15.0))
    dt = float(arg(3,0.002))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            weights = json.loads(line)
            score = compute(weights,ts_target,sim_time,dt)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            score = 0.0
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

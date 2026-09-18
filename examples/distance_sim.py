import sys
import json
from math import sqrt
from math import trunc

# the wiring check. genome is a 2d point, fitness is how close it got to the
# goal, 1 being right on it. one smooth basin, so every nudge function looks
# the same here. rastrigin_sim.py is the one that tells them apart
# args: goal x, goal y, span (half width of each range, max dist is span*sqrt(2))

def compute(weights,goal,max_dist):
    dist = sqrt((weights[0]-goal[0])**2 + (weights[1]-goal[1])**2)
    return trunc(max(0.0, 1 - dist / max_dist) * 10000) / 10000

def main():
    goal = (float(sys.argv[1]),float(sys.argv[2]))
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights,goal,float(sys.argv[3]) * sqrt(2))
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

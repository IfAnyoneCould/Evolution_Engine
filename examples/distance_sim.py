import sys
import json
from math import sqrt
from math import trunc

# the simplest possible worker, useful for checking the engine is wired up
# before pointing it at anything hard. the genome is a 2d point and fitness is
# just how close it got to a goal point, normalized so 1 is exactly on it.
#
# one smooth basin and no local minima, so a constant nudge does fine here and
# the decaying ones have nothing to show off. use rastrigin_sim.py for that.
#
# args:
#   argv[1]  goal x
#   argv[2]  goal y
#   argv[3]  span, half width of the range. the max distance is span*sqrt(2)

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

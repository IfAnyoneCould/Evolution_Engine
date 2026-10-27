import os
import sys
import time
import subprocess

# sits between the engine and a sim and logs every eval, so the engine's eval
# count comes from the sim side and not from what the engine reports
# args: log dir, then the sim command. one log per worker so they don't share a file

log = open(os.path.join(sys.argv[1], f"{os.getpid()}.txt"), "w", buffering=1)
sim = subprocess.Popen(sys.argv[2:], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

for line in sys.stdin:
    sim.stdin.write(line)
    sim.stdin.flush()
    out = sim.stdout.readline()
    log.write(f"{time.perf_counter_ns()} {out.strip()}\n")
    sys.stdout.write(out)
    sys.stdout.flush()

sim.stdin.close()
sim.wait()

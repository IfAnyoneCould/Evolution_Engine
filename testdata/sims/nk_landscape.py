import sys
import json
import random
from math import trunc

# seeded nk landscape over thresholded bits (weight > 0.5 is a 1). each bit's
# contribution depends on itself and the next k bits around a ring, k=0 is one
# smooth hill and cranking k makes it more rugged. with adjacent neighbors the
# true max and min can be found exactly by dp, so the score is rescaled to [0, 1]
# args: n bits (genome is n), k, seed

def window(bits):
    idx = 0
    for b in bits:
        idx = idx * 2 + b
    return idx

def raw(bits,tables,n,k):
    total = 0.0
    for i in range(n):
        total += tables[i][window([bits[(i+j)%n] for j in range(k + 1)])]
    return total / n

def extreme(tables,n,k,pick):
    # fix the first k bits, walk the ring keeping the last k bits as state, then
    # close the loop with the wrapped windows at the end
    best = None
    for p in range(2 ** k):
        prefix = tuple((p >> (k - 1 - j)) & 1 for j in range(k))
        states = {prefix: 0.0}
        for j in range(k,n):
            nxt = {}
            for s,v in states.items():
                for bit in (0,1):
                    w = s + (bit,)
                    val = v + tables[j-k][window(w)]
                    key = w[1:]
                    if key not in nxt or pick(val,nxt[key]) == val:
                        nxt[key] = val
            states = nxt
        for s,v in states.items():
            seq = s + prefix
            for m in range(k):
                v += tables[n-k+m][window(seq[m:m+k+1])]
            if best is None or pick(v,best) == v:
                best = v
    return best / n

def compute(weights,tables,n,k,lo,hi):
    if len(weights) != n:
        return 0.0

    bits = [1 if x > 0.5 else 0 for x in weights]
    score = (raw(bits,tables,n,k) - lo) / (hi - lo)
    return trunc(max(0.0, min(1.0, score)) * 10000) / 10000

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    k = max(0, min(k, n // 2)) # the dp needs n >= 2k

    rng = random.Random(seed)
    tables = [[rng.random() for _ in range(2 ** (k + 1))] for _ in range(n)]
    lo = extreme(tables,n,k,min)
    hi = extreme(tables,n,k,max)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        weights = json.loads(line)
        score = compute(weights,tables,n,k,lo,hi)
        print(score,flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy", "scipy", "optuna", "cmaes", "cma", "nevergrad", "matplotlib"]
# ///

import os
import sys
import json
import time
import shutil
import argparse
import warnings
import tempfile
import subprocess
from math import ceil
from math import log

import numpy as np

# runs the engine and the free optimizers on the same sims with the same eval
# budget and compares best fitness by eval count. every tool searches the unit
# cube over the free weights and talks to the real sim over the same pipe
# run from anywhere: uv run bench/compare.py

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = os.path.join(ROOT, "testdata", "sims", "configs")
OUT = os.path.join(ROOT, "bench", "out")
SHIM = os.path.join(ROOT, "bench", "shim.py")

# shifted ones only, sphere and rastrigin have their optimum dead center and nevergrad starts there
PROBLEMS = ["shifted_sphere", "shifted_rastrigin", "rosenbrock", "pressure_vessel", "cartpole", "xor_net",
            "lunar_lander", "pendulum_swingup"]
TOOLS = ["engine-ga", "engine-cmaes", "random", "optuna-tpe", "optuna-cmaes", "pycma", "nevergrad", "scipy-de"]
GP_CAP = 200 # gp cost grows with n^3, past this it's hours per run

class Budget(Exception):
    pass

class Sim:
    def __init__(self, prog):
        self.p = subprocess.Popen([prog["path"], *prog.get("args", [])], cwd=ROOT, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)

    def __call__(self, w):
        self.p.stdin.write(json.dumps([float(v) for v in w]) + "\n")
        self.p.stdin.flush()
        try:
            return float(self.p.stdout.readline())
        except ValueError:
            return 0.0

    def close(self):
        self.p.stdin.close()
        self.p.wait()

class Problem:
    def __init__(self, name, budget):
        with open(os.path.join(CONFIGS, name + ".json")) as f:
            self.cfg = json.load(f)
        b = np.array(self.cfg["bounds"], dtype=float)
        self.lo, self.hi = b[:, 0], b[:, 1]
        self.free = self.hi > self.lo
        self.n = int(self.free.sum())
        self.budget = budget
        self.trace = []
        self.sim = None

    def full(self, u):
        w = self.lo.copy()
        w[self.free] = self.lo[self.free] + np.clip(u, 0, 1) * (self.hi - self.lo)[self.free]
        return w

    # fitness of a unit cube point, maximize
    def __call__(self, u):
        if len(self.trace) >= self.budget:
            raise Budget()
        f = self.sim(self.full(np.asarray(u, dtype=float)))
        self.trace.append(f)
        return f

def run_random(p, seed):
    r = np.random.default_rng(seed)
    while True:
        p(r.random(p.n))

def run_optuna(sampler):
    def run(p, seed):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        s = optuna.create_study(direction="maximize", sampler=sampler(seed))
        s.optimize(lambda t: p([t.suggest_float(f"x{i}", 0, 1) for i in range(p.n)]), n_trials=p.budget)
    return run

def tpe(seed):
    import optuna
    return optuna.samplers.TPESampler(seed=seed)

def optuna_cma(seed):
    import optuna
    return optuna.samplers.CmaEsSampler(seed=seed)

def gp(seed):
    import optuna
    return optuna.samplers.GPSampler(seed=seed)

# ipop restarts so it uses the whole budget instead of stopping on convergence
def run_pycma(p, seed):
    import cma
    pop = None
    while True:
        opts = {"bounds": [0, 1], "seed": seed + 1, "verbose": -9}
        if pop:
            opts["popsize"] = pop
        es = cma.CMAEvolutionStrategy(p.n * [0.5], 0.3, opts)
        while not es.stop():
            xs = es.ask()
            es.tell(xs, [-p(x) for x in xs])
        pop = 2 * es.popsize
        seed += 1000

def run_nevergrad(p, seed):
    import nevergrad as ng
    par = ng.p.Array(shape=(p.n,), lower=0, upper=1)
    par.random_state = np.random.RandomState(seed)
    ng.optimizers.NGOpt(parametrization=par, budget=p.budget, num_workers=1).minimize(lambda x: -p(x))

def run_de(p, seed):
    from scipy.optimize import differential_evolution
    differential_evolution(lambda x: -p(x), [(0, 1)] * p.n, seed=seed, maxiter=10**6, tol=0, atol=0, polish=False)

def run_engine(opt):
    def run(p, seed):
        logs = tempfile.mkdtemp(dir=OUT)
        cfg = dict(p.cfg)
        prog = cfg["program"]
        if opt == "ga":
            per = 59 # population 60 minus the elite, which doesn't get re-evaluated
        else:
            per = 4 + int(3 * log(p.n))
        cycles = ceil(p.budget / per) + 1
        cfg.update({
            "optimizer": opt,
            "program": {"path": sys.executable,
                        "args": [SHIM, logs, prog["path"], *prog.get("args", [])]},
            "run_settings": {"target_fitness": 1, "max_cycles": cycles},
            "stagnation_detection": {"patience": cycles},
            "sim_settings": {"rand_seed": seed},
            "workers": 4,
        })
        path = os.path.join(logs, "cfg.json.cfg")
        with open(path, "w") as f:
            json.dump(cfg, f)
        out = subprocess.run([ENGINE, path, "json"], cwd=ROOT, capture_output=True, text=True)
        for line in out.stdout.splitlines():
            e = json.loads(line)
            if e["type"] == "error":
                print(f"  engine error: {e['data']['message']}")
        evals = []
        for log_file in os.listdir(logs):
            if not log_file.endswith(".txt"):
                continue
            with open(os.path.join(logs, log_file)) as f:
                for line in f:
                    t, _, v = line.strip().partition(" ")
                    try:
                        evals.append((int(t), float(v)))
                    except ValueError:
                        evals.append((int(t), 0.0))
        evals.sort()
        p.trace = [v for _, v in evals[:p.budget]]
        shutil.rmtree(logs, ignore_errors=True)
    return run

RUNNERS = {
    "engine-ga": run_engine("ga"),
    "engine-cmaes": run_engine("cmaes"),
    "random": run_random,
    "optuna-tpe": run_optuna(tpe),
    "optuna-cmaes": run_optuna(optuna_cma),
    "optuna-gp": run_optuna(gp),
    "pycma": run_pycma,
    "nevergrad": run_nevergrad,
    "scipy-de": run_de,
}

def run_one(name, tool, seed, budget):
    if tool == "optuna-gp":
        budget = min(budget, GP_CAP)
    p = Problem(name, budget)
    start = time.time()
    if not tool.startswith("engine"):
        p.sim = Sim(p.cfg["program"])
    try:
        RUNNERS[tool](p, seed)
    except Budget:
        pass
    finally:
        if p.sim:
            p.sim.close()
    best = np.maximum.accumulate(p.trace) if p.trace else np.zeros(1)
    return {"problem": name, "tool": tool, "seed": seed, "dims": p.n, "evals": len(p.trace),
            "seconds": time.time() - start, "best": best.tolist()}

def at(best, i):
    return best[min(i, len(best)) - 1]

def summarize(runs, budget, target):
    rows = ["| problem | tool | best @10% | best @100% | evals to target | hits |", "|---|---|---|---|---|---|"]
    for name in dict.fromkeys(r["problem"] for r in runs):
        for tool in dict.fromkeys(r["tool"] for r in runs):
            rs = [r for r in runs if r["problem"] == name and r["tool"] == tool]
            if not rs:
                continue
            b = budget if tool != "optuna-gp" else min(budget, GP_CAP)
            early = np.median([at(r["best"], b // 10) for r in rs])
            final = np.median([r["best"][-1] for r in rs])
            hits = [next((i + 1 for i, v in enumerate(r["best"]) if v >= target), None) for r in rs]
            ok = [h for h in hits if h]
            ett = f"{int(np.median(ok))}" if ok else "-"
            rows.append(f"| {name} | {tool} | {early:.4f} | {final:.4f} | {ett} | {len(ok)}/{len(rs)} |")
    return "\n".join(rows)

def plot(runs, budget, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(dict.fromkeys(r["problem"] for r in runs))
    cols = 3
    fig, axs = plt.subplots(ceil(len(names) / cols), cols, figsize=(5 * cols, 3.6 * ceil(len(names) / cols)),
                            squeeze=False)
    for ax, name in zip(axs.flat, names):
        for tool in dict.fromkeys(r["tool"] for r in runs):
            rs = [r["best"] for r in runs if r["problem"] == name and r["tool"] == tool]
            if not rs:
                continue
            n = max(len(b) for b in rs)
            m = np.median([b + [b[-1]] * (n - len(b)) for b in rs], axis=0)
            ax.plot(np.arange(1, n + 1), m, label=tool, lw=1.5 if tool.startswith("engine") else 1)
        ax.set_title(name)
        ax.set_xscale("log")
        ax.set_xlabel("evals")
        ax.set_ylabel("best fitness")
    for ax in list(axs.flat)[len(names):]:
        ax.axis("off")
    axs.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)

def main():
    global ENGINE
    warnings.filterwarnings("ignore")
    a = argparse.ArgumentParser()
    a.add_argument("--problems", nargs="+", default=PROBLEMS)
    a.add_argument("--tools", nargs="+", default=TOOLS, choices=list(RUNNERS))
    a.add_argument("--budget", type=int, default=1000)
    a.add_argument("--seeds", type=int, default=3)
    a.add_argument("--target", type=float, default=0.99)
    args = a.parse_args()

    os.makedirs(OUT, exist_ok=True)
    ENGINE = os.path.join(OUT, "engine.exe" if os.name == "nt" else "engine")
    if any(t.startswith("engine") for t in args.tools):
        subprocess.run(["go", "build", "-o", ENGINE, "./cmd/engine"], cwd=ROOT, check=True)

    runs = []
    for name in args.problems:
        for tool in args.tools:
            for seed in range(args.seeds):
                r = run_one(name, tool, seed, args.budget)
                print(f"{name:18} {tool:13} seed {seed}  best {r['best'][-1]:.4f}  evals {r['evals']:5}  "
                      f"{r['seconds']:.1f}s", flush=True)
                runs.append(r)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    with open(os.path.join(OUT, f"runs-{stamp}.json"), "w") as f:
        json.dump(runs, f)
    table = summarize(runs, args.budget, args.target)
    with open(os.path.join(OUT, f"summary-{stamp}.md"), "w") as f:
        f.write(table + "\n")
    plot(runs, args.budget, os.path.join(OUT, f"curves-{stamp}.png"))
    print("\n" + table)
    print(f"\nwrote bench/out/*-{stamp}.*")

if __name__ == "__main__":
    main()

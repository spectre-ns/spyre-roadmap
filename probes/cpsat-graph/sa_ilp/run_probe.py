"""Solve every instance with pywraplp (SCIP, CP-SAT) and SaSolver; one JSON line each.

Usage: run_probe.py <out.jsonl> [sa_seconds=10] [exact_seconds=60] [name-filter]
"""

from __future__ import annotations

import json
import os
import random
import sys
import time

from ortools.linear_solver import pywraplp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import problems  # noqa: E402
from sa_solver import SaSolver  # noqa: E402

STATUS = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}


def run_exact(backend, build, seed, limit_s):
    s = pywraplp.Solver.CreateSolver(backend)
    build(s, random.Random(seed))
    s.SetTimeLimit(int(limit_s * 1000))
    if backend == "CP_SAT":
        s.SetNumThreads(8)
    t0 = time.perf_counter()
    st = s.Solve()
    wall = time.perf_counter() - t0
    ok = st in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
    return dict(
        status=STATUS.get(st, str(st)),
        obj=s.Objective().Value() if ok else None,
        bound=s.Objective().BestBound() if ok else None,
        wall=wall,
        vars=s.NumVariables(),
        rows=s.NumConstraints(),
    )


def run_sa(build, seed, sa_seed, limit_s):
    s = SaSolver(seed=sa_seed)
    build(s, random.Random(seed))
    s.SetTimeLimit(int(limit_s * 1000))
    st = s.Solve()
    return dict(
        status=STATUS.get(st, str(st)),
        obj=s.Objective().Value(),
        wall=s.WallTime() / 1000,
        time_to_best=s.stats.get("time_to_best"),
        iterations=s.stats.get("iterations"),
        vars=s.NumVariables(),
        rows=s.NumConstraints(),
    )


def main():
    out = sys.argv[1]
    sa_s = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    ex_s = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
    filt = sys.argv[4] if len(sys.argv) > 4 else ""
    with open(out, "a") as f:
        for name, build, seed in problems.instances():
            if filt and filt not in name:
                continue
            for backend in ("SCIP", "CP_SAT"):
                r = run_exact(backend, build, seed, ex_s)
                f.write(json.dumps(dict(instance=name, solver=backend, **r)) + "\n")
                f.flush()
                print(name, backend, r["status"], r["obj"], f"{r['wall']:.2f}s", flush=True)
            for sa_seed in (0, 1, 2):
                r = run_sa(build, seed, sa_seed, sa_s)
                f.write(json.dumps(dict(instance=name, solver=f"SA.{sa_seed}", **r)) + "\n")
                f.flush()
                print(name, f"SA.{sa_seed}", r["status"], r["obj"], f"ttb={r['time_to_best']}", flush=True)


if __name__ == "__main__":
    main()

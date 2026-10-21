"""ILP instances written only against the API pywraplp.Solver and SaSolver share.

Each ``build_*(s, rng)`` takes a solver ``s`` and a seeded ``random.Random`` and
posts variables, linear rows and an objective. Nothing here knows which backend
it is talking to.
"""

from __future__ import annotations

import glob
import os
import pickle


def build_mknap(s, rng, n=100, m=5):
    """Multi-dimensional 0-1 knapsack: maximize profit under m capacities."""
    x = [s.BoolVar(f"x{i}") for i in range(n)]
    for _ in range(m):
        w = [rng.randint(1, 100) for _ in range(n)]
        s.Add(s.Sum(w[i] * x[i] for i in range(n)) <= sum(w) // 2)
    p = [rng.randint(1, 100) for _ in range(n)]
    s.Maximize(s.Sum(p[i] * x[i] for i in range(n)))


def build_setcover(s, rng, n_elem=200, n_sets=120):
    """Weighted set cover: every element covered at least once, minimize cost."""
    sets = [set(rng.sample(range(n_elem), rng.randint(5, 15))) for _ in range(n_sets)]
    for e in range(n_elem):  # make it feasible
        if not any(e in st for st in sets):
            sets[rng.randrange(n_sets)].add(e)
    y = [s.BoolVar(f"y{j}") for j in range(n_sets)]
    for e in range(n_elem):
        s.Add(s.Sum(y[j] for j in range(n_sets) if e in sets[j]) >= 1)
    c = [rng.randint(1, 20) for _ in range(n_sets)]
    s.Minimize(s.Sum(c[j] * y[j] for j in range(n_sets)))


def build_gap(s, rng, agents=10, jobs=60):
    """Generalized assignment: each job to exactly one agent within capacities."""
    cost = [[rng.randint(10, 50) for _ in range(jobs)] for _ in range(agents)]
    res = [[rng.randint(5, 25) for _ in range(jobs)] for _ in range(agents)]
    cap = [int(0.8 * sum(res[a]) / agents) for a in range(agents)]
    x = [[s.BoolVar(f"x{a}_{j}") for j in range(jobs)] for a in range(agents)]
    for j in range(jobs):
        s.Add(s.Sum(x[a][j] for a in range(agents)) == 1)
    for a in range(agents):
        s.Add(s.Sum(res[a][j] * x[a][j] for j in range(jobs)) <= cap[a])
    s.Minimize(s.Sum(cost[a][j] * x[a][j] for a in range(agents) for j in range(jobs)))


def build_binpack(s, rng, items=40, cap=100):
    """Bin packing with an assignment formulation."""
    size = [rng.randint(10, 60) for _ in range(items)]
    bins = items  # trivial upper bound
    y = [s.BoolVar(f"y{b}") for b in range(bins)]
    x = [[s.BoolVar(f"x{i}_{b}") for b in range(bins)] for i in range(items)]
    for i in range(items):
        s.Add(s.Sum(x[i][b] for b in range(bins)) == 1)
    for b in range(bins):
        s.Add(s.Sum(size[i] * x[i][b] for i in range(items)) - cap * y[b] <= 0)
    s.Minimize(s.Sum(y))


def build_randilp(s, rng, n=60, m=40):
    """Bounded general-integer ILP with mixed-sign rows."""
    x = [s.IntVar(0, 10, f"x{i}") for i in range(n)]
    for _ in range(m):
        a = [rng.randint(-5, 10) for _ in range(n)]
        s.Add(s.Sum(a[i] * x[i] for i in range(n)) <= rng.randint(50, 300))
    c = [rng.randint(1, 20) for _ in range(n)]
    s.Maximize(s.Sum(c[i] * x[i] for i in range(n)))


CAPS = os.environ.get(
    "PROBE_CAPS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deenum", "runs")
)


def lx_residency_builder(cap_path):
    """LX residency from a Phase 2.0 capture: keep the buffers whose spill cost
    is largest without exceeding LX capacity at any tick. Linear: one binary
    per buffer and one capacity row per tick; division is not chosen here."""
    import torch  # noqa: F401  (unpickling needs torch_spyre classes)
    import torch_spyre  # noqa: F401

    cap = pickle.load(open(cap_path, "rb"))
    align = cap["alignment"]
    limit = cap["limit"] // align
    bufs = [b for b in cap["buffers"] if b.uses]

    def units(b):
        return -(-b.size // align)

    def life(b):
        st = b.lifetime_start_override if b.lifetime_start_override is not None else b.uses[0]
        en = b.lifetime_end_override if b.lifetime_end_override is not None else b.uses[-1]
        return st, en

    def spill(b):
        first_read = b.first_use_is_read
        reads = len(b.uses) - (0 if first_read else 1) - (1 if first_read else 0)
        inter = str(getattr(b, "boundary", "")).endswith("Intermediate")
        return max(0, reads + (1 if inter else 0)) * units(b)

    def build(s, rng):
        r = {}
        for b in bufs:
            r[b.name] = s.IntVar(0, 0 if b.residency_reason is not None else 1, f"r_{b.name}")
        ticks = sorted({t for b in bufs for t in range(life(b)[0], life(b)[1] + 1)})
        for t in ticks:
            live = [b for b in bufs if life(b)[0] <= t <= life(b)[1] and units(b) <= limit]
            if live:
                s.Add(s.Sum(units(b) * r[b.name] for b in live) <= limit)
        for b in bufs:
            if units(b) > limit:
                s.Add(r[b.name] <= 0)
        s.Maximize(s.Sum(spill(b) * r[b.name] for b in bufs))

    return build


def instances():
    out = []
    for fam, fn in [
        ("mknap", build_mknap),
        ("setcover", build_setcover),
        ("gap", build_gap),
        ("binpack", build_binpack),
        ("randilp", build_randilp),
    ]:
        for seed in (1, 2, 3):
            out.append((f"{fam}.{seed}", fn, seed))
    for name in [
        "caps_default/attn_block_256x1024x16.0.pkl",
        "caps_default/transformer_block_256x1024x16x2816.0.pkl",
        "caps_default/sdpa_gqa_1x32x8x512x128.0.pkl",
        "caps_granite_default/granite_micro.3.pkl",
    ]:
        path = os.path.join(CAPS, name)
        if os.path.exists(path):
            out.append((f"lx.{os.path.basename(name)[:-4]}", lx_residency_builder(path), 0))
    return out

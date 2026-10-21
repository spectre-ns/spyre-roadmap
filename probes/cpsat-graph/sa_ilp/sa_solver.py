"""A pywraplp-shaped solver backed by simulated annealing (plan Phase 3a probe).

Mirrors the modeling half of ``ortools.linear_solver.pywraplp.Solver`` so one
model-building function can run unchanged on SCIP / CP-SAT (through pywraplp)
and on this class, plus the operations our layout model needs beyond linear
rows (the same names the CP-SAT adapter in ``cpsat_backend.py`` implements):

    solver.IntVar(lb, ub, name) / solver.BoolVar(name)
    solver.Add(expr <= rhs | expr >= rhs | expr == rhs)[.OnlyEnforceIf(lit, ...)]
    solver.NewIntervalVar(start, size, name)
    solver.NewOptionalIntervalVar(start, size, presence, name)
    solver.AddNoOverlap2D(x_intervals, y_intervals)
    solver.Minimize(expr) / solver.Maximize(expr)
    solver.SetTimeLimit(ms); status = solver.Solve()
    solver.Value(var) / var.solution_value(); solver.Objective().Value()

Integer and Boolean variables only. The search is simulated annealing over
complete integer assignments with energy

    objective + weight * (row violation + global violation)

``weight`` adapts: it grows while the walk stays infeasible and decays while
feasible. Rows are linear (``lo <= sum(a_i x_i) <= hi``), optionally enforced
only when literals are 1. Globals (``NoOverlap2D``) are constraints with their
own incremental violation and repair moves: SA has no propagation for them, so
they are scored, not optimized, but they do not need linearizing. Moves:
row repair, global repair, row-preserving swaps (feasible region only), and
random single-variable changes. The best *feasible* assignment seen is returned;
SA proves nothing, so the status is FEASIBLE or NOT_SOLVED, never OPTIMAL.
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict

from ortools.linear_solver import pywraplp

INF = math.inf


# --------------------------------------------------------------------------
# Expressions and rows
# --------------------------------------------------------------------------


class LinearExpr:
    __slots__ = ("coeffs", "const")

    def __init__(self, coeffs=None, const=0.0):
        self.coeffs = dict(coeffs or {})
        self.const = const

    @staticmethod
    def of(x) -> "LinearExpr":
        if isinstance(x, LinearExpr):
            return x
        if isinstance(x, Var):
            return LinearExpr({x: 1.0})
        return LinearExpr({}, float(x))

    def value(self, x, override=None):
        s = self.const
        for v, a in self.coeffs.items():
            s += a * (override[1] if override and override[0] == v.index else x[v.index])
        return s

    def __add__(self, other):
        o = LinearExpr.of(other)
        c = dict(self.coeffs)
        for v, a in o.coeffs.items():
            c[v] = c.get(v, 0.0) + a
        return LinearExpr(c, self.const + o.const)

    __radd__ = __add__

    def __neg__(self):
        return LinearExpr({v: -a for v, a in self.coeffs.items()}, -self.const)

    def __sub__(self, other):
        return self + (-LinearExpr.of(other))

    def __rsub__(self, other):
        return LinearExpr.of(other) - self

    def __mul__(self, k):
        if isinstance(k, (Var, LinearExpr)):
            raise TypeError("products of variables are not linear")
        return LinearExpr({v: a * k for v, a in self.coeffs.items()}, self.const * k)

    __rmul__ = __mul__

    def __le__(self, other):
        e = self - other
        return Row(e.coeffs, -INF, -e.const)

    def __ge__(self, other):
        e = self - other
        return Row(e.coeffs, -e.const, INF)

    def __eq__(self, other):  # noqa: D105 (pywraplp semantics: builds a row)
        e = self - other
        return Row(e.coeffs, -e.const, -e.const)

    __hash__ = None


class Var:
    __slots__ = ("lb", "ub", "name", "index", "value", "is_bool")

    def __init__(self, lb, ub, name, index):
        self.lb, self.ub, self.name, self.index = int(lb), int(ub), name, index
        self.value = self.lb
        self.is_bool = self.lb == 0 and self.ub == 1

    def solution_value(self):
        return self.value

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return LinearExpr.of(self) == other

    def __add__(self, o):
        return LinearExpr.of(self) + o

    __radd__ = __add__

    def __sub__(self, o):
        return LinearExpr.of(self) - o

    def __rsub__(self, o):
        return LinearExpr.of(o) - self

    def __neg__(self):
        return -LinearExpr.of(self)

    def __mul__(self, k):
        return LinearExpr.of(self) * k

    __rmul__ = __mul__

    def __le__(self, o):
        return LinearExpr.of(self) <= o

    def __ge__(self, o):
        return LinearExpr.of(self) >= o


class Row:
    __slots__ = ("coeffs", "lo", "hi", "enforce")

    def __init__(self, coeffs, lo, hi):
        self.coeffs, self.lo, self.hi = coeffs, lo, hi
        self.enforce: list[Var] = []

    def OnlyEnforceIf(self, *lits):
        for lit in lits:
            self.enforce.extend(lit if isinstance(lit, (list, tuple)) else [lit])
        return self


class Interval:
    __slots__ = ("start", "size", "presence", "name")

    def __init__(self, start, size, presence, name):
        self.start, self.size = LinearExpr.of(start), LinearExpr.of(size)
        self.presence, self.name = presence, name


# --------------------------------------------------------------------------
# Global constraints
# --------------------------------------------------------------------------


class NoOverlap2D:
    """Present rectangles must not overlap. Violation is the summed overlap
    area of present pairs. Candidate pairs are precomputed from the x axis when
    every x interval is fixed (our case: lifetimes), else all pairs."""

    def __init__(self, xs, ys):
        self.xs, self.ys = xs, ys
        n = len(xs)
        self.n = n
        self.var_rects = defaultdict(set)
        for r in range(n):
            for e in (xs[r].start, xs[r].size, ys[r].start, ys[r].size):
                for v in e.coeffs:
                    self.var_rects[v.index].add(r)
            for itv in (xs[r], ys[r]):
                if isinstance(itv.presence, Var):
                    self.var_rects[itv.presence.index].add(r)
        x_fixed = all(not xs[r].start.coeffs and not xs[r].size.coeffs for r in range(n))
        self.nbrs = [[] for _ in range(n)]
        for a in range(n):
            for b in range(a + 1, n):
                if x_fixed:
                    sa, ea = xs[a].start.const, xs[a].start.const + xs[a].size.const
                    sb, eb = xs[b].start.const, xs[b].start.const + xs[b].size.const
                    if min(ea, eb) <= max(sa, sb):
                        continue
                self.nbrs[a].append(b)
                self.nbrs[b].append(a)
        self.deps = set(self.var_rects)

    def geom(self, r, x, override=None):
        def present(itv):
            p = itv.presence
            if p is None:
                return True
            if isinstance(p, Var):
                val = override[1] if override and override[0] == p.index else x[p.index]
                return val >= 1
            return bool(p)

        X, Y = self.xs[r], self.ys[r]
        if not (present(X) and present(Y)):
            return None
        sx, sy = X.start.value(x, override), Y.start.value(x, override)
        return (sx, sx + X.size.value(x, override), sy, sy + Y.size.value(x, override))

    @staticmethod
    def ov(g, h):
        if g is None or h is None:
            return 0.0
        ox = min(g[1], h[1]) - max(g[0], h[0])
        oy = min(g[3], h[3]) - max(g[2], h[2])
        return ox * oy if ox > 0 and oy > 0 else 0.0

    def init(self, x):
        self.g = [self.geom(r, x) for r in range(self.n)]
        self.pairs = {}
        for a in range(self.n):
            for b in self.nbrs[a]:
                if a < b:
                    o = self.ov(self.g[a], self.g[b])
                    if o:
                        self.pairs[(a, b)] = o
        self.total = sum(self.pairs.values())

    def _changes(self, i, new, x):
        rects = self.var_rects.get(i, ())
        newg = {r: self.geom(r, x, (i, new)) for r in rects}
        seen, out = set(), []
        for r in rects:
            for q in self.nbrs[r]:
                key = (r, q) if r < q else (q, r)
                if key in seen:
                    continue
                seen.add(key)
                g1 = newg.get(key[0], self.g[key[0]])
                g2 = newg.get(key[1], self.g[key[1]])
                out.append((key, self.ov(g1, g2)))
        return newg, out

    def delta(self, i, new, x):
        _, changes = self._changes(i, new, x)
        return sum(o - self.pairs.get(k, 0.0) for k, o in changes)

    def apply(self, i, new, x):
        newg, changes = self._changes(i, new, x)
        for k, o in changes:
            self.total += o - self.pairs.get(k, 0.0)
            if o:
                self.pairs[k] = o
            else:
                self.pairs.pop(k, None)
        for r, g in newg.items():
            self.g[r] = g

    def violated(self):
        return self.total > 1e-9

    def repair(self, x, rng, lb, ub):
        """Move one rectangle of a random overlapping pair just past the other
        on the y axis, or drop it if its presence is a variable."""
        if not self.pairs:
            return None
        a, b = rng.choice(list(self.pairs))
        if rng.random() < 0.5:
            a, b = b, a
        Y = self.ys[a]
        pres = Y.presence if isinstance(Y.presence, Var) else self.xs[a].presence
        movable = [(v, c) for v, c in Y.start.coeffs.items() if c]
        if movable and rng.random() < 0.8:
            v, c = movable[0]
            gb = self.g[b]
            size = Y.size.value(x)
            target = gb[3] if rng.random() < 0.5 else gb[2] - size  # above or below b
            cur = Y.start.value(x)
            new = x[v.index] + round((target - cur) / c)
            new = min(ub[v.index], max(lb[v.index], new))
            if new != x[v.index]:
                return v.index, new
        if isinstance(pres, Var) and x[pres.index] == 1 and lb[pres.index] == 0:
            return pres.index, 0
        return None


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------


class _Objective:
    def __init__(self, solver):
        self._solver = solver

    def Value(self):
        return self._solver._best_obj_value


class SaSolver:
    FEASIBLE = pywraplp.Solver.FEASIBLE
    NOT_SOLVED = pywraplp.Solver.NOT_SOLVED

    def __init__(self, seed=0):
        self.vars: list[Var] = []
        self.rows: list[Row] = []
        self.globals: list = []
        self.obj = LinearExpr()
        self.sense = 1.0  # +1 minimize, -1 maximize (internally minimize sense*obj)
        self.time_limit_s = 10.0
        self.rng = random.Random(seed)
        self._best_obj_value = None
        self._wall = 0.0
        self.stats = {}

    # ---- modeling API ----------------------------------------------------
    def IntVar(self, lb, ub, name=""):
        v = Var(lb, ub, name, len(self.vars))
        self.vars.append(v)
        return v

    def BoolVar(self, name=""):
        return self.IntVar(0, 1, name)

    def NumVar(self, *a, **k):
        raise NotImplementedError("SaSolver handles integer variables only")

    def Add(self, row, name=""):
        if not isinstance(row, Row):
            raise TypeError("Add expects a linear comparison")
        self.rows.append(row)
        return row

    def NewIntervalVar(self, start, size, name=""):
        return Interval(start, size, None, name)

    def NewOptionalIntervalVar(self, start, size, presence, name=""):
        return Interval(start, size, presence, name)

    def AddNoOverlap2D(self, xs, ys):
        g = NoOverlap2D(list(xs), list(ys))
        self.globals.append(g)
        return g

    def Sum(self, xs):
        e = LinearExpr()
        for x in xs:
            e = e + x
        return e

    def Minimize(self, expr):
        self.obj, self.sense = LinearExpr.of(expr), 1.0

    def Maximize(self, expr):
        self.obj, self.sense = LinearExpr.of(expr), -1.0

    def Objective(self):
        return _Objective(self)

    def Value(self, v):
        return LinearExpr.of(v).value([w.value for w in self.vars])

    def SetTimeLimit(self, ms):
        self.time_limit_s = ms / 1000.0

    def WallTime(self):
        return int(self._wall * 1000)

    def NumVariables(self):
        return len(self.vars)

    def NumConstraints(self):
        return len(self.rows) + len(self.globals)

    # ---- search ----------------------------------------------------------
    def Solve(self):
        t0 = time.perf_counter()
        n = len(self.vars)
        rows = self.rows
        cols = defaultdict(list)  # var -> [(row, coeff)]
        lit_rows = defaultdict(list)  # var -> rows it enforces
        for ri, r in enumerate(rows):
            for v, a in r.coeffs.items():
                if a:
                    cols[v.index].append((ri, a))
            for lit in r.enforce:
                lit_rows[lit.index].append(ri)
        gcols = defaultdict(list)
        for g in self.globals:
            for i in g.deps:
                gcols[i].append(g)
        objc = [0.0] * n
        for v, a in self.obj.coeffs.items():
            objc[v.index] = self.sense * a
        lb = [v.lb for v in self.vars]
        ub = [v.ub for v in self.vars]
        x = [v.lb for v in self.vars]
        act = [sum(a * x[v.index] for v, a in r.coeffs.items()) for r in rows]
        enf = [[lit.index for lit in r.enforce] for r in rows]

        def raw(ri, a_):
            r = rows[ri]
            if a_ < r.lo:
                return r.lo - a_
            if a_ > r.hi:
                return a_ - r.hi
            return 0.0

        def on(ri, override=None):
            for li in enf[ri]:
                val = override[1] if override and override[0] == li else x[li]
                if val < 1:
                    return False
            return True

        def viol(ri, a_, override=None):
            return raw(ri, a_) if on(ri, override) else 0.0

        violated = {ri for ri in range(len(rows)) if viol(ri, act[ri]) > 1e-9}
        for g in self.globals:
            g.init(x)
        obj = sum(objc[i] * x[i] for i in range(n))

        scale = max(1.0, max((abs(c) for c in objc), default=1.0))
        weight = scale
        T = scale
        best_obj, best_x = INF, None
        it = accepted = 0
        last_feasible_it = 0

        def infeasible():
            return bool(violated) or any(g.violated() for g in self.globals)

        def delta_for(i, new):
            d = new - x[i]
            dv = 0.0
            touched = set()
            for ri, a in cols[i]:
                touched.add(ri)
                dv += viol(ri, act[ri] + a * d, (i, new)) - viol(ri, act[ri])
            for ri in lit_rows[i]:
                if ri not in touched:
                    dv += viol(ri, act[ri], (i, new)) - viol(ri, act[ri])
            for g in gcols[i]:
                dv += g.delta(i, new, x)
            return objc[i] * d, dv

        def apply(i, new):
            nonlocal obj
            d = new - x[i]
            if not d:
                return
            for g in gcols[i]:
                g.apply(i, new, x)
            obj += objc[i] * d
            for ri, a in cols[i]:
                act[ri] += a * d
            x[i] = new
            for ri in set(r for r, _ in cols[i]) | set(lit_rows[i]):
                if viol(ri, act[ri]) > 1e-9:
                    violated.add(ri)
                else:
                    violated.discard(ri)

        def try_move(i, new):
            nonlocal accepted, best_obj, best_x, last_feasible_it
            if new == x[i]:
                return
            do, dv = delta_for(i, new)
            dE = do + weight * dv
            if dE <= 0 or rng.random() < math.exp(-dE / T):
                apply(i, new)
                accepted += 1
                if not infeasible():
                    last_feasible_it = it
                    if obj < best_obj - 1e-9:
                        best_obj, best_x = obj, list(x)
                        self.stats["time_to_best"] = time.perf_counter() - t0

        deadline = t0 + self.time_limit_s
        rng = self.rng
        while True:
            it += 1
            if (it & 255) == 0:
                if time.perf_counter() >= deadline:
                    break
                T = max(T * 0.97, 1e-6 * scale)
                if infeasible():
                    if it - last_feasible_it > 2000:
                        weight *= 1.3
                else:
                    weight = max(scale, weight * 0.98)
                if T < 1e-4 * scale and rng.random() < 0.02:
                    T = scale * 0.5  # reheat
            bad_globals = [g for g in self.globals if g.violated()]
            if bad_globals and rng.random() < 0.5:
                mv = rng.choice(bad_globals).repair(x, rng, lb, ub)
                if mv:
                    try_move(*mv)
                continue
            if rows and not infeasible() and rng.random() < 0.3:
                # row-preserving swap: +1 / -1 on two vars with equal coefficients
                r = rows[rng.randrange(len(rows))]
                items = [(v.index, a) for v, a in r.coeffs.items() if a]
                if len(items) < 2:
                    continue
                (i, ai), (j, aj) = rng.sample(items, 2)
                if ai != aj or x[i] >= ub[i] or x[j] <= lb[j]:
                    continue
                do1, dv1 = delta_for(i, x[i] + 1)
                apply(i, x[i] + 1)
                do2, dv2 = delta_for(j, x[j] - 1)
                dE = do1 + do2 + weight * (dv1 + dv2)
                if dE <= 0 or rng.random() < math.exp(-dE / T):
                    apply(j, x[j] - 1)
                    accepted += 1
                    if not infeasible():
                        last_feasible_it = it
                        if obj < best_obj - 1e-9:
                            best_obj, best_x = obj, list(x)
                            self.stats["time_to_best"] = time.perf_counter() - t0
                else:
                    apply(i, x[i] - 1)  # revert
                continue
            if violated and rng.random() < 0.6:
                ri = rng.choice(list(violated))
                r = rows[ri]
                items = [(v.index, a) for v, a in r.coeffs.items() if a]
                if not items:
                    continue
                i, a = items[rng.randrange(len(items))]
                need = (r.lo - act[ri]) if act[ri] < r.lo else (r.hi - act[ri])
                step = need / a
                step = math.ceil(step) if step > 0 else math.floor(step)
                try_move(i, min(ub[i], max(lb[i], x[i] + step)))
                continue
            i = rng.randrange(n)
            if lb[i] == ub[i]:
                continue
            if ub[i] - lb[i] == 1:
                new = lb[i] + ub[i] - x[i]
            elif rng.random() < 0.7:
                span = max(1, (ub[i] - lb[i]) // 16)
                new = x[i] + rng.choice((-1, 1)) * (1 + int(rng.expovariate(1.0 / span)))
                new = min(ub[i], max(lb[i], new))
            else:
                new = rng.randint(lb[i], ub[i])
            try_move(i, new)
        self._wall = time.perf_counter() - t0
        self.stats.update(iterations=it, accepted=accepted)
        if best_x is None:
            self._best_obj_value = None
            return self.NOT_SOLVED
        for v, val in zip(self.vars, best_x):
            v.value = val
        self._best_obj_value = self.sense * best_obj + self.obj.const
        return self.FEASIBLE

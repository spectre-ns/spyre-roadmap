"""The same modeling interface as SaSolver, implemented on CP-SAT (cp_model).

pywraplp cannot express intervals or 2-D no-overlap, so the exact reference for
the layout model is CP-SAT itself, behind the same method names.
"""

from __future__ import annotations

from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model


class _Objective:
    def __init__(self, s):
        self._s = s

    def Value(self):
        return self._s._obj_value


class CpSatProtocol:
    OPTIMAL = pywraplp.Solver.OPTIMAL
    FEASIBLE = pywraplp.Solver.FEASIBLE
    NOT_SOLVED = pywraplp.Solver.NOT_SOLVED

    def __init__(self, workers=16):
        self.m = cp_model.CpModel()
        self.workers = workers
        self.time_limit_s = 60.0
        self._obj_value = None
        self._wall = 0.0
        self._solver = None

    def IntVar(self, lb, ub, name=""):
        return self.m.new_int_var(int(lb), int(ub), name)

    def BoolVar(self, name=""):
        return self.m.new_bool_var(name)

    def Add(self, row, name=""):
        return self.m.add(row)  # returned Constraint has OnlyEnforceIf

    def NewIntervalVar(self, start, size, name=""):
        return self.m.new_interval_var(start, size, start + size, name)

    def NewOptionalIntervalVar(self, start, size, presence, name=""):
        return self.m.new_optional_interval_var(start, size, start + size, presence, name)

    def AddNoOverlap2D(self, xs, ys):
        return self.m.add_no_overlap_2d(list(xs), list(ys))

    def Sum(self, xs):
        return sum(xs)

    def Minimize(self, e):
        self.m.minimize(e)

    def Maximize(self, e):
        self.m.maximize(e)

    def Objective(self):
        return _Objective(self)

    def SetTimeLimit(self, ms):
        self.time_limit_s = ms / 1000.0

    def Solve(self):
        s = cp_model.CpSolver()
        s.parameters.num_workers = self.workers
        s.parameters.max_time_in_seconds = self.time_limit_s
        s.parameters.random_seed = 0
        st = s.solve(self.m)
        self._solver = s
        self._wall = s.wall_time
        if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            self._obj_value = s.objective_value
            self.bound = s.best_objective_bound
        return {cp_model.OPTIMAL: self.OPTIMAL, cp_model.FEASIBLE: self.FEASIBLE}.get(st, self.NOT_SOLVED)

    def Value(self, v):
        return self._solver.value(v)

    def WallTime(self):
        return int(self._wall * 1000)

    def NumVariables(self):
        return len(self.m.proto.variables)

    def NumConstraints(self):
        return len(self.m.proto.constraints)

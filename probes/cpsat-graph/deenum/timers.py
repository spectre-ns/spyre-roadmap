"""Stage-0 profile for the enumeration study: where LX planning time goes.

Process-local monkeypatches only. ``install(out_jsonl, tag, caps_dir=None)``
wraps the co-optimizing allocator's enumeration steps and the CP-SAT solver's
build/solve steps in reentrant timers and writes one JSON line per
``scratchpad_planning`` call:

* ``excl``  seconds spent in each label and in no nested label (these
  partition the planning wall time, so they add up without double counting);
* ``incl``  inclusive seconds per label;
* ``calls`` call counts, including the untimed hot helpers;
* ``census`` what the solver was handed (menus, edges, pairs, classes,
  relayout candidates) and what it built (variables, constraints).

Harness work (census, optional input pickles) runs under the ``harness`` label
so it is separable from the planner's own time.
"""

from __future__ import annotations

import copy
import functools
import json
import os
import pickle
import time
from collections import defaultdict


class T:
    stack: list = []
    incl: dict = defaultdict(float)
    excl: dict = defaultdict(float)
    calls: dict = defaultdict(int)
    helper_s: dict = defaultdict(float)  # (helper, enclosing label) -> seconds
    plan_idx = 0
    census: list = []
    edge_keys: list = []
    edge_raw: list = []  # per-candidate view keys, for the lazy-encoding probe
    solves: list = []
    fired: set = set()


def timed(label, fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        T.calls[label] += 1
        T.fired.add(label)
        reentrant = any(f[0] == label for f in T.stack)
        frame = [label, time.perf_counter(), 0.0]
        T.stack.append(frame)
        try:
            return fn(*a, **k)
        finally:
            dt = time.perf_counter() - frame[1]
            T.stack.pop()
            if not reentrant:
                T.incl[label] += dt
            T.excl[label] += dt - frame[2]
            if T.stack:
                T.stack[-1][2] += dt

    return wrapper


def counted(label, fn):
    """Hot helper: count calls and seconds by enclosing label, no frame."""

    @functools.wraps(fn)
    def wrapper(*a, **k):
        T.calls[label] += 1
        T.fired.add(label)
        if any(f[0] == "h:" + label for f in T.stack):
            return fn(*a, **k)
        parent = T.stack[-1][0] if T.stack else "<none>"
        T.stack.append(["h:" + label, 0.0, 0.0])
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            dt = time.perf_counter() - t0
            frame = T.stack.pop()
            if T.stack:
                # Timed work nested under the helper still belongs to a child
                # label, so hand it up instead of losing it with this frame.
                T.stack[-1][2] += frame[2]
            # Keep the enclosing timed frame's exclusive time intact: a helper
            # frame is transparent, its seconds are reported on the side.
            T.helper_s[f"{label}@{parent}"] += dt

    return wrapper


def _patch_method(cls, name, label, kind=timed):
    for klass in cls.__mro__:
        if name in vars(klass):
            raw = vars(klass)[name]
            if isinstance(raw, staticmethod):
                setattr(klass, name, staticmethod(kind(label, raw.__func__)))
            elif isinstance(raw, classmethod):
                raise TypeError(name)
            else:
                setattr(klass, name, kind(label, raw))
            return
    raise AttributeError(f"{cls.__name__}.{name}")


def _classes(n_rows, n_cols, pairs):
    """Row/column equivalence classes of a pair table and its rectangularity."""
    rows, cols = defaultdict(set), defaultdict(set)
    for i, j in pairs:
        rows[i].add(j)
        cols[j].add(i)
    row_sigs = {frozenset(v) for v in rows.values()}
    col_sigs = {frozenset(v) for v in cols.values()}
    # Rectangular: the table is a disjoint union of (row class x col class)
    # blocks, i.e. distinct row signatures are pairwise disjoint.
    seen: set = set()
    rect = True
    for sig in row_sigs:
        if seen & sig:
            rect = False
            break
        seen |= sig
    return len(row_sigs), len(col_sigs), rect


def _digit_stride(expr, split, n):
    """Stride S when ``expr`` is the digit ``floor(core_id / S) mod split`` on
    every core below ``n``; ``None`` when it is not such a digit."""
    from sympy.utilities.misc import as_int

    from torch_spyre._inductor.core_mapping import _owner_at_core

    try:
        vals = [as_int(_owner_at_core(expr, c)) for c in range(n)]
    except Exception:
        return None
    stride = next((c for c, v in enumerate(vals) if v != 0), None)
    if stride is None:
        return None
    if all(v == (c // stride) % split for c, v in enumerate(vals)):
        return stride
    return None


def _view_key(view):
    """Canonical (cores, ((dim, split, stride), ...)) of a view: what per-dim
    integer variables would have to agree on. ``None`` = no usable view;
    a key starting with "irregular" = an owner formula outside the digit form.
    """
    if view is None:
        return None
    n = view.num_cores if view.num_cores is not None else view.split_product
    slots = dict(view.core_to_slot)
    digits = []
    for dim, split in sorted(view.work_slice_dims):
        if split <= 1:
            continue
        stride = _digit_stride(slots[dim], split, n) if dim in slots else None
        if stride is None:
            return ("irregular", repr(view))
        digits.append((dim, split, stride))
    return (n, tuple(digits))


_EXACT_KEYS: dict = {}


def _exact_key(view):
    """Hashable key with ``a.same_partition(b)  <=>  key(a) == key(b) != None``:
    the core count plus, per split dim, the split and the owner slot evaluated
    on every core -- exactly what ``same_owner_maps`` compares pairwise."""
    if view is None:
        return None
    if view in _EXACT_KEYS:
        return _EXACT_KEYS[view]
    from sympy import sympify
    from sympy.utilities.misc import as_int

    from torch_spyre._inductor.core_mapping import _owner_at_core

    n = view.num_cores if view.num_cores is not None else view.split_product
    key = None
    if n > 0:
        slots = dict(view.core_to_slot)
        rows = []
        try:
            for dim, split in sorted(view.work_slice_dims):
                if split <= 1:
                    continue
                expr = sympify(slots[dim])
                rows.append(
                    (dim, split, tuple(as_int(_owner_at_core(expr, c)) for c in range(n)))
                )
            key = (n, tuple(rows))
        except (KeyError, TypeError, ValueError):
            key = None  # same_owner_maps answers False for such a view
    _EXACT_KEYS[view] = key
    return key


def _hash_join_match_pairs(self, parent_divisions, consumer_divisions, alloc):
    """``ResidencyEdge.match_pairs`` as a hash join on the exact view key."""
    from torch_spyre._inductor.work_division import _tile_view_for_div as tile
    cores = self._cores_used
    index = defaultdict(list)
    for j, cd in enumerate(consumer_divisions):
        view, tview = _exact_key(self.consumer_view(cd)), _exact_key(self.consumer_tile_view(cd))
        if view is not None and tview is not None:
            index[(view, tview, cores(cd))].append(j)
    pairs = []
    for i, cd in enumerate(parent_divisions):
        view = _exact_key(self.parent_view(cd))
        if view is None:
            continue
        tview = _exact_key(
            tile(self.parent_op, self.write_dep, self.buf_name, cd, self.prep_cache)
        )
        if tview is None:
            continue
        pairs.extend((i, j) for j in index.get((view, tview, cores(cd)), ()))
    return pairs


def _edge_key_census(edge, parent_divisions, consumer_divisions, pairs, alloc):
    from torch_spyre._inductor.work_division import _tile_view_for_div as tile
    pk = [_view_key(edge.parent_view(cd)) for cd in parent_divisions]
    ck = [_view_key(edge.consumer_view(cd)) for cd in consumer_divisions]
    pt = [
        _view_key(tile(edge.parent_op, edge.write_dep, edge.buf_name, cd, edge.prep_cache))
        for cd in parent_divisions
    ]
    ct = [_view_key(edge.consumer_tile_view(cd)) for cd in consumer_divisions]
    cores = edge._cores_used
    joined = [
        (i, j)
        for i in range(len(pk))
        if pk[i] is not None and pt[i] is not None
        for j in range(len(ck))
        if ck[j] is not None
        and ct[j] is not None
        and pk[i] == ck[j]
        and pt[i] == ct[j]
        and cores(parent_divisions[i]) == cores(consumer_divisions[j])
    ]

    def irregular(keys):
        return sum(1 for k in keys if k is not None and k[0] == "irregular")

    def none(keys):
        return sum(1 for k in keys if k is None)

    split_dims = [len(k[1]) for k in pk + ck if k is not None and k[0] != "irregular"]

    def cand(cd):
        return dict(
            splits={str(k): int(v) for k, v in cd.splits.items()},
            red=sorted(str(k) for k in cd.reduction_syms),
            tile=[(str(k), int(v)) for k, v in getattr(cd, "tile_splits", ())],
            cores=int(cores(cd)),
        )

    T.edge_raw.append(
        dict(
            consumer=edge.consumer_op.get_name(),
            parent=edge.buf_name,
            pk=pk, ck=ck, pt=pt, ct=ct,
            pairs=[tuple(p) for p in pairs],
            pc=[cand(cd) for cd in parent_divisions],
            cc=[cand(cd) for cd in consumer_divisions],
        )
    )
    return dict(
        consumer=edge.consumer_op.get_name(),
        parent=edge.buf_name,
        n_parent=len(pk),
        n_consumer=len(ck),
        pairs=len(pairs),
        exact=sorted(joined) == sorted(pairs),
        joined=len(joined),
        irregular=[irregular(pk), irregular(ck), irregular(pt), irregular(ct)],
        none=[none(pk), none(ck)],
        parent_classes=len({k for k in pk if k is not None}),
        consumer_classes=len({k for k in ck if k is not None}),
        max_split_dims=max(split_dims, default=0),
    )


def _census(solver, cost_expr):
    bufs = list(solver.buffers)
    by_name = {b.name: b for b in bufs}
    menus, tiled, edges, relay = [], 0, [], []
    for b in bufs:
        divs = getattr(b, "core_divisions", None) or []
        menus.append(len(divs))
        tiled += sum(1 for cd in divs if getattr(cd, "tile_splits", ()))
        for parent, pairs in (getattr(b, "cd_parent_matches", None) or {}).items():
            p = by_name.get(parent)
            n_p = len(getattr(p, "core_divisions", None) or []) if p else 0
            rc, cc, rect = _classes(n_p, len(divs), pairs)
            edges.append([n_p, len(divs), len(pairs), rc, cc, int(rect)])
        for parent, cands in (getattr(b, "cd_parent_relayouts", None) or {}).items():
            relay.append([len(cands), len({c.group for c in cands})])
    return dict(
        buffers=len(bufs),
        menus=menus,
        tiled_candidates=tiled,
        edges=edges,
        relayout_edges=relay,
        priced=cost_expr is not None,
        relayout_copies=[type(b).__name__ for b in bufs].count("RelayoutCopyBuffer"),
    )


def install(out_jsonl, tag, caps_dir=None):
    import torch_spyre._inductor.cost_model as cost_model
    import torch_spyre._inductor.pass_utils as pass_utils
    import torch_spyre._inductor.passes as passes
    import torch_spyre._inductor.scratchpad.allocator as alloc
    import torch_spyre._inductor.scratchpad.ilp_solver_ortools as ilp
    from torch_spyre._inductor import config

    A = alloc.CoOptimizingAllocator
    C = ilp.CpSatLayoutSolver

    for name, label in [
        ("_prepare_buffers", "prepare_buffers"),
        ("_determine_in_place_division_invariant", "in_place_invariant"),
        ("_division_map", "menu:division_map"),
        ("_enumerate_core_divisions", "menu:enumerate_core_divisions"),
        ("_tiling_candidates", "menu:tiling_candidates"),
        ("_build_cd_bound_buffers", "build_buffers"),
        ("_cd_parent_matches", "pairs:parent_matches"),
        ("_cd_parent_relayouts", "pairs:relayout_candidates"),
        ("_clone_divisions_and_matches", "pairs:clone_matches"),
        ("_loop_carry_update_edge", "pairs:loop_carry"),
        ("_loop_carry_read_edges", "pairs:loop_carry"),
        ("_relayout_copy_buffers", "pairs:relayout_copy_buffers"),
        ("_pricing_operations", "pricing_operations"),
        ("_direct_read_candidates_priced", "menu:direct_read_priced"),
        ("_solve", "solve_wrapper"),
        ("_extract_op_features", "cost:op_features"),
        ("_post_solve", "post_solve"),
    ]:
        _patch_method(A, name, label)

    for name, label in [
        ("_plan_layout_generic", "cpsat:other"),
        ("_wrap", "cpsat:build_menu_tables"),
        ("_add_core_division", "cpsat:build_gates"),
        ("_constrain_relayout_copies", "cpsat:build_other"),
        ("_add_inplace_relaxation", "cpsat:build_other"),
        ("_cut_literals", "cpsat:build_other"),
        ("_tile_count_terms", "cpsat:build_other"),
        ("_solve_and_record", "cpsat:solve"),
        ("_extract", "cpsat:extract"),
    ]:
        _patch_method(C, name, label)
    import sys as _sys
    cep = _sys.modules[ilp.cost_expr_to_cpsat.__module__]
    _patch_method(cep._SympyExprToCpSat, "convert", "cpsat:cost_lowering")

    cost_model.predict_bundles = timed("cost:predict_bundles", cost_model.predict_bundles)

    # Hot helpers, patched where they are looked up.
    for mod in (pass_utils, alloc):
        for name in ("_per_core_view_from_prep", "_prepare_per_core_view"):
            if hasattr(mod, name):
                setattr(mod, name, counted(name, getattr(mod, name)))
    if hasattr(alloc, "enumerate_work_division_candidates"):
        alloc.enumerate_work_division_candidates = counted(
            "enumerate_work_division_candidates",
            alloc.enumerate_work_division_candidates,
        )
    _patch_method(pass_utils.PerCoreView, "same_partition", "same_partition", counted)

    # Does per-dim (split, stride) + core-count equality reproduce each pair
    # table? Recomputes the views, so it runs under the harness label.
    RE = alloc.ResidencyEdge
    orig_match = RE.match_pairs

    hash_join = os.environ.get("DEENUM_HASHJOIN", "0") == "1"

    def match_pairs(self, parent_divisions, consumer_divisions):
        if hash_join:
            pairs = _hash_join_match_pairs(self, parent_divisions, consumer_divisions, alloc)
            # Exactness against the production nested loop, in harness time.
            ref = timed("harness", orig_match)(self, parent_divisions, consumer_divisions)
            T.edge_keys.append(dict(hash_join_exact=pairs == ref, pairs=len(ref)))
            return pairs
        pairs = orig_match(self, parent_divisions, consumer_divisions)
        if os.environ.get("DEENUM_EDGE_KEYS", "1") == "1":
            try:
                T.edge_keys.append(
                    timed("harness", _edge_key_census)(
                        self, parent_divisions, consumer_divisions, pairs, alloc
                    )
                )
            except Exception as e:  # never let the census break a compile
                T.edge_keys.append(dict(error=f"{type(e).__name__}: {e}"))
        return pairs

    RE.match_pairs = match_pairs

    # Solver input census (+ optional pickle), under the harness label.
    orig_plan = C.plan_layout_and_core_divisions

    def plan(self, cost_expr=None):
        @functools.partial(timed, "harness")
        def before():
            T.census.append(_census(self, cost_expr))
            if caps_dir:
                idx = T.plan_idx
                try:
                    bufs = []
                    for b in self.buffers:  # drop live compiler objects first
                        b2 = copy.copy(b)
                        for attr, empty in (
                            ("residency_edges", {}),
                            ("division_space", None),
                            ("tile_reads", {}),
                        ):
                            if hasattr(b2, attr):
                                setattr(b2, attr, empty)
                        bufs.append(b2)
                    bufs = copy.deepcopy(bufs)
                except Exception as e:  # never break the solve over a capture
                    with open(os.path.join(caps_dir, f"{tag}.{idx}.err"), "w") as f:
                        f.write(f"deepcopy {type(e).__name__}: {e}")
                    return
                rec = dict(
                    graph=tag,
                    solve=idx,
                    buffers=bufs,
                    edge_raw=list(T.edge_raw),
                    limit=self.limit,
                    alignment=self.alignment,
                    cost_expr=cost_expr,
                    sencores=config.sencores,
                    time_limit=self._time_limit_seconds,
                    auto_coarse_tiling=config.auto_coarse_tiling,
                )
                path = os.path.join(caps_dir, f"{tag}.{idx}.pkl")
                try:
                    with open(path, "wb") as f:
                        pickle.dump(rec, f)
                except Exception as e:
                    with open(path + ".err", "w") as f:
                        f.write(f"{type(e).__name__}: {e}")

        before()
        try:
            return orig_plan(self, cost_expr)
        finally:
            T.solves.append(dict(getattr(self, "last_solve_stats", {}) or {}))

    C.plan_layout_and_core_divisions = timed("cpsat:entry", plan)

    if caps_dir:
        os.makedirs(caps_dir, exist_ok=True)

    orig_planning = passes.scratchpad_planning

    def planning(graph, *a, **k):
        for d in (T.incl, T.excl, T.calls, T.helper_s):
            d.clear()
        T.census.clear()
        T.edge_keys.clear()
        T.edge_raw.clear()
        T.solves.clear()
        t0 = time.perf_counter()
        try:
            return timed("planning", orig_planning)(graph, *a, **k)
        finally:
            wall = time.perf_counter() - t0
            rec = dict(
                tag=tag,
                plan=T.plan_idx,
                wall_s=wall,
                ops=len(graph.operations),
                excl=dict(T.excl),
                incl=dict(T.incl),
                calls=dict(T.calls),
                helper_s=dict(T.helper_s),
                census=list(T.census),
                edge_keys=list(T.edge_keys),
                solves=list(T.solves),
                cfg=dict(
                    auto_coarse_tiling=config.auto_coarse_tiling,
                    layout_solver=config.layout_solver,
                    sencores=config.sencores,
                    workers=config.get_cpu_count(),
                    relayout=config.lx_planner_relayout,
                ),
            )
            T.plan_idx += 1
            with open(out_jsonl, "a") as f:
                f.write(json.dumps(rec) + "\n")

    passes.scratchpad_planning = planning

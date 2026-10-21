"""Lazy vs eager encoding probe (Phase 2.0 of the CP-SAT graph refactor plan).

Usage: lazy_probe.py <caps_dir> <out.jsonl> [workers=1,16] [limit_s=60]

For every captured solve (timers.py capture: sanitized buffers + per-edge view
keys), build ONE reduced layout model three ways and solve each:

  E   eager, today's encoding: a menu-index var per buffer, element tables for
      footprint / cores, and a pair table per READ edge gating residency.
  L1  lazy relations, menu legality: per-axis split vars; an edge is gated by
      per-device-dim (split, stride) equality plus core-count equality, strides
      as prefix products of the op's splits; each node's legality is ONE table
      over its split tuple (no pair tables).
  L2  fully lazy: like L1 but legality is only per-axis domains, cores <= the
      menu's max, and at most one reduction split. No menu at all.

Reduced model, identical across arms: residency literal per buffer (pinned off
by residency_reason), per-core footprint ceil(size_units / output_partition),
LX capacity, 2-D no-overlap over (lifetime x address), match gate on READ
edges between two menu buffers, objective = spill cost of non-resident buffers
(weighted) minus total cores (parallelism tie-break). Not modelled (same for all
arms): relayout copies, in-place merges, the sympy cost expression, tiling.

Before solving, each READ edge is checked: does the per-axis relation inferred
from its candidates reproduce the allocator's final pair table exactly? Edges
that do not fall back to a table over the two split tuples in L1/L2 (counted).
"""

from __future__ import annotations

import glob
import itertools
import json
import math
import os
import pickle
import sys
import time

import torch  # noqa: F401  (unpickling needs torch_spyre classes)
import torch_spyre  # noqa: F401
from ortools.sat.python import cp_model


def ceil_div(a, b):
    return -(-a // b)


# --------------------------------------------------------------------------
# Per-axis relation inference from per-candidate view keys
# --------------------------------------------------------------------------


def predicted_key(cand, sym_dim, order, contiguous, whole=False):
    """View key the per-axis rule predicts for a candidate on one edge side."""
    splits = cand["splits"]
    seq = [s for s in order if s in splits]
    if contiguous is not None and splits.get(contiguous, 1) > 1 and contiguous in seq:
        seq.remove(contiguous)
        seq.insert(0, contiguous)
    stride, digits = 1, []
    for s in seq:
        f = splits[s]
        d = sym_dim.get(s)
        if f > 1 and d is not None:
            digits.append((d, f, stride))
        stride *= f
    if whole and not digits:
        return (1, ())  # a tiling that slices nothing of the buffer reads it whole
    return (cand["cores"], tuple(sorted(digits)))


def infer_side(cands, keys):
    """Find (sym->device dim or None, symbol order, contiguous symbol) that
    reproduces every regular key on this side; None if no such rule exists."""
    syms = sorted({s for c in cands for s in c["splits"]})
    # symbol -> device dim votes from single-split candidates
    sym_dim: dict = {}
    for c, k in zip(cands, keys):
        if k is None or k[0] == "irregular":
            continue
        live = [s for s, f in c["splits"].items() if f > 1]
        if len(live) == 1:
            s = live[0]
            d = k[1][0][0] if k[1] else None
            if s in sym_dim and sym_dim[s] != d:
                return None
            sym_dim[s] = d
    unknown = [s for s in syms if s not in sym_dim]
    dims = sorted({dg[0] for k in keys if k and k[0] != "irregular" for dg in k[1]})
    for assign in itertools.product([None, *dims], repeat=len(unknown)):
        sd = dict(sym_dim, **dict(zip(unknown, assign)))
        orders = itertools.permutations(syms) if len(syms) <= 5 else [tuple(syms)]
        for order in orders:
            for contiguous in [None, *syms]:
                for whole in ([False, True] if any(c.get("tile") for c in cands) else [False]):
                    if all(
                        k is None
                        or k[0] == "irregular"
                        or predicted_key(c, sd, order, contiguous, whole) == k
                        for c, k in zip(cands, keys)
                    ):
                        return dict(sym_dim=sd, order=list(order), contiguous=contiguous, whole=whole)
    return None


def tile_cands(cands):
    """The tile side of each candidate, in the shape infer_side expects."""
    return [
        dict(splits=dict(c["tile"]), cores=math.prod(n for _, n in c["tile"]), tile=True)
        for c in cands
    ]


def side_view_none(cands, keys):
    """Which candidates have no usable view (e.g. partial-reduction writes)."""
    return [k is None or (k and k[0] == "irregular") for k in keys]


def normalize_keys(keys):
    out = []
    for k in keys:
        if k is None or k[0] == "irregular":
            out.append(k)
        else:
            out.append((k[0], tuple(tuple(d) for d in k[1])))
    return out


def analyse_edge(e, final_pairs, menu_sizes):
    """Is the edge exactly a per-axis relation? Returns a dict with the rule."""
    pk, ck = normalize_keys(e["pk"]), normalize_keys(e["ck"])
    pt, ct = normalize_keys(e["pt"]), normalize_keys(e["ct"])
    trivial_tiles = all(t == (1, ()) or t is None for t in pt + ct)
    res = dict(
        parent=e["parent"],
        consumer=e["consumer"],
        n_parent=len(pk),
        n_consumer=len(ck),
        pairs=len(final_pairs),
        allocator_modified=sorted(map(tuple, e["pairs"])) != sorted(map(tuple, final_pairs)),
        trivial_tiles=trivial_tiles,
        irregular=sum(1 for k in pk + ck if k is not None and k[0] == "irregular"),
    )
    res["menu_mismatch"] = (len(pk), len(ck)) != menu_sizes
    res["irregular"] += sum(1 for k in pt + ct if k is not None and k[0] == "irregular")
    if res["allocator_modified"] or res["irregular"] or res["menu_mismatch"]:
        res["lazy_exact"] = False
        return res
    prule, crule = infer_side(e["pc"], pk), infer_side(e["cc"], ck)
    ptc, ctc = tile_cands(e["pc"]), tile_cands(e["cc"])
    prule_t, crule_t = infer_side(ptc, pt), infer_side(ctc, ct)
    res["parent_rule"], res["consumer_rule"] = prule, crule
    res["parent_tile_rule"], res["consumer_tile_rule"] = prule_t, crule_t
    if None in (prule, crule, prule_t, crule_t):
        res["lazy_exact"] = False
        return res

    def key(c, r):
        return predicted_key(c, r["sym_dim"], r["order"], r["contiguous"], r.get("whole", False))

    # The relation the lazy encoding posts: both views usable, core and tile keys equal.
    pred = [
        (i, j)
        for i in range(len(pk))
        if pk[i] is not None and pt[i] is not None
        for j in range(len(ck))
        if ck[j] is not None and ct[j] is not None
        and key(e["pc"][i], prule) == key(e["cc"][j], crule)
        and key(ptc[i], prule_t) == key(ctc[j], crule_t)
    ]
    res["lazy_exact"] = sorted(pred) == sorted(map(tuple, final_pairs))

    # Candidates with no usable view (or tile view): the gate must forbid them. A
    # producer whose only reason is a split reduction axis is already covered by
    # the "reduction split == 1" constraint; anything else needs the side's table.
    def unexplained(keys, tkeys, cands, producer):
        out = []
        for k, t, c in zip(keys, tkeys, cands):
            if k is None or t is None:
                red = producer and k is None and t is not None and any(
                    c["splits"].get(s, 1) > 1 for s in c["red"]
                )
                if not red:
                    out.append(True)
                    continue
            out.append(False)
        return out

    pu = unexplained(pk, pt, e["pc"], True)
    cu = unexplained(ck, ct, e["cc"], False)
    res["p_ok"] = [i for i in range(len(pk)) if pk[i] is not None and pt[i] is not None] if any(pu) else None
    res["c_ok"] = [j for j in range(len(ck)) if ck[j] is not None and ct[j] is not None] if any(cu) else None
    return res


# --------------------------------------------------------------------------
# Model building
# --------------------------------------------------------------------------


def lifetime(b):
    start = b.lifetime_start_override if b.lifetime_start_override is not None else b.uses[0]
    end = b.lifetime_end_override if b.lifetime_end_override is not None else b.uses[-1]
    return start, end


def spill_cost(b, units):
    first_read = b.first_use_is_read
    read_count = len(b.uses) - (0 if first_read else 1)
    served = read_count - (1 if first_read else 0)
    boundary = getattr(b, "boundary", None)
    inter = (str(boundary).endswith("Intermediate")) if boundary is not None else not first_read
    return max(0, served + (1 if inter else 0)) * units


class Builder:
    def __init__(self, cap, arm, edges_info):
        self.cap, self.arm, self.edges_info = cap, arm, edges_info
        self.m = cp_model.CpModel()
        self.align = cap["alignment"]
        self.limit_units = cap["limit"] // self.align
        self.bufs = [b for b in cap["buffers"] if b.uses]
        self.node = {}

    def menu(self, b):
        return list(getattr(b, "core_divisions", None) or [])

    def post_node(self, b):
        m, units = self.m, ceil_div(b.size, self.align)
        menu = self.menu(b)
        v = dict(units=units)
        v["in"] = m.new_bool_var(f"in_{b.name}")
        if b.residency_reason is not None:
            m.add(v["in"] == 0)
        if not menu or len(menu) == 1:
            cd = menu[0] if menu else None
            outp = cd.output_partition * cd.tiling.output_tile_count if cd else 1
            v["eff"] = ceil_div(units, outp)
            v["cores"] = cd.cores_used if cd else 1
            v["fixed"] = True
            v["split"] = {str(s): f for s, f in (cd.splits.items() if cd else [])}
        elif self.arm == "E":
            n = len(menu)
            div = m.new_int_var(0, n - 1, f"div_{b.name}")
            effs = [ceil_div(units, cd.output_partition * cd.tiling.output_tile_count) for cd in menu]
            cores = [cd.cores_used for cd in menu]
            v["div"] = div
            v["eff"] = m.new_int_var(min(effs), max(effs), f"eff_{b.name}")
            m.add_element(div, effs, v["eff"])
            v["cores"] = m.new_int_var(min(cores), max(cores), f"cores_{b.name}")
            m.add_element(div, cores, v["cores"])
            v["fixed"] = False
        else:  # L1 / L2: per-axis split vars
            syms = sorted({str(s) for cd in menu for s in cd.splits})
            red = {str(s) for cd in menu for s in cd.reduction_syms}
            dom = {s: sorted({cd.splits.get(_sym(cd, s), 1) for cd in menu}) for s in syms}
            sv = {
                s: m.new_int_var_from_domain(cp_model.Domain.FromValues(dom[s]), f"s_{b.name}_{s}")
                for s in syms
            }
            v["split"], v["fixed"] = sv, False
            tsyms = sorted({str(k) for cd in menu for k, _ in cd.tile_splits})
            tdom = {t: sorted({dict((str(k), n) for k, n in cd.tile_splits).get(t, 1) for cd in menu}) for t in tsyms}
            tv = {
                t: m.new_int_var_from_domain(cp_model.Domain.FromValues(tdom[t]), f"t_{b.name}_{t}")
                for t in tsyms
            }
            tred = set()
            for cd in menu:
                for (k, _), ax in zip(cd.tile_splits, cd.tiling.axes):
                    if ax.is_reduction:
                        tred.add(str(k))
            v["tile"], v["tsyms"] = tv, tsyms
            if self.arm.startswith("L1"):
                tuples = [
                    [cd.splits.get(_sym(cd, s), 1) for s in syms]
                    + [dict((str(k), n) for k, n in cd.tile_splits).get(t, 1) for t in tsyms]
                    for cd in menu
                ]
                m.add_allowed_assignments([sv[s] for s in syms] + [tv[t] for t in tsyms], tuples)
            else:
                rs = [s for s in syms if s in red]
                if len(rs) > 1:
                    lits = []
                    for s in rs:
                        lit = m.new_bool_var("")
                        m.add(sv[s] > 1).only_enforce_if(lit)
                        m.add(sv[s] == 1).only_enforce_if(~lit)
                        lits.append(lit)
                    m.add(sum(lits) <= 1)
            maxc = max(cd.cores_used for cd in menu)
            v["cores"] = self.product([sv[s] for s in syms], maxc, f"cores_{b.name}")
            maxt = max(cd.tiling.output_tile_count for cd in menu) * maxc
            outp = self.product(
                [sv[s] for s in syms if s not in red] + [tv[t] for t in tsyms if t not in tred],
                maxt, f"outp_{b.name}",
            )
            if isinstance(outp, int):
                v["eff"] = ceil_div(units, outp)
            else:
                eff = m.new_int_var(0, max(units, 0), f"eff_{b.name}")
                m.add_division_equality(eff, units + outp - 1, outp)
                v["eff"] = eff
            v["syms"], v["red"] = syms, red
        start, end = lifetime(b)
        v["start"], v["end"] = start, end
        v["off"] = m.new_int_var(0, self.limit_units, f"off_{b.name}")
        m.add(v["off"] + v["eff"] <= self.limit_units).only_enforce_if(v["in"])
        v["spill"] = spill_cost(b, units)
        self.node[b.name] = v

    def product(self, vars_, ub, name):
        if not vars_:
            return 1
        acc = vars_[0]
        for i, x in enumerate(vars_[1:]):
            p = self.m.new_int_var(1, ub, f"{name}_p{i}")
            self.m.add_multiplication_equality(p, [acc, x])
            acc = p
        return acc

    def post_no_overlap(self):
        m, xs, ys = self.m, [], []
        for name, v in self.node.items():
            t = m.new_optional_fixed_size_interval_var(
                v["start"], v["end"] - v["start"] + 1, v["in"], f"t_{name}"
            )
            if isinstance(v["eff"], int):
                a = m.new_optional_fixed_size_interval_var(v["off"], v["eff"], v["in"], f"a_{name}")
            else:
                endv = m.new_int_var(0, self.limit_units, f"aend_{name}")
                a = m.new_optional_interval_var(v["off"], v["eff"], endv, v["in"], f"a_{name}")
            xs.append(t)
            ys.append(a)
        m.add_no_overlap_2d(xs, ys)

    def post_edges(self):
        m, stats = self.m, dict(gated=0, table=0, lazy=0, pinned=0)
        by_name = {b.name: b for b in self.bufs}
        for b in self.bufs:
            for parent, pairs in (getattr(b, "cd_parent_matches", None) or {}).items():
                if parent not in self.node or b.name not in self.node:
                    continue
                p, c = self.node[parent], self.node[b.name]
                pb = by_name[parent]
                if p["fixed"] or c["fixed"]:
                    continue  # today: no gate unless both choose (placement-only)
                stats["gated"] += 1
                if not pairs:
                    m.add(p["in"] == 0)
                    stats["pinned"] += 1
                    continue
                if self.arm == "E":
                    m.add_allowed_assignments([p["div"], c["div"]], pairs).only_enforce_if(p["in"])
                    stats["table"] += 1
                    continue
                info = self.edges_info.get((parent, b.name))
                if info is None or not info.get("lazy_exact"):
                    # fallback: a table over the two (split, tile) tuples
                    pm, cm = self.menu(pb), self.menu(b)

                    def tup(cd, v):
                        tiles = dict((str(k), n) for k, n in cd.tile_splits)
                        return [cd.splits.get(_sym(cd, s), 1) for s in v["syms"]] + [
                            tiles.get(t, 1) for t in v["tsyms"]
                        ]

                    def vars_of(v):
                        return [v["split"][s] for s in v["syms"]] + [v["tile"][t] for t in v["tsyms"]]

                    tuples = [tup(pm[i], p) + tup(cm[j], c) for i, j in pairs]
                    m.add_allowed_assignments(vars_of(p) + vars_of(c), tuples).only_enforce_if(p["in"])
                    stats["table"] += 1
                    continue
                stats["lazy"] += 1
                if info.get("p_ok") is not None:
                    self.post_usable(p, self.menu(pb), info["p_ok"])
                if info.get("c_ok") is not None:
                    self.post_usable(c, self.menu(b), info["c_ok"], en=p["in"])
                self.post_lazy_gate(p, c, info["parent_rule"], info["consumer_rule"])
                self.post_tile_gate(p, c, info["parent_tile_rule"], info["consumer_tile_rule"])
        return stats

    def post_lazy_gate(self, p, c, prule, crule):
        m, en = self.m, p["in"]
        # producer: a split reduction axis leaves a partial write -> no view
        for s in p["syms"]:
            if s in p["red"]:
                m.add(p["split"][s] == 1).only_enforce_if(en)
        m.add(p["cores"] == c["cores"]).only_enforce_if(en)
        self._gate_dims(en, self.dims(p, prule), self.dims(c, crule))

    def tile_count(self, v, rule, dims):
        """The tile key's count: all tiles, or 1 when the rule reads the buffer
        whole and no buffer dim is tiled."""
        n = self.product(list(v["tile"].values()), 4096, "")
        if not rule.get("whole"):
            return n
        lives = [f for f, _ in dims.values() if not isinstance(f, int)]
        if not lives:
            return 1
        any_live = self.m.new_bool_var("")
        lits = []
        for f in lives:
            lit = self.m.new_bool_var("")
            self.m.add(f > 1).only_enforce_if(lit)
            self.m.add(f == 1).only_enforce_if(~lit)
            lits.append(lit)
        self.m.add_max_equality(any_live, lits)
        out = self.m.new_int_var(1, 4096, "")
        self.m.add(out == n).only_enforce_if(any_live)
        self.m.add(out == 1).only_enforce_if(~any_live)
        return out

    def post_usable(self, v, menu, ok, en=None):
        """Per-view table: the side's split/tile tuple must be one whose view exists."""
        syms, tsyms = v["syms"], v["tsyms"]
        tuples = [
            [menu[i].splits.get(_sym(menu[i], s), 1) for s in syms]
            + [dict((str(k), n) for k, n in menu[i].tile_splits).get(t, 1) for t in tsyms]
            for i in ok
        ]
        vars_ = [v["split"][s] for s in syms] + [v["tile"][t] for t in tsyms]
        lit = v["in"] if en is None else en
        if tuples:
            self.m.add_allowed_assignments(vars_, tuples).only_enforce_if(lit)
        else:
            self.m.add(lit == 0)

    def post_tile_gate(self, p, c, prule, crule):
        """Tile keys equal: same tile count and per-dim (count, stride)."""
        m, en = self.m, p["in"]
        pd, cd = self.dims(p, prule, "tile"), self.dims(c, crule, "tile")
        pn, cn = self.tile_count(p, prule, pd), self.tile_count(c, crule, cd)
        if not (isinstance(pn, int) and isinstance(cn, int)):
            m.add(pn == cn).only_enforce_if(en)
        elif pn != cn:
            m.add(en == 0)
        self._gate_dims(en, pd, cd)

    def _gate_dims(self, en, pdims, cdims):
        m = self.m
        for d in sorted(set(pdims) | set(cdims)):
            pf, pst = pdims.get(d, (1, 1))
            cf, cst = cdims.get(d, (1, 1))
            if isinstance(pf, int) and isinstance(cf, int):
                if pf != cf:
                    m.add(en == 0)
                continue
            m.add(pf == cf).only_enforce_if(en)
            if isinstance(pst, int) and isinstance(cst, int) and pst == cst:
                continue
            f = pf if not isinstance(pf, int) else cf
            live = m.new_bool_var("")
            m.add(f > 1).only_enforce_if(live)
            m.add(f == 1).only_enforce_if(~live)
            m.add(pst == cst).only_enforce_if([en, live])

    def dims(self, v, rule, which="split"):
        """device dim -> (var, stride expr) under the side's rule."""
        sd, order, contiguous = rule["sym_dim"], rule["order"], rule["contiguous"]
        seq = [s for s in order if s in v[which]]
        if contiguous is not None and contiguous in seq:
            # contiguous moves first only when split > 1: a reified choice; the
            # probe keeps it simple by requiring the rule's choice to be static
            seq.remove(contiguous)
            seq.insert(0, contiguous)
        out, stride = {}, 1
        for i, s in enumerate(seq):
            d = sd.get(s)
            if d is not None:
                out[d] = (v[which][s], stride)
            if i < len(seq) - 1:
                stride = self.product_expr(stride, v[which][s])
        return out

    def product_expr(self, a, b):
        if isinstance(a, int) and a == 1:
            return b
        p = self.m.new_int_var(1, 1024, "")
        self.m.add_multiplication_equality(p, [a, b])
        return p

    def post_objective(self):
        nodes = self.node.values()
        weight = 1 + sum(v["cores"] if isinstance(v["cores"], int) else 64 for v in nodes)
        spill = sum(v["spill"] * (1 - v["in"]) for v in nodes)
        cores = sum(v["cores"] for v in nodes)
        self.m.minimize(weight * spill - cores)

    def build(self):
        for b in self.bufs:
            self.post_node(b)
        self.post_no_overlap()
        stats = self.post_edges()
        self.post_objective()
        if self.arm == "L1d":  # branch on residency first, resident before spilled
            lits = [v["in"] for v in self.node.values()]
            self.m.add_decision_strategy(
                lits, cp_model.CHOOSE_FIRST, cp_model.SELECT_MAX_VALUE
            )
        return stats


def _sym(cd, name):
    for s in cd.splits:
        if str(s) == name:
            return s
    return name


def solve(model, workers, limit, arm=""):
    s = cp_model.CpSolver()
    s.parameters.num_workers = workers
    if arm == "L1l":
        s.parameters.linearization_level = 2
    if os.environ.get("INTERLEAVE") == "1":  # deterministic multi-worker search
        s.parameters.interleave_search = True
    s.parameters.random_seed = 0
    s.parameters.max_time_in_seconds = limit
    t0 = time.perf_counter()
    st = s.solve(model)
    wall = time.perf_counter() - t0
    ok = st in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    return dict(
        status=s.status_name(st),
        obj=s.objective_value if ok else None,
        wall=wall,
        det_time=s.response_proto.deterministic_time,
        sol_hash=hash(tuple(s.response_proto.solution)) if ok else None,
    )


def main():
    caps, out = sys.argv[1], sys.argv[2]
    workers = [int(w) for w in (sys.argv[3] if len(sys.argv) > 3 else "1,16").split(",")]
    limit = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            done.add((r["capture"], r.get("arm"), r.get("workers")))
    with open(out, "a") as f:
        for path in sorted(glob.glob(os.path.join(caps, "*.pkl"))):
            cap = pickle.load(open(path, "rb"))
            name = os.path.basename(path)[:-4]
            final = {}
            for b in cap["buffers"]:
                for parent, pairs in (getattr(b, "cd_parent_matches", None) or {}).items():
                    final[(parent, b.name)] = pairs
            menus = {b.name: len(getattr(b, "core_divisions", None) or []) for b in cap["buffers"]}
            info = {}
            for e in cap.get("edge_raw", []):
                key = (e["parent"], e["consumer"])
                if key in final:
                    info[key] = analyse_edge(e, final[key], (menus.get(key[0]), menus.get(key[1])))
            if (name, "edges", None) not in done:
                f.write(json.dumps(dict(
                    capture=name, arm="edges", workers=None,
                    edges=[{k: v for k, v in i.items() if not k.endswith("_rule")} for i in info.values()],
                )) + "\n")
                f.flush()
            for arm in os.environ.get("ARMS", "E,L1,L2").split(","):
                for w in workers:
                    if (name, arm, w) in done:
                        continue
                    t0 = time.perf_counter()
                    try:
                        bld = Builder(cap, arm, info)
                        stats = bld.build()
                        build_s = time.perf_counter() - t0
                        proto = bld.m.Proto()
                        size = dict(vars=len(proto.variables), cons=len(proto.constraints))
                        r = solve(bld.m, w, limit, arm)
                    except Exception as e:  # record and move on
                        r, stats, build_s, size = dict(status=f"ERROR {type(e).__name__}: {e}"), {}, None, {}
                    f.write(json.dumps(dict(
                        capture=name, arm=arm, workers=w, build_s=build_s,
                        **size, **stats, **r,
                    )) + "\n")
                    f.flush()
                    print(name, arm, w, r.get("status"), r.get("obj"), f"{r.get('wall', 0):.2f}s", flush=True)


if __name__ == "__main__":
    main()

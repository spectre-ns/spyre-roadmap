"""LX layout (residency + offsets + 2-D no-overlap) from a Phase 2.0 capture,
written once against the shared interface and solved by CP-SAT and by SA.

Usage: lx_layout.py <out.jsonl> <caps glob> [sa_seconds=10] [cpsat_seconds=60]

Per buffer: a presence literal (pinned off by residency_reason or when it cannot
fit), an LX offset, a fixed lifetime interval and an optional address interval;
all address intervals under one 2-D no-overlap with the lifetimes. Footprint is
the smallest per-core footprint on the buffer's menu (division is not chosen
here). Objective: maximize the spill cost kept resident. SA's answer is checked
by an independent verifier (pairwise overlap and capacity from the values).
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import sys

import torch  # noqa: F401  (unpickling needs torch_spyre classes)
import torch_spyre  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cpsat_backend import CpSatProtocol  # noqa: E402
from sa_solver import SaSolver  # noqa: E402


def ceil_div(a, b):
    return -(-a // b)


def layout_data(cap):
    align = cap["alignment"]
    limit = cap["limit"] // align
    out = []
    for b in cap["buffers"]:
        if not b.uses:
            continue
        units = ceil_div(b.size, align)
        menu = getattr(b, "core_divisions", None) or []
        size = min(
            (ceil_div(units, cd.output_partition * cd.tiling.output_tile_count) for cd in menu),
            default=units,
        )
        st = b.lifetime_start_override if b.lifetime_start_override is not None else b.uses[0]
        en = b.lifetime_end_override if b.lifetime_end_override is not None else b.uses[-1]
        first_read = b.first_use_is_read
        reads = len(b.uses) - (0 if first_read else 1) - (1 if first_read else 0)
        inter = str(getattr(b, "boundary", "")).endswith("Intermediate")
        spill = max(0, reads + (1 if inter else 0)) * units
        allowed = b.residency_reason is None and 0 < size <= limit
        out.append(dict(name=b.name, size=max(size, 1), start=st, end=en, spill=spill, allowed=allowed))
    return limit, out


def build(s, limit, bufs):
    pres, offs, xs, ys = {}, {}, [], []
    for b in bufs:
        p = s.IntVar(0, 1 if b["allowed"] else 0, f"p_{b['name']}")
        o = s.IntVar(0, max(0, limit - b["size"]), f"o_{b['name']}")
        pres[b["name"]], offs[b["name"]] = p, o
        xs.append(s.NewOptionalIntervalVar(b["start"], b["end"] - b["start"] + 1, p, f"t_{b['name']}"))
        ys.append(s.NewOptionalIntervalVar(o, b["size"], p, f"a_{b['name']}"))
    s.AddNoOverlap2D(xs, ys)
    s.Maximize(s.Sum(b["spill"] * pres[b["name"]] for b in bufs))
    return pres, offs


def verify(limit, bufs, p, o):
    placed = [b for b in bufs if p[b["name"]] >= 1]
    for b in placed:
        if not b["allowed"] or o[b["name"]] < 0 or o[b["name"]] + b["size"] > limit:
            return False
    for i, a in enumerate(placed):
        for c in placed[i + 1 :]:
            if min(a["end"], c["end"]) < max(a["start"], c["start"]):
                continue
            if min(o[a["name"]] + a["size"], o[c["name"]] + c["size"]) > max(o[a["name"]], o[c["name"]]):
                return False
    return True


def main():
    out, pattern = sys.argv[1], sys.argv[2]
    sa_s = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    cp_s = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0
    with open(out, "a") as f:
        for path in sorted(glob.glob(pattern)):
            name = os.path.basename(path)[:-4]
            limit, bufs = layout_data(pickle.load(open(path, "rb")))
            if not any(b["allowed"] for b in bufs):
                continue
            arms = [("CPSAT", CpSatProtocol(workers=16), cp_s)]
            arms += [(f"SA.{k}", SaSolver(seed=k), sa_s) for k in (0, 1, 2)]
            for arm, s, limit_s in arms:
                pres, offs = build(s, limit, bufs)
                s.SetTimeLimit(int(limit_s * 1000))
                st = s.Solve()
                ok = st in (s.FEASIBLE, getattr(s, "OPTIMAL", -1))
                rec = dict(
                    capture=name, arm=arm, buffers=len(bufs),
                    status={0: "OPTIMAL", 1: "FEASIBLE"}.get(st, "NOT_SOLVED"),
                    obj=s.Objective().Value() if ok else None,
                    wall=s.WallTime() / 1000,
                )
                if ok:
                    p = {k: round(s.Value(v)) for k, v in pres.items()}
                    o = {k: round(s.Value(v)) for k, v in offs.items()}
                    rec["verified"] = verify(limit, bufs, p, o)
                if arm.startswith("SA"):
                    rec["time_to_best"] = s.stats.get("time_to_best")
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(name, arm, rec["status"], rec["obj"], rec.get("verified"), f"{rec['wall']:.2f}s", flush=True)


if __name__ == "__main__":
    main()

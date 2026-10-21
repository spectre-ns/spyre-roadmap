"""Summarize lazy_probe.py rows: edge coverage and E vs L1 vs L2 solve cost.

Usage: report_probe.py <probe.jsonl> [<probe.jsonl> ...]
"""

import json
import math
import sys
from collections import Counter, defaultdict


def gmean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def main():
    rows = [json.loads(l) for p in sys.argv[1:] for l in open(p)]
    edges = [e for r in rows if r["arm"] == "edges" for e in r["edges"]]
    why = Counter()
    for e in edges:
        if e["lazy_exact"]:
            why["lazy exact"] += 1
        elif e["allocator_modified"]:
            why["allocator-modified table (carry/clone)"] += 1
        elif e.get("menu_mismatch"):
            why["menu mismatch"] += 1
        elif not e["trivial_tiles"]:
            why["tiled, no exact per-axis rule"] += 1
        elif e["irregular"]:
            why["irregular owner formula"] += 1
        else:
            why["no per-axis rule / rule inexact"] += 1
    print(f"READ edges with a pair table: {len(edges)}")
    for k, v in why.most_common():
        print(f"  {k:40s} {v:6d}  {100 * v / max(1, len(edges)):5.1f}%")
    pairs = sum(e["pairs"] for e in edges)
    cands = sum(e["n_parent"] * e["n_consumer"] for e in edges)
    print(f"  pair-table entries {pairs}, candidate pairs compared {cands}")

    by = defaultdict(dict)  # (capture, workers) -> arm -> row
    for r in rows:
        if r["arm"] != "edges":
            by[(r["capture"], r["workers"])][r["arm"]] = r
    for w in sorted({k[1] for k in by}):
        keys = [k for k in by if k[1] == w and {"E", "L1", "L2"} <= set(by[k])]
        print(f"\nworkers={w}: {len(keys)} solves")
        for arm in ("E", "L1", "L2"):
            st = Counter(by[k][arm]["status"] for k in keys)
            wall = sum(by[k][arm].get("wall", 0) for k in keys)
            build = sum(by[k][arm].get("build_s") or 0 for k in keys)
            size = sum(by[k][arm].get("cons", 0) for k in keys)
            ratio = gmean(
                [by[k][arm]["wall"] / by[k]["E"]["wall"] for k in keys
                 if by[k][arm].get("wall") and by[k]["E"].get("wall")]
            )
            same = sum(1 for k in keys if by[k][arm].get("obj") == by[k]["E"].get("obj"))
            better = sum(
                1 for k in keys
                if by[k][arm].get("obj") is not None and by[k]["E"].get("obj") is not None
                and by[k][arm]["obj"] < by[k]["E"]["obj"]
            )
            print(f"  {arm:3s} solve {wall:8.2f}s  build {build:6.2f}s  constraints {size:8d}  "
                  f"gmean vs E {ratio:5.2f}x  obj==E {same}/{len(keys)}  obj<E {better}  "
                  f"{dict(st)}")
        worst = sorted(keys, key=lambda k: by[k]["L1"].get("wall", 0) / max(1e-3, by[k]["E"].get("wall", 1e-3)), reverse=True)[:5]
        for k in worst:
            print("   slowest L1/E:", k[0], {a: (by[k][a].get("status"), round(by[k][a].get("wall", 0), 2)) for a in ("E", "L1", "L2")})


if __name__ == "__main__":
    main()

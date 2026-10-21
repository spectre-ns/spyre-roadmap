"""Summarize timers.py records: where LX planning time goes, per graph.

Usage: report_profile.py <profile.jsonl> [min_wall_s]
"""

import json
import sys
from collections import defaultdict

FAMILY = {
    "menu": "menu enumeration",
    "pairs": "pair/relayout tables",
    "cost": "cost-expr build",
    "cpsat:build_menu_tables": "model build: menu tables",
    "cpsat:build_gates": "model build: gates",
    "cpsat:build_other": "model build: other",
    "cpsat:cost_lowering": "cost lowering",
    "cpsat:solve": "CP-SAT solve",
}


def fam(label):
    if label in FAMILY:
        return FAMILY[label]
    head = label.split(":")[0]
    if head in FAMILY:
        return FAMILY[head]
    if head == "cpsat":
        return "solver other"
    if label == "harness":
        return "harness"
    return "planning other"


def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    floor = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    order = [
        "menu enumeration", "pair/relayout tables", "cost-expr build",
        "model build: menu tables", "model build: gates", "model build: other",
        "cost lowering", "CP-SAT solve", "solver other", "planning other",
    ]
    tot = defaultdict(float)
    print("graph | ops | bufs | menu max/sum | edges | pairs | relay cands | wall s | "
          + " | ".join(order) + " | enum% | status")
    n = 0
    for r in rows:
        by = defaultdict(float)
        for label, s in r["excl"].items():
            by[fam(label)] += s
        wall = r["wall_s"] - by.get("harness", 0.0)
        if wall < floor:
            continue
        n += 1
        c = r["census"][-1] if r["census"] else {}
        menus = c.get("menus", [0])
        edges = c.get("edges", [])
        enum = by["menu enumeration"] + by["pair/relayout tables"]
        for k in order:
            tot[k] += by[k]
        tot["wall"] += wall
        tot["enum"] += enum
        st = ",".join(s.get("status", "?") for s in r["solves"])
        print(
            f"{r['tag']}.{r['plan']} | {r['ops']} | {c.get('buffers', 0)} | "
            f"{max(menus)}/{sum(menus)} | {len(edges)} | {sum(e[2] for e in edges)} | "
            f"{sum(x[0] for x in c.get('relayout_edges', []))} | {wall:.2f} | "
            + " | ".join(f"{by[k]:.2f}" for k in order)
            + f" | {100 * enum / wall:.0f}% | {st}"
        )
    w = tot["wall"] or 1
    print(f"\nTOTAL over {n} plans: wall {tot['wall']:.1f}s")
    for k in order:
        print(f"  {k:28s} {tot[k]:8.2f}s  {100 * tot[k] / w:5.1f}%")
    print(f"  {'enumeration (menu + pairs)':28s} {tot['enum']:8.2f}s  {100 * tot['enum'] / w:5.1f}%")

    # Exactness of the per-dim key join against each pair table.
    ek = [e for r in rows for e in r.get("edge_keys", [])]
    good = [e for e in ek if "error" not in e]
    if ek:
        exact = sum(e["exact"] for e in good)
        irr = sum(1 for e in good if any(e["irregular"]))
        print(f"\nmatch_pairs calls: {len(ek)} (census errors {len(ek) - len(good)}); "
              f"key join == pair table on {exact}/{len(good)}; "
              f"calls with an irregular owner formula: {irr}; "
              f"max split dims on one view: {max((e['max_split_dims'] for e in good), default=0)}")
        for e in [e for e in good if not e["exact"]][:5]:
            print("  inexact:", e)
    helpers = defaultdict(float)
    calls = defaultdict(int)
    for r in rows:
        for k, v in r["helper_s"].items():
            if not k.endswith("@harness"):
                helpers[k.split("@")[0]] += v
        for k, v in r["calls"].items():
            calls[k] += v
    print("\nhot helpers (seconds outside harness / total calls incl. harness):")
    for k in sorted(helpers, key=helpers.get, reverse=True):
        print(f"  {k:36s} {helpers[k]:8.2f}s  {calls[k]} calls")


if __name__ == "__main__":
    main()

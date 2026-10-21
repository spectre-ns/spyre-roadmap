"""Test a closed form for relayout legality against recorded decisions.

Usage: ms_check.py <ms.jsonl> [...]

Each record (from zoo_ms.py) holds both views' digit keys
``(cores, ((dim, factor, stride), ...))`` (factor > 1 only), the two core
counts, the split products and ``movement_supported``'s answer. The closed form
below decides the same question from the digits alone:

1. domain: C_dst >= C_src, C_src | C_dst, each key's core count equals its
   domain, the source split product equals C_src; equal domains with identical
   keys are not a relayout;
2. source slices distinct: the view's core -> slot map is a bijection onto its
   slices;
3. destination covered: the view's map reaches every slice (each the same
   number of times when the domains are equal); the destination split product
   divides C_dst.

2 and 3 are properties of one view, not of the pair: evaluated over at most 32
cores and cached per view, they are a small per-node table. For views
``core_to_slice_mapping`` produces (prefix-product strides, replication gaps)
they reduce to "exact mixed radix" and "nested strides with gaps", but coprime
or overlapping strides can also be bijective, so the general form is the map
itself;
4. per device dim, one factor divides the other (uniform overlaps); then
   fan-in = prod max(1, f_src / f_dst), fan-out = prod max(1, f_dst / f_src) *
   (C_dst / split_dst); the larger-domain rule is fan-in 1 and, for each source
   slice, the destination slices nested in it replicated C_dst / C_src times in
   total (a function of the destination view and the source's factors);
5. fan-in <= 8.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter

MAX_FANIN = 8


def digits(key):
    return sorted((tuple(d) for d in key[1]), key=lambda d: d[2])


_SLICES: dict = {}


def slices(key):
    """Each core's slot tuple under the view's digits (cached per view)."""
    k = (key[0], tuple(tuple(d) for d in key[1]))
    if k not in _SLICES:
        c, ds = k
        _SLICES[k] = [tuple((core // s) % f for _, f, s in ds) for core in range(c)]
    return _SLICES[k]


def mixed_radix_exact(ds):
    stride = 1
    for _, f, s in ds:
        if s != stride:
            return False
        stride *= f
    return True


def nested_with_gaps(ds, cores):
    top = 1
    for _, f, s in ds:
        if s % top:
            return False
        top = s * f
    return cores % top == 0


def closed_form(r):
    sk, dk = r["src_key"], r["dst_key"]
    cs, cd = r["src_cores"], r["dst_cores"]
    if cs <= 0 or cd < cs or cd % cs or sk[0] != cs or dk[0] != cd:
        return False, "domain"
    if cs == cd and digits(sk) == digits(dk):
        return False, "same view"
    spd, sps = r["dst_split_product"], r["src_split_product"]
    if spd <= 0 or cd % spd or sps != cs:
        return False, "split products"
    sd, dd = digits(sk), digits(dk)
    if math.prod(f for _, f, _ in sd) != cs or len(set(slices(sk))) != cs:
        return False, "source slices"
    reps = Counter(slices(dk))
    if math.prod(f for _, f, _ in dd) != spd or len(reps) != spd:
        return False, "destination coverage"
    if cs == cd and len(set(reps.values())) != 1:  # equal domains: even replication
        return False, "destination coverage"
    fs = {d: f for d, f, _ in sd}
    fd = {d: f for d, f, _ in dd}
    fanin = 1
    for dim in fs.keys() | fd.keys():
        a, b = fs.get(dim, 1), fd.get(dim, 1)
        if a % b and b % a:
            return False, "non-uniform overlap"
        fanin *= max(1, a // b)
    if fanin > MAX_FANIN:
        return False, "fan-in"
    if cd > cs:
        if fanin != 1:
            return False, "broadcast not one-to-one"
        # each source slice must reach exactly C_dst / C_src destination cores: sum
        # the replication of the destination slices nested in it (depends on the
        # destination view and the source's factors only, not its strides)
        dims = [d for d, _, _ in dk[1]]  # slices() tuples follow the key's order
        per_src = Counter()
        for tup, n in reps.items():
            parent = tuple(
                (dim, t // (fd[dim] // fs.get(dim, 1))) for dim, t in zip(dims, tup)
            )
            per_src[parent] += n
        if len(per_src) * math.prod(
            fs.get(d, 1) for d in fs if d not in fd
        ) != cs or any(v != cd // cs for v in per_src.values()):
            return False, "broadcast not one-to-one"
    return True, "ok"


def main():
    recs = [json.loads(line) for p in sys.argv[1:] for line in open(p)]
    recs = [r for r in recs if "error" not in r]
    regular = [
        r for r in recs
        if r["src_key"] and r["dst_key"]
        and r["src_key"][0] != "irregular" and r["dst_key"][0] != "irregular"
    ]
    conf = Counter()
    reasons = Counter()
    misses = []
    for r in regular:
        pred, why = closed_form(r)
        conf[(r["result"], pred)] += 1
        if not pred:
            reasons[why] += 1
        if pred != r["result"]:
            misses.append((r, why))
    print(f"records {len(recs)}, regular digit keys {len(regular)}, "
          f"irregular / no view {len(recs) - len(regular)}")
    print(f"agree {conf[(True, True)] + conf[(False, False)]} / {len(regular)}  "
          f"(true/true {conf[(True, True)]}, false/false {conf[(False, False)]}, "
          f"actual true but predicted false {conf[(True, False)]}, "
          f"actual false but predicted true {conf[(False, True)]})")
    print("closed-form rejections by reason:", dict(reasons))
    for r, why in misses[:10]:
        print("  MISS", r["model"], "actual", r["result"], "predicted", not r["result"], why,
              r["src_key"], r["dst_key"], r["src_cores"], r["dst_cores"],
              r["src_split_product"], r["dst_split_product"])


if __name__ == "__main__":
    main()

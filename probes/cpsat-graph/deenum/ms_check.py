"""Test a closed form for relayout legality against recorded decisions.

Usage: ms_check.py <ms.jsonl> [...]

Each record (from zoo_ms.py) holds both views' digit keys
``(cores, ((dim, factor, stride), ...))`` (factor > 1 only), the two core
counts, the split products and ``movement_supported``'s answer. The closed form
below decides the same question from the digits alone:

1. domain: C_dst >= C_src, C_src | C_dst, each key's core count equals its
   domain, the source split product equals C_src; equal domains with identical
   keys are not a relayout;
2. source slices distinct: sorted by stride, source digits are an exact mixed
   radix (stride_k = product of the factors before it);
3. destination covered evenly: sorted by stride, each stride is a multiple of
   the previous digit's range (gaps are replication) and C_dst is a multiple of
   the top digit's range; the destination split product divides C_dst;
4. per device dim, one factor divides the other (uniform overlaps); then
   fan-in = prod max(1, f_src / f_dst) and the larger-domain rule is fan-in 1;
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
    if math.prod(f for _, f, _ in sd) != cs or not mixed_radix_exact(sd):
        return False, "source slices"
    if math.prod(f for _, f, _ in dd) != spd or not nested_with_gaps(dd, cd):
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
    if cd > cs and fanin != 1:
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

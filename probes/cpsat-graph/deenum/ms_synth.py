"""Check the relayout-legality closed form (ms_check.py) against the real
movement_supported on synthetic digit-form views, including the invalid
geometries real compiles never produce.

Usage: ms_synth.py <out.json> [random_pairs=150000] [per_reason=3000] [seed=0]

A view is (core count C, digits) over 2 device dims; each split digit
(dim, f, s) owns slot Mod(floor(core_id / s), f). Factors come from
{2, 3, 4, 6, 8} (3 and 6 make non-uniform overlaps possible), C from
{1, 2, 3, 4, 6, 8, 12, 16, 24, 32}, strides from the divisors of C, so the
space includes overlapping digits, replication gaps and wrap-around. Pairs are
drawn uniformly, then topped up per closed-form outcome so every rejection
reason is exercised; agreement is reported per reason.
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter, defaultdict

import sympy
import torch  # noqa: F401  (torch before torch_spyre)
import torch_spyre  # noqa: F401
from torch_spyre._inductor.pass_utils import PerCoreView
from torch_spyre._inductor.scratchpad.lx_relayout import movement_supported

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ms_check import closed_form  # noqa: E402

CORES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
FACTORS = (2, 3, 4, 6, 8)
core_id = sympy.Symbol("core_id")


def slot(f, s):
    return sympy.Mod(core_id, f) if s == 1 else sympy.Mod(sympy.floor(core_id / s), f)


def all_views():
    views = []
    for c in CORES:
        divs = [d for d in range(1, c + 1) if c % d == 0]
        for f0 in (1, *FACTORS):
            for f1 in (1, *FACTORS):
                if f0 * f1 > c or c % (f0 * f1):
                    continue
                split = [(d, f) for d, f in ((0, f0), (1, f1)) if f > 1]
                for strides in _stride_choices(len(split), divs):
                    digits = tuple((d, f, s) for (d, f), s in zip(split, strides))
                    views.append((c, digits))
    return views


def _stride_choices(k, divs):
    if k == 0:
        return [()]
    if k == 1:
        return [(s,) for s in divs]
    return [(a, b) for a in divs for b in divs]


def make(c, digits):
    return PerCoreView(
        tuple((d, f) for d, f, _ in digits),
        tuple((d, slot(f, s)) for d, f, s in digits),
        num_cores=c,
    )


def record(src, dst):
    (cs, sd), (cd, dd) = src, dst
    return dict(
        src_key=(cs, sd),
        dst_key=(cd, dd),
        src_cores=cs,
        dst_cores=cd,
        src_split_product=_prod(sd),
        dst_split_product=_prod(dd),
    )


def _prod(digits):
    p = 1
    for _, f, _ in digits:
        p *= f
    return p


def main():
    out = sys.argv[1]
    n_random = int(sys.argv[2]) if len(sys.argv) > 2 else 150000
    per_reason = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    rng = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    views = all_views()
    built = {}

    def view(v):
        if v not in built:
            built[v] = make(*v)
        return built[v]

    tally = defaultdict(Counter)  # reason -> Counter(agree / disagree)
    misses = []
    seen = set()

    def check(src, dst):
        key = (src, dst)
        if key in seen:
            return None
        seen.add(key)
        rec = record(src, dst)
        pred, why = closed_form(rec)
        try:
            actual = bool(movement_supported(view(src), view(dst), src[0], dst[0]))
        except ValueError:
            tally[why]["raised"] += 1  # an invalid view the real code refuses outright
            return why
        tally[why]["agree" if pred == actual else "disagree"] += 1
        if pred != actual and len(misses) < 50:
            misses.append(dict(rec, actual=actual, predicted=pred, reason=why))
        return why

    for _ in range(n_random):
        check(rng.choice(views), rng.choice(views))
    # top up rare outcomes: keep drawing until each reason has per_reason pairs
    reasons = ("ok", "domain", "same view", "split products", "source slices",
               "destination coverage", "non-uniform overlap", "fan-in",
               "broadcast not one-to-one")
    budget = 40 * per_reason * len(reasons)
    while budget > 0 and any(sum(tally[r].values()) < per_reason for r in reasons):
        budget -= 1
        src, dst = rng.choice(views), rng.choice(views)
        rec = record(src, dst)
        _, why = closed_form(rec)
        if sum(tally[why].values()) < per_reason:
            check(src, dst)
    result = dict(
        views=len(views),
        pairs=len(seen),
        by_reason={k: dict(v) for k, v in tally.items()},
        misses=misses,
    )
    with open(out, "w") as f:
        json.dump(result, f, indent=1, default=str)
    total = Counter()
    for v in tally.values():
        total.update(v)
    print(f"views {len(views)}, pairs checked {len(seen)}, agree {total['agree']}, "
          f"disagree {total['disagree']}")
    for k in reasons:
        print(f"  {k:28s} {dict(tally.get(k, {}))}")


if __name__ == "__main__":
    main()

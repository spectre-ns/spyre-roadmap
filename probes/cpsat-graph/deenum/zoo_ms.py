"""Record every movement_supported decision while compiling one zoo model.

Usage: zoo_ms.py <model name> <out.jsonl>

One JSON line per distinct call: both views' digit keys (timers._view_key), raw
work_slice_dims, split products, core counts, and the result. Used to test a
closed form for relayout legality over view digits (plan Phase 2d).
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch
import torch_spyre  # noqa: F401  (after torch)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import timers  # noqa: E402  (_view_key)
import zoo  # noqa: E402


class Done(Exception):
    pass


def main():
    name, out = sys.argv[1], sys.argv[2]
    torch._inductor.config.fx_graph_cache = False
    torch._inductor.config.force_disable_caches = True

    import torch_spyre._inductor.passes as passes
    import torch_spyre._inductor.scratchpad.lx_relayout as lx

    orig = lx.movement_supported
    seen: dict = {}

    def recorded(source, destination, source_num_cores, destination_num_cores):
        result = orig(source, destination, source_num_cores, destination_num_cores)
        try:
            sk, dk = timers._view_key(source), timers._view_key(destination)
            key = (repr(sk), repr(dk), source_num_cores, destination_num_cores)
            if key not in seen:
                seen[key] = dict(
                    src_key=sk,
                    dst_key=dk,
                    src_dims=[list(map(int, d)) for d in source.work_slice_dims],
                    dst_dims=[list(map(int, d)) for d in destination.work_slice_dims],
                    src_split_product=int(source.split_product),
                    dst_split_product=int(destination.split_product),
                    src_cores=int(source_num_cores),
                    dst_cores=int(destination_num_cores),
                    result=bool(result),
                )
        except Exception as e:  # never break the compile over the census
            seen.setdefault(("error", repr(e)), dict(error=repr(e)))
        return result

    lx.movement_supported = recorded
    inner = passes.scratchpad_planning

    def planning(*a, **k):
        inner(*a, **k)
        raise Done(name)

    passes.scratchpad_planning = planning
    torch.manual_seed(0)
    fn, args = zoo.MODELS[name]()
    dargs = [a.to(torch.device("spyre")) for a in args]
    t0 = time.time()
    status = "no abort"
    try:
        torch.compile(fn)(*dargs)
    except Exception as e:
        chain, cur = [], e
        while cur is not None:
            chain.append(cur)
            cur = cur.__cause__ or cur.__context__
        status = "ok" if any(isinstance(c, Done) for c in chain) else f"error {e!r}"[:300]
    with open(out, "a") as f:
        for rec in seen.values():
            f.write(json.dumps(dict(model=name, **rec), default=str) + "\n")
    print(f"{name}\t{status}\t{len(seen)} distinct calls\t{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

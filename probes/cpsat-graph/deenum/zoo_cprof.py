"""cProfile one allocator method while compiling a zoo model on Spyre.

Usage: zoo_cprof.py <model name> <method name> <out.txt>
"""

from __future__ import annotations

import cProfile
import io
import os
import pstats
import sys
import time

import torch
import torch_spyre  # noqa: F401  (after torch)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import zoo  # noqa: E402


class Done(Exception):
    pass


def main():
    name, method, out = sys.argv[1], sys.argv[2], sys.argv[3]
    torch._inductor.config.fx_graph_cache = False
    torch._inductor.config.force_disable_caches = True

    import torch_spyre._inductor.passes as passes
    import torch_spyre._inductor.scratchpad.allocator as alloc

    prof = cProfile.Profile()
    A = alloc.CoOptimizingAllocator
    orig = getattr(A, method)
    wall = [0.0, 0]

    def wrapped(*a, **k):
        t0 = time.perf_counter()
        prof.enable()
        try:
            return orig(*a, **k)
        finally:
            prof.disable()
            wall[0] += time.perf_counter() - t0
            wall[1] += 1

    setattr(A, method, wrapped)
    inner = passes.scratchpad_planning

    def planning(*a, **k):
        inner(*a, **k)
        raise Done(name)

    passes.scratchpad_planning = planning
    torch.manual_seed(0)
    fn, args = zoo.MODELS[name]()
    dargs = [a.to(torch.device("spyre")) for a in args]
    status = "no abort"
    try:
        torch.compile(fn)(*dargs)
    except Exception as e:
        chain, cur = [], e
        while cur is not None:
            chain.append(cur)
            cur = cur.__cause__ or cur.__context__
        status = "ok" if any(isinstance(c, Done) for c in chain) else f"error {e!r}"[:300]
    buf = io.StringIO()
    buf.write(f"{name} {method}: {wall[1]} calls, {wall[0]:.2f}s under cProfile, {status}\n")
    st = pstats.Stats(prof, stream=buf)
    st.sort_stats("tottime").print_stats(22)
    st.sort_stats("cumulative").print_stats(30)
    with open(out, "w") as f:
        f.write(buf.getvalue())
    print(f"{name}\t{status}\t{wall[0]:.2f}s", flush=True)


if __name__ == "__main__":
    main()

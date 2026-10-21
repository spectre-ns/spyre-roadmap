"""Compile one zoo model on Spyre under the Stage-0 timers.

Usage: zoo_t.py <model name> <out.jsonl> [caps_dir]
"""

from __future__ import annotations

import faulthandler
import os
import signal
import sys
import time
import traceback

import torch
import torch_spyre  # noqa: F401  (after torch)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import timers  # noqa: E402
import zoo  # noqa: E402


class Done(Exception):
    pass


def main():
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    name, out = sys.argv[1], sys.argv[2]
    caps = sys.argv[3] if len(sys.argv) > 3 else None
    torch._inductor.config.fx_graph_cache = False
    torch._inductor.config.force_disable_caches = True
    timers.install(out, name, caps)

    import torch_spyre._inductor.passes as passes

    inner = passes.scratchpad_planning

    def planning(*a, **k):
        inner(*a, **k)
        raise Done(name)

    passes.scratchpad_planning = planning

    torch.manual_seed(0)
    fn, args = zoo.MODELS[name]()
    dargs = [a.to(torch.device("spyre")) for a in args]
    t0 = time.time()
    status = "completed (no abort?)"
    try:
        torch.compile(fn)(*dargs)
    except Exception as e:
        chain, cur = [], e
        while cur is not None:
            chain.append(cur)
            cur = cur.__cause__ or cur.__context__
        if any(isinstance(c, Done) for c in chain) or "planning" in timers.T.fired:
            status = "ok"
        else:
            status = "error: " + "".join(
                traceback.format_exception_only(type(e), e)
            ).strip()[-300:]
    missing = sorted(
        {"planning", "prepare_buffers", "menu:division_map", "cpsat:entry"}
        - timers.T.fired
    )
    print(
        f"{name}\t{status}\t{time.time() - t0:.1f}s\tmissing={missing}\t"
        f"{torch_spyre.__file__}",
        flush=True,
    )


if __name__ == "__main__":
    main()

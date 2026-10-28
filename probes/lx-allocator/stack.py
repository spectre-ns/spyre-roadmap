"""Placement under LX pressure: native 2-D no-overlap against packing.

Builds a decoder stack from the sketch's own classes (14 ops per layer: q_proj,
scores, max, sub_exp, sum, probs_v, div, o_proj, resid1, gate, up, silu_mul,
down, resid2; k/v cache and five weights per layer as InputNodes), shrinks LX
capacity, and plans it twice through SpyreOptimizer: once with CpSatBackend
placing natively and once packing (a cumulative under a roof, then the Packer,
lowering the roof on overflow). Each plan is then judged by check_decisions.

    python3 stack.py LAYERS TIME_LIMIT_S WORKERS SCALES [TOKENS]
    python3 stack.py 8 60 8 1,0.25,0.1
"""

import math
import sys
import time

from lx_design import (Axis, InputNode, MatmulNode, PointwiseNode, ReductionNode, SolveSettings,
                       SpyreOptimizationGraph, SpyreOptimizer, check_decisions, live_boxes)


def build(layers, T=32, H=4096, I=12800, heads=32, hd=128, kv=100) -> SpyreOptimizationGraph:
    g = SpyreOptimizationGraph()
    ins, ops, edges = [], [], []

    def inp(name, dims):
        n = InputNode(name, axes=tuple(Axis(f"d{i}", e) for i, e in enumerate(dims)), reads=(),
                      writes={f"d{i}": i for i in range(len(dims))}, shape=dims, size=math.prod(dims) * 2)
        ins.append(n)
        return n

    def op(n):
        ops.append(n)
        return n

    def mm(name, M, K, N):  # y is an nn.Linear weight [out=N, in=K]
        return op(MatmulNode(name, axes=(Axis("M", M), Axis("K", K, reduction=True), Axis("N", N)),
                             reads=({"M": 0, "K": 1}, {"K": 1, "N": 0}), writes={"M": 0, "N": 1},
                             shape=(M, N), size=M * N * 2))

    def pw(name, axes, reads, shape):
        return op(PointwiseNode(name, axes=axes, reads=reads, writes={a.name: i for i, a in enumerate(axes)},
                                shape=shape, size=math.prod(shape) * 2))

    def red(name):
        return op(ReductionNode(name, axes=(Axis("h", heads), Axis("L", kv, reduction=True)),
                                reads=({"h": 0, "L": 1},), writes={"h": 0}, shape=(heads, 1), size=heads * 2))

    x, q = inp("x0", (T, H)), inp("q0", (heads, hd))
    for l in range(layers):
        k, v = inp(f"k{l}", (heads, kv, hd)), inp(f"v{l}", (heads, kv, hd))
        Wo, Wg, Wu, Wd = inp(f"Wo{l}", (H, hd)), inp(f"Wg{l}", (I, H)), inp(f"Wu{l}", (I, H)), inp(f"Wd{l}", (H, I))
        if l > 0:
            Wq = inp(f"Wq{l}", (hd, H))
            q = mm(f"q_proj{l}", T, H, hd)
            edges += [(x, q, 0), (Wq, q, 1)]
        sc = op(MatmulNode(f"scores{l}", axes=(Axis("h", heads), Axis("K", hd, reduction=True), Axis("N", kv)),
                           reads=({"h": 0, "K": 1}, {"h": 0, "N": 1, "K": 2}), writes={"h": 0, "N": 1},
                           shape=(heads, kv), size=heads * kv * 2))
        mx = red(f"max{l}")
        se = pw(f"sub_exp{l}", (Axis("h", heads), Axis("L", kv)), ({"h": 0, "L": 1}, {"h": 0}), (heads, kv))
        sm = red(f"sum{l}")
        pv = op(MatmulNode(f"probs_v{l}", axes=(Axis("h", heads), Axis("K", kv, reduction=True), Axis("N", hd)),
                           reads=({"h": 0, "K": 1}, {"h": 0, "K": 1, "N": 2}), writes={"h": 0, "N": 1},
                           shape=(heads, hd), size=heads * hd * 2))
        dv = pw(f"div{l}", (Axis("h", heads), Axis("N", hd)), ({"h": 0, "N": 1}, {"h": 0}), (heads, hd))
        o = mm(f"o_proj{l}", T, hd, H)
        r1 = pw(f"resid1_{l}", (Axis("M", T), Axis("N", H)), ({"M": 0, "N": 1}, {"M": 0, "N": 1}), (T, H))
        gt, up = mm(f"gate{l}", T, H, I), mm(f"up{l}", T, H, I)
        h = pw(f"silu_mul{l}", (Axis("M", T), Axis("N", I)), ({"M": 0, "N": 1}, {"M": 0, "N": 1}), (T, I))
        dn = mm(f"down{l}", T, I, H)
        r2 = pw(f"resid2_{l}", (Axis("M", T), Axis("N", H)), ({"M": 0, "N": 1}, {"M": 0, "N": 1}), (T, H))
        edges += [(q, sc, 0), (k, sc, 1), (sc, mx, 0), (sc, se, 0), (mx, se, 1), (se, sm, 0), (se, pv, 0),
                  (v, pv, 1), (pv, dv, 0), (sm, dv, 1), (dv, o, 0), (Wo, o, 1), (o, r1, 0), (x, r1, 1),
                  (r1, gt, 0), (Wg, gt, 1), (r1, up, 0), (Wu, up, 1), (gt, h, 0), (up, h, 1), (h, dn, 0),
                  (Wd, dn, 1), (dn, r2, 0), (r1, r2, 1)]
        x = r2
    for n in ins + ops:  # every input arrives before the first op, as in the sketch's scenarios
        g.add_node(n)
    for n in ins:
        g.pin(n, "stick", len(n.shape) - 1, "graph input")
    for s, d, i in edges:
        g.add_read(s, d, i)
    return g


class Probe(SpyreOptimizer):
    def commit(self, graph):
        pass


def peak(g):
    """Most LX units live at once over the plan's boxes (an in-place pair is one box)."""
    boxes = live_boxes(g)
    return max((sum(size for s, e, size, _ in boxes if s <= t < e) for t in range(len(g.nodes))), default=0)


def run(layers, scale, limit, workers, tokens):
    for placement in ("no_overlap", "pack"):
        g = build(layers, T=tokens)
        g.capacity = int(g.capacity * scale)
        opt = Probe(settings=SolveSettings(time_limit_s=limit, workers=workers), placement=placement)
        t0 = time.perf_counter()
        opt.optimize(g)
        wall = time.perf_counter() - t0
        s = opt.solution
        line = (f"L={layers} T={tokens} cap x{scale:<5} {placement:10}: {s.status:11} {wall:6.1f}s "
                f"objective {int(s.objective) if s.ok else '-':>11}")
        if s.ok:
            line += (f"  spilled {sum(1 for n in g.nodes if n.runs and not n.resident):3d}"
                     f"  peak {peak(g)}/{g.capacity // g.alignment}  check: {check_decisions(g) or 'ok'}")
        print(line, flush=True)


if __name__ == "__main__":
    layers, limit, workers = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 32
    for scale in (float(x) for x in sys.argv[4].split(",")):
        run(layers, scale, limit, workers, tokens)

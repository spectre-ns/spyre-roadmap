"""Small models compiled for Spyre with CP-SAT model capture.

Usage: zoo.py <model name|list> <out_dir>
Compiles one model with torch.compile on the spyre device; the capture hook
dumps every CpSatLayoutSolver model and aborts after LX planning.
"""

from __future__ import annotations

import faulthandler
import math
import os
import signal
import sys
import time
import traceback

import torch
import torch.nn.functional as F
import torch_spyre  # noqa: F401  (after torch)

F16 = torch.float16


def rnd(*shape):
    return torch.randn(*shape, dtype=F16)


def w(*shape):
    return rnd(*shape) * 0.02


def softmax_rows(S, D):
    return (lambda x: torch.softmax(x, dim=-1)), (rnd(S, D),)


def rms(x, g):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * g


def rmsnorm(T, D):
    return rms, (rnd(T, D), rnd(D))


def residual_rmsnorm(T, D):
    def f(x, r, g):
        h = x + r
        return rms(h, g), h

    return f, (rnd(T, D), rnd(T, D), rnd(D))


def mlp_swiglu(S, E, M):
    def f(x, g, u, d):
        return (x @ u * F.silu(x @ g)) @ d

    return f, (rnd(S, E), w(E, M), w(E, M), w(M, E))


def mlp_gelu(S, E, M):
    def f(x, w1, w2):
        return F.gelu(x @ w1) @ w2

    return f, (rnd(S, E), w(E, M), w(M, E))


def linear_bias_gelu(S, E, M):
    def f(x, w1, b1):
        return F.gelu(x @ w1 + b1)

    return f, (rnd(S, E), w(E, M), rnd(M))


def simple_attn(H, Q, L, D):
    def f(q, k, v, s):
        qk = q @ k.transpose(-1, -2).contiguous()
        return (qk * s).softmax(dim=-1) @ v

    return f, (rnd(H * Q, D), rnd(L, D), rnd(L, D), torch.full((L,), 1 / math.sqrt(D), dtype=F16))


def sdpa(B, H, S, D, causal=False):
    def f(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    return f, (rnd(B, H, S, D), rnd(B, H, S, D), rnd(B, H, S, D))


def sdpa_g(B, Hq, Hkv, Lq, Lk, D, causal=False):
    """General SDPA: grouped-query when Hkv < Hq, cross-length when Lq != Lk."""
    gqa = Hkv != Hq

    def f(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=gqa)

    return f, (rnd(B, Hq, Lq, D), rnd(B, Hkv, Lk, D), rnd(B, Hkv, Lk, D))


def attn_block(S, E, H):
    D = E // H

    def heads(t):  # SDPA on Spyre takes (B, H, S, D)
        return t.view(1, S, H, D).transpose(1, 2)

    def f(x, wq, wk, wv, wo):
        o = F.scaled_dot_product_attention(heads(x @ wq), heads(x @ wk), heads(x @ wv))
        return x + o.transpose(1, 2).contiguous().view(S, E) @ wo

    return f, (rnd(S, E), w(E, E), w(E, E), w(E, E), w(E, E))


def transformer_block(S, E, H, M):
    D = E // H

    def heads(t):  # SDPA on Spyre takes (B, H, S, D)
        return t.view(1, S, H, D).transpose(1, 2)

    def f(x, n1, wq, wk, wv, wo, n2, g, u, d):
        h = rms(x, n1)
        o = F.scaled_dot_product_attention(heads(h @ wq), heads(h @ wk), heads(h @ wv))
        o = o.transpose(1, 2).contiguous().view(S, E)
        x = x + o @ wo
        h = rms(x, n2)
        return x + (h @ u * F.silu(h @ g)) @ d

    return f, (rnd(S, E), rnd(E), w(E, E), w(E, E), w(E, E), w(E, E), rnd(E), w(E, M), w(E, M), w(M, E))


def softplus(T, D):
    def f(x, beta, threshold, one, minus):
        bx = beta * x
        return torch.where(bx >= threshold, x, torch.log(one + torch.exp(minus * abs(bx))) + F.relu(bx))

    full = lambda v: torch.full((D, T), v, dtype=F16)  # noqa: E731
    return f, (rnd(D, T), full(1.0), full(20.0), full(1.0), full(-1.0))


def mm_chain(M, K, N, P):
    return (lambda a, b, c: (a @ b) @ c), (rnd(M, K), w(K, N), w(N, P))


def pointwise_chain(S, D):
    def f(x, y, g, b):
        h = torch.exp(x * g + b) - y
        return torch.tanh(h) * y + x

    return f, (rnd(S, D), rnd(S, D), rnd(D), rnd(D))


def row_stats(S, D):
    def f(x):
        m = x.amax(-1, keepdim=True)
        e = torch.exp(x - m)
        return e / e.sum(-1, keepdim=True), x.sum(-1)

    return f, (rnd(S, D),)


def mlp_stack(S, E, M):
    def f(x, a1, b1, a2, b2):
        h = x + F.gelu(x @ a1) @ b1
        return h + F.gelu(h @ a2) @ b2

    return f, (rnd(S, E), w(E, M), w(M, E), w(E, M), w(M, E))


MODELS = {
    "softmax_256x1024": lambda: softmax_rows(256, 1024),
    "softmax_1024x2048": lambda: softmax_rows(1024, 2048),
    "rmsnorm_128x256": lambda: rmsnorm(128, 256),
    "rmsnorm_512x2048": lambda: rmsnorm(512, 2048),
    "residual_rmsnorm_256x1024": lambda: residual_rmsnorm(256, 1024),
    "row_stats_512x1024": lambda: row_stats(512, 1024),
    "pointwise_chain_512x1024": lambda: pointwise_chain(512, 1024),
    "softplus_128x64": lambda: softplus(128, 64),
    "linear_bias_gelu_256x1024x4096": lambda: linear_bias_gelu(256, 1024, 4096),
    "mm_chain_256x512x1024x256": lambda: mm_chain(256, 512, 1024, 256),
    "mlp_gelu_256x1024x4096": lambda: mlp_gelu(256, 1024, 4096),
    "mlp_swiglu_128x512x1536": lambda: mlp_swiglu(128, 512, 1536),
    "mlp_swiglu_256x1024x4096": lambda: mlp_swiglu(256, 1024, 4096),
    "mlp_stack_128x512x2048": lambda: mlp_stack(128, 512, 2048),
    "simple_attn_4x64x256x128": lambda: simple_attn(4, 64, 256, 128),
    "sdpa_1x8x256x64": lambda: sdpa(1, 8, 256, 64),
    "sdpa_1x4x512x128": lambda: sdpa(1, 4, 512, 128),
    "sdpa_causal_1x8x128x128": lambda: sdpa(1, 8, 128, 128, causal=True),
    "sdpa_1x12x512x128": lambda: sdpa_g(1, 12, 12, 512, 512, 128),
    "sdpa_1x32x512x64": lambda: sdpa_g(1, 32, 32, 512, 512, 64),
    "sdpa_1x16x1024x64": lambda: sdpa_g(1, 16, 16, 1024, 1024, 64),
    "sdpa_1x8x2048x64": lambda: sdpa_g(1, 8, 8, 2048, 2048, 64),
    "sdpa_4x8x256x64": lambda: sdpa_g(4, 8, 8, 256, 256, 64),
    "sdpa_causal_1x12x512x128": lambda: sdpa_g(1, 12, 12, 512, 512, 128, causal=True),
    "sdpa_causal_1x8x1024x64": lambda: sdpa_g(1, 8, 8, 1024, 1024, 64, causal=True),
    "sdpa_gqa_1x32x8x512x128": lambda: sdpa_g(1, 32, 8, 512, 512, 128),
    "sdpa_gqa_1x16x4x1024x64": lambda: sdpa_g(1, 16, 4, 1024, 1024, 64),
    "sdpa_q64_kv1024_1x8x64": lambda: sdpa_g(1, 8, 8, 64, 1024, 64),
    "attn_block_128x512x8": lambda: attn_block(128, 512, 8),
    "attn_block_256x1024x16": lambda: attn_block(256, 1024, 16),
    "transformer_block_128x512x8x1536": lambda: transformer_block(128, 512, 8, 1536),
    "transformer_block_256x1024x16x2816": lambda: transformer_block(256, 1024, 16, 2816),
}


def main():
    # guard.py sends SIGUSR1 before killing a runaway compile: dump all stacks
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    name, out_dir = sys.argv[1], sys.argv[2]
    if name == "list":
        print("\n".join(MODELS))
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import spyre_capture

    torch._inductor.config.fx_graph_cache = False
    torch._inductor.config.force_disable_caches = True
    spyre_capture.install(out_dir, name)
    torch.manual_seed(0)
    fn, args = MODELS[name]()
    dev = torch.device("spyre")
    dargs = [a.to(dev) for a in args]
    t0 = time.time()
    status = "completed (no abort?)"
    try:
        torch.compile(fn)(*dargs)
    except Exception as e:  # CaptureDone arrives wrapped by dynamo
        chain, cur = [], e
        while cur is not None:
            chain.append(cur)
            cur = cur.__cause__ or cur.__context__
        if any(isinstance(c, spyre_capture.CaptureDone) for c in chain):
            status = "captured"
        else:
            status = "error: " + "".join(traceback.format_exception_only(type(e), e)).strip()[-400:]
    print(f"{name}\t{status}\t{spyre_capture.S.solve_idx} solves\t{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

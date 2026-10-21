"""Granite 4.0 micro generate on Spyre under the Stage-0 timers.

Usage: granite_t.py <hf model path> <tag> <max_new_tokens> <out.jsonl> [caps_dir]
"""
import faulthandler
import os
import signal
import sys
import time

os.environ.setdefault("HF_HOME", os.path.expanduser("~"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.environ.get("HF_ADAPTERS", os.path.expanduser("~/hf-adapters")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch_spyre  # noqa: E402,F401

# transformers & deps from the hf-adapters venv, appended only AFTER torch has
# autoloaded this checkout's torch_spyre (see ../pr5132/granite_cap.py).
sys.path.append(os.environ.get("HF_PATH_SHIM", os.path.expanduser("~/cpsat_bench/hfpath")))

import timers  # noqa: E402


def main():
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    path, tag, ntok, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    caps = sys.argv[5] if len(sys.argv) > 5 else None
    timers.install(out, tag, caps)
    from transformers import AutoTokenizer

    import hf_adapters.hf_granitemoehybrid as granite4
    from hf_adapters import AutoSpyreModelForCausalLM
    from hf_adapters.auto_spyre_model import torch_dtype_for_model_path
    from hf_adapters.hf_common import prepare_rope_and_heads

    adapter_prepare = granite4.prepare_for_spyre

    def prepare_padded(model):
        adapter_prepare(model)
        prepare_rope_and_heads(model)

    granite4.prepare_for_spyre = prepare_padded

    t0 = time.time()
    model = AutoSpyreModelForCausalLM.from_pretrained(
        path, dtype=torch_dtype_for_model_path(path)
    )
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    tok = AutoTokenizer.from_pretrained(path)
    t0 = time.time()
    res = model.generate(
        tok, ["The capital of France is"], max_new_tokens=ntok, do_sample=False,
        timing=True,
    )
    print(f"{tag}\tok\tgenerate {time.time() - t0:.0f}s -> {res!r}\t"
          f"plans={timers.T.plan_idx}\t{torch_spyre.__file__}", flush=True)


if __name__ == "__main__":
    main()

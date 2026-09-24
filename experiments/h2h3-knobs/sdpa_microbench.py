"""SDPA-level microbenchmark: does a sliding-window *mask* cost less than a dense global mask?

Motivation. In the pilot, making all 22 layers global cost only ~3% latency at pad=7000. If the
sliding layers were really cheap, converting 14 of them to global should cost much more. The
hypothesis this measures is that they are not cheap: ``sdpa_attention_forward`` in
transformers 4.57.6 hands ``F.scaled_dot_product_attention`` a **dense** ``[1,1,L,L]`` float bias
for both branches (``sliding_window_mask`` for local layers, ``global_attention_mask`` for global
ones), so every layer pays dense O(L^2) attention regardless of the window; the window only zeroes
weights after the fact.

This is a microbenchmark of the kernel, not of the model: it isolates the mask's cost from
everything else. Reported as ``measured`` on this box, this GPU, this torch build.

    env/venv/bin/python experiments/h2h3-knobs/sdpa_microbench.py --L 7068
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = os.path.dirname(os.path.abspath(__file__))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def make_masks(L, half, dtype=torch.float32, device="cuda"):
    rows = torch.arange(L, device=device).unsqueeze(0)
    dist = (rows - rows.T).abs()
    win = dist <= half
    neg = torch.finfo(dtype).min
    global_mask = torch.zeros(1, 1, L, L, dtype=dtype, device=device)
    sliding = global_mask.masked_fill(win.logical_not(), neg)
    # a mask of the same shape that blocks everything except a tiny corner: cheapest possible
    # *content* with identical structure -> isolates structure/shape cost from content cost
    return global_mask, sliding


def timeit(fn, iters=5, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return {"median_s": round(statistics.median(ts), 5), "min_s": round(min(ts), 5),
            "samples": [round(t, 5) for t in ts]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=7068)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HERE, "sdpa_microbench.json"))
    a = ap.parse_args()

    L, H, D = a.L, a.heads, a.dim
    dev = "cuda"
    torch.manual_seed(0)
    q = torch.randn(1, H, L, D, device=dev, dtype=torch.float32)
    k = torch.randn(1, H, L, D, device=dev, dtype=torch.float32)
    v = torch.randn(1, H, L, D, device=dev, dtype=torch.float32)
    global_mask, sliding = make_masks(L, 64)

    torch.cuda.reset_peak_memory_stats()
    out = {
        "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "L": L, "heads": H, "head_dim": D, "dtype": "float32", "iters": a.iters,
        "note": "one SDPA call per timing; the full model runs 22 such calls per forward",
        "cases": {},
    }
    cases = {
        "no_mask": lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=None),
        "dense_global_mask_all_zeros": lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=global_mask),
        "dense_sliding_mask_half_64": lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=sliding),
    }
    for name, fn in cases.items():
        r = timeit(fn, iters=a.iters)
        out["cases"][name] = r
        print("%-30s median=%.5fs min=%.5fs" % (name, r["median_s"], r["min_s"]), flush=True)

    base = out["cases"]["no_mask"]["median_s"]
    for name, r in out["cases"].items():
        r["ratio_vs_no_mask"] = round(r["median_s"] / base, 3)
    g = out["cases"]["dense_global_mask_all_zeros"]["median_s"]
    s = out["cases"]["dense_sliding_mask_half_64"]["median_s"]
    out["sliding_over_global"] = round(s / g, 3)
    out["peak_vram_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    print("sliding/global mask cost ratio: %.3f  (1.0 == the window buys no kernel time)"
          % out["sliding_over_global"], flush=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote %s" % os.path.basename(a.out))


if __name__ == "__main__":
    main()

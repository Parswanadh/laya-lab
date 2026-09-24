"""Per-layer attention-mask evidence, dumped in full.

Why this exists. In the arm sweep, making **all 22 layers global** cost only ~6% latency at
pad=7000, and *narrowing* the window to 64 changed nothing either. The explanation is a property of
the execution path, not of the algorithm: in transformers 4.57.6 ``sdpa_attention_forward`` hands
``F.scaled_dot_product_attention`` a **dense** ``[1, 1, L, L]`` float32 bias on *every* layer —
``sliding_window_mask`` for the 14 local layers, ``global_attention_mask`` for the 8 global ones
(both built in ``ModernBertModel._update_attention_mask``). A sliding-window *bias* does not reduce
attention work; it only zeroes weights after the full L x L score matrix exists. So the shipped
model already pays dense-attention cost in all 22 layers while getting a ±64 receptive field in 14.

This script records that directly: one forward per arm at ``--L`` tokens, with the full per-layer
mask record (shape, dtype, bytes, sampled max attended distances) written to mask_evidence.json.

    env/venv/bin/python experiments/h2h3-knobs/mask_evidence.py --pad 7000
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(LAB, "fork"))
sys.path.insert(0, os.path.join(LAB, "experiments", "harness"))

import torch  # noqa: E402

import laya  # noqa: E402


def load_runner():
    spec = importlib.util.spec_from_file_location("e002run", os.path.join(HERE, "run.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pad", type=int, default=7000)
    ap.add_argument("--model", default=os.path.join(LAB, "models", "multilingual"))
    ap.add_argument("--out", default=os.path.join(HERE, "mask_evidence.json"))
    a = ap.parse_args()

    R = load_runner()
    p1 = R.load_p1_module()
    import pools
    pool = pools.load_pool("upstream_multilingual")
    agent = laya.load(a.model, device="cuda")
    agent.dtype = torch.float32
    enc = agent.model.encoder
    orig = R.snapshot_config(enc)
    tok = agent.tok
    text = R.build_doc(p1, tok, "p1_exact", a.pad, p1.REQUESTS[0][0], pool["filler_unit"])

    probe = R.AttentionProbe()
    probe.install()
    out = {"pad": a.pad, "text_tokens": len(tok(text, add_special_tokens=False)["input_ids"]),
           "arms": {}, "config_at_load": orig}
    for arm in ("baseline", "w64", "w512", "allglobal"):
        R.apply_arm(enc, R.ARM_SPECS[arm], orig)
        probe.records = []
        probe.armed = True
        agent.predict({"text": text}, pool["question"], max_len=8192)
        probe.armed = False
        recs = probe.records
        L = recs[0]["L"] if recs else 0
        per_layer = [{"layer_id": r["layer_id"], "branch": r["branch"],
                      "attn_local_attention_arg": r["attn_local_attention_arg"],
                      "mask_shape": r["mask_shape"], "mask_is_4d_float": r["mask_is_4d_float"],
                      "last_row_max_dist": r["rows"][str(L - 1)]["max_dist"],
                      "last_row_allowed_keys": r["rows"][str(L - 1)]["allowed_keys"]}
                     for r in recs]
        dense_bytes = sum((r["mask_shape"][-1] * r["mask_shape"][-2] * 4)
                          for r in recs if r["mask_shape"])
        out["arms"][arm] = {
            "L": L, "layers": len(recs),
            "n_sliding": sum(1 for r in recs if r["branch"] == "sliding"),
            "n_global": sum(1 for r in recs if r["branch"] == "global"),
            "mask_shape_histogram": _hist([tuple(r["mask_shape"]) if r["mask_shape"] else None
                                           for r in recs]),
            "dense_mask_bytes_per_forward_sum_over_layers": dense_bytes,
            "dense_attention_score_bytes_if_materialised_per_layer": L * L * 12 * 4 if L else 0,
            "modelled_dense_attention_flops_per_forward":
                22 * 2 * 2 * L * L * 768 if L else 0,
            "per_layer": per_layer,
        }
        print("%-10s L=%d sliding=%d global=%d mask_shapes=%s dense_mask_bytes/forward=%.1f MiB"
              % (arm, L, out["arms"][arm]["n_sliding"], out["arms"][arm]["n_global"],
                 out["arms"][arm]["mask_shape_histogram"], dense_bytes / 2**20), flush=True)

    R.apply_arm(enc, R.ARM_SPECS["baseline"], orig)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote %s" % os.path.basename(a.out))


def _hist(seq):
    h = {}
    for x in seq:
        k = str(x)
        h[k] = h.get(k, 0) + 1
    return h


if __name__ == "__main__":
    main()

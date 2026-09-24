"""Diagnose why arm3 (cross-attention from random init) did not fit the real training data.

    env/venv/bin/python experiments/h5-adapter/_probe_lr.py --mode real   [GPU]
    env/venv/bin/python experiments/h5-adapter/_probe_lr.py --mode synth  [CPU]

Two hypotheses, and the two modes separate them:

* `synth` — the synthetic cache whose state tokens carry the label. If the cross-attention head
  cannot fit *that*, the architecture or its implementation is the problem.
* `real` — the real 1008-item training cache over a few epochs at several learning rates. If the
  synthetic task fits but the real one does not, it is an optimisation problem on real data, not an
  architectural one.

Prints the per-epoch train loss/accuracy so the trajectory is visible; writes nothing to runs/.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

os.environ.setdefault("USE_TF", "0")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
from transformers import AutoConfig, AutoModel  # noqa: E402

import arms as A  # noqa: E402
import common_h5 as C  # noqa: E402
import features as FEAT  # noqa: E402
import train as T  # noqa: E402


def run_one(arm: str, lr: float, epochs: int, items, cache_dir: str, shipped, tmp: str) -> dict:
    C.CACHE_DIR = cache_dir
    T.RUNS_DIR = os.path.join(tmp, "runs")     # never touch the real runs/ directory
    res = T.train_arm(arm, seed=0, epochs=epochs, lr=lr, token_budget=12288, max_batch=8,
                      device_name="cpu" if shipped.encoder.config.hidden_size < 100 else "cuda",
                      log_every=0, train_items_override=items, shipped_override=shipped)
    return {
        "arm": arm, "lr": lr, "epochs": epochs,
        "loss_first": res["history"][0]["train_loss"],
        "loss_last": res["history"][-1]["train_loss"],
        "acc_first": res["history"][0]["train_accuracy"],
        "acc_last": res["history"][-1]["train_accuracy"],
        "acc_best": max(h["train_accuracy"] for h in res["history"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["real", "synth"])
    ap.add_argument("--lrs", default="3e-4,1e-3,3e-3,1e-2")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--arms", default="arm3_xattn,arm2_random_init")
    a = ap.parse_args()
    lrs = [float(x) for x in a.lrs.split(",")]
    arms = a.arms.split(",")
    tmp = tempfile.mkdtemp(prefix="h5-probe-")
    try:
        if a.mode == "real":
            import common_h5 as _C
            agent = _C.load_agent("cuda")
            shipped = agent.model
            cache_dir = os.path.join(_C.CACHE_DIR)
            plan = C.load_plan()
            items = plan["train_items"]
        else:
            from check_learnability import HIDDEN, synth_cache
            cfg = AutoConfig.for_model("bert", hidden_size=HIDDEN, num_hidden_layers=2,
                                       num_attention_heads=2, intermediate_size=64, vocab_size=128)
            shipped = A.DecisionModel(AutoModel.from_config(cfg), head_layers=2, n_act=2)
            shipped.eval()
            plan = C.load_plan()
            items = plan["train_items"][:128]
            cache_dir = os.path.join(tmp, "synth")
            synth_cache(os.path.join(cache_dir, "train"), items, correlated=True)

        print("=== mode=%s  items=%d  epochs=%d  lrs=%s ===" % (a.mode, len(items), a.epochs, lrs))
        rows = []
        for arm in arms:
            for lr in lrs:
                r = run_one(arm, lr, a.epochs, items, cache_dir, shipped, tmp)
                rows.append(r)
                print("  %-18s lr=%-7g loss %.4f -> %.4f   acc %.3f -> %.3f (best %.3f)"
                      % (arm, lr, r["loss_first"], r["loss_last"], r["acc_first"], r["acc_last"],
                         r["acc_best"]), flush=True)
        out = os.path.join(HERE, "_probe_lr_%s.json" % a.mode)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"mode": a.mode, "n_items": len(items), "epochs": a.epochs, "rows": rows},
                      fh, indent=1)
        print("wrote %s" % os.path.relpath(out, os.path.dirname(HERE)))
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

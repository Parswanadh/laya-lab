"""Short learning-rate probe for the zero-initialised parallel cross-attention branch (issue #11).

    flock -n ../../.gpu.lock -c 'env/venv/bin/python experiments/h5-adapter/probe_arm3r.py'

A randomly initialised branch usually needs a larger step than a fine-tuned one, and the branch's
*inner* weights are exactly zero-gradient at step 0 (``dL/dz = W^T dL/ddelta = 0`` while ``W = 0``),
so the rate on ``cross.*`` is a real recipe choice. The issue asks for it to be reported and
justified rather than defaulted silently.

The probe holds the **rest of the recipe fixed at arm 2's** (`lr = 5e-4`, the difference between the
two arms has to stay the architecture) and varies only the branch's rate. Every config runs the real
1008-item training set for a few epochs on cached features, with dropout on and the same batching,
and reports the per-epoch curve plus the measured branch contribution to the logits
(`train.py`'s zeroed-`out_proj` probe). Writes `probe_arm3r.json` next to this file; checkpoints go
to a temporary directory and are removed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import common_h5 as C  # noqa: E402
import train as T  # noqa: E402

ARM = "arm3r_residual"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=ARM)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--main-lr", type=float, default=5e-4)
    ap.add_argument("--cross-lrs", default="5e-4,3e-3,1e-2")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cross_lrs = [float(x) for x in a.cross_lrs.split(",")]

    tmp = tempfile.mkdtemp(prefix="h5-arm3r-probe-")
    rows = []
    try:
        T.RUNS_DIR = os.path.join(tmp, "runs")   # never touch the production runs/ directory
        agent = C.load_agent("cuda")
        print("=== arm3r branch-lr probe: main lr=%g, epochs=%d, cross lrs=%s ==="
              % (a.main_lr, a.epochs, cross_lrs), flush=True)
        for lr_cross in cross_lrs:
            t0 = time.time()
            res = T.train_arm(a.arm, seed=a.seed, epochs=a.epochs, lr=a.main_lr,
                              lr_cross=lr_cross, token_budget=12288, max_batch=8,
                              device_name="cuda", log_every=0, shipped_override=agent.model)
            hist = res["history"]
            row = {
                "arm": a.arm, "lr": a.main_lr, "lr_cross": lr_cross, "epochs": a.epochs,
                "loss_per_epoch": [round(h["train_loss"], 4) for h in hist],
                "acc_per_epoch": [round(h["train_accuracy"], 4) for h in hist],
                "branch_contribution_per_epoch": [h.get("branch_logit_contribution_max")
                                                  for h in hist],
                "out_proj_weight_fro_per_epoch": [h.get("out_proj_weight_fro") for h in hist],
                "step0_out_proj_grad_sum": (res["step0_gradients"] or {}).get("out_proj_weight_grad_sum"),
                "final_train_loss": hist[-1]["train_loss"],
                "final_train_accuracy": hist[-1]["train_accuracy"],
                "seconds": round(time.time() - t0, 1),
                "peak_vram_gb": res["peak_vram_gb"],
            }
            rows.append(row)
            print("  cross_lr=%-7g loss %s  acc %s" % (
                lr_cross, " ".join("%.4f" % v for v in row["loss_per_epoch"]),
                " ".join("%.3f" % v for v in row["acc_per_epoch"])), flush=True)
            print("               branch|dlogit| %s   ||W_out|| %s" % (
                " ".join("%.3g" % (v or 0.0) for v in row["branch_contribution_per_epoch"]),
                " ".join("%.3g" % (v or 0.0) for v in row["out_proj_weight_fro_per_epoch"])),
                flush=True)
        out = os.path.join(HERE, "probe_arm3r.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"arm": a.arm, "seed": a.seed, "main_lr": a.main_lr, "epochs": a.epochs,
                       "n_train_items": rows[0]["epochs"] and 1008, "configs": rows}, fh, indent=1)
            fh.write("\n")
        print("wrote %s" % os.path.relpath(out, os.path.dirname(HERE)))
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

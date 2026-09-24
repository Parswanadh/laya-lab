"""Driver for the init-fair arm-3 re-run (issue #11): train -> eval -> delete the head -> stats.

    flock -n ../../.gpu.lock -c 'env/venv/bin/python experiments/h5-adapter/run_arm3r.py \
        --stages train,eval --arms arm3r_residual --epochs 40 --lr 5e-4 --lr-cross 3e-3'
    flock -n ../../.gpu.lock -c 'env/venv/bin/python experiments/h5-adapter/run_arm3r.py \
        --stages train,eval --arms arm2long_shipped_init --epochs 40 --lr 5e-4'

Deliberately small and stage-separated so the GPU lock is taken per stage rather than held for the
whole afternoon, and so a run killed part-way leaves a complete (arm, seed) result behind.

`head.pt` is deleted **immediately after the eval that needs it**: the filesystem reached 100 % full
during the previous run of this experiment, the checkpoint is 59 MB, and it is regenerable exactly
from the `(arm, seed, lr, epochs)` recorded in the committed `training.json`. Per-item predictions
and the training record are the kept artifacts. `--keep-heads` overrides the deletion.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(os.path.dirname(HERE))
PY = os.path.join(LAB, "env", "venv", "bin", "python")
LOG = os.path.join(HERE, "run_arm3r.log")


def log(msg: str) -> None:
    line = "[%s] %s" % (time.strftime("%F %T"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(cmd: List[str]) -> int:
    log("$ " + " ".join(cmd))
    out = subprocess.run(cmd, cwd=LAB, capture_output=True, text=True)
    sys.stdout.write(out.stdout)
    sys.stderr.write(out.stderr)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(out.stdout)
        fh.write(out.stderr)
    return out.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="train,eval",
                    help="comma-separated: train,eval,stats")
    ap.add_argument("--arms", default="arm3r_residual,arm2long_shipped_init")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr-cross", type=float, default=None)
    ap.add_argument("--keep-heads", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true",
                    help="do not run the zeroed-branch control for arm3r_residual")
    ap.add_argument("--skip-eval-if-predictions-exist", action="store_true", default=True)
    a = ap.parse_args()
    stages = [s for s in a.stages.split(",") if s]
    arms = [s for s in a.arms.split(",") if s]
    seeds = [int(x) for x in a.seeds.split(",") if x != ""]

    log("=== run_arm3r stages=%s arms=%s seeds=%s epochs=%d lr=%g lr_cross=%s ==="
        % (stages, arms, seeds, a.epochs, a.lr, a.lr_cross))

    if "train" in stages:
        for arm in arms:
            for seed in seeds:
                cmd = [PY, "experiments/h5-adapter/train.py", "--arm", arm, "--seed", str(seed),
                       "--epochs", str(a.epochs), "--lr", str(a.lr)]
                if a.lr_cross is not None:
                    cmd += ["--lr-cross", str(a.lr_cross)]
                rc = run(cmd)
                if rc != 0:
                    log("FAILED training %s seed %d (rc=%d)" % (arm, seed, rc))
                    return rc

    if "eval" in stages:
        for arm in arms:
            for seed in seeds:
                pred = os.path.join(HERE, "predictions", "%s-seed%d.jsonl" % (arm, seed))
                if a.skip_eval_if_predictions_exist and os.path.exists(pred):
                    log("skip eval %s seed %d: %s already exists"
                        % (arm, seed, os.path.basename(pred)))
                else:
                    rc = run([PY, "experiments/h5-adapter/eval.py", "--arm", arm,
                              "--seed", str(seed)])
                    if rc != 0:
                        log("FAILED eval %s seed %d (rc=%d)" % (arm, seed, rc))
                        return rc
                # The cheapest control: the same trained head with the added branch re-zeroed.
                # Costs one extra eval pass and answers "how much of arm3r is the branch?".
                if arm == "arm3r_residual" and not a.skip_ablation:
                    abl = os.path.join(HERE, "predictions", "arm3r_residual_ablated-seed%d.jsonl" % seed)
                    if os.path.exists(abl):
                        log("skip ablation for %s seed %d: %s already exists"
                            % (arm, seed, os.path.basename(abl)))
                    else:
                        rc = run([PY, "experiments/h5-adapter/eval.py", "--arm", arm,
                                  "--seed", str(seed), "--zero-branch",
                                  "--arm-label", "arm3r_residual_ablated", "--out", abl])
                        if rc != 0:
                            log("FAILED branch ablation %s seed %d (rc=%d)" % (arm, seed, rc))
                            return rc
                if not a.keep_heads:
                    head = os.path.join(HERE, "runs", arm, "seed%d" % seed, "head.pt")
                    if os.path.exists(head):
                        mb = os.path.getsize(head) / 1e6
                        os.remove(head)
                        log("deleted %s (%.1f MB)" % (os.path.relpath(head, LAB), mb))
                tp = os.path.join(HERE, "runs", arm, "seed%d" % seed, "training.json")
                if os.path.exists(tp):
                    with open(tp, encoding="utf-8") as fh:
                        rec: Dict[str, Any] = json.load(fh)
                    h = rec["history"][-1]
                    log("  %s seed%d: %d epochs, %.0fs, final loss %.4f acc %.4f, branch|dlogit|=%.4g"
                        % (arm, seed, rec["epochs"], rec["train_seconds"], h["train_loss"],
                           h["train_accuracy"], h.get("branch_logit_contribution_max") or 0.0))

    if "stats" in stages:
        rc = run([PY, "experiments/h5-adapter/stats.py"])
        if rc != 0:
            return rc

    log("=== done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

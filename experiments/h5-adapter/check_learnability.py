"""Can the training loop learn anything at all?

    env/venv/bin/python experiments/h5-adapter/check_learnability.py

A tiny *random* encoder has nothing to learn, so an end-to-end smoke test with one cannot tell a
working training loop from a broken one. This builds a synthetic feature cache in the exact format
`features.FeatureStore` reads, whose state positions literally carry the label, and trains the real
`train.train_arm` loop on it. A working head plus optimiser must reach high train accuracy.

It also trains on a cache whose state carries an *uncorrelated* label. That second arm is the point:
without it, "it learned" could just mean "it memorised the item ids" or "the target pairing is not
actually being checked".

The sweep over learning rate and epoch budget is what sets the real run's settings. Those are
chosen from **train** accuracy only -- the evaluation set is never read here -- so the choice cannot
be tuning on the test set.

Outcome is written to `experiments/h5-adapter/learnability.json`.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from typing import Any, Dict, List

os.environ.setdefault("USE_TF", "0")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "worktrees", "h5"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoConfig, AutoModel  # noqa: E402

import arms as A  # noqa: E402
import common_h5 as C  # noqa: E402
import features as FEAT  # noqa: E402
import train as T  # noqa: E402

HIDDEN = 32
LENGTH = 64
STATE_START = 8
MARKERS = [2, 3, 4, 5]


def synth_cache(path: str, items: List[Dict[str, Any]], correlated: bool, seed: int = 0) -> str:
    """A `FeatureStore`-readable cache whose state tokens carry the label (or a decoy)."""
    os.makedirs(path, exist_ok=True)
    mm = np.memmap(os.path.join(path, "features.f16"), dtype=np.float16, mode="w+",
                   shape=(len(items) * LENGTH, HIDDEN))
    rng = np.random.default_rng(seed)
    idx_items = []
    for i, it in enumerate(items):
        h = np.zeros((LENGTH, HIDDEN), dtype=np.float16)
        gold = C.LABELS.index(it["label"])
        shown = gold if correlated else int(rng.integers(0, len(C.LABELS)))
        for pos in range(STATE_START, STATE_START + 8):
            h[pos, shown] = 1.0
        # Each option marker must be *distinguishable*. The decision head has no positional
        # embedding of its own -- it is a transformer over the encoder's contextualised states --
        # so if every marker carried the same vector the four options would be interchangeable and
        # the softmax over them could never beat 1/n_options no matter how well the label was read.
        # In the real model the encoder's RoPE makes the markers distinct; here a slot code does.
        for slot, pos in enumerate(MARKERS):
            h[pos, len(C.LABELS) + slot] = 1.0
        mm[i * LENGTH:(i + 1) * LENGTH] = h
        idx_items.append({
            "item_id": it["item_id"], "split": "train", "cell": it["cell"], "label": it["label"],
            "lang": "en", "template_id": it["template_id"], "option_order": None,
            "pad": it["pad"], "needle_position": it["needle_position"],
            "offset": i * LENGTH, "length": LENGTH, "hidden": HIDDEN, "markers": list(MARKERS),
            "state_start": STATE_START, "head_len": STATE_START,
            "state_tokens_full": LENGTH - STATE_START, "state_tokens_kept": LENGTH - STATE_START,
            "truncated": False, "needle_tokens": 1, "needle_token_start": STATE_START,
            "needle_tokens_kept": 1, "needle_kept": True, "request_kept": True, "pad_exact": True,
            "state_sha256": "synthetic", "filler_tokens_before": 0,
        })
    mm.flush()
    del mm
    with open(os.path.join(path, "index.json"), "w", encoding="utf-8") as fh:
        json.dump({"split": "train", "file": "features.f16", "dtype": "float16", "hidden": HIDDEN,
                   "total_tokens": len(items) * LENGTH, "n_items": len(idx_items),
                   "items": idx_items, "synthetic": True,
                   "what": ("state tokens 8..15 carry a one-hot label; markers 2..5 carry a "
                            "one-hot slot code") if correlated
                           else "state tokens 8..15 carry an unrelated label"},
                  fh)
    return path


def main() -> int:
    plan = C.load_plan()
    items = plan["train_items"][:256]
    tmp = tempfile.mkdtemp(prefix="h5-learn-")
    try:
        cfg = AutoConfig.for_model("bert", hidden_size=HIDDEN, num_hidden_layers=2,
                                   num_attention_heads=2, intermediate_size=64, vocab_size=128)
        shipped = A.DecisionModel(AutoModel.from_config(cfg), head_layers=2, n_act=2)
        shipped.eval()

        results: List[Dict[str, Any]] = []
        for arm in ("arm2_random_init", "arm3_xattn"):
            for correlated in (True, False):
                for lr, epochs in ((1e-4, 60), (5e-4, 60), (2e-3, 60), (1e-4, 250), (5e-4, 250)):
                    sub = os.path.join(tmp, "c%d_%s_%g_%d" % (correlated, arm, lr, epochs))
                    synth_cache(os.path.join(sub, "train"), items, correlated)
                    C.CACHE_DIR = sub
                    T.RUNS_DIR = os.path.join(sub, "runs")
                    res = T.train_arm(arm, seed=0, epochs=epochs, lr=lr, token_budget=4096,
                                      max_batch=8, device_name="cpu", log_every=0,
                                      train_items_override=items, shipped_override=shipped)
                    row = {
                        "arm": arm, "state_carries_label": correlated, "lr": lr,
                        "epochs": epochs, "steps": res["steps"],
                        "final_train_accuracy": res["history"][-1]["train_accuracy"],
                        "final_train_loss": res["history"][-1]["train_loss"],
                        "best_train_accuracy": max(h["train_accuracy"] for h in res["history"]),
                        "chance": 0.25,
                        "batches_with_verified_item_target_pairing":
                            res["batches_with_verified_item_target_pairing"],
                    }
                    results.append(row)
                    print("  %-18s label=%d lr=%-8g epochs=%3d  train_acc=%.3f loss=%.4f"
                          % (arm, correlated, lr, epochs, row["final_train_accuracy"],
                             row["final_train_loss"]), flush=True)
        out = {
            "artifact": "experiments/h5-adapter/learnability.json",
            "reproduce": "env/venv/bin/python experiments/h5-adapter/check_learnability.py",
            "what": ("the real train_arm loop, driven by a synthetic feature cache whose state "
                     "tokens carry the label. Not a model of anything -- a test of the loop."),
            "n_items": len(items), "hidden": HIDDEN, "length": LENGTH,
            "results": results,
            "reading": ("`best_on_a_real_label` high together with `highest_on_a_decoy_label` at "
                        "chance is the evidence that the loop can learn and that it is not merely "
                        "fitting the item identity or the target pairing"),
            "learns": {},
        }
        for arm in ("arm2_random_init", "arm3_xattn"):
            good = [r for r in results if r["arm"] == arm and r["state_carries_label"]]
            bad = [r for r in results if r["arm"] == arm and not r["state_carries_label"]]
            out["learns"][arm] = {
                "best_on_a_real_label": max(r["final_train_accuracy"] for r in good),
                "best_lr_and_epochs": max(good, key=lambda r: r["final_train_accuracy"])["lr"],
                "best_epochs": max(good, key=lambda r: r["final_train_accuracy"])["epochs"],
                "highest_on_a_decoy_label": max(r["final_train_accuracy"] for r in bad),
            }
        with open(os.path.join(HERE, "learnability.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
            fh.write("\n")
        print(json.dumps(out["learns"], indent=1))
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

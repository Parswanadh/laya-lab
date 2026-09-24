"""Build the frozen-encoder feature cache for one split.

    env/venv/bin/python experiments/h5-adapter/build_cache.py --split eval
    env/venv/bin/python experiments/h5-adapter/build_cache.py --split train
    env/venv/bin/python experiments/h5-adapter/build_cache.py --split eval --fidelity

`--fidelity` re-runs a small sample one item at a time, with no batch padding, and compares the
encoder's output against the cached values. Batching must not change the representation the arms
train on; that is measured here rather than assumed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import common_h5 as C  # noqa: E402
import features as FEAT  # noqa: E402


def fidelity_check(store: FEAT.FeatureStore, items, builder, model, device, n: int = 8,
                   cell: str = "L4000-p100") -> dict:
    """Batched cache vs unbatched single-item forward, one document at a time."""
    model.eval()
    idx = [i for i, it in enumerate(store.items) if it["cell"] == cell][:n]
    if not idx:
        idx = list(range(min(n, len(store.items))))
    worst = 0.0
    worst_item = None
    with torch.inference_mode():
        for i in idx:
            it = store.items[i]
            cond = next(c for c in items if c["item_id"] == it["item_id"])
            doc = builder.build(cond, option_order=cond.get("option_order"))
            ids = torch.tensor([doc["input_ids"]], dtype=torch.long, device=device)
            att = torch.ones_like(ids)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                h = model.encoder(input_ids=ids, attention_mask=att).last_hidden_state
            h = h.float().cpu().numpy()[0]
            cached = store.hidden_states(i).astype(np.float32)
            d = float(np.abs(h - cached).max())
            if d > worst:
                worst, worst_item = d, it["item_id"]
    return {
        "n_items": len(idx), "cell": cell,
        "max_abs_difference": worst, "worst_item": worst_item,
        "tolerance": 2e-3,
        "passed": bool(worst <= 2e-3),
        "meaning": ("the cached fp16 features equal a fresh unbatched encoder forward, so batch "
                    "padding did not change the representation the arms train on"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["eval", "train"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--token-budget", type=int, default=12288)
    ap.add_argument("--max-batch", type=int, default=4)
    ap.add_argument("--fidelity", action="store_true")
    a = ap.parse_args()
    device = torch.device(a.device)

    plan = C.load_plan()
    if a.split == "eval":
        items = FEAT.build_eval_conditions(plan)
    else:
        items = plan["train_items"]
    agent = C.load_agent(a.device)
    pool = C.load_needle_pool("needles-h5-%s-v1.json" % a.split)
    builder = C.make_builder(agent.tok, C.load_filler(), pool)
    out = os.path.join(C.CACHE_DIR, a.split)
    FEAT.build_cache(items, a.split, out, agent.model, builder, device,
                     token_budget=a.token_budget, max_batch=a.max_batch)
    if a.fidelity:
        store = FEAT.FeatureStore(out)
        res = fidelity_check(store, items, builder, agent.model, device)
        with open(os.path.join(out, "fidelity.json"), "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
            fh.write("\n")
        print("fidelity: max |cached - fresh| = %.5f over n=%d (%s)"
              % (res["max_abs_difference"], res["n_items"], "PASS" if res["passed"] else "FAIL"))
        return 0 if res["passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

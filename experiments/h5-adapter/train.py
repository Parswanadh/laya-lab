"""Train one H5 arm for one seed, on cached frozen-encoder features.

    env/venv/bin/python experiments/h5-adapter/train.py --arm arm3_xattn --seed 0

Writes `experiments/h5-adapter/runs/<arm>/seed<k>/{head.pt,training.json}`. The head state dict is
small (~60 MB fp32) and is committed; optimiser state is not.

Design notes that matter for the comparison:

* **Same optimiser, schedule, epoch count, batch token budget and data for every trained arm.**
  Only the head module differs.
* **Gradients never touch the encoder**: the cached `h` was produced under `inference_mode`, and
  the encoder's parameters have `requires_grad=False`, so a backward that reached it would raise
  rather than silently fine-tune it. `train.py` additionally asserts the encoder's grad is `None`
  after the first step.
* Head forward/backward runs under bf16 autocast with fp32 master weights, which is how the
  shipped head was trained (`amp_dtype: bf16` in the checkpoint config). bf16 needs no loss
  scaling, so there is no GradScaler to get wrong.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import arms as A  # noqa: E402
import common_h5 as C  # noqa: E402
import features as FEAT  # noqa: E402

RUNS_DIR = os.path.join(HERE, "runs")


def batches_by_length(items: List[Dict[str, Any]], token_budget: int, max_batch: int,
                      seed: int) -> List[List[int]]:
    """Index batches sharing a token budget, shuffled group order per epoch.

    Reusing ``features._token_budget_batches`` keeps training and cache building on one definition
    of "a batch", so arm 2's self-attention never sees a longer padded sequence than the cache
    builder did.
    """
    lengths = [it["length"] for it in items]
    groups = list(FEAT._token_budget_batches(lengths, token_budget, max_batch))
    random.Random(seed).shuffle(groups)
    return groups


def train_arm(arm: str, seed: int, epochs: int = 8, lr: float = 1e-4, weight_decay: float = 0.01,
              token_budget: int = 12288, max_batch: int = 8, warmup_frac: float = 0.05,
              device_name: str = "cuda", limit_train: Optional[int] = None,
              log_every: int = 50, shipped_override=None,
              train_items_override: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """``shipped_override`` and ``train_items_override`` let the offline selftest drive this loop
    with a tiny encoder and a stratified item subset, so the loop under test is this loop and not a
    copy of it. A subset whose positions are not its store row indices is the point: the arms train
    on stratified blocks, and an identity assumption between the two would be invisible otherwise.
    """

    spec = A.ARMS[arm]
    if spec["train_mode"] is None:
        raise ValueError("%s is not a trained arm" % arm)
    device = torch.device(device_name)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    plan = C.load_plan()
    train_items = C.checked_train_items(plan, spec["train_mode"])
    if train_items_override is not None:
        train_items = list(train_items_override)
    elif limit_train:
        train_items = train_items[:limit_train]
    store = FEAT.FeatureStore(os.path.join(C.CACHE_DIR, "train"))
    by_id = {it["item_id"]: i for i, it in enumerate(store.items)}
    idx = [by_id[it["item_id"]] for it in train_items]
    if len(idx) != len(train_items):
        raise AssertionError("cache holds %d of the %d training items" % (len(idx), len(train_items)))
    labels = C.LABELS
    targets = {i: FEAT.target_index(store.items[i], labels) for i in idx}
    subset = [store.items[i] for i in idx]

    shipped = shipped_override if shipped_override is not None else C.load_agent(device_name).model
    model = A.build_arm_model(shipped, arm, seed).to(device)
    acc = A.freeze_for_training(model)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    # batches_by_length indexes the *subset*; every use below maps back through `idx`, so the
    # collate call and the target lookup address the same store rows the batch was built from
    groups = batches_by_length(subset, token_budget, max_batch, seed)
    steps_per_epoch = len(groups)
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(warmup_frac * total_steps))

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr * (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * lr * (1 + math.cos(math.pi * p))

    history: List[Dict[str, Any]] = []
    verified_batches = 0
    step = 0
    t0 = time.time()
    for epoch in range(epochs):
        ep_loss, ep_correct, ep_n = 0.0, 0, 0
        for group in groups:
            rows = [idx[g] for g in group]
            b = FEAT.collate(store, rows, device)
            tgt = torch.tensor([targets[i] for i in rows], dtype=torch.long, device=device)
            # The pairing invariant, checked on every batch rather than argued: the store rows the
            # features came from must be the plan items the targets were derived from. Without it a
            # silent index-space mismatch trains the head against the wrong labels and every number
            # downstream is meaningless while looking entirely normal.
            for g, meta in zip(group, b["meta"]):
                if meta["item_id"] != train_items[g]["item_id"]:
                    raise AssertionError(
                        "batch row %r does not correspond to the plan item its target came from (%r)"
                        % (meta["item_id"], train_items[g]["item_id"]))
            verified_batches += 1
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits, _act = model(None, b["attention_mask"], b["marker_pos"], b["marker_mask"],
                                     b["qtype"], state_start=b["state_start"], encoder_hidden=b["h"])
            loss = F.cross_entropy(logits.float(), tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step == 0:
                grads = [n for n, p in model.named_parameters()
                         if p.grad is not None and n.startswith("encoder.")]
                if grads:
                    raise AssertionError("encoder received gradients: %s" % grads[:3])
                encoder_grad = None
                for p in model.encoder.parameters():
                    if p.grad is not None:
                        encoder_grad = p.grad.abs().sum().item()
                        break
                if encoder_grad not in (None, 0.0):
                    raise AssertionError("encoder grad is %r after one step" % encoder_grad)
            with torch.no_grad():
                ep_loss += float(loss) * len(group)
                ep_correct += int((logits.float().argmax(-1) == tgt).sum())
                ep_n += len(group)
            if log_every and step % log_every == 0:
                print("  [%s s%d] step %4d/%d lr %.2e loss %.4f" % (arm, seed, step, total_steps,
                                                                    lr_at(step), float(loss)), flush=True)
            step += 1
        row = {"epoch": epoch, "step": step, "train_loss": ep_loss / ep_n,
               "train_accuracy": ep_correct / ep_n, "lr": lr_at(max(0, step - 1)),
               "seconds": round(time.time() - t0, 1)}
        history.append(row)
        print("  [%s s%d] epoch %d  train_loss=%.4f train_acc=%.4f  (%.0fs)"
              % (arm, seed, epoch, row["train_loss"], row["train_accuracy"], row["seconds"]), flush=True)

    out_dir = os.path.join(RUNS_DIR, arm, "seed%d" % seed)
    os.makedirs(out_dir, exist_ok=True)
    sd = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save({"arm": arm, "seed": seed, "head_kind": spec["kind"],
                "head_nhead": spec.get("nhead"), "state_dict": sd,
                "trainable_names": sorted(sd)}, os.path.join(out_dir, "head.pt"))
    peak_vram = (torch.cuda.max_memory_allocated(device) / 1e9) if device.type == "cuda" else None
    result = {
        "arm": arm, "seed": seed, "epochs": epochs, "lr": lr, "weight_decay": weight_decay,
        "token_budget": token_budget, "max_batch": max_batch, "warmup_steps": warmup,
        "steps": step, "steps_per_epoch": steps_per_epoch,
        "n_train_items": len(idx), "train_mode": spec["train_mode"],
        "batches_with_verified_item_target_pairing": verified_batches,
        "parameter_accounting": acc,
        "train_seconds": round(time.time() - t0, 1),
        "peak_vram_gb": peak_vram,
        "history": history,
        "head_state_dict_path": "experiments/h5-adapter/runs/%s/seed%d/head.pt" % (arm, seed),
        "git": C.git_state(),
        "environment": C.environment(device),
        "pool_hashes": C.pool_hashes(),
    }
    with open(os.path.join(out_dir, "training.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
        fh.write("\n")
    print("  [%s s%d] wrote %s (peak VRAM %.2f GB)" % (arm, seed, out_dir, peak_vram or 0.0))
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=A.TRAINED_ARMS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--token-budget", type=int, default=12288)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-train", type=int, default=None,
                    help="smoke test only: train on the first N items")
    a = ap.parse_args()
    train_arm(a.arm, a.seed, epochs=a.epochs, lr=a.lr, token_budget=a.token_budget,
              max_batch=a.max_batch, device_name=a.device, limit_train=a.limit_train)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

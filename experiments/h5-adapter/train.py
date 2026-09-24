"""Train one H5 arm for one seed, on cached frozen-encoder features.

    env/venv/bin/python experiments/h5-adapter/train.py --arm arm3r_residual --seed 0 \\
        --epochs 40 --lr 5e-4 [--lr-cross 5e-3]

Writes `experiments/h5-adapter/runs/<arm>/seed<k>/{head.pt,training.json}`. The head state dict is
~60 MB fp32, is **gitignored**, and `run_arm3r.py` deletes it immediately after the eval that needs
it -- the filesystem reached 100 % full during the previous run of this experiment. `training.json`,
which carries the per-epoch curve and the parameter accounting, is the kept artifact.

Design notes that matter for the comparison:

* **Same optimiser, schedule, epoch count, batch token budget and data for every trained arm.**
  Only the head module differs -- plus, for an arm that carries a zero-initialised parallel branch,
  an optional separate rate for that branch (`--lr-cross`), reported in the run record.
* **Gradients never touch the encoder**: the cached `h` was produced under `inference_mode`, and
  the encoder's parameters have `requires_grad=False`, so a backward that reached it would raise
  rather than silently fine-tune it. `train.py` additionally asserts the encoder's grad is `None`
  after the first step.
* Head forward/backward runs under bf16 autocast with fp32 master weights, which is how the
  shipped head was trained (`amp_dtype: bf16` in the checkpoint config). bf16 needs no loss
  scaling, so there is no GradScaler to get wrong.
* **The added branch is measured every epoch**, not assumed: one fixed batch is scored with the
  branch as trained and again with its output projection zeroed, and the max logit difference is
  recorded. It is exactly `0` at step 0 (that is the identity the arm rests on) and has to grow for
  the arm to be evidence at all -- an arm whose new module never moves is invalid, not null
  (`protocol.md` §10).
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


def probe_cell_accuracy(model, store: "FEAT.FeatureStore", conditions: List[Dict[str, Any]],
                        device, cell: str, token_budget: int = 16384,
                        max_batch: int = 8) -> Optional[Dict[str, Any]]:
    """Accuracy on one evaluation cell, from the cache, with the head in eval mode.

    Training loss on a cosine schedule always flattens as the rate goes to zero, so it cannot by
    itself answer "did this converge?". This probes held-out **accuracy** at the cell the comparison
    is about, at a few points in the schedule. It is a probe, not a selection rule: the schedule and
    the stopping point are fixed in advance for both arms (`findings/E-004.md` §4), and nothing here
    feeds back into training.
    """
    want = [c for c in conditions if c["cell"] == cell]
    by_id = {it["item_id"]: i for i, it in enumerate(store.items)}
    keep = [c for c in want if c["item_id"] in by_id]
    if not keep:
        return None
    idx = [by_id[c["item_id"]] for c in keep]
    lengths = [store.items[i]["length"] for i in idx]
    was_training = model.training
    model.eval()
    correct = 0
    with torch.inference_mode():
        for group in FEAT._token_budget_batches(lengths, token_budget, max_batch):
            real = [idx[i] for i in group]
            b = FEAT.collate(store, real, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits, _act = model(None, b["attention_mask"], b["marker_pos"], b["marker_mask"],
                                     b["qtype"], state_start=b["state_start"], encoder_hidden=b["h"])
            logits = logits.float()
            for j, gi in enumerate(group):
                slot = int(torch.argmax(logits[j][b["marker_mask"][j]]).item())
                correct += int(slot == FEAT.target_index(keep[gi], C.LABELS))
    if was_training:
        model.train()
    return {"cell": cell, "n": len(keep), "accuracy": correct / len(keep)}


def train_arm(arm: str, seed: int, epochs: int = 8, lr: float = 1e-4, weight_decay: float = 0.01,
              token_budget: int = 12288, max_batch: int = 8, warmup_frac: float = 0.05,
              device_name: str = "cuda", limit_train: Optional[int] = None,
              log_every: int = 50, shipped_override=None,
              train_items_override: Optional[List[Dict[str, Any]]] = None,
              lr_cross: Optional[float] = None, probe_every_epoch: bool = True,
              eval_probe_cell: Optional[str] = None, eval_probe_every: int = 8) -> Dict[str, Any]:
    """``shipped_override`` and ``train_items_override`` let the offline selftest drive this loop
    with a tiny encoder and a stratified item subset, so the loop under test is this loop and not a
    copy of it. A subset whose positions are not its store row indices is the point: the arms train
    on stratified blocks, and an identity assumption between the two would be invisible otherwise.

    ``lr_cross`` gives the parallel cross-attention branch its own learning rate (default: the same
    as everything else). A zero-initialised branch starts from a linear read-out of random features
    and has to travel further than a fine-tuned module does, so a higher rate for it is a *recipe*
    choice that has to be reported, not a silent one. ``probe_every_epoch`` measures the branch's
    contribution to the logits once per epoch by re-running one fixed batch with the branch's output
    projection zeroed: an arm whose new module never moves is reported as invalid (protocol.md
    §10), and this is the number that decides it.
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
    # One optimiser, one schedule, one recipe -- except that the zero-initialised parallel branch
    # may carry its own rate. Every parameter is in exactly one group, so `lr_at` scales both the
    # same way and the ratio between them is constant for the whole run.
    cross_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and n.startswith("cross.")]
    rest_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("cross.")]
    groups_def = [{"params": rest_params, "lr": lr}]
    if cross_params:
        groups_def.append({"params": cross_params, "lr": lr_cross if lr_cross else lr})
    opt = torch.optim.AdamW(groups_def, lr=lr, weight_decay=weight_decay)
    base_lrs = [g["lr"] for g in opt.param_groups]
    # batches_by_length indexes the *subset*; every use below maps back through `idx`, so the
    # collate call and the target lookup address the same store rows the batch was built from
    groups = batches_by_length(subset, token_budget, max_batch, seed)
    steps_per_epoch = len(groups)
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(warmup_frac * total_steps))

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    def probe_branch_contribution(batch: Dict[str, Any]) -> Optional[float]:
        """max |logit difference| between the branch as trained and the branch with its output
        projection zeroed, on one fixed batch. Exactly 0 at step 0 by construction; non-zero and
        growing means the added stage is doing something. Returns None for arms without a branch."""
        if model.cross is None:
            return None
        was_training = model.training
        model.eval()
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                with_branch, _ = model(None, batch["attention_mask"], batch["marker_pos"],
                                       batch["marker_mask"], batch["qtype"],
                                       state_start=batch["state_start"], encoder_hidden=batch["h"])
            w = model.cross.out_proj.weight.detach().clone()
            b = model.cross.out_proj.bias.detach().clone()
            model.cross.zero_init_out_proj()
            without, _ = model(None, batch["attention_mask"], batch["marker_pos"],
                               batch["marker_mask"], batch["qtype"],
                               state_start=batch["state_start"], encoder_hidden=batch["h"])
            with torch.no_grad():
                model.cross.out_proj.weight.copy_(w)
                model.cross.out_proj.bias.copy_(b)
        if was_training:
            model.train()
        return float((with_branch.float() - without.float()).abs().max())

    def branch_norms() -> Optional[Dict[str, float]]:
        if model.cross is None:
            return None
        return {
            "out_proj_weight_fro": float(model.cross.out_proj.weight.detach().float().norm()),
            "out_proj_bias_norm": float(model.cross.out_proj.bias.detach().float().norm()),
            "branch_in_proj_weight_fro": float(
                model.cross.branch.layers[0].cross_attn.in_proj_weight.detach().float().norm()),
        }

    # the held-out accuracy probe (see probe_cell_accuracy): loaded once, only if asked for
    probe_store = None
    probe_conditions: List[Dict[str, Any]] = []
    if eval_probe_cell:
        try:
            probe_store = FEAT.FeatureStore(os.path.join(C.CACHE_DIR, "eval"))
            probe_conditions = FEAT.build_eval_conditions(
                plan, pool=C.load_needle_pool("needles-h5-eval-v1.json"))
        except Exception as e:  # never let the probe take the run down with it
            print("  [%s s%d] eval probe unavailable: %s" % (arm, seed, e), flush=True)
            probe_store, probe_conditions = None, []

    history: List[Dict[str, Any]] = []
    verified_batches = 0
    step = 0
    t0 = time.time()
    step0_grads: Optional[Dict[str, float]] = None
    probe_batch = None
    for epoch in range(epochs):
        ep_loss, ep_correct, ep_n = 0.0, 0, 0
        for group in groups:
            rows = [idx[g] for g in group]
            b = FEAT.collate(store, rows, device)
            if probe_batch is None and probe_every_epoch:
                probe_batch = {k: v for k, v in b.items() if k != "meta"}
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
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = base * lr_at(step)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits, _act = model(None, b["attention_mask"], b["marker_pos"], b["marker_mask"],
                                     b["qtype"], state_start=b["state_start"], encoder_hidden=b["h"])
            loss = F.cross_entropy(logits.float(), tgt)
            if not torch.isfinite(loss):
                # A non-finite loss is a dead run, and at these rates it is the failure mode that
                # looks like "the arm did not learn" from the epoch curve alone. Stop loudly.
                raise AssertionError("%s seed %d: non-finite loss at step %d (%r)"
                                     % (arm, seed, step, float(loss)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
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
                step0_grads = {
                    "out_proj_weight_grad_sum": (
                        float(model.cross.out_proj.weight.grad.abs().sum())
                        if model.cross is not None and model.cross.out_proj.weight.grad is not None
                        else None),
                    "branch_in_proj_grad_sum": (
                        float(model.cross.branch.layers[0].cross_attn.in_proj_weight.grad.abs().sum())
                        if model.cross is not None
                        and model.cross.branch.layers[0].cross_attn.in_proj_weight.grad is not None
                        else None),
                    "encoder_grad_sum": encoder_grad,
                }
            opt.step()
            with torch.no_grad():
                ep_loss += float(loss) * len(group)
                ep_correct += int((logits.float().argmax(-1) == tgt).sum())
                ep_n += len(group)
            if log_every and step % log_every == 0:
                msg = "  [%s s%d] step %4d/%d lr %.2e loss %.4f" % (
                    arm, seed, step, total_steps, opt.param_groups[0]["lr"], float(loss))
                print(msg, flush=True)
            step += 1
        row = {"epoch": epoch, "step": step, "train_loss": ep_loss / ep_n,
               "train_accuracy": ep_correct / ep_n,
               "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - t0, 1)}
        if probe_batch is not None:
            row["branch_logit_contribution_max"] = probe_branch_contribution(probe_batch)
        norms = branch_norms()
        if norms is not None:
            row.update(norms)
        if probe_store is not None and probe_conditions and (
                (epoch + 1) % max(1, eval_probe_every) == 0 or epoch == epochs - 1):
            try:
                row["eval_probe"] = probe_cell_accuracy(model, probe_store, probe_conditions,
                                                        device, eval_probe_cell)
            except Exception as e:
                row["eval_probe_error"] = str(e)
        history.append(row)
        extra_note = ""
        if row.get("branch_logit_contribution_max") is not None:
            extra_note = "  branch|dlogit|=%.4g  ||W||=%.3g" % (
                row["branch_logit_contribution_max"], row["out_proj_weight_fro"])
        if row.get("eval_probe"):
            extra_note += "  eval[%s]=%.3f (n=%d)" % (row["eval_probe"]["cell"],
                                                      row["eval_probe"]["accuracy"],
                                                      row["eval_probe"]["n"])
        print("  [%s s%d] epoch %d  train_loss=%.4f train_acc=%.4f  (%.0fs)%s"
              % (arm, seed, epoch, row["train_loss"], row["train_accuracy"], row["seconds"],
                 extra_note), flush=True)

    out_dir = os.path.join(RUNS_DIR, arm, "seed%d" % seed)
    os.makedirs(out_dir, exist_ok=True)
    sd = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save({"arm": arm, "seed": seed, "head_kind": spec["kind"],
                "head_nhead": spec.get("nhead"), "state_dict": sd,
                "trainable_names": sorted(sd)}, os.path.join(out_dir, "head.pt"))
    peak_vram = (torch.cuda.max_memory_allocated(device) / 1e9) if device.type == "cuda" else None
    result = {
        "arm": arm, "seed": seed, "epochs": epochs, "lr": lr, "lr_cross": lr_cross,
        "weight_decay": weight_decay,
        "eval_probe_cell": eval_probe_cell, "eval_probe_every": eval_probe_every,
        "token_budget": token_budget, "max_batch": max_batch, "warmup_steps": warmup,
        "steps": step, "steps_per_epoch": steps_per_epoch,
        "n_train_items": len(idx), "train_mode": spec["train_mode"],
        "batches_with_verified_item_target_pairing": verified_batches,
        "step0_gradients": step0_grads,
        "final_branch_norms": branch_norms(),
        "branch_contribution_last_epoch": (
            history[-1].get("branch_logit_contribution_max") if history else None),
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
    ap.add_argument("--arm", required=True, choices=A.ALL_TRAINED_ARMS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-cross", type=float, default=None,
                    help="learning rate for the parallel cross-attention branch (arm3r_residual); "
                         "default: same as --lr")
    ap.add_argument("--token-budget", type=int, default=12288)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-probe-cell", default=None,
                    help="held-out cell to score every --eval-probe-every epochs (probe only; the "
                         "schedule and stopping point are fixed in advance for every arm)")
    ap.add_argument("--eval-probe-every", type=int, default=8)
    ap.add_argument("--limit-train", type=int, default=None,
                    help="smoke test only: train on the first N items")
    a = ap.parse_args()
    train_arm(a.arm, a.seed, epochs=a.epochs, lr=a.lr, token_budget=a.token_budget,
              max_batch=a.max_batch, device_name=a.device, limit_train=a.limit_train,
              lr_cross=a.lr_cross, eval_probe_cell=a.eval_probe_cell,
              eval_probe_every=a.eval_probe_every)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

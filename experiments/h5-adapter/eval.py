"""Evaluate one H5 arm over every evaluation cell, and write raw per-item predictions.

    env/venv/bin/python experiments/h5-adapter/eval.py --arm arm1_frozen
    env/venv/bin/python experiments/h5-adapter/eval.py --arm arm3_xattn --seed 0

Rows land in `experiments/h5-adapter/predictions/<arm>-seed<k>.jsonl`, one per (cell, item), with
the gold label, the predicted label, the marker slot chosen, the full probability vector and the
latency of the forward that produced it. Every statistic in the finding is recomputed from these
files by `stats.py`; nothing is aggregated here.

Arm 1 runs the **shipped model end to end** (encoder plus its own head), because that is the
baseline the protocol names. Trained arms run the head over cached encoder features. The two paths
are reconciled by `arm2_step0`: arm 2's architecture at its shipped initialisation, evaluated from
the cache, must reproduce arm 1. That is what makes "the cache is the same representation the
shipped path computes" a measurement instead of an assumption.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import arms as A  # noqa: E402
import common_h5 as C  # noqa: E402
import features as FEAT  # noqa: E402

PRED_DIR = os.path.join(HERE, "predictions")
RUNS_DIR = os.path.join(HERE, "runs")
EVAL_TOKEN_BUDGET = 32768
EVAL_MAX_BATCH = 8


def _row(condition: Dict[str, Any], store_item: Dict[str, Any], probs: List[float],
         latency_s: float, arm: str, seed: Any, dtype_note: str = "") -> Dict[str, Any]:
    labels = C.LABELS
    slot = int(np.argmax(probs))
    prediction = FEAT.slot_to_label(condition, labels, slot)
    gold_slot = FEAT.target_index(condition, labels)
    p = float(probs[slot])
    row = {
        "arm": arm,
        "seed": seed,
        "cell": condition["cell"],
        "item_id": store_item["item_id"],
        "label": condition["label"],
        "template_id": condition["template_id"],
        "pad": condition["pad"],
        "needle_position": condition["needle_position"],
        "needle_token_start": store_item.get("needle_token_start"),
        "filler_tokens_before": store_item.get("filler_tokens_before"),
        "input_tokens": store_item.get("length"),
        "state_tokens_kept": store_item.get("state_tokens_kept"),
        "truncated": store_item.get("truncated"),
        "needle_kept": store_item.get("needle_kept"),
        "option_order": condition.get("option_order"),
        "slot_predicted": slot,
        "slot_gold": gold_slot,
        "prediction": prediction,
        "correct": bool(prediction == condition["label"]),
        # the swapped-needle control asks a different question: does the answer follow the text that
        # is actually in the document? `label_if_needle_read` is that text's class.
        "label_if_needle_read": condition.get("label_if_needle_read"),
        "follows_needle": (None if condition.get("label_if_needle_read") is None
                           else bool(prediction == condition["label_if_needle_read"])),
        "control": condition.get("control"),
        "probability": p,
        "probabilities": [round(float(x), 6) for x in probs],
        "options": list(labels),
        "latency_s": latency_s,
        "latency_scope": "full_model_encoder_and_head" if arm == "arm1_frozen" else "head_only_over_cached_features",
        "dtype_note": dtype_note,
    }
    return row


def evaluate_arm1(agent, tok, builder, conditions: List[Dict[str, Any]], device,
                  warmup_items: int = 3, dtype_note: str = "shipped fp16 weights, fp16 autocast") -> List[Dict[str, Any]]:
    """The shipped model, end to end, over every condition."""
    model = agent.model
    model.eval()
    # len per condition from the builder's estimate; documents are materialised one batch at a
    # time so the full set is never resident
    lengths = [builder.estimate_length(c) for c in conditions]
    rows: List[Dict[str, Any]] = []
    done_warmup = 0
    with torch.inference_mode():
        for group in FEAT._token_budget_batches(lengths, EVAL_TOKEN_BUDGET, EVAL_MAX_BATCH):
            L = max(lengths[i] for i in group)
            b = len(group)
            built = {i: (conditions[i], builder.build(conditions[i],
                                                      option_order=conditions[i].get("option_order")))
                     for i in group}
            ids = torch.zeros(b, L, dtype=torch.long)
            att = torch.zeros(b, L, dtype=torch.long)
            kmax = max(len(built[i][1]["markers"]) for i in group)
            mpos = torch.zeros(b, kmax, dtype=torch.long)
            mmask = torch.zeros(b, kmax, dtype=torch.bool)
            for j, i in enumerate(group):
                seq, mk = built[i][1]["input_ids"], built[i][1]["markers"]
                ids[j, :len(seq)] = torch.tensor(seq, dtype=torch.long)
                att[j, :len(seq)] = 1
                mpos[j, :len(mk)] = torch.tensor(mk, dtype=torch.long)
                mmask[j, :len(mk)] = True
            ids, att, mpos, mmask = ids.to(device), att.to(device), mpos.to(device), mmask.to(device)
            qtype = torch.zeros(b, dtype=torch.long, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits, _act = model(ids, att, mpos, mmask, qtype)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / b
            logits = logits.float()
            for j, i in enumerate(group):
                warm = done_warmup < warmup_items
                done_warmup += 1
                probs = torch.softmax(logits[j][mmask[j]], -1).cpu().numpy().tolist()
                rows.append(_row(conditions[i], {"item_id": conditions[i]["item_id"],
                                                 "length": lengths[i],
                                                 "template_id": conditions[i]["template_id"],
                                                 "needle_token_start": built[i][1]["needle_token_start"],
                                                 "filler_tokens_before": built[i][1]["filler_tokens_before"],
                                                 "state_tokens_kept": built[i][1]["state_tokens_kept"],
                                                 "truncated": built[i][1]["truncated"],
                                                 "needle_kept": built[i][1]["needle_kept"]},
                                 probs, 0.0 if warm else dt, "arm1_frozen", None, dtype_note))
            del built
    return rows


def evaluate_from_cache(model, store: FEAT.FeatureStore, conditions: List[Dict[str, Any]],
                        device, arm: str, seed: Any, warmup_items: int = 3,
                        dtype_note: str = "head over cached fp16 encoder features, bf16 autocast",
                        with_latency: bool = False) -> List[Dict[str, Any]]:
    model.eval()
    by_id = {it["item_id"]: i for i, it in enumerate(store.items)}
    idx = [by_id[c["item_id"]] for c in conditions]
    lengths = [store.items[i]["length"] for i in idx]
    rows: List[Dict[str, Any]] = []
    done_warmup = 0
    order = list(FEAT._token_budget_batches(lengths, EVAL_TOKEN_BUDGET, EVAL_MAX_BATCH))
    with torch.inference_mode():
        for group in order:
            real = [idx[i] for i in group]
            batch = FEAT.collate(store, real, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits, _act = model(None, batch["attention_mask"], batch["marker_pos"],
                                     batch["marker_mask"], batch["qtype"],
                                     state_start=batch["state_start"],
                                     encoder_hidden=batch["h"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / len(group)
            logits = logits.float()
            for j, gi in enumerate(group):
                cond = conditions[gi]
                warm = done_warmup < warmup_items
                done_warmup += 1
                probs = torch.softmax(logits[j][batch["marker_mask"][j]], -1).cpu().numpy().tolist()
                rows.append(_row(cond, store.items[real[j]], probs,
                                 0.0 if (warm or not with_latency) else dt, arm, seed, dtype_note))
    return rows


def write_rows(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_trained(model, arm: str, seed: int, device) -> Dict[str, Any]:
    p = os.path.join(RUNS_DIR, arm, "seed%d" % seed, "head.pt")
    if not os.path.exists(p):
        raise FileNotFoundError("no trained head at %s -- run train.py first" % p)
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    own = model.state_dict()
    for k, v in ckpt["state_dict"].items():
        if k not in own:
            raise KeyError("trained head has %r, which the model does not" % k)
        own[k] = v
    model.load_state_dict(own)
    return ckpt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["arm1_frozen", "arm2_step0"] + A.TRAINED_ARMS)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke test only")
    ap.add_argument("--latency", action="store_true",
                    help="record head latency in the rows (off by default: timing rows are noise)")
    a = ap.parse_args()
    device = torch.device(a.device)
    plan = C.load_plan()
    conditions = FEAT.build_eval_conditions(plan, pool=C.load_needle_pool("needles-h5-eval-v1.json"))
    if a.limit:
        conditions = conditions[:a.limit]
    out = a.out or (
        os.path.join(PRED_DIR, "%s.jsonl" % a.arm) if a.arm == "arm1_frozen"
        else os.path.join(PRED_DIR, "%s-seed%d.jsonl" % (a.arm, a.seed)))

    if a.arm == "arm1_frozen":
        agent = C.load_agent(a.device)
        tok = agent.tok
        pool = C.load_needle_pool("needles-h5-eval-v1.json")
        builder = C.make_builder(tok, C.load_filler(), pool)
        rows = evaluate_arm1(agent, tok, builder, conditions, device)
    else:
        store = FEAT.FeatureStore(os.path.join(C.CACHE_DIR, "eval"))
        agent = C.load_agent(a.device)
        if a.arm == "arm2_step0":
            model = A.build_arm_model(agent.model, "arm2_shipped_init", seed=0).to(device)
            A.freeze_for_training(model)
            arm_name, seed = "arm2_step0", 0
        else:
            if a.seed is None:
                raise SystemExit("--seed is required for a trained arm")
            model = A.build_arm_model(agent.model, a.arm, a.seed).to(device)
            A.freeze_for_training(model)
            meta = load_trained(model, a.arm, a.seed, device)
            arm_name, seed = a.arm, a.seed
            print("  loaded %s (%s)" % (meta["arm"], meta["state_dict"] and "state_dict ok"))
        rows = evaluate_from_cache(model, store, conditions, device, arm_name, seed,
                                   with_latency=a.latency)
    write_rows(out, rows)
    by_cell: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_cell.setdefault(r["cell"], []).append(r)
    print("wrote %s  n=%d" % (os.path.relpath(out, os.path.dirname(HERE)), len(rows)))
    for cell in sorted(by_cell):
        rs = by_cell[cell]
        acc = sum(1 for r in rs if r["correct"]) / len(rs)
        lats = [r["latency_s"] for r in rs if r["latency_s"]]
        lat = " p50=%.3fs p95=%.3fs" % (statistics.median(lats),
                                        sorted(lats)[int(0.95 * (len(lats) - 1))]) if lats else ""
        print("  %-18s n=%3d acc=%.3f%s" % (cell, len(rs), acc, lat))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

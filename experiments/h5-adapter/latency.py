"""Measure the actual cost of each arm: end-to-end warm latency and peak VRAM.

    env/venv/bin/python experiments/h5-adapter/latency.py

protocol.md s5 asks for p50/p95 latency, peak VRAM and tokens/sec, and s8 asks for >=3 discarded
warm-up passes before any timing and for the GPU lock to be recorded as held.

Six rows per arm: the two-ended anchors of the length axis at the needle-at-END position
(L=0, L=4000, L=7000) plus L=4000 with the needle at the START, so the aggregation path's cost is
visible at the lengths the arms are actually compared at. Timings run the **full** model --
encoder and head -- because that is what a caller pays; the head-only marginal cost is reported
separately from the cache path.
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

import torch  # noqa: E402

import arms as A  # noqa: E402
import common_h5 as C  # noqa: E402
import features as FEAT  # noqa: E402

SAMPLED_CELLS = ("L0", "L4000-p000", "L4000-p100", "L7000-p100")


def time_arm(arm: str, seed: Optional[int], conditions: List[Dict[str, Any]], builder, agent,
             device, n: int = 30, warmup: int = 3) -> Dict[str, Any]:
    if arm == "arm1_frozen":
        model = agent.model
    else:
        model = A.build_arm_model(agent.model, arm, seed if seed is not None else 0).to(device)
        A.freeze_for_training(model)
        if seed is not None:
            import eval as E
            E.load_trained(model, arm, seed, device)
    model.eval()

    by_cell: Dict[str, List[Dict[str, Any]]] = {}
    for c in conditions:
        by_cell.setdefault(c["cell"], []).append(c)
    out: Dict[str, Any] = {"arm": arm, "seed": seed, "warmup_discarded": warmup,
                           "cells": {}, "per_cell": {}}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for cell in SAMPLED_CELLS:
        sample = by_cell.get(cell, [])[:n]
        if not sample:
            continue
        built = [builder.build(c, option_order=c.get("option_order")) for c in sample]
        lats: List[float] = []
        toks = 0
        for k, doc in enumerate(built):
            seq, mk = doc["input_ids"], doc["markers"]
            ids = torch.tensor([seq], dtype=torch.long, device=device)
            att = torch.ones_like(ids)
            mpos = torch.tensor([mk], dtype=torch.long, device=device)
            mmask = torch.ones_like(mpos, dtype=torch.bool)
            qtype = torch.zeros(1, dtype=torch.long, device=device)
            sstart = torch.tensor([doc["state_start"]], dtype=torch.long, device=device)
            kw = {"state_start": sstart} if arm != "arm1_frozen" else {}
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                                 enabled=device.type == "cuda"):
                model(ids, att, mpos, mmask, qtype, **kw)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if k >= warmup:
                lats.append(dt)
                toks += len(seq)
        lats_sorted = sorted(lats)
        out["per_cell"][cell] = {
            "n": len(lats),
            "input_tokens_median": int(statistics.median([len(d["input_ids"]) for d in built])),
            "p50_latency_ms": 1000.0 * statistics.median(lats),
            "p95_latency_ms": 1000.0 * lats_sorted[min(len(lats_sorted) - 1, int(0.95 * len(lats_sorted)))],
            "mean_latency_ms": 1000.0 * statistics.fmean(lats),
            "tokens_per_second": (toks / sum(lats)) if lats else None,
            "batched": False,
        }
    if device.type == "cuda":
        out["peak_vram_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
        out["peak_vram_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 1e9
    return out


def head_only_cost(conditions: List[Dict[str, Any]], store: FEAT.FeatureStore, agent, device,
                   arm: str = "arm3_xattn", n: int = 60) -> Dict[str, Any]:
    """The marginal cost of the aggregation path: head forward over cached features.

    Reported separately precisely because it is *not* the shippable number -- it excludes the
    encoder, which every arm pays identically.
    """
    out: Dict[str, Any] = {}
    for kind, nhead in (("self_attention", None), ("cross_attention", 4)):
        model = A.DecisionModel(agent.model.encoder, head_layers=2, n_act=2, dropout=0.1,
                                head_kind=kind, head_nhead=nhead).to(device)
        model.eval()
        by_id = {it["item_id"]: i for i, it in enumerate(store.items)}
        # the needle-at-END cell at 4000 tokens: the regime under test
        sample = [c for c in conditions if c["cell"] == "L4000-p100"][:n]
        idx = [by_id[c["item_id"]] for c in sample]
        for it in idx:                      # warm-up, discarded
            b = FEAT.collate(store, [it], device)
            with torch.no_grad():
                model(None, b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"],
                      state_start=b["state_start"], encoder_hidden=b["h"])
        lats = []
        for it in idx:
            b = FEAT.collate(store, [it], device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                 enabled=device.type == "cuda"):
                model(None, b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"],
                      state_start=b["state_start"], encoder_hidden=b["h"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            lats.append(time.perf_counter() - t0)
        out[kind] = {
            "n": len(lats),
            "p50_latency_ms": 1000.0 * statistics.median(lats),
            "p95_latency_ms": 1000.0 * sorted(lats)[min(len(lats) - 1, int(0.95 * len(lats)))],
            "note": "head only, over cached encoder features; excludes the encoder every arm pays",
        }
        del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(HERE, "latency.json"))
    ap.add_argument("--arms", default="arm1_frozen," + ",".join(A.TRAINED_ARMS))
    ap.add_argument("--gpu-lock-held", action="store_true",
                    help="set by the launcher when it ran under flock; recorded, not inferred")
    a = ap.parse_args()
    device = torch.device(a.device)
    seeds = [int(x) for x in a.seeds.split(",") if x != ""]

    plan = C.load_plan()
    conditions = FEAT.build_eval_conditions(plan)
    agent = C.load_agent(a.device)
    builder = C.make_builder(agent.tok, C.load_filler(), C.load_needle_pool("needles-h5-eval-v1.json"))

    result: Dict[str, Any] = {
        "artifact": "experiments/h5-adapter/latency.json",
        "reproduce": "flock -n .gpu.lock -c 'env/venv/bin/python experiments/h5-adapter/latency.py --gpu-lock-held'",
        "gpu_lock_held": bool(a.gpu_lock_held),
        "environment": C.environment(device),
        "git": C.git_state(),
        "warmup_discarded": 3,
        "protocol_note": ("protocol.md s8: timings taken with the GPU lock held, single process, "
                          "after >=3 discarded warm-up passes. `gpu_lock_held` records whether the "
                          "launcher actually held it; it is not inferred from the numbers."),
        "arms": {},
    }
    for arm in a.arms.split(","):
        if arm == "arm1_frozen":
            print("timing %s ..." % arm, flush=True)
            result["arms"][arm] = time_arm(arm, None, conditions, builder, agent, device, n=a.n)
        else:
            for seed in seeds:
                print("timing %s seed %d ..." % (arm, seed), flush=True)
                result["arms"]["%s|seed%d" % (arm, seed)] = time_arm(
                    arm, seed, conditions, builder, agent, device, n=a.n)
    store_path = os.path.join(C.CACHE_DIR, "eval")
    if os.path.exists(os.path.join(store_path, "index.json")):
        store = FEAT.FeatureStore(store_path)
        print("head-only cost ...", flush=True)
        result["head_only"] = head_only_cost(conditions, store, agent, device)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
        fh.write("\n")
    print("wrote %s" % os.path.relpath(a.out, os.path.dirname(HERE)))
    for arm, rows in result["arms"].items():
        for cell, st in rows["per_cell"].items():
            print("  %-24s %-12s p50=%7.1fms p95=%7.1fms %6.0f tok/s"
                  % (arm, cell, st["p50_latency_ms"], st["p95_latency_ms"], st["tokens_per_second"]))
    if "head_only" in result:
        for kind, st in result["head_only"].items():
            print("  head-only %-16s p50=%7.1fms n=%d" % (kind, st["p50_latency_ms"], st["n"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

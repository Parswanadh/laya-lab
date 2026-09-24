"""Interleaved cost measurement: baseline vs w512 vs all-global at pad=7000.

The sweep's per-arm latencies are single medians taken in sequence inside one process; two arms
that are *deterministic duplicates* (allglobal and allglobal_w512_combo, bit-identical outputs)
differed by 4.6% in median latency, so a single non-interleaved ratio cannot support a cost claim.
This script measures the same three arms **interleaved** (round-robin over repetitions), so drift
in clock/thermals/allocator affects all arms equally, and reports the paired ratio per repetition.

    env/venv/bin/python experiments/h2h3-knobs/cost_interleave.py --pad 7000 --reps 3 --per-rep 5
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(LAB, "fork"))
sys.path.insert(0, os.path.join(LAB, "experiments", "harness"))

import torch  # noqa: E402

import laya  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pad", type=int, default=7000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--per-rep", type=int, default=5, help="items per arm per repetition")
    ap.add_argument("--arms", default="baseline,w512,allglobal")
    ap.add_argument("--out", default=os.path.join(HERE, "cost_interleave.json"))
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("e002run", os.path.join(HERE, "run.py"))
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    p1 = R.load_p1_module()
    import pools
    pool = pools.load_pool("upstream_multilingual")
    items = pools.build_items(pool, 20, 20260924)["items"][: a.per_rep]

    agent = laya.load(os.path.join(LAB, "models", "multilingual"), device="cuda")
    agent.dtype = torch.float32
    enc = agent.model.encoder
    orig = R.snapshot_config(enc)
    texts = [R.build_doc(p1, agent.tok, "p1_exact", a.pad, it["text"], pool["filler_unit"])
             for it in items]
    arms = [x for x in a.arms.split(",") if x]

    # warm every arm once (kernel selection per mask shape)
    for arm in arms:
        R.apply_arm(enc, R.ARM_SPECS[arm], orig)
        agent.predict({"text": texts[0]}, pool["question"], max_len=8192)

    samples = {arm: [] for arm in arms}
    per_rep = {arm: [] for arm in arms}
    order = []
    for rep in range(a.reps):
        for arm in arms:
            R.apply_arm(enc, R.ARM_SPECS[arm], orig)
            torch.cuda.reset_peak_memory_stats()
            ts = []
            for text in texts:
                t0 = time.perf_counter()
                agent.predict({"text": text}, pool["question"], max_len=8192)
                ts.append(time.perf_counter() - t0)
            samples[arm].extend(ts)
            per_rep[arm].append(round(statistics.median(ts), 4))
            order.append({"rep": rep, "arm": arm, "median_s": round(statistics.median(ts), 4),
                          "peak_vram_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1)})
            print("  rep=%d arm=%-10s median=%.4fs  peak_vram=%.0f MiB"
                  % (rep, arm, statistics.median(ts), order[-1]["peak_vram_reserved_mib"]), flush=True)

    out = {"pad": a.pad, "reps": a.reps, "per_rep": a.per_rep, "arms": arms,
           "n_per_arm": len(items) * a.reps, "order": order,
           "arm_median_s": {arm: round(statistics.median(samples[arm]), 4) for arm in arms},
           "arm_min_s": {arm: round(min(samples[arm]), 4) for arm in arms},
           "arm_median_of_rep_medians_s": {arm: round(statistics.median(per_rep[arm]), 4)
                                           for arm in arms},
           "per_rep_medians": per_rep}
    base = out["arm_median_s"].get("baseline")
    if base:
        out["ratio_vs_baseline"] = {arm: round(out["arm_median_s"][arm] / base, 3) for arm in arms}
    # spread of *identical-work* repeats: the largest per-rep median spread across arms that must
    # be identical is the practical noise floor for a cost claim
    if len(arms) > 1:
        spreads = []
        for rep in range(a.reps):
            vals = [per_rep[arm][rep] for arm in arms]
            spreads.append(round((max(vals) - min(vals)) / min(vals), 3))
        out["within_rep_arm_spread"] = spreads
    print(json.dumps({k: out[k] for k in ("arm_median_s", "ratio_vs_baseline",
                                          "arm_median_of_rep_medians_s", "within_rep_arm_spread")},
                     indent=1), flush=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote %s" % os.path.basename(a.out))


if __name__ == "__main__":
    main()

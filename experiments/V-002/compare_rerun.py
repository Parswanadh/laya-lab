"""V-002 verification E: compare a fresh end-to-end re-run of an eval arm against the stored JSONL.

Usage:
    env/venv/bin/python experiments/V-002/compare_rerun.py \
        --stored experiments/h5-adapter/predictions/arm1_frozen.jsonl \
        --rerun  experiments/V-002/repro_arm1_frozen.jsonl \
        --out    experiments/V-002/compare_arm1.json

Reports, per cell: accuracy in each file, the per-item prediction agreement rate, the largest and
mean absolute probability difference, and the McNemar p the *re-run* would produce against arm1 --
so a re-run that changes the verdict is visible rather than averaged away.

Run: see above.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List


def load(path: str) -> Dict[str, Dict[str, Any]]:
    return {json.loads(ln)["item_id"]: json.loads(ln)
            for ln in open(path, encoding="utf-8") if ln.strip()}


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stored", required=True)
    ap.add_argument("--rerun", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S, R = load(a.stored), load(a.rerun)
    shared = sorted(set(S) & set(R))
    out: Dict[str, Any] = {
        "stored": a.stored, "rerun": a.rerun,
        "n_stored": len(S), "n_rerun": len(R), "n_shared": len(shared),
        "only_stored": sorted(set(S) - set(R))[:5], "only_rerun": sorted(set(R) - set(S))[:5],
    }
    pred_mismatch = [k for k in shared if S[k]["prediction"] != R[k]["prediction"]]
    correct_mismatch = [k for k in shared if bool(S[k]["correct"]) != bool(R[k]["correct"])]
    slot_mismatch = [k for k in shared if S[k]["slot_predicted"] != R[k]["slot_predicted"]]
    label_mismatch = [k for k in shared if S[k]["label"] != R[k]["label"]]
    max_pd = 0.0
    sum_pd = 0.0
    for k in shared:
        for x, y in zip(S[k]["probabilities"], R[k]["probabilities"]):
            d = abs(x - y)
            max_pd = max(max_pd, d)
            sum_pd += d
    out.update({
        "prediction_mismatches": len(pred_mismatch),
        "slot_mismatches": len(slot_mismatch),
        "correct_mismatches": len(correct_mismatch),
        "label_mismatches": len(label_mismatch),
        "examples": [{"item_id": k, "stored": S[k]["prediction"], "rerun": R[k]["prediction"],
                      "stored_probs": S[k]["probabilities"], "rerun_probs": R[k]["probabilities"]}
                     for k in pred_mismatch[:5]],
        "max_abs_prob_diff": max_pd,
        "mean_abs_prob_diff": sum_pd / (len(shared) * 4) if shared else None,
        "per_cell": {},
    })
    cells = sorted({S[k]["cell"] for k in shared})
    for cell in cells:
        ks = [k for k in shared if S[k]["cell"] == cell]
        s_acc = sum(1 for k in ks if S[k]["correct"]) / len(ks)
        r_acc = sum(1 for k in ks if R[k]["correct"]) / len(ks)
        b = sum(1 for k in ks if R[k]["correct"] and not S[k]["correct"])
        c = sum(1 for k in ks if (not R[k]["correct"]) and S[k]["correct"])
        out["per_cell"][cell] = {
            "n": len(ks), "stored_accuracy": s_acc, "rerun_accuracy": r_acc,
            "agreement_rate": sum(1 for k in ks if S[k]["prediction"] == R[k]["prediction"]) / len(ks),
            "b_rerun_only_right": b, "c_stored_only_right": c,
            "mcnemar_p_rerun_vs_stored": mcnemar(b, c),
        }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    print("stored n=%d  rerun n=%d  shared=%d" % (len(S), len(R), len(shared)))
    print("prediction mismatches=%d  slot=%d  correct=%d  label=%d  max|dp|=%.6f mean|dp|=%.2e"
          % (out["prediction_mismatches"], out["slot_mismatches"], out["correct_mismatches"],
             out["label_mismatches"], out["max_abs_prob_diff"], out["mean_abs_prob_diff"]))
    for cell, v in out["per_cell"].items():
        print("  %-20s n=%3d stored=%.3f rerun=%.3f agree=%.3f b=%d c=%d p=%.3g"
              % (cell, v["n"], v["stored_accuracy"], v["rerun_accuracy"], v["agreement_rate"],
                 v["b_rerun_only_right"], v["c_stored_only_right"], v["mcnemar_p_rerun_vs_stored"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

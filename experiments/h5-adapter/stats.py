"""Recompute every reported number from the raw per-item prediction files.

    env/venv/bin/python experiments/h5-adapter/stats.py

Reads `experiments/h5-adapter/predictions/*.jsonl` and writes
`experiments/h5-adapter/summary.json` plus a printed table. Nothing here runs a model.

The statistics primitives come from `experiments/harness/metrics.py` -- the program's canonical
implementation -- so an H5 number is computed the same way every other arm's number is. Only the
row schema differs (this experiment's rows carry `pad` rather than `pad_tokens`), so the
grouping-based helpers are called with this experiment's field names. The metrics module's sha256
is recorded in the output, because a number is only reproducible against the revision that made it.

Reported for every cell, as protocol.md requires:
  random (1/n_options), majority class, and the **position-only oracle** -- an arm that does not
  beat the position-only oracle has learned nothing about content and is reported in those words.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(LAB, "experiments", "harness"))
sys.path.insert(0, HERE)

import metrics as M  # noqa: E402

PRED_DIR = os.path.join(HERE, "predictions")
BOOT = 10000

# The comparison family for Holm-Bonferroni. Baselines only -- arm4 is a *candidate variant* of
# arm3, not a control, so putting it in the baseline list would both duplicate the arm3/arm4
# comparison and inflate the family the correction is applied over.
BASELINES = ("arm1_frozen", "arm2_shipped_init", "arm2_random_init")
CANDIDATE_FAMILY = ("arm3_xattn", "arm4_xattn_long")
PRIMARY_CELL = "L4000-p100"


def load_predictions(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def discover(pred_dir: str) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
    """(arm, seed) -> rows, from the per-file naming convention."""
    out: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for path in sorted(glob.glob(os.path.join(pred_dir, "*.jsonl"))):
        base = os.path.basename(path)[:-len(".jsonl")]
        if base == "arm1_frozen":
            arm, seed = "arm1_frozen", -1
        elif base == "arm2_step0":
            arm, seed = "arm2_step0", 0
        else:
            arm, _, s = base.rpartition("-seed")
            seed = int(s)
        out[(arm, seed)] = load_predictions(path)
    return out


def ece(rows: Sequence[Dict[str, Any]], bins: int = 15) -> float:
    import numpy as np
    conf = np.array([r["probability"] for r in rows], dtype=float)
    corr = np.array([1.0 if r["correct"] else 0.0 for r in rows], dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (conf >= lo if i == 0 else conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - corr[sel].mean())
    return float(e)


def fitted_temperature(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """One-parameter temperature fitted on this cell's own rows, by grid search on NLL.

    Reported beside the raw ECE because the shipped temperatures are unfitted (protocol.md s5) --
    and because a fitted-on-the-evaluated-cell temperature is descriptive, not a calibration claim.
    """
    import numpy as np
    probs = np.array([r["probabilities"] for r in rows], dtype=float)
    gold = np.array([r["slot_gold"] for r in rows], dtype=int)
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    best_t, best_nll = 1.0, float("inf")
    for t in np.concatenate([np.linspace(0.25, 1.0, 31), np.linspace(1.05, 5.0, 80)]):
        z = logits / t
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        nll = float(-np.log(np.clip(p[np.arange(len(gold)), gold], 1e-12, 1.0)).mean())
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return {"fitted_temperature": best_t, "nll_at_fitted": best_nll,
            "nll_at_raw": float(-np.log(np.clip(probs[np.arange(len(gold)), gold], 1e-12, 1.0)).mean())}


CONTROL_CELLS = ("L4000-p100-ablated", "L4000-p100-swapped", "L4000-p100-perm")


def control_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The mechanism controls, which need their own reading.

    * **ablated** (the needle replaced by a sentence with no class cue): accuracy must be chance.
      Anything above chance means the arm was scoring the haystack, the prompt or the position.
    * **swapped** (the needle replaced by one from another class): accuracy against the *original*
      label must collapse, and ``follows_needle_accuracy`` -- accuracy against the class of the
      text actually present -- must be high. That pair is the sharpest available evidence that the
      answer tracks the document's content.
    * **perm** (same items, options reordered): if accuracy holds, the answer is not an option-index
      artefact.
    """
    out: Dict[str, Any] = {"n": len(rows), "accuracy": M.accuracy(rows) if rows else None}
    fn = [bool(r["follows_needle"]) for r in rows if r.get("follows_needle") is not None]
    if fn:
        out["follows_needle_n"] = len(fn)
        out["follows_needle_accuracy"] = sum(fn) / len(fn)
    preds = [r["prediction"] for r in rows]
    out["predicted_distribution"] = {lb: preds.count(lb) for lb in sorted(set(preds))}
    out["modal_prediction"] = max(sorted(set(preds)), key=lambda p: preds.count(p)) if preds else None
    out["random"] = 1.0 / len(rows[0]["options"]) if rows else None
    return out


def cell_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    correct = [bool(r["correct"]) for r in rows]
    labels = [r["label"] for r in rows]
    lo, hi = M.bootstrap_ci(correct, n_boot=BOOT, seed=12345)
    preds = [r["prediction"] for r in rows]
    modal = max(sorted(set(preds)), key=lambda p: preds.count(p)) if preds else None
    return {
        "n": len(rows),
        "correct": sum(correct),
        "accuracy": M.accuracy(rows),
        "macro_f1": M.macro_f1(rows),
        "accuracy_ci95_low": lo,
        "accuracy_ci95_high": hi,
        "ci_method": "percentile bootstrap over items, n_boot=%d, seed=12345" % BOOT,
        "majority_class_accuracy": M.majority_class_accuracy(labels),
        "random": 1.0 / len(rows[0]["options"]) if rows else None,
        "label_distribution": {lb: labels.count(lb) for lb in sorted(set(labels))},
        "modal_prediction": modal,
        "modal_share": (preds.count(modal) / len(preds)) if preds else None,
        "ece_raw": ece(rows),
        **fitted_temperature(rows),
    }


def paired(a: Sequence[Dict[str, Any]], b: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    ca, cb = M.pair_rows(a, b, key_fields=("item_id",))
    if not ca:
        return None
    out = M.mcnemar_exact(ca, cb)
    out["accuracy_a"] = sum(ca) / len(ca)
    out["accuracy_b"] = sum(cb) / len(cb)
    out["effect_size_pp"] = 100.0 * (out["accuracy_a"] - out["accuracy_b"])
    return out


def holm(pvals: Sequence[Tuple[str, float]]) -> Dict[str, float]:
    """Holm-Bonferroni adjusted p-values for a family of ``(label, p)`` pairs."""
    fam = sorted(pvals, key=lambda kv: kv[1])
    m = len(fam)
    adj: Dict[str, float] = {}
    running = 0.0
    for i, (label, p) in enumerate(fam):
        val = min(1.0, (m - i) * p)
        running = max(running, val)
        adj[label] = running
    return adj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default=PRED_DIR)
    ap.add_argument("--out", default=os.path.join(HERE, "summary.json"))
    a = ap.parse_args()

    with open(os.path.join(LAB, "experiments", "harness", "metrics.py"), "rb") as fh:
        metrics_sha = hashlib.sha256(fh.read()).hexdigest()

    runs = discover(a.pred_dir)
    if not runs:
        print("no predictions found in %s" % a.pred_dir)
        return 1
    arms = sorted({k[0] for k in runs})
    seeds = sorted({k[1] for k in runs if k[1] >= 0})
    cells = sorted({r["cell"] for k, rows in runs.items() for r in rows})

    # ---- per (arm, seed, cell) ------------------------------------------------------------
    per_seed: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for (arm, seed), rows in sorted(runs.items()):
        by_cell: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            by_cell.setdefault(r["cell"], []).append(r)
        per_seed.setdefault(arm, {})["seed%d" % seed] = {
            cell: cell_stats(rs) for cell, rs in sorted(by_cell.items())
        }

    # ---- seed spread and mean over seeds --------------------------------------------------
    train_seeds = [s for s in seeds if s >= 0]
    spread: Dict[str, Dict[str, Any]] = {}
    for arm in arms:
        keys = [k for k in per_seed.get(arm, {}) if k.startswith("seed") and k != "seed-1"]
        if not keys:
            keys = list(per_seed.get(arm, {}))
        spread[arm] = {}
        for cell in cells:
            vals = [per_seed[arm][k][cell]["accuracy"] for k in keys if cell in per_seed[arm][k]]
            if not vals:
                continue
            spread[arm][cell] = {
                "seeds": [k for k in keys if cell in per_seed[arm][k]],
                "n_seeds": len(vals),
                "accuracy_mean": sum(vals) / len(vals),
                "accuracy_min": min(vals),
                "accuracy_max": max(vals),
                "seed_spread_pp": 100.0 * (max(vals) - min(vals)),
                "accuracy_per_seed": vals,
            }

    # ---- position-only oracle, computed over the pooled rows of every arm ----------------
    # Independent of which arm produced the rows: it is a property of the item set.
    pooled: Dict[str, List[Dict[str, Any]]] = {}
    for (arm, seed), rows in runs.items():
        for r in rows:
            pooled.setdefault(r["cell"], []).append(r)
    oracle = {}
    for cell, rows in sorted(pooled.items()):
        seen, uniq = set(), []
        for r in rows:
            if r["item_id"] in seen:
                continue
            seen.add(r["item_id"])
            uniq.append(r)
        oracle[cell] = M.position_only_oracle(uniq, group_fields=("pad", "needle_position"))

    # ---- paired tests, per seed and pooled -------------------------------------------------
    comparisons: List[Dict[str, Any]] = []
    pvals: List[Tuple[str, float]] = []
    for cand in CANDIDATE_FAMILY:
        for base in BASELINES:
            if (cand, train_seeds[0] if train_seeds else -1) not in runs:
                continue
            for cell in cells:
                for seed in (train_seeds or [-1]):
                    ka = (cand, seed)
                    kb = (base, -1) if base == "arm1_frozen" else (base, seed)
                    if ka not in runs or kb not in runs:
                        continue
                    ra = [r for r in runs[ka] if r["cell"] == cell]
                    rb = [r for r in runs[kb] if r["cell"] == cell]
                    st = paired(ra, rb)
                    if st is None:
                        continue
                    label = "%s_vs_%s|%s|seed%d" % (cand, base, cell, seed)
                    rec = {"candidate": cand, "baseline": base, "cell": cell, "seed": seed, **st}
                    comparisons.append(rec)
                    pvals.append((label, st["p_value_two_sided_exact"]))
    adj = holm(pvals)
    for rec in comparisons:
        label = "%s_vs_%s|%s|seed%d" % (rec["candidate"], rec["baseline"], rec["cell"], rec["seed"])
        rec["p_holm"] = adj[label]
        rec["survives_holm_0.05"] = adj[label] < 0.05
        rec["sign"] = "candidate_better" if rec["effect_size_pp"] > 0 else (
            "baseline_better" if rec["effect_size_pp"] < 0 else "tie")

    # ---- seed sign agreement: a gain inside the seed spread is not a gain -----------------
    sign_summary: Dict[str, Any] = {}
    for cand in CANDIDATE_FAMILY:
        for base in BASELINES:
            for cell in cells:
                recs = [c for c in comparisons
                        if c["candidate"] == cand and c["baseline"] == base and c["cell"] == cell]
                if not recs:
                    continue
                better = sum(1 for c in recs if c["effect_size_pp"] > 0)
                sig = sum(1 for c in recs if c["p_holm"] < 0.05 and c["effect_size_pp"] > 0)
                key = "%s_vs_%s|%s" % (cand, base, cell)
                sign_summary[key] = {
                    "n_seeds": len(recs),
                    "seeds_candidate_better": better,
                    "seeds_candidate_worse": sum(1 for c in recs if c["effect_size_pp"] < 0),
                    "seeds_surviving_holm_0.05_in_candidate_favour": sig,
                    "min_effect_pp": min(c["effect_size_pp"] for c in recs),
                    "max_effect_pp": max(c["effect_size_pp"] for c in recs),
                }

    # ---- headline verdict ----------------------------------------------------------------
    def spread_of(arm: str, cell: str) -> Optional[Dict[str, Any]]:
        return spread.get(arm, {}).get(cell)

    verdict = {}
    for cell in (PRIMARY_CELL, "L7000-p100", "L4000-p000", "L0"):
        a3 = spread_of("arm3_xattn", cell)
        a2s = spread_of("arm2_shipped_init", cell)
        a2r = spread_of("arm2_random_init", cell)
        a1 = spread_of("arm1_frozen", cell)
        a4 = spread_of("arm4_xattn_long", cell)
        row: Dict[str, Any] = {
            "cell": cell,
            "position_only_oracle": oracle.get(cell, {}).get("oracle_accuracy"),
            "arm1_frozen": a1 and a1["accuracy_mean"],
            "arm2_shipped_init": a2s and a2s["accuracy_mean"],
            "arm2_random_init": a2r and a2r["accuracy_mean"],
            "arm3_xattn": a3 and a3["accuracy_mean"],
            "arm4_xattn_long": a4 and a4["accuracy_mean"],
        }
        if a3 and a2s:
            row["arm3_minus_arm2_shipped_pp"] = 100.0 * (a3["accuracy_mean"] - a2s["accuracy_mean"])
            row["inside_arm3_seed_spread"] = abs(
                a3["accuracy_mean"] - a2s["accuracy_mean"]) <= (a3["seed_spread_pp"] / 100.0)
            row["inside_arm2_seed_spread"] = abs(
                a3["accuracy_mean"] - a2s["accuracy_mean"]) <= (a2s["seed_spread_pp"] / 100.0)
        if a3 and a2r:
            row["arm3_minus_arm2_random_pp"] = 100.0 * (a3["accuracy_mean"] - a2r["accuracy_mean"])
        orc = row["position_only_oracle"]
        if a3 and orc is not None:
            row["arm3_beats_position_only_oracle"] = a3["accuracy_mean"] > orc + 1e-9
        verdict[cell] = row

    out = {
        "artifact": "experiments/h5-adapter/summary.json",
        "reproduce": "env/venv/bin/python experiments/h5-adapter/stats.py",
        "prediction_files": sorted(os.path.basename(p) for p in glob.glob(
            os.path.join(a.pred_dir, "*.jsonl"))),
        "metrics_module_sha256": metrics_sha,
        "metrics_module_path": "experiments/harness/metrics.py",
        "arms": arms,
        "seeds": seeds,
        "cells": cells,
        "per_arm_seed_cell": per_seed,
        "seed_spread": spread,
        "position_only_oracle": oracle,
        "mechanism_controls": {
            cell: {arm: control_stats([r for r in runs[k] if r["cell"] == cell])
                   for (arm, _seed), rws in sorted(runs.items())
                   for k in [(arm, _seed)] if any(r["cell"] == cell for r in rws)}
            for cell in CONTROL_CELLS
        },
        "comparisons": comparisons,
        "seed_sign_agreement": sign_summary,
        "holm_family_size": len(pvals),
        "verdict": verdict,
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")

    # ---- print ---------------------------------------------------------------------------
    print("metrics.py sha256 %s" % metrics_sha[:16])
    print("\nper-cell accuracy (mean over seeds; oracle = position-only)")
    hdr = "%-18s %5s %6s | %8s %8s %8s %8s %8s | %7s" % (
        "cell", "n", "oracle", "arm1", "arm2_shp", "arm2_rnd", "arm3_xa", "arm4_lng", "3-2shp")
    print(hdr)
    print("-" * len(hdr))
    for cell in cells:
        orc = oracle.get(cell, {}).get("oracle_accuracy")
        vals = []
        for arm in ("arm1_frozen", "arm2_shipped_init", "arm2_random_init", "arm3_xattn",
                    "arm4_xattn_long"):
            s = spread.get(arm, {}).get(cell)
            vals.append("%8.3f" % s["accuracy_mean"] if s else "       -")
        n = next((per_seed[arm]["seed%d" % (train_seeds[0] if train_seeds else 0)][cell]["n"]
                  for arm in arms
                  if "seed%d" % (train_seeds[0] if train_seeds else 0) in per_seed.get(arm, {})
                  and cell in per_seed[arm]["seed%d" % (train_seeds[0] if train_seeds else 0)]), 0)
        d = verdict.get(cell, {}).get("arm3_minus_arm2_shipped_pp")
        print("%-18s %5d %6.3f | %s | %+7.2f" % (cell, n, orc if orc is not None else float("nan"),
                                                 " ".join(vals), d if d is not None else float("nan")))
    print("\nseed spreads (pp)")
    for arm in arms:
        if arm == "arm1_frozen":
            continue
        for cell in cells:
            s = spread.get(arm, {}).get(cell)
            if s and s["n_seeds"] > 1:
                print("  %-18s %-18s mean=%.3f spread=%.2fpp %s"
                      % (arm, cell, s["accuracy_mean"], s["seed_spread_pp"],
                         ["%.3f" % v for v in s["accuracy_per_seed"]]))
    print("\nmechanism controls (the needle is absent in `ablated`, replaced by another class in "
          "`swapped`, options reordered in `perm`)")
    for cell in CONTROL_CELLS:
        block = out["mechanism_controls"].get(cell, {})
        if not block:
            continue
        print("  %s" % cell)
        for arm in sorted(block):
            st = block[arm]
            extra = (" follows_needle=%.3f (n=%d)" % (st["follows_needle_accuracy"],
                                                      st["follows_needle_n"])
                     if st.get("follows_needle_accuracy") is not None else "")
            print("    %-22s n=%3d acc=%.3f random=%.3f%s"
                  % (arm, st["n"], st["accuracy"], st["random"], extra))

    print("\nverdict cells")
    for cell, row in verdict.items():
        print("  %s: %s" % (cell, json.dumps(row, sort_keys=True)))
    print("\nwrote %s" % os.path.relpath(a.out, LAB))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

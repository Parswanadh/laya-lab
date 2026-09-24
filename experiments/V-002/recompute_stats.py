"""V-002 verification C: recompute every arm1-vs-arm2 statistic from the raw JSONL predictions.

No model, no GPU. Written from scratch by the verifier: McNemar exact, the percentile bootstrap,
Holm-Bonferroni and the position-only oracle are all re-implemented here rather than imported, so
agreement with `stats.py`/`summary.json` is evidence and not a shared-code artefact.

Reported per cell per arm:
  n, correct, accuracy, Wilson CI, seeded percentile bootstrap CI (10,000 resamples),
  gold label distribution, majority-class accuracy, modal prediction + its share,
  accuracy above the position-only oracle.

Paired arm2-vs-arm1 per cell:
  discordant counts (b=arm2 right & arm1 wrong, c=arm2 wrong & arm1 right),
  exact two-sided McNemar p, effect in pp, odds ratio,
  the same table computed as if the arms were *independent* (two-proportion z / Fisher) to show
  how much the wrong test changes the answer,
  Holm-adjusted p over three families: the 8 protocol cells, all 11 cells, and 9 cells.

Pairing integrity:
  same item_id -> same gold label, same template, same pad/position, same document state hash,
  same input length, same needle token start; no duplicate ids; equal id sets across arms.

Run: env/venv/bin/python experiments/V-002/recompute_stats.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(os.path.dirname(HERE))
H5 = os.path.join(LAB, "experiments", "h5-adapter")

PRED_DIR = os.path.join(H5, "predictions")
CACHE_INDEX = os.path.join(H5, "cache", "eval", "index.json")
BOOT = 10000
BOOT_SEED = 12345
PROTOCOL_CELLS = ("L0", "L4000-p000", "L4000-p025", "L4000-p050", "L4000-p075",
                  "L4000-p100", "L7000-p000", "L7000-p100")
CONTROL_CELLS = ("L4000-p100-perm", "L4000-p100-ablated", "L4000-p100-swapped")


def load_rows(path: str) -> List[Dict[str, Any]]:
    return [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]


def wilson(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def boot_ci(correct: Sequence[bool], n_boot: int = BOOT, seed: int = BOOT_SEED) -> Tuple[float, float]:
    """Percentile bootstrap over items (my own implementation, same seed convention as stats.py)."""
    n = len(correct)
    if n == 0:
        return float("nan"), float("nan")
    rng = random.Random("%d|boot|%d" % (seed, n))
    vals = []
    for _ in range(n_boot):
        s = 0
        for _ in range(n):
            s += 1 if correct[rng.randrange(n)] else 0
        vals.append(s / n)
    vals.sort()
    lo = vals[max(0, int(0.025 * n_boot) - 1)]
    hi = vals[min(n_boot - 1, int(0.975 * n_boot))]
    return lo, hi


def mcnemar_exact(b: int, c: int) -> Dict[str, Any]:
    """Exact two-sided McNemar from the two discordant counts.

    b = arm2 right / arm1 wrong, c = arm2 wrong / arm1 right. Conditional on b+c discordant
    pairs, b ~ Binomial(b+c, 0.5) under H0; the two-sided exact p is 2*P(X <= min(b,c)), capped
    at 1. (This is the same statistic as stats.py's `mcnemar_exact`; re-derived here.)
    """
    n = b + c
    if n == 0:
        return {"n_discordant": 0, "p_value_two_sided_exact": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p = min(1.0, 2 * tail)
    # mid-p and the chi-square version for contrast
    chi2 = ((abs(b - c) - 1) ** 2 / n) if n else 0.0
    return {"n_discordant": n, "p_value_two_sided_exact": p,
            "p_midp": max(0.0, min(1.0, tail - math.comb(n, k) / (2 ** n))),
            "chi2_continuity_corrected": chi2,
            "odds_ratio_b_over_c": (b / c) if c else float("inf")}


def two_proportion(k1: int, n1: int, k2: int, n2: int) -> Dict[str, Any]:
    """Independent two-proportion z test with pooled variance -- the WRONG test for this design,
    computed only to quantify how much it changes the answer."""
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return {"z": float("nan"), "p_two_sided": float("nan")}
    z = (p1 - p2) / se
    pv = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return {"z": z, "p_two_sided": pv, "diff_pp": 100 * (p1 - p2)}


def holm(pairs: Sequence[Tuple[str, float]]) -> Dict[str, float]:
    """Holm-Bonferroni step-down adjusted p-values (my own implementation)."""
    fam = sorted(pairs, key=lambda kv: kv[1])
    m = len(fam)
    adj: Dict[str, float] = {}
    running = 0.0
    for i, (label, p) in enumerate(fam):
        val = min(1.0, (m - i) * p)
        running = max(running, val)
        adj[label] = running
    return adj


def oracle(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Position-only oracle by my own definition: best constant-label accuracy inside each
    (pad, needle_position) group, and over the whole set for the strongest single constant."""
    groups: Dict[Tuple[Any, ...], Dict[str, int]] = {}
    for r in rows:
        g = (r["pad"], r["needle_position"])
        groups.setdefault(g, {})
        groups[g][r["label"]] = groups[g].get(r["label"], 0) + 1
    total = hits = 0
    per = {}
    for g, c in sorted(groups.items(), key=lambda kv: str(kv[0])):
        best = max(c.values())
        n = sum(c.values())
        total += n
        hits += best
        per["%s|%s" % g] = {"n": n, "best": best, "oracle_accuracy": best / n, "counts": c}
    overall_counts: Dict[str, int] = {}
    for r in rows:
        overall_counts[r["label"]] = overall_counts.get(r["label"], 0) + 1
    return {"oracle_accuracy": hits / total if total else None, "n": total, "per_group": per,
            "single_constant_best": (max(overall_counts.values()) / total) if total else None,
            "overall_label_counts": overall_counts}


def main() -> int:
    files = sorted(f for f in os.listdir(PRED_DIR) if f.endswith(".jsonl"))
    runs: Dict[str, List[Dict[str, Any]]] = {f: load_rows(os.path.join(PRED_DIR, f)) for f in files}
    out: Dict[str, Any] = {
        "artifact": "experiments/V-002/verify_stats.json",
        "reproduce": "env/venv/bin/python experiments/V-002/recompute_stats.py",
        "source_files": {f: {"n_rows": len(r), "sha256": hashlib.sha256(
            open(os.path.join(PRED_DIR, f), "rb").read()).hexdigest()} for f, r in runs.items()},
        "bootstrap": {"n_boot": BOOT, "seed": BOOT_SEED, "method": "percentile over items"},
    }

    # ---- pairing integrity -------------------------------------------------------------------
    cache = json.load(open(CACHE_INDEX, encoding="utf-8"))
    state_by_id = {it["item_id"]: it.get("state_sha256") for it in cache["items"]}
    len_by_id = {it["item_id"]: it["length"] for it in cache["items"]}
    arm1 = runs.get("arm1_frozen.jsonl")
    arm2 = runs.get("arm2_shipped_init-seed0.jsonl")
    if arm1 is None or arm2 is None:
        print("missing arm1/arm2 prediction files")
        return 1
    a1 = {r["item_id"]: r for r in arm1}
    a2 = {r["item_id"]: r for r in arm2}
    pair = {
        "n_arm1": len(arm1), "n_arm2": len(arm2),
        "n_unique_arm1": len(a1), "n_unique_arm2": len(a2),
        "ids_only_in_arm1": sorted(set(a1) - set(a2))[:5],
        "ids_only_in_arm2": sorted(set(a2) - set(a1))[:5],
        "n_shared": len(set(a1) & set(a2)),
    }
    fields = ("label", "template_id", "pad", "needle_position", "slot_gold", "option_order")
    diff = {f: [k for k in a1 if k in a2 and a1[k].get(f) != a2[k].get(f)] for f in fields}
    pair["field_diffs"] = {f: {"n": len(v), "examples": v[:3]} for f, v in diff.items()}
    meta = ("input_tokens", "needle_token_start", "state_tokens_kept", "truncated", "needle_kept")
    diffm = {f: [k for k in a1 if k in a2 and a1[k].get(f) != a2[k].get(f)] for f in meta}
    pair["doc_metadata_diffs"] = {f: {"n": len(v), "examples": v[:3]} for f, v in diffm.items()}
    # document identity: one cache state hash per item id, and the gold label agrees with the plan
    plan = json.load(open(os.path.join(H5, "plan.json"), encoding="utf-8"))
    plan_by_id = {it["item_id"]: it for it in plan["eval_items"]}
    n_state_missing = sum(1 for k in a1 if k not in state_by_id)
    n_plan_missing = sum(1 for k in a1 if k not in plan_by_id)
    label_vs_plan = sum(1 for k in a1 if k in plan_by_id and a1[k]["label"] != plan_by_id[k]["label"])
    # a state hash is unique per (needle, pad, position, request text) -- count collisions
    state_counts: Dict[str, int] = {}
    for k, v in state_by_id.items():
        state_counts[v] = state_counts.get(v, 0) + 1
    pair.update({
        "cache_state_hashes": {
            "n_items_in_cache": len(state_by_id),
            "n_items_in_arm_files_missing_from_cache": n_state_missing,
            "n_items_missing_from_plan": n_plan_missing,
            "n_label_mismatch_vs_plan": label_vs_plan,
            "n_distinct_state_hashes": len(set(state_by_id.values())),
            "n_state_hashes_shared_by_multiple_items": sum(1 for v in state_counts.values() if v > 1),
            "note": ("state_sha256 is the sha256 of the exact token-id list the encoder was given; "
                     "a shared value across two item ids would mean the same document was scored "
                     "under two labels"),
        },
        "length_match_vs_cache": sum(1 for k in a1 if k in len_by_id
                                     and a1[k].get("input_tokens") == len_by_id[k]),
    })
    # per-cell: do the paired arms see the same documents? (arm1 stores the same metadata)
    out["pairing"] = pair

    # ---- per-cell stats --------------------------------------------------------------------
    arms = {"arm1_frozen": arm1, "arm2_shipped_init": arm2}
    if "arm3_xattn-seed0.jsonl" in runs:
        arms["arm3_xattn"] = runs["arm3_xattn-seed0.jsonl"]
    cells = [c for c in PROTOCOL_CELLS + CONTROL_CELLS
             if any(r["cell"] == c for r in arm1)]
    per_cell: Dict[str, Dict[str, Any]] = {}
    for cell in cells:
        block: Dict[str, Any] = {}
        for arm, rows in arms.items():
            rs = [r for r in rows if r["cell"] == cell]
            if not rs:
                continue
            correct = [bool(r["correct"]) for r in rs]
            k = sum(correct)
            n = len(rs)
            labels: Dict[str, int] = {}
            preds: Dict[str, int] = {}
            for r in rs:
                labels[r["label"]] = labels.get(r["label"], 0) + 1
                preds[r["prediction"]] = preds.get(r["prediction"], 0) + 1
            modal = max(sorted(preds), key=lambda p: preds[p])
            lo, hi = boot_ci(correct)
            wlo, whi = wilson(k, n)
            # macro F1 over the gold labels present
            f1s = []
            for lb in sorted(labels):
                tp = sum(1 for r in rs if r["prediction"] == lb and r["label"] == lb)
                fp = sum(1 for r in rs if r["prediction"] == lb and r["label"] != lb)
                fn = sum(1 for r in rs if r["prediction"] != lb and r["label"] == lb)
                f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
            block[arm] = {
                "n": n, "correct": k, "accuracy": k / n,
                "boot_ci95": [lo, hi], "wilson_ci95": [wlo, whi],
                "macro_f1": sum(f1s) / len(f1s),
                "label_distribution": dict(sorted(labels.items())),
                "majority_class_accuracy": max(labels.values()) / n,
                "prediction_distribution": dict(sorted(preds.items())),
                "modal_prediction": modal, "modal_share": preds[modal] / n,
                "random": 1.0 / len(rs[0]["options"]),
            }
        per_cell[cell] = block
    out["per_cell"] = per_cell

    # ---- position-only oracle (recomputed from label fields only) ----------------------------
    out["position_oracle"] = {
        "whole_eval_set_arm1_rows": oracle(arm1),
        "by_cell": {},
    }
    for cell in cells:
        rs = [r for r in arm1 if r["cell"] == cell]
        out["position_oracle"]["by_cell"][cell] = {
            "n": len(rs),
            "oracle_accuracy_by_pad_position": oracle(rs)["oracle_accuracy"],
            "single_constant_best": oracle(rs)["single_constant_best"],
            "n_distinct_groups": len(oracle(rs)["per_group"]),
        }

    # ---- paired arm2 vs arm1 ----------------------------------------------------------------
    comparisons: List[Dict[str, Any]] = []
    for cell in cells:
        X = [r for r in arm2 if r["cell"] == cell]
        Y = {r["item_id"]: r for r in arm1 if r["cell"] == cell}
        pairs = [(r, Y[r["item_id"]]) for r in X if r["item_id"] in Y]
        b = sum(1 for x, y in pairs if x["correct"] and not y["correct"])
        c = sum(1 for x, y in pairs if (not x["correct"]) and y["correct"])
        n = len(pairs)
        k2 = sum(1 for x, _ in pairs if x["correct"])
        k1 = sum(1 for _, y in pairs if y["correct"])
        mc = mcnemar_exact(b, c)
        ind = two_proportion(k2, n, k1, n)
        comparisons.append({
            "cell": cell, "n_pairs": n,
            "arm2_correct": k2, "arm1_correct": k1,
            "arm2_accuracy": k2 / n, "arm1_accuracy": k1 / n,
            "effect_pp": 100.0 * (k2 - k1) / n,
            "b_arm2_only_right": b, "c_arm1_only_right": c,
            "mcnemar": mc,
            "independent_two_proportion": ind,
            "p_independent_over_p_paired": (ind["p_two_sided"] / mc["p_value_two_sided_exact"]
                                            if mc["p_value_two_sided_exact"] else None),
        })
    # Holm over three families
    fams = {
        "protocol_8": [c for c in comparisons if c["cell"] in PROTOCOL_CELLS],
        "all_11": comparisons,
        "scored_9_incl_perm": [c for c in comparisons
                               if c["cell"] in PROTOCOL_CELLS + ("L4000-p100-perm",)],
    }
    for name, fam in fams.items():
        adj = holm([(c["cell"], c["mcnemar"]["p_value_two_sided_exact"]) for c in fam])
        for c in fam:
            c.setdefault("holm", {})[name] = {
                "p_holm": adj[c["cell"]], "survives_0.05": adj[c["cell"]] < 0.05,
                "family_size": len(fam),
            }
    out["arm2_vs_arm1"] = comparisons

    # ---- does any scored cell beat the oracle / collapse to a constant? ----------------------
    attacks = {}
    for cell in cells:
        blk = per_cell[cell]
        orc = out["position_oracle"]["by_cell"][cell]["oracle_accuracy_by_pad_position"]
        attacks[cell] = {
            "oracle": orc,
            "per_arm": {a: {"accuracy": v["accuracy"], "beats_oracle": v["accuracy"] > orc + 1e-9,
                            "modal_share": v["modal_share"], "modal_prediction": v["modal_prediction"],
                            "majority_class_accuracy": v["majority_class_accuracy"]}
                        for a, v in blk.items()},
        }
    out["attacks"] = attacks

    with open(os.path.join(HERE, "verify_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")

    # ---- print -----------------------------------------------------------------------------
    print("pairing: n_arm1=%d n_arm2=%d shared=%d" % (pair["n_arm1"], pair["n_arm2"], pair["n_shared"]))
    print("  field diffs: %s" % json.dumps({k: v["n"] for k, v in pair["field_diffs"].items()}))
    print("  doc metadata diffs: %s" % json.dumps({k: v["n"] for k, v in pair["doc_metadata_diffs"].items()}))
    print("  cache state hashes: %s" % json.dumps(pair["cache_state_hashes"]))
    print("\nper-cell accuracy (n, arm1, arm2, oracle, modal share)")
    for cell in cells:
        blk = per_cell[cell]
        a1s = blk["arm1_frozen"]; a2s = blk["arm2_shipped_init"]
        print("  %-20s n=%3d arm1=%.3f arm2=%.3f oracle=%.2f | arm1 modal=%.2f (%s) arm2 modal=%.2f (%s)"
              % (cell, a1s["n"], a1s["accuracy"], a2s["accuracy"],
                 out["position_oracle"]["by_cell"][cell]["oracle_accuracy_by_pad_position"],
                 a1s["modal_share"], a1s["modal_prediction"], a2s["modal_share"], a2s["modal_prediction"]))
    print("\narm2 vs arm1 (paired McNemar exact); holm p over each family")
    for c in comparisons:
        h = c["holm"]

        def g(fam):
            return h[fam]["p_holm"] if fam in h else float("nan")
        print("  %-20s n=%3d acc %.3f -> %.3f  %+6.2fpp  b=%3d c=%3d  p=%.3e | holm8=%.3g holm9=%.3g holm11=%.3g"
              % (c["cell"], c["n_pairs"], c["arm1_accuracy"], c["arm2_accuracy"], c["effect_pp"],
                 c["b_arm2_only_right"], c["c_arm1_only_right"],
                 c["mcnemar"]["p_value_two_sided_exact"],
                 g("protocol_8"), g("scored_9_incl_perm"), g("all_11")))
    print("\nindependent-test contrast (p_paired -> p_independent, ratio)")
    for c in comparisons:
        print("  %-20s %.3e -> %.3e  x%.2f"
              % (c["cell"], c["mcnemar"]["p_value_two_sided_exact"],
                 c["independent_two_proportion"]["p_two_sided"],
                 c["p_independent_over_p_paired"] or float("nan")))
    print("\nswapped-cell content test (accuracy vs substituted needle's class)")
    for arm, rows in arms.items():
        rs = [r for r in rows if r["cell"] == "L4000-p100-swapped"]
        fn = [bool(r["follows_needle"]) for r in rs if r.get("follows_needle") is not None]
        acc = sum(1 for r in rs if r["correct"]) / len(rs) if rs else float("nan")
        print("  %-20s n=%3d acc_vs_original=%.3f follows_needle=%.3f (n=%d)"
              % (arm, len(rs), acc, (sum(fn) / len(fn)) if fn else float("nan"), len(fn)))
    print("\nablated-cell collapse (needle removed by content)")
    for arm, rows in arms.items():
        rs = [r for r in rows if r["cell"] == "L4000-p100-ablated"]
        preds: Dict[str, int] = {}
        for r in rs:
            preds[r["prediction"]] = preds.get(r["prediction"], 0) + 1
        modal = max(sorted(preds), key=lambda p: preds[p])
        print("  %-20s n=%3d acc=%.3f modal=%s share=%.3f"
              % (arm, len(rs), sum(1 for r in rs if r["correct"]) / len(rs), modal, preds[modal] / len(rs)))

    print("\nlabel distributions (arm1 rows; gold labels)")
    for cell in cells:
        print("  %-20s %s" % (cell, json.dumps(per_cell[cell]["arm1_frozen"]["label_distribution"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

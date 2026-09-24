"""E-002 analysis — read the raw per-item JSONL and produce every number in findings/E-002.md.

    env/venv/bin/python experiments/h2h3-knobs/analyse.py

Nothing here re-runs the model: it reads experiments/h2h3-knobs/raw/*.jsonl (+ .summary.json)
and writes experiments/h2h3-knobs/analysis.json plus a printed table.

Reported per (pool, construction, arm, pad):
  accuracy, n, delta vs the baseline arm on the *same items*, paired McNemar exact p,
  paired bootstrap 95% CI of the delta, median / p95 latency (+ ratio vs baseline),
  peak reserved VRAM, and the live-mask histogram from the liveness probe.

Also computed (mandatory controls):
  * pad=0 regression: every arm vs baseline at pad=0.
  * identity checks: w128 should be bit-identical to baseline; allglobal_w512_combo to allglobal.
  * truncation control: no item may have been truncated at limit=8192.
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
BOOT = 10000
BOOT_SEED = 20260924


def load_rows():
    rows = {}
    for path in sorted(glob.glob(os.path.join(RAW, "*.jsonl"))):
        tag = os.path.basename(path)[:-len(".jsonl")]
        with open(path, encoding="utf-8") as fh:
            rows[tag] = [json.loads(l) for l in fh if l.strip()]
    return rows


def load_summaries():
    out = {}
    for path in sorted(glob.glob(os.path.join(RAW, "*.summary.json"))):
        with open(path, encoding="utf-8") as fh:
            out[os.path.basename(path)[:-len(".summary.json")]] = json.load(fh)
    return out


def key(tag, pad):
    return (tag, pad)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar (binomial) p for discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2.0**n
    return min(1.0, 2.0 * tail)


def paired_delta_ci(base, arm, n_boot=BOOT, seed=BOOT_SEED):
    """Percentile bootstrap CI of accuracy(arm) - accuracy(base), resampling items."""
    ids = sorted(set(base) & set(arm))
    if not ids:
        return None
    diffs = [arm[i] - base[i] for i in ids]
    n = len(diffs)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = 0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[min(n_boot - 1, int(0.975 * n_boot))]
    return {"delta": round(sum(diffs) / n, 6), "ci95": [round(lo, 6), round(hi, 6)],
            "n_pairs": n, "b_base_right_arm_wrong": sum(1 for d in diffs if d == -1),
            "c_base_wrong_arm_right": sum(1 for d in diffs if d == 1)}


def main():
    rows = load_rows()
    sums = load_summaries()
    if not rows:
        print("no raw rows yet")
        return

    # tag -> pad -> item_id -> correct
    correctness, p_gold, per_item = {}, {}, {}
    for tag, rs in rows.items():
        for r in rs:
            correctness.setdefault(tag, {}).setdefault(r["pad"], {})[r["item_id"]] = int(r["correct"])
            p_gold.setdefault(tag, {}).setdefault(r["pad"], {})[r["item_id"]] = r["p_gold"]
            per_item[(tag, r["pad"], r["item_id"])] = r

    order = sorted(rows)
    out = {"cells": [], "identity_checks": [], "pad0_regression": [], "truncation": []}

    print("%-46s %5s %4s %8s %9s %8s %9s %9s %10s" %
          ("cell (tag | pad)", "n", "corr", "acc", "delta_vs_base", "mcnemar_p", "median_s",
           "p95_s", "peakVRAM"))
    for tag in order:
        s = sums.get(tag, {})
        base_tag = tag.replace("__baseline__", "__baseline__")  # baseline is its own arm
        for cell in s.get("cells", []):
            pad = cell["pad"]
            arm_tag = tag
            base_candidates = [t for t in order
                              if sums.get(t, {}).get("arm") == "baseline"
                              and sums.get(t, {}).get("construction") == s.get("construction")
                              and (sums.get(t, {}).get("pool") == s.get("pool"))
                              and (sums.get(t, {}).get("n") == s.get("n"))
                              and (sums.get(t, {}).get("seed") == s.get("seed"))
                              and (sums.get(t, {}).get("limit") == s.get("limit"))]
            base_tag = base_candidates[0] if base_candidates else None
            d = None
            if base_tag and base_tag in correctness and pad in correctness[base_tag]:
                d = paired_delta_ci(correctness[base_tag][pad], correctness.get(arm_tag, {}).get(pad, {}))
            lat_ratio = None
            if base_tag and base_tag in sums:
                bc = next((c for c in sums[base_tag]["cells"] if c["pad"] == pad), None)
                if bc and bc.get("median_latency_s") and cell.get("median_latency_s"):
                    lat_ratio = round(cell["median_latency_s"] / bc["median_latency_s"], 3)
            row = {
                "tag": tag, "arm": s.get("arm"), "construction": s.get("construction"),
                "pool": s.get("pool"), "n": cell["n"], "seed": s.get("seed"),
                "dtype": s.get("dtype"), "limit": s.get("limit"),
                "correct": cell["correct"], "accuracy": cell["accuracy"],
                "majority_class_accuracy": s.get("majority_class_accuracy"),
                "delta_vs_baseline": None if d is None else d["delta"],
                "delta_ci95": None if d is None else d["ci95"],
                "mcnemar_exact_p": None if d is None else round(
                    mcnemar_exact(d["b_base_right_arm_wrong"], d["c_base_wrong_arm_right"]), 5),
                "b_base_right_arm_wrong": None if d is None else d["b_base_right_arm_wrong"],
                "c_base_wrong_arm_right": None if d is None else d["c_base_wrong_arm_right"],
                "median_latency_s": cell["median_latency_s"],
                "p95_latency_s": cell["p95_latency_s"],
                "latency_ratio_vs_baseline": lat_ratio,
                "peak_vram_alloc_mib": cell["peak_vram_alloc_mib"],
                "peak_vram_reserved_mib": cell["peak_vram_reserved_mib"],
                "truncated_items": cell["truncated_items"],
                "input_tokens_max": cell["input_tokens_max"],
                "state_tokens_median": cell["state_tokens_median"],
                "liveness": cell["liveness"],
                "pad": pad,
            }
            out["cells"].append(row)
            print("%-46s %5d %4d %8.3f %9s %8s %9s %9s %10.0f" %
                  ("%s|%d" % (tag.replace("__", " / "), pad), row["n"], row["correct"],
                   row["accuracy"], "n/a" if d is None else "%+.3f" % d["delta"],
                   "n/a" if d is None else "%.4f" % (row["mcnemar_exact_p"] or 1.0),
                   row["median_latency_s"], row["p95_latency_s"], row["peak_vram_reserved_mib"]))

    # ---- identity checks (arms that must be bit-identical) ----
    def compare(tag_a, tag_b, label):
        res = {"check": label, "a": tag_a, "b": tag_b, "pads": {}}
        for pad in sorted(set(correctness.get(tag_a, {})) & set(correctness.get(tag_b, {}))):
            ids = sorted(set(correctness[tag_a][pad]) & set(correctness[tag_b][pad]))
            pred_same = sum(1 for i in ids
                            if per_item[(tag_a, pad, i)]["pred"] == per_item[(tag_b, pad, i)]["pred"])
            max_dp = max((abs(p_gold[tag_a][pad][i] - p_gold[tag_b][pad][i]) for i in ids), default=None)
            res["pads"][str(pad)] = {"n": len(ids), "identical_predictions": pred_same,
                                     "max_abs_delta_p_gold": round(max_dp, 8) if max_dp is not None else None}
        out["identity_checks"].append(res)
        print("\nidentity %s: %s vs %s -> %s" % (label, tag_a, tag_b, json.dumps(res["pads"])))

    for tag in order:
        s = sums.get(tag, {})
        pool, n, seed = s.get("pool"), s.get("n"), s.get("seed")
        def find(arm, construction="p1_exact"):
            for t in order:
                ss = sums.get(t, {})
                if (ss.get("arm") == arm and ss.get("construction") == construction
                        and ss.get("pool") == pool and ss.get("n") == n and ss.get("seed") == seed):
                    return t
            return None
        b, w128 = find("baseline"), find("w128")
        if b and w128 and s.get("arm") == "w128":
            compare(b, w128, "w128 is the shipped config re-applied: must be identical to baseline")
        ag, combo = find("allglobal"), find("allglobal_w512_combo")
        if ag and combo and s.get("arm") == "allglobal_w512_combo":
            compare(ag, combo, "all-global with config.local_attention=512 must equal all-global "
                               "(falsifies 'the sliding mask is still in use')")

    # ---- pad=0 regression ----
    for tag in order:
        s = sums.get(tag, {})
        cell = next((c for c in s.get("cells", []) if c["pad"] == 0), None)
        if not cell:
            continue
        pool, n, seed = s.get("pool"), s.get("n"), s.get("seed")
        base = None
        for t in order:
            ss = sums.get(t, {})
            if (ss.get("arm") == "baseline" and ss.get("construction") == s.get("construction")
                    and ss.get("pool") == pool and ss.get("n") == n and ss.get("seed") == seed):
                base = t
                break
        if base is None or base == tag:
            continue
        d = paired_delta_ci(correctness[base][0], correctness.get(tag, {}).get(0, {}))
        if d is None:
            continue
        rec = {"tag": tag, "arm": s.get("arm"), "pool": pool, "n": cell["n"],
               "baseline_accuracy": next(c["accuracy"] for c in sums[base]["cells"] if c["pad"] == 0),
               "arm_accuracy": cell["accuracy"], "delta": d["delta"], "delta_ci95": d["ci95"],
               "mcnemar_exact_p": round(mcnemar_exact(d["b_base_right_arm_wrong"],
                                                     d["c_base_wrong_arm_right"]), 5),
               "b": d["b_base_right_arm_wrong"], "c": d["c_base_wrong_arm_right"]}
        out["pad0_regression"].append(rec)
        print("pad=0 %-44s base=%.3f arm=%.3f delta=%+.3f p=%.4f"
              % (tag.replace("__", " / "), rec["baseline_accuracy"], rec["arm_accuracy"],
                 rec["delta"], rec["mcnemar_exact_p"]))

    # ---- truncation control ----
    for tag in order:
        s = sums.get(tag, {})
        for c in s.get("cells", []):
            out["truncation"].append({"tag": tag, "pad": c["pad"], "truncated_items": c["truncated_items"],
                                      "input_tokens_max": c["input_tokens_max"], "limit": s.get("limit")})
    bad = [t for t in out["truncation"] if t["truncated_items"]]
    print("\ntruncation control: %d cells, %d with truncated items" % (len(out["truncation"]), len(bad)))

    # ---- P1 reproduction: per-item, not just aggregate ----
    p1_path = os.path.join(os.path.dirname(HERE), "orch-diagnostic", "results.json")
    rep = {"artifact": os.path.relpath(p1_path, os.path.dirname(HERE)), "pads": {}}
    if os.path.exists(p1_path):
        with open(p1_path, encoding="utf-8") as fh:
            p1 = json.load(fh)
        p1_rows = p1.get("probes", {}).get("C_length", {})
        base_tags = [t for t in order if sums.get(t, {}).get("arm") == "baseline"
                     and sums.get(t, {}).get("construction") == "p1_exact"
                     and sums.get(t, {}).get("n") == 20]
        if base_tags:
            bt = base_tags[0]
            idx = {}
            for r in rows[bt]:
                # item_id "upstream|NN|lang" -> NN is the index into P1's own REQUESTS order
                nn = int(r["item_id"].split("|")[1])
                idx[(r["pad"], nn)] = r
            for pad_s, cell in p1_rows.items():
                pad = int(pad_s)
                mine = idx
                agree = tot = 0
                max_dp = 0.0
                for nn, item in enumerate(cell.get("per_item", [])):
                    r = mine.get((pad, nn))
                    if r is None:
                        continue
                    tot += 1
                    agree += int(r["pred"] == item["pred"])
                    if r["gold"] == item["gold"]:
                        max_dp = max(max_dp, abs(r["p_gold"] - float(item["p_gold"])))
                rep["pads"][pad_s] = {"n": tot, "prediction_agreement": agree,
                                      "p1_accuracy": cell["accuracy"],
                                      "max_abs_delta_p_gold_same_gold": round(max_dp, 6)}
            rep["baseline_tag"] = bt
            out["p1_reproduction"] = rep
            print("\nP1 reproduction (per item, baseline %s vs probes.C_length):" % bt)
            for pad_s, v in rep["pads"].items():
                print("  pad=%-5s n=%d identical_predictions=%d/%d max|dp_gold|=%.5f (P1 acc %.3f)"
                      % (pad_s, v["n"], v["prediction_agreement"], v["n"],
                         v["max_abs_delta_p_gold_same_gold"], v["p1_accuracy"]))

    with open(os.path.join(HERE, "analysis.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print("wrote analysis.json (%d cells)" % len(out["cells"]))


if __name__ == "__main__":
    main()

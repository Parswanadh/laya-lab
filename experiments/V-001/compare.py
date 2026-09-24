"""V-001 per-cell comparison: stored artifact vs fresh reproduction.

    env/venv/bin/python experiments/V-001/compare.py
"""
import json
import os
import sys

LAB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V = os.path.join(LAB, "experiments", "V-001")


def j(p):
    with open(p) as f:
        return json.load(f)


def stored(name, live):
    """Prefer the frozen copy taken before the reproduction; fall back to the live artifact."""
    p = os.path.join(V, "stored", name)
    return j(p if os.path.exists(p) else live)


def pct(a, b):
    return "same" if a == b else "%+d" % (b - a)


def main():
    out = {}
    stored_b = stored("experiments_orch-baseline_results.json", os.path.join(LAB, "experiments", "orch-baseline", "results.json"))
    fresh_b = j(os.path.join(V, "fresh", "orch-baseline-results.json"))
    stored_d = stored("experiments_orch-diagnostic_results.json", os.path.join(LAB, "experiments", "orch-diagnostic", "results.json"))
    fresh_d = j(os.path.join(V, "fresh", "orch-diagnostic-results.json"))
    rep = j(os.path.join(V, "repeat.json"))

    print("=== BASELINE: stored vs fresh, per cell ===")
    print("%-6s %-6s | %-11s | %-15s | %-15s | %-21s | %s"
          % ("limit", "pad", "correct s/f", "accuracy s/f", "probe_tok s/f",
             "median_lat s/f", "verdict"))
    rows = []
    for s, f in zip(stored_b["rows"], fresh_b["rows"]):
        assert (s["limit"], s["pad"]) == (f["limit"], f["pad"]), "cell order differs"
        ok = (s["correct"] == f["correct"] and s["accuracy"] == f["accuracy"]
              and s["probe_tokens"] == f["probe_tokens"] and s["n"] == f["n"])
        rows.append({"limit": s["limit"], "pad": s["pad"], "n": s["n"],
                     "correct_stored": s["correct"], "correct_fresh": f["correct"],
                     "accuracy_stored": s["accuracy"], "accuracy_fresh": f["accuracy"],
                     "probe_tokens_stored": s["probe_tokens"], "probe_tokens_fresh": f["probe_tokens"],
                     "median_latency_stored": s["median_latency_s"],
                     "median_latency_fresh": f["median_latency_s"],
                     "latency_ratio": round(f["median_latency_s"] / s["median_latency_s"], 3),
                     "verdict": "REPRODUCED" if ok else "NOT REPRODUCED"})
        lat = ("%.4f/%.4f (x%.2f)" % (s["median_latency_s"], f["median_latency_s"],
                                        f["median_latency_s"] / s["median_latency_s"])
               if s["median_latency_s"] else "%.4f/n/a" % f["median_latency_s"])
        print("%-6d %-6d | %-11s | %-15s | %-15s | %-21s | %s"
              % (s["limit"], s["pad"],
                 "%d/%d" % (s["correct"], f["correct"]),
                 "%.3f/%.3f" % (s["accuracy"], f["accuracy"]),
                 "%d/%d" % (s["probe_tokens"], f["probe_tokens"]),
                 lat, "REPRODUCED" if ok else "NOT REPRODUCED"))
    out["baseline_rows"] = rows
    out["baseline_header_match"] = {k: (stored_b.get(k) == fresh_b.get(k)) for k in
                                    ("device", "gpu", "n_per_cell")}

    print("\n=== DIAGNOSTIC A_dtype (per_item is discarded by the script) ===")
    a_rows = []
    for name in stored_d["probes"]["A_dtype"]:
        s = stored_d["probes"]["A_dtype"][name]
        f = fresh_d["probes"]["A_dtype"][name]
        a_rows.append({"name": name, "accuracy_stored": s["accuracy"], "accuracy_fresh": f["accuracy"],
                       "n_stored": s["n"], "n_fresh": f["n"],
                       "median_latency_stored": s["median_latency_s"],
                       "median_latency_fresh": f["median_latency_s"],
                       "verdict": "REPRODUCED" if s["accuracy"] == f["accuracy"] and s["n"] == f["n"]
                                  else "NOT REPRODUCED"})
        print("  %-24s acc %.3f/%.3f  n %d/%d  lat %.4f/%.4f  %s"
              % (name, s["accuracy"], f["accuracy"], s["n"], f["n"],
                 s["median_latency_s"], f["median_latency_s"], a_rows[-1]["verdict"]))
    out["A_dtype"] = a_rows

    for probe in ("B_position", "C_length"):
        print("\n=== DIAGNOSTIC %s: stored vs fresh, per cell ===" % probe)
        p_rows = []
        for cell in stored_d["probes"][probe]:
            s = stored_d["probes"][probe][cell]
            f = fresh_d["probes"][probe][cell]
            acc_ok = s["accuracy"] == f["accuracy"] and s["n"] == f["n"]
            n_diff_pred = sum(1 for a, b in zip(s["per_item"], f["per_item"]) if a["pred"] != b["pred"])
            worst_p = max(abs(a["p_gold"] - b["p_gold"]) for a, b in zip(s["per_item"], f["per_item"]))
            p_rows.append({"cell": cell, "accuracy_stored": s["accuracy"], "accuracy_fresh": f["accuracy"],
                           "n_stored": s["n"], "n_fresh": f["n"], "n_pred_differs": n_diff_pred,
                           "max_abs_p_gold_diff": round(worst_p, 6),
                           "median_latency_stored": s["median_latency_s"],
                           "median_latency_fresh": f["median_latency_s"],
                           "verdict": "REPRODUCED" if acc_ok else "NOT REPRODUCED"})
            print("  %-6s acc %.3f/%.3f  n %d/%d  items_with_different_pred %2d/%-2d  "
                  "max|dp_gold| %.6f  lat %.4f/%.4f  %s"
                  % (cell, s["accuracy"], f["accuracy"], s["n"], f["n"], n_diff_pred, len(s["per_item"]),
                     worst_p, s["median_latency_s"], f["median_latency_s"], p_rows[-1]["verdict"]))
        out[probe] = p_rows

    print("\n=== DIAGNOSTIC derived fields ===")
    out["derived"] = {k: {"stored": stored_d.get(k), "fresh": fresh_d.get(k),
                          "match": stored_d.get(k) == fresh_d.get(k)}
                      for k in ("majority_class_accuracy", "label_counts")}
    for k, v in out["derived"].items():
        print("  %-26s stored=%s fresh=%s match=%s" % (k, v["stored"], v["fresh"], v["match"]))

    print("\n=== instrumented repeat run: per-cell predictions (baseline dtype = %s) ==="
          % rep["dtype_at_load"])
    cell_rows = []
    by_key = {}
    for k, c in rep["cells"].items():
        lim = int(k.split("_")[1][5:])
        pad = int(k.split("_")[2][3:])
        by_key[(lim, pad)] = c
    for r in rows:
        c = by_key.get((r["limit"], r["pad"]))
        if c is None:
            continue
        rows_seen = sorted({i["input_tokens"] for i in c["per_item"]})
        cell_rows.append({"limit": r["limit"], "pad": r["pad"], "accuracy_repeat_run": c["accuracy"],
                          "accuracy_fresh_run": r["accuracy_fresh"],
                          "matches_script": c["accuracy"] == r["accuracy_fresh"],
                          "pred_histogram": c["pred_histogram"], "distinct_predictions": c["distinct_predictions"],
                          "input_tokens_seen": rows_seen,
                          "n_input_tokens_values": len(rows_seen)})
        print("  limit=%-5d pad=%-5d acc(script/repeat)=%.3f/%.3f  preds=%s  input_tokens=%s"
              % (r["limit"], r["pad"], r["accuracy_fresh"], c["accuracy"], c["pred_histogram"], rows_seen))
    out["instrumented_baseline"] = cell_rows

    print("\n=== nondeterminism: identical cell repeated in-process ===")
    nd = []
    pairs = [("baseline_limit8192_pad7000", "baseline_8192_7000_run2"),
             ("baseline_8192_7000_run2", "baseline_8192_7000_run3"),
             ("diag_C4000_fp32_run1", "diag_C4000_fp32_run2"),
             ("diag_B0.5_fp32_run1", "diag_B0.5_fp32_run2")]
    for a, b in pairs:
        ca = rep["cells"].get(a) or rep["repeats"][a]
        cb = rep["repeats"][b]
        npred = sum(1 for x, y in zip(ca["per_item"], cb["per_item"]) if x["pred"] != y["pred"])
        maxp = max(abs(x["p_gold"] - y["p_gold"]) for x, y in zip(ca["per_item"], cb["per_item"]))
        nd.append({"a": a, "b": b, "acc_a": ca["accuracy"], "acc_b": cb["accuracy"],
                   "n_pred_differs": npred, "max_abs_p_gold_diff": maxp})
        print("  %-32s vs %-32s acc %.3f/%.3f  pred-diff %d/20  max|dp_gold| %.8f"
              % (a, b, ca["accuracy"], cb["accuracy"], npred, maxp))
    # the diagnostic's own internal duplicate: B 1.0 and C 4000 are the same documents
    same = stored_d["probes"]["B_position"]["1.0"]["per_item"] == stored_d["probes"]["C_length"]["4000"]["per_item"]
    nd.append({"a": "stored B_position/1.0", "b": "stored C_length/4000", "identical_per_item": same})
    print("  stored B_position[1.0].per_item == C_length[4000].per_item: %s (same documents twice)" % same)
    out["nondeterminism"] = nd

    print("\n=== injection: does the same document under bf16 vs fp32 agree? ===")
    print("  baseline(8192,pad) uses bf16 and diagnostic C_length(pad) fp32 on token-identical docs")
    inj = []
    for pad in (0, 1000, 2000, 4000, 7000):
        b = [r for r in rows if r["limit"] == 8192 and r["pad"] == pad][0]
        d = out["C_length"][[c["cell"] for c in out["C_length"]].index(str(pad))]
        inj.append({"pad": pad, "bf16_baseline_acc": b["accuracy_fresh"],
                    "fp32_diagnostic_acc": d["accuracy_fresh"],
                    "agree": b["accuracy_fresh"] == d["accuracy_fresh"]})
        print("  pad=%-5d bf16 %.3f vs fp32 %.3f  agree=%s"
              % (pad, b["accuracy_fresh"], d["accuracy_fresh"], inj[-1]["agree"]))
    out["bf16_vs_fp32_same_docs"] = inj

    with open(os.path.join(V, "compare.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote experiments/V-001/compare.json")


if __name__ == "__main__":
    main()

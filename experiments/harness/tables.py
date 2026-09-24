#!/usr/bin/env python
"""Regenerate the tables of a finding from the raw per-item JSONL. No model, no GPU.

    env/venv/bin/python experiments/harness/tables.py experiments/baseline-repro \
        experiments/baseline-repro-upstream experiments/position-sweep

Prints, for every run directory given:

* the headroom table: accuracy, bootstrap CI, macro-F1, majority-class and random references, the
  modal prediction and its share, and how much of the document the model actually got;
* latency (p50/p95) and tokens;
* for every pair of runs, exact McNemar over the cells they share, paired by item id.

This is the command that regenerates every number quoted from these runs in ``findings/E-001.md``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import checks  # noqa: E402
import metrics  # noqa: E402


def load_run(run_dir: Path):
    rows = checks.load_rows(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return rows, summary, manifest


def fmt(x, nd=3):
    return "n/a" if x is None else ("%.*f" % (nd, x))


def collapse_stats(rows) -> dict:
    """Per-cell: how many distinct probability vectors and predictions the model produced.

    One distinct vector across 20 different documents is the signature of a total collapse: the
    input no longer reaches the answer at all. Upstream's own pad>=2000/limit=1024 cells have
    exactly one.
    """
    by_cell = {}
    for r in rows:
        key = r["cell"]
        d = by_cell.setdefault(key, {"vecs": set(), "preds": set(), "n": 0})
        d["vecs"].add(tuple(round(float(v), 6) for v in r["prob_vector"]))
        d["preds"].add(r["prediction"])
        d["n"] += 1
    return {k: {"n": v["n"], "distinct_prob_vectors": len(v["vecs"]),
                "distinct_predictions": len(v["preds"])} for k, v in by_cell.items()}


def collapse_detail(name: str, rows) -> None:
    """Per cell: the confusion matrix and how wide the modal answer's probability moves.

    A collapsed cell has one prediction for every item; the probability range says whether that
    constant decision came with byte-identical probabilities (upstream's single repeated filler
    sentence gives that) or merely a constant argmax over slightly different inputs.
    """
    by_cell = {}
    for r in rows:
        by_cell.setdefault(r["cell"], []).append(r)
    print("\n#### %s — confusion and collapse detail\n" % name)
    print("| cell | n | gold \\ pred | modal pred p (min / median / max) |")
    print("|---|---|---|---|")
    for cell, rs in sorted(by_cell.items()):
        gold = sorted({r["label"] for r in rs})
        preds = sorted({r["prediction"] for r in rs})
        parts = []
        for g in gold:
            counts = {p: sum(1 for r in rs if r["label"] == g and r["prediction"] == p) for p in preds}
            parts.append("%s -> %s" % (g, ", ".join("%s:%d" % (p, c) for p, c in counts.items() if c)))
        mp, share = metrics.modal_prediction(rs)
        ps = [float(r["probabilities"][mp]) for r in rs]
        print("| %s | %d | %s | %s (%.4f / %.4f / %.4f) |"
              % (cell, len(rs), "; ".join(parts), mp, min(ps), sorted(ps)[len(ps) // 2], max(ps)))


def cell_table(name: str, summary: dict, rows=None) -> None:
    print("\n### %s — cells (n per cell in the table)\n" % name)
    cs = collapse_stats(rows) if rows else {}
    print("| pad | pos | max_len | n | correct | acc | CI95 | macroF1 | maj.cls | rand | modal pred (share) | distinct prob vecs / preds | req kept (med kept/med tot tok) | med state kept/full tok | p50 s | p95 s | tok/s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in summary["cells"]:
        rb = c.get("random_baselines") or {}
        col = cs.get(c["cell_key"]) or {}
        cells_out = [
            str(c["pad_tokens"]), "%.2f" % c["needle_position"], str(c["max_len_requested"]),
            str(c["n"]), str(c["correct"]), fmt(c["accuracy"]),
            "[%s, %s]" % (fmt(c["accuracy_ci95_low"]), fmt(c["accuracy_ci95_high"])),
            fmt(c.get("macro_f1")), fmt(c["majority_class_accuracy"]),
            fmt(rb.get("random_over_options")),
            "%s (%s)" % (c["modal_prediction"], fmt(c["modal_prediction_share"], 2)),
            "%s / %s" % (col.get("distinct_prob_vectors", "?"), col.get("distinct_predictions", "?")),
            "%s (%s/%s of %s)" % (fmt(c["request_kept_fraction"], 2),
                                  c.get("median_request_tokens_kept"), c.get("median_request_tokens"),
                                  c.get("median_needle_tokens_kept")),
            "%s / %s" % (c["median_state_tokens_kept"], c["median_state_tokens_full"]),
            fmt(c["median_latency_s"]), fmt(c.get("p95_latency_s")),
            fmt(c.get("tokens_per_second"), 1),
        ]
        print("| " + " | ".join(cells_out) + " |")


def baselines_block(name: str, summary: dict) -> None:
    b = summary.get("baselines") or {}
    orc = b.get("position_only_oracle") or {}
    print("\n**%s baselines.** random over options %s; random over the 3 gold labels %s; "
          "majority class per cell %s; position-only oracle (grouped by pad x position) %s."
          % (name, fmt(b.get("random_over_options")), fmt(b.get("random_over_gold_labels")),
             json.dumps({k: round(v, 3) for k, v in (b.get("majority_class_accuracy_per_cell") or {}).items()}),
             fmt(orc.get("oracle_accuracy"))))


def mcnemar_table(name_a: str, name_b: str, rows_a, rows_b) -> None:
    cells_a = {r["cell"] for r in rows_a}
    cells_b = {r["cell"] for r in rows_b}
    shared = sorted(cells_a & cells_b)
    if not shared:
        print("\n**McNemar %s vs %s:** no shared cells (different cell keys)." % (name_a, name_b))
        return
    print("\n**McNemar %s vs %s** (paired by item_id; A = %s, B = %s)\n" % (name_a, name_b, name_a, name_b))
    print("| cell | n pairs | A acc | B acc | A wrong / B right | A right / B wrong | p (exact, two-sided) |")
    print("|---|---|---|---|---|---|---|")
    for cell in shared:
        a = [r for r in rows_a if r["cell"] == cell]
        b = [r for r in rows_b if r["cell"] == cell]
        ca, cb = metrics.pair_rows(a, b)
        if not ca:
            continue
        m = metrics.mcnemar_exact(ca, cb)
        print("| %s | %d | %s | %s | %d | %d | %s |"
              % (cell, m["n_pairs"], fmt(metrics.accuracy(a)), fmt(metrics.accuracy(b)),
                 m["a_wrong_b_right"], m["a_right_b_wrong"], ("%.3g" % m["p_value_two_sided_exact"])))


def within_run_table(name: str, rows, axis: str, group_fields) -> None:
    """Paired comparison *inside* one run: same items, same documents, one axis changed.

    `axis` is the field that varies (max_len_effective or needle_position); the groups are the
    cells that share everything else. Exact McNemar, paired by item id.
    """
    groups = {}
    for r in rows:
        key = tuple(r[f] for f in group_fields)
        groups.setdefault(key, {}).setdefault(r[axis], []).append(r)
    print("\n**%s — paired comparison over %s** (same items and documents, exact McNemar)\n"
          % (name, axis))
    print("| group | low | high | low acc | high acc | low wrong / high right | low right / high wrong | p |")
    print("|---|---|---|---|---|---|---|---|")
    for key, by_val in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        vals = sorted(by_val)
        if len(vals) < 2:
            continue
        lo, hi = vals[0], vals[-1]
        ca, cb = metrics.pair_rows(by_val[lo], by_val[hi])
        if not ca:
            continue
        m = metrics.mcnemar_exact(ca, cb)
        print("| %s | %s | %s | %s | %s | %d | %d | %s |"
              % ("|".join(str(x) for x in key), lo, hi, fmt(metrics.accuracy(by_val[lo])),
                 fmt(metrics.accuracy(by_val[hi])), m["a_wrong_b_right"], m["a_right_b_wrong"],
                 "%.3g" % m["p_value_two_sided_exact"]))


def main(argv):
    dirs = [Path(a) for a in argv] or [Path("experiments/baseline-repro")]
    loaded = []
    for d in dirs:
        rows, summary, manifest = load_run(d)
        name = d.name
        print("\n## %s" % name)
        print("run_id: `%s`  pool: `%s`  seed: %s  n per cell: %s  branch/commit: `%s`  status: %s"
              % (summary.get("run_id"), summary.get("pool"), summary.get("seed"),
                 summary.get("n_per_cell"),
                 (manifest.get("git", {}).get("lab", {}) or {}).get("sha", "?")[:12],
                 manifest.get("status")))
        print("items: %s  |  label_counts: %s" % (manifest["items"]["n"], manifest["items"]["label_counts"]))
        checks_ok = summary.get("notes")
        print("checks: %s" % checks_ok)
        if summary.get("repeat_check"):
            print("repeat check: %s" % json.dumps(summary["repeat_check"]))
        cell_table(name, summary, rows)
        collapse_detail(name, rows)
        within_run_table(name, rows, "max_len_effective", ("pad_tokens", "needle_position"))
        within_run_table(name, rows, "needle_position", ("pad_tokens", "max_len_effective"))
        baselines_block(name, summary)
        loaded.append((name, rows))
    for i in range(len(loaded)):
        for j in range(i + 1, len(loaded)):
            mcnemar_table(loaded[i][0], loaded[j][0], loaded[i][1], loaded[j][1])


if __name__ == "__main__":
    main(sys.argv[1:])

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


def cell_table(name: str, summary: dict) -> None:
    print("\n### %s — cells (n per cell in the table)\n" % name)
    print("| pad | pos | max_len | n | correct | acc | CI95 | macroF1 | maj.cls | rand | modal pred (share) | req kept | med kept tok | p50 s | p95 s | tok/s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in summary["cells"]:
        rb = c.get("random_baselines") or {}
        print("| %d | %.2f | %s | %d | %d | %s | [%s, %s] | %s | %s | %s | %s (%s) | %s | %s | %s | %s | %s |"
              % (c["pad_tokens"], c["needle_position"], c["max_len_requested"], c["n"], c["correct"],
                 fmt(c["accuracy"]), fmt(c["accuracy_ci95_low"]), fmt(c["accuracy_ci95_high"]),
                 fmt(c.get("macro_f1")), fmt(c["majority_class_accuracy"]),
                 fmt(rb.get("random_over_options")), c["modal_prediction"],
                 fmt(c["modal_prediction_share"], 2), fmt(c["request_kept_fraction"], 2),
                 c["median_state_tokens_kept"], fmt(c["median_latency_s"]),
                 fmt(c.get("p95_latency_s")), fmt(c.get("tokens_per_second"), 1)))


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
        cell_table(name, summary)
        baselines_block(name, summary)
        loaded.append((name, rows))
    for i in range(len(loaded)):
        for j in range(i + 1, len(loaded)):
            mcnemar_table(loaded[i][0], loaded[j][0], loaded[i][1], loaded[j][1])


if __name__ == "__main__":
    main(sys.argv[1:])

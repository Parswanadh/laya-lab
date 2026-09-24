"""Pipeline checks that do not need a verdict from anyone.

``structural_checks`` reads only the published artifacts (JSONL + summary), so it can be run
against a real run long after the model is gone -- a verifier can point it at a run directory.
``mechanism_checks`` rebuilds documents from the manifest's seed and pool, so it only works while
the pool files are present (they are committed).

These are harness self-tests, not evidence about the model. They assert that the pipeline did what
it claims it did (budgets hit, pairing preserved, truncation rule as documented, JSONL parseable),
never that a number is "good".
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

REQUIRED_ROW_FIELDS = (
    "run_id", "cell", "item_index", "item_id", "label", "lang", "template_id",
    "prediction", "probabilities", "prob_vector", "options", "correct",
    "pad_tokens", "needle_position", "max_len_effective", "head_max_len_effective",
    "latency_s", "input_tokens", "state_tokens_full", "state_tokens_kept", "truncated",
    "needle_tokens", "needle_tokens_kept", "needle_kept",
    "request_tokens", "request_tokens_kept", "request_kept",
    "doc_sha256", "seed", "device",
)


def _res(name: str, passed: bool, detail: Any = None) -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def load_rows(out_dir) -> List[Dict[str, Any]]:
    rows = []
    path = Path(out_dir) / "predictions.jsonl"
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError("predictions.jsonl line %d is not JSON: %s" % (i + 1, e))
    return rows


def structural_checks(out_dir, manifest: Optional[Dict[str, Any]] = None,
                      summary: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    out_dir = Path(out_dir)
    if manifest is None:
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    if summary is None:
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    rows = load_rows(out_dir)
    out: List[Dict[str, Any]] = []

    cells = [c["cell_key"] for c in summary["cells"]]
    per_cell: Dict[str, List[Dict[str, Any]]] = {k: [] for k in cells}
    stray = []
    for r in rows:
        if r["cell"] in per_cell:
            per_cell[r["cell"]].append(r)
        else:
            stray.append(r["cell"])
    out.append(_res("cell_keys_match_summary", not stray, {"stray": stray[:5], "n_cells": len(cells)}))
    out.append(_res("row_count_matches_summary",
                    len(rows) == sum(c["n"] for c in summary["cells"]),
                    {"rows": len(rows), "summary_n": sum(c["n"] for c in summary["cells"])}))

    missing = {}
    bad_type = {}
    for r in rows:
        for f in REQUIRED_ROW_FIELDS:
            if f not in r:
                missing[f] = missing.get(f, 0) + 1
    out.append(_res("required_fields_present", not missing, missing))

    prob_bad, pred_bad, argmax_bad = [], [], []
    for r in rows:
        p = r["probabilities"]
        s = sum(p.values())
        if abs(s - 1.0) > 0.02:
            prob_bad.append({"item": r["item_id"], "sum": s})
        if r["prediction"] not in r["options"]:
            pred_bad.append(r["item_id"])
        if r["prediction"] != max(p.items(), key=lambda kv: (kv[1], kv[0]))[0]:
            argmax_bad.append(r["item_id"])
    out.append(_res("probabilities_sum_to_one", not prob_bad, prob_bad[:5]))
    out.append(_res("prediction_is_an_option", not pred_bad, pred_bad[:5]))
    out.append(_res("prediction_is_probability_argmax", not argmax_bad, argmax_bad[:5]))

    trunc_ok = [r["item_id"] for r in rows if not r.get("trunc_rule_ok", True)]
    out.append(_res("truncation_matches_documented_rule", not trunc_ok, trunc_ok[:5]))

    over = [{"item": r["item_id"], "input_tokens": r["input_tokens"],
             "max_len": r["max_len_effective"]} for r in rows
            if r["input_tokens"] > r["max_len_effective"]]
    out.append(_res("input_tokens_within_budget", not over, over[:5]))

    dupes = len(rows) - len({(r["cell"], r["item_id"]) for r in rows})
    out.append(_res("no_duplicate_cell_item_rows", dupes == 0, {"duplicates": dupes}))

    # pairing: the same (pad, position) must run the same item ids at every max_len
    pairing_bad = []
    groups: Dict[Any, Dict[Any, set]] = {}
    for r in rows:
        groups.setdefault((r["pad_tokens"], r["needle_position"]), {}).setdefault(
            r["max_len_effective"], set()).add(r["item_id"])
    for key, by_ml in groups.items():
        if len(by_ml) > 1:
            sets = list(by_ml.values())
            if any(s != sets[0] for s in sets[1:]):
                pairing_bad.append({"pad": key[0], "position": key[1],
                                    "sizes": {m: len(s) for m, s in by_ml.items()}})
    out.append(_res("items_paired_across_max_len", not pairing_bad, pairing_bad[:5]))

    # documents must not depend on max_len (a difference would break the paired design)
    doc_by_cell = {}
    for r in rows:
        doc_by_cell.setdefault((r["pad_tokens"], r["needle_position"], r["item_id"]), set()).add(
            r["doc_sha256"])
    doc_bad = [k for k, v in doc_by_cell.items() if len(v) > 1]
    out.append(_res("document_independent_of_max_len", not doc_bad, doc_bad[:5]))

    # accuracy recomputed from raw rows must equal what the summary claims
    acc_bad = []
    for c in summary["cells"]:
        rs = per_cell[c["cell_key"]]
        if not rs:
            continue
        acc = sum(1 for r in rs if r["correct"]) / len(rs)
        if abs(acc - c["accuracy"]) > 1e-9:
            acc_bad.append({"cell": c["cell_key"], "summary": c["accuracy"], "raw": acc})
    out.append(_res("summary_accuracy_recomputes_from_jsonl", not acc_bad, acc_bad[:5]))

    items = manifest.get("items", {})
    lc = items.get("label_counts", {})
    out.append(_res("label_balance", bool(items.get("balance_ok", False)),
                    {"label_counts": lc, "max_label_share": items.get("max_label_share")}))
    out.append(_res("leakage_check_passed", bool(manifest.get("pools", {}).get("leakage_check", {}).get("passed")),
                    manifest.get("pools", {}).get("leakage_check", {}).get("stem_hits", [])[:3]))
    return out


def mechanism_checks(builder, items: Sequence[Dict[str, Any]], cells: Sequence[Dict[str, Any]],
                     seed: int, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rebuild documents and check the truncation mechanism the way a sceptic would."""
    out: List[Dict[str, Any]] = []
    by_key = {(r["item_id"], r["pad_tokens"], r["needle_position"], r["max_len_effective"]): r
              for r in rows}
    sample = list(items)[:3]

    # determinism of the document given the seed
    det_ok, det_bad = True, []
    for item in sample:
        for cell in cells[:3]:
            d1 = builder.build(item, cell["pad_tokens"], cell["needle_position"], seed)
            d2 = builder.build(item, cell["pad_tokens"], cell["needle_position"], seed)
            if d1["doc_sha256"] != d2["doc_sha256"]:
                det_ok = False
                det_bad.append(item["item_id"])
    out.append(_res("documents_deterministic_for_seed", det_ok, det_bad[:3]))

    # a different seed must move the documents (only meaningful where there is filler to move)
    filler_cells = [c for c in cells if c["pad_tokens"] > 0]
    moved, total = 0, 0
    for item in sample:
        for cell in filler_cells[:3]:
            d1 = builder.build(item, cell["pad_tokens"], cell["needle_position"], seed)
            d2 = builder.build(item, cell["pad_tokens"], cell["needle_position"], seed + 1)
            total += 1
            moved += 1 if d1["doc_sha256"] != d2["doc_sha256"] else 0
    if total == 0:
        out.append(_res("seed_changes_documents", True,
                        {"skipped": "no cell in this plan has filler (all pad_tokens == 0)"}))
    else:
        out.append(_res("seed_changes_documents", moved == total, {"moved": moved, "total": total}))

    # the row's recorded document hash must be the one the seed rebuilds
    rebuild_ok, rebuild_bad = True, []
    checked = 0
    for r in rows[:40]:
        item = next((it for it in items if it["item_id"] == r["item_id"]), None)
        if item is None:
            continue
        d = builder.build(item, r["pad_tokens"], r["needle_position"], seed)
        checked += 1
        if d["doc_sha256"] != r["doc_sha256"]:
            rebuild_ok = False
            rebuild_bad.append(r["item_id"])
    out.append(_res("row_doc_sha256_rebuilds_from_seed", rebuild_ok,
                    {"checked": checked, "mismatches": rebuild_bad[:3]}))

    # mechanism: with the needle at the end and a small budget the request must be cut away;
    # at the start of the document it must survive. This is the collapse, stated as a fact about
    # the sequence rather than about accuracy.
    def frac(pos: float, ml_eff: int) -> Optional[float]:
        sel = [r for r in rows if r["needle_position"] == pos
               and r["max_len_effective"] == ml_eff and r["pad_tokens"] > 0]
        if not sel:
            return None
        return sum(1 for r in sel if r["request_kept"]) / len(sel)

    small = min((c["max_len_effective"] for c in cells), default=None)
    big = max((c["max_len_effective"] for c in cells), default=None)
    if small and big and small != big:
        tail = frac(1.0, small)
        head = frac(0.0, small)
        large_tail = frac(1.0, big)
        if tail is not None and head is not None:
            out.append(_res("small_budget_drops_the_tail_request", tail == 0.0 and head > 0.0,
                            {"request_kept_fraction_at_position_1.0": tail,
                             "request_kept_fraction_at_position_0.0": head,
                             "max_len": small}))
        if large_tail is not None:
            longest = max(int(r["state_tokens_full"]) for r in rows)
            if big > longest:
                out.append(_res("large_budget_keeps_the_tail_request", large_tail == 1.0,
                                {"request_kept_fraction_at_position_1.0": large_tail, "max_len": big,
                                 "longest_document_tokens": longest}))
            else:
                out.append(_res("large_budget_keeps_the_tail_request", True,
                                {"skipped": ("largest budget %d is smaller than the longest document "
                                             "%d in this plan, so full retention is not expected"
                                             % (big, longest))}))
    return out


def report(results: Iterable[Dict[str, Any]], log=print) -> bool:
    ok = True
    for r in results:
        flag = "PASS" if r["passed"] else "FAIL"
        ok = ok and r["passed"]
        detail = "" if r["detail"] in (None, {}, []) else "  %s" % json.dumps(r["detail"], ensure_ascii=False)
        log("[%s] %s%s" % (flag, r["name"], detail[:400]))
    return ok

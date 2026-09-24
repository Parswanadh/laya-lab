"""Descriptive statistics over raw per-item rows.

This module reports numbers; it does not assign confidence. Every function takes the raw rows so
a verifier can recompute the same quantity from ``predictions.jsonl`` without re-running a model.

* ``accuracy``                  — the cell's point estimate
* ``bootstrap_ci``              — percentile CI, resampling *items* (the cluster), not rows
* ``mcnemar_exact``             — paired test for two arms over the same items
* ``modal_prediction_share``    — 1.0 means every item in the cell got the same answer, which is
                                  how a truncation collapse is told apart from ordinary error
* ``majority_class_accuracy``   — what a constant answer would score on this cell's own labels
"""
from __future__ import annotations

import math
import random
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def accuracy(rows: Sequence[Dict[str, Any]]) -> float:
    if not rows:
        return float("nan")
    return sum(1 for r in rows if r["correct"]) / len(rows)


def bootstrap_ci(correct: Sequence[bool], n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 12345) -> Tuple[float, float]:
    """Percentile bootstrap CI over items. Deterministic for a given seed."""
    n = len(correct)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(correct[0]), float(correct[0])
    rng = random.Random("%d|boot|%d" % (seed, n))
    vals = []
    for _ in range(n_boot):
        s = 0
        for _ in range(n):
            s += 1 if correct[rng.randrange(n)] else 0
        vals.append(s / n)
    vals.sort()
    lo = vals[max(0, int((alpha / 2) * n_boot) - 1)]
    hi = vals[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> Dict[str, Any]:
    """Exact two-sided McNemar for two arms evaluated on the same items, in the same order.

    ``a`` and ``b`` must be aligned per item (paired design). Reports the discordant counts so the
    test can be checked by hand.
    """
    if len(a) != len(b):
        raise ValueError("mcnemar_exact needs aligned, equal-length arms (%d vs %d)" % (len(a), len(b)))
    n01 = sum(1 for x, y in zip(a, b) if (not x) and y)
    n10 = sum(1 for x, y in zip(a, b) if x and (not y))
    n = n01 + n10
    if n == 0:
        p = 1.0
    else:
        k = min(n01, n10)
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
        p = min(1.0, 2 * tail)
    return {"n_pairs": len(a), "a_wrong_b_right": n01, "a_right_b_wrong": n10,
            "n_discordant": n, "p_value_two_sided_exact": p}


def modal_prediction(rows: Sequence[Dict[str, Any]]) -> Tuple[Optional[str], float]:
    if not rows:
        return None, float("nan")
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["prediction"]] = counts.get(r["prediction"], 0) + 1
    label, k = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    return label, k / len(rows)


def predicted_distribution(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["prediction"]] = counts.get(r["prediction"], 0) + 1
    return dict(sorted(counts.items()))


def majority_class_accuracy(labels: Sequence[str]) -> float:
    if not labels:
        return float("nan")
    counts: Dict[str, int] = {}
    for lb in labels:
        counts[lb] = counts.get(lb, 0) + 1
    return max(counts.values()) / len(labels)


def _pct(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def latency_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    lats = [float(r["latency_s"]) for r in rows]
    if not lats:
        return {}
    return {"median_latency_s": statistics.median(lats),
            "mean_latency_s": statistics.fmean(lats),
            "p95_latency_s": _pct(lats, 0.95),
            "min_latency_s": min(lats),
            "max_latency_s": max(lats)}


def cell_summary(rows: Sequence[Dict[str, Any]], n_boot: int = 10000, seed: int = 12345) -> Dict[str, Any]:
    """One cell's descriptive numbers, plus the diagnostics that make a collapse visible."""
    correct = [bool(r["correct"]) for r in rows]
    labels = [r["label"] for r in rows]
    lo, hi = bootstrap_ci(correct, n_boot=n_boot, seed=seed)
    modal_label, modal_share = modal_prediction(rows)
    tokens = [int(r["input_tokens"]) for r in rows]
    funded = [int(r["state_tokens_full"]) for r in rows]
    kept = [int(r["state_tokens_kept"]) for r in rows]
    out = {
        "n": len(rows),
        "correct": sum(1 for c in correct if c),
        "accuracy": accuracy(rows),
        "accuracy_ci95_low": lo,
        "accuracy_ci95_high": hi,
        "accuracy_ci_method": "percentile bootstrap over items, n_boot=%d, seed=%d" % (n_boot, seed),
        "majority_class_accuracy": majority_class_accuracy(labels),
        "modal_prediction": modal_label,
        "modal_prediction_share": modal_share,
        "predicted_distribution": predicted_distribution(rows),
        "label_distribution": {lb: labels.count(lb) for lb in sorted(set(labels))},
        "median_input_tokens": int(statistics.median(tokens)) if tokens else None,
        "min_input_tokens": min(tokens) if tokens else None,
        "max_input_tokens": max(tokens) if tokens else None,
        "median_state_tokens_full": int(statistics.median(funded)) if funded else None,
        "median_state_tokens_kept": int(statistics.median(kept)) if kept else None,
        "truncated_fraction": (sum(1 for r in rows if r["truncated"]) / len(rows)) if rows else None,
        "request_kept_fraction": (sum(1 for r in rows if r["request_kept"]) / len(rows)) if rows else None,
        "needle_kept_fraction": (sum(1 for r in rows if r["needle_kept"]) / len(rows)) if rows else None,
        "trunc_rule_ok_all": all(r["trunc_rule_ok"] for r in rows) if rows else None,
        "pad_exact_all": all(bool(r.get("pad_exact")) for r in rows) if rows else None,
    }
    out.update(latency_stats(rows))
    return out


def pair_rows(rows_a: Sequence[Dict[str, Any]], rows_b: Sequence[Dict[str, Any]],
              key_fields: Tuple[str, ...] = ("item_id",)) -> Tuple[List[bool], List[bool]]:
    """Align two arms' rows by key (default: item id) and return their correctness vectors."""
    b_by_key = {}
    for r in rows_b:
        b_by_key[tuple(r[k] for k in key_fields)] = bool(r["correct"])
    a_map: Dict[Tuple[Any, ...], bool] = {}
    for r in rows_a:
        a_map[tuple(r[k] for k in key_fields)] = bool(r["correct"])
    keys = [k for k in a_map if k in b_by_key]
    keys.sort(key=lambda k: tuple(str(x) for x in k))
    return [a_map[k] for k in keys], [b_by_key[k] for k in keys]

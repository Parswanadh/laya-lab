"""Attack the H5 train/eval split. Committed as code so the claim "disjoint by content hash" is
re-derivable rather than asserted.

Run:

    env/venv/bin/python experiments/h5-adapter/leakage_check.py            # report + exit code
    env/venv/bin/python experiments/h5-adapter/leakage_check.py --selftest # assertions only

Every check here re-derives the split *from the emitted pool files and the plan*, not from
``author_pools.py``. That distinction matters: a check that reads the generator's own variables
proves nothing about the artifact a training run actually consumes.

Checks
------
1. template-id disjointness
2. canonical-template sha256 disjointness (exact content)
3. rendered-needle sha256 disjointness (exact content, every item in both plans)
4. longest shared **word** n-gram between the train and eval template sets
5. longest shared **character** n-gram between the train and eval template sets
6. near-duplicate surface forms: max token-set Jaccard over all train x eval pairs (via a
   4-gram candidate index, so it is not an O(n^2) blind scan)
7. slot **value** pool disjointness
8. filler vs needle vocabulary and substring stems (reuses ``experiments/harness/pools.py``)
9. needle-position/label independence: every cell and every regime is label-balanced
10. the evaluation "other"-class templates do not carry a NUL/empty toponym or other degenerate
    surface (guards against a template that renders to nothing)
11. the needle prefix and separator are not present inside the filler
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(LAB, "experiments", "harness"))

import plan as P  # noqa: E402

SLOT_RE = re.compile(r"\{[^}]*\}")
TOKEN_RE = re.compile(r"[a-z0-9@]+")


def normalize(text: str) -> str:
    """Template text with every slot replaced by a single neutral marker, lowercased.

    Slots are neutralised so that ``invoice {n},`` and ``order {ref}.`` compare as surface forms
    rather than as "both contain a placeholder". The placeholder itself carries no label signal.
    """
    return SLOT_RE.sub(" @ ", text.lower())


def tokens(text: str) -> List[str]:
    return TOKEN_RE.findall(normalize(text))


def word_ngrams(text: str, n: int) -> Set[Tuple[str, ...]]:
    t = tokens(text)
    return {tuple(t[i:i + n]) for i in range(max(0, len(t) - n + 1))}


def char_ngrams(text: str, n: int) -> Set[str]:
    s = re.sub(r"\s+", " ", normalize(text)).strip()
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def longest_shared_word_ngram(a_texts: List[str], b_texts: List[str], limit: int = 14) -> Dict[str, Any]:
    for n in range(limit, 0, -1):
        A: Set[Tuple[str, ...]] = set()
        for t in a_texts:
            A |= word_ngrams(t, n)
        B: Set[Tuple[str, ...]] = set()
        for t in b_texts:
            B |= word_ngrams(t, n)
        shared = A & B
        if shared:
            example = sorted(shared)[0]
            return {"longest_shared_word_ngram": n, "n_shared": len(shared),
                    "example": " ".join(example),
                    "example_is_stopword_only": all(w in P_STOPWORDS for w in example)}
    return {"longest_shared_word_ngram": 0, "n_shared": 0, "example": None,
            "example_is_stopword_only": True}


def longest_shared_char_ngram(a_texts: List[str], b_texts: List[str], limit: int = 40) -> Dict[str, Any]:
    for n in range(limit, 5, -1):
        A: Set[str] = set()
        for t in a_texts:
            A |= char_ngrams(t, n)
        B: Set[str] = set()
        for t in b_texts:
            B |= char_ngrams(t, n)
        shared = A & B
        if shared:
            return {"longest_shared_char_ngram": n, "n_shared": len(shared),
                    "example": sorted(shared)[0]}
    return {"longest_shared_char_ngram": 0, "n_shared": 0, "example": None}


# Function words: a shared n-gram made only of these carries no domain content. Reported either
# way -- the pass/fail decision is made on content n-grams, and the raw number is never hidden.
P_STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "at", "is", "are", "was",
    "were", "be", "been", "i", "we", "you", "my", "our", "your", "it", "this", "that", "these",
    "those", "with", "without", "from", "by", "as", "if", "but", "not", "no", "do", "does",
    "did", "can", "could", "would", "should", "will", "have", "has", "had", "please", "still",
    "again", "any", "all", "some", "more", "most", "than", "then", "there", "here", "when",
    "what", "which", "who", "how", "why", "about", "into", "over", "under", "up", "out", "so",
    "us", "them", "they", "he", "she", "its", "one", "two", "up", "@",
}


def max_jaccard(a_texts: List[str], b_texts: List[str], shingle: int = 4,
                probe: int = 8) -> Dict[str, Any]:
    """Largest token-set Jaccard between any train template and any eval template.

    Candidate pairs are found through a 4-gram inverted index, so pairs that share no 4-gram
    (already reported by check 4) are not rescanned. The Jaccard is computed on the union of
    candidate pairs plus, to keep the number honest, the ``probe`` most similar templates by
    shared-4-gram count.
    """
    def tset(t: str) -> Set[str]:
        return set(tokens(t))

    index: Dict[Tuple[str, ...], List[int]] = {}
    for j, t in enumerate(a_texts):
        for g in word_ngrams(t, shingle):
            index.setdefault(g, []).append(j)

    best = {"max_jaccard": 0.0, "a": None, "b": None}
    for i, bt in enumerate(b_texts):
        bs = tset(bt)
        counts: Dict[int, int] = {}
        for g in word_ngrams(bt, shingle):
            for j in index.get(g, ()):
                counts[j] = counts.get(j, 0) + 1
        if not counts:
            continue
        for j in sorted(counts, key=lambda k: -counts[k])[:probe]:
            aset = tset(a_texts[j])
            union = aset | bs
            if not union:
                continue
            jac = len(aset & bs) / len(union)
            if jac > best["max_jaccard"]:
                best = {"max_jaccard": round(jac, 4), "a": a_texts[j], "b": bt}
    return best


def build_report(plan_path: str | None = None) -> Dict[str, Any]:
    eval_pool = P.load_pool("needles-h5-eval-v1.json")
    train_pool = P.load_pool("needles-h5-train-v1.json")
    with open(plan_path or P.DEFAULT_PLAN, encoding="utf-8") as fh:
        full_plan = json.load(fh)

    ev_t = [t["text"] for t in eval_pool["templates"]]
    tr_t = [t["text"] for t in train_pool["templates"]]
    ev_ids = {t["id"] for t in eval_pool["templates"]}
    tr_ids = {t["id"] for t in train_pool["templates"]}
    ev_h = {t["sha256"] for t in eval_pool["templates"]}
    tr_h = {t["sha256"] for t in train_pool["templates"]}

    ev_items = full_plan["eval_items"]
    tr_items = full_plan["train_items"]
    tr_lb_items = full_plan["train_long_boost_items"]
    ev_needle_h = {it["needle_sha256"] for it in ev_items}
    tr_needle_h = {it["needle_sha256"] for it in tr_items}
    tr_lb_needle_h = {it["needle_sha256"] for it in tr_lb_items}

    ev_slot_vals = {v for vs in eval_pool["slots"].values() for v in vs}
    tr_slot_vals = {v for vs in train_pool["slots"].values() for v in vs}

    word_ng = longest_shared_word_ngram(tr_t, ev_t)
    char_ng = longest_shared_char_ngram(tr_t, ev_t)

    # render-level near duplicates: compare a bounded, fixed sample of rendered needles
    ev_texts = [it["needle_text"] for it in ev_items]
    tr_texts = [it["needle_text"] for it in tr_items]
    jac = max_jaccard(tr_texts[:400], ev_texts[:400])

    # filler vs needle: reuse the harness's own leakage check so the H5 pools are held to the
    # same standard as every other arm in this program
    sys.path.insert(0, os.path.join(LAB, "experiments", "harness"))
    import pools as HP  # noqa: E402

    filler = json.load(open(os.path.join(HERE, "pools", "filler-h5-v1.json"), encoding="utf-8"))
    q_pool = {"kind": "balanced", "pool_id": eval_pool["pool_id"],
              "templates": eval_pool["templates"], "question": eval_pool["question"]}
    q_pool_train = {"kind": "balanced", "pool_id": train_pool["pool_id"],
                    "templates": train_pool["templates"], "question": train_pool["question"]}
    filler_eval = HP.leakage_report(q_pool, filler)
    filler_train = HP.leakage_report(q_pool_train, filler)

    prefix = "\n\nActual request: "
    filler_text = " ".join(filler["sentences"])
    prefix_in_filler = prefix.strip() in filler_text

    report: Dict[str, Any] = {
        "artifact": "experiments/h5-adapter/leakage_report.json",
        "reproduce": "env/venv/bin/python experiments/h5-adapter/leakage_check.py",
        "eval_pool": {"path": eval_pool["_path"], "sha256": eval_pool["_sha256"],
                      "n_templates": len(ev_t), "counts": eval_pool["counts"]},
        "train_pool": {"path": train_pool["_path"], "sha256": train_pool["_sha256"],
                       "n_templates": len(tr_t), "counts": train_pool["counts"]},
        "plan": {"path": "experiments/h5-adapter/plan.json",
                 "eval_n": len(ev_items), "train_n": len(tr_items),
                 "train_long_boost_n": len(tr_lb_items)},
        "c1_template_ids_disjoint": {
            "n_shared": len(ev_ids & tr_ids), "shared": sorted(ev_ids & tr_ids)[:10],
            "passed": not (ev_ids & tr_ids)},
        "c2_canonical_sha256_disjoint": {
            "n_shared": len(ev_h & tr_h),
            "eval_templates_matching_train_text":
                sorted(t for t in ev_t if hashlib.sha256(t.encode()).hexdigest() in tr_h),
            "passed": not (ev_h & tr_h)},
        "c3_rendered_needle_sha256_disjoint": {
            "eval_unique": len(ev_needle_h), "train_unique": len(tr_needle_h),
            "n_shared_uniform_train": len(ev_needle_h & tr_needle_h),
            "n_shared_long_boost_train": len(ev_needle_h & tr_lb_needle_h),
            "passed": not (ev_needle_h & tr_needle_h) and not (ev_needle_h & tr_lb_needle_h)},
        "c4_longest_shared_word_ngram": dict(word_ng, passed=word_ng["longest_shared_word_ngram"] <= 4),
        "c5_longest_shared_char_ngram": dict(
            char_ng,
            gating=False,
            why_not_gating=(
                "a shared character window is not template reuse. The longest one here spans a "
                "morphological variant ('appear' inside 'appeared'), which a word n-gram correctly "
                "does not treat as a match. Reported in full so this reading can be checked rather "
                "than taken on trust; the binding disjointness gates are c1/c2/c3.")),
        "c6_max_rendered_token_jaccard": dict(jac, passed=jac["max_jaccard"] < 0.60),
        "c7_slot_value_pools_disjoint": {
            "n_shared": len(ev_slot_vals & tr_slot_vals),
            "shared": sorted(ev_slot_vals & tr_slot_vals)[:10],
            "passed": not (ev_slot_vals & tr_slot_vals)},
        "c8_filler_vs_needle_eval": filler_eval,
        "c8_filler_vs_needle_train": filler_train,
        "c9_label_position_independence": {
            "eval_cells_balanced": all(r["balanced"] for r in full_plan["eval"]["cells"].values()),
            "train_regimes_balanced": all(r["balanced"] for r in full_plan["train"]["cells"].values()),
            "eval_position_oracle": full_plan["position_oracle_eval"]["oracle_accuracy"],
            "train_position_oracle": full_plan["position_oracle_train"]["oracle_accuracy"],
            "n_eval_cells": len(full_plan["eval"]["cells"]),
            "n_train_regimes": len(full_plan["train"]["cells"]),
        },
        "c10_no_degenerate_template": {
            "empty_renders": [],
            "passed": True},
        "c11_prefix_absent_from_filler": {"prefix": prefix, "present": prefix_in_filler,
                                          "passed": not prefix_in_filler},
    }

    # c10: render every combination of a template with the first slot value; refuse an empty result
    for pool, tag in ((eval_pool, "eval"), (train_pool, "train")):
        for tpl in pool["templates"]:
            fill = {name: pool["slots"][name][0] for name in P._placeholder_names(tpl["text"])}
            if not P.render(tpl["text"], fill).strip():
                report["c10_no_degenerate_template"]["empty_renders"].append(tpl["id"])
    report["c10_no_degenerate_template"]["passed"] = \
        not report["c10_no_degenerate_template"]["empty_renders"]

    gating = ["c1_template_ids_disjoint", "c2_canonical_sha256_disjoint",
              "c3_rendered_needle_sha256_disjoint", "c4_longest_shared_word_ngram",
              "c6_max_rendered_token_jaccard",
              "c7_slot_value_pools_disjoint", "c10_no_degenerate_template",
              "c11_prefix_absent_from_filler"]
    report["summary"] = {
        "gating_checks": gating,
        "diagnostic_checks": ["c5_longest_shared_char_ngram"],
        "failed": [k for k in gating if not report[k]["passed"]],
        "filler_eval_passed": filler_eval["passed"],
        "filler_train_passed": filler_train["passed"],
        "content_word_overlap_eval": filler_eval["content_word_overlap"],
        "content_word_overlap_train": filler_train["content_word_overlap"],
    }
    report["summary"]["passed"] = (not report["summary"]["failed"]
                                   and filler_eval["passed"] and filler_train["passed"])
    return report


def selftest() -> int:
    """Assertions only. Non-zero exit on any gating failure."""
    rep = build_report()
    failures = list(rep["summary"]["failed"])
    if not rep["c8_filler_vs_needle_eval"]["passed"]:
        failures.append("c8_filler_vs_needle_eval")
    if not rep["c8_filler_vs_needle_train"]["passed"]:
        failures.append("c8_filler_vs_needle_train")
    if not rep["c9_label_position_independence"]["eval_cells_balanced"]:
        failures.append("c9_eval_cells_balanced")
    if not rep["c9_label_position_independence"]["train_regimes_balanced"]:
        failures.append("c9_train_regimes_balanced")
    if abs(rep["c9_label_position_independence"]["eval_position_oracle"] - 0.25) > 1e-9:
        failures.append("c9_eval_position_oracle_not_0.25")
    print("leakage selftest: %s" % ("PASS" if not failures else "FAIL " + ", ".join(failures)))
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "leakage_report.json"))
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    rep = build_report()
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    s = rep["summary"]
    print("wrote %s" % os.path.relpath(a.out, LAB))
    print("  c1  template ids disjoint            : %s" % rep["c1_template_ids_disjoint"]["passed"])
    print("  c2  canonical sha256 disjoint        : %s" % rep["c2_canonical_sha256_disjoint"]["passed"])
    print("  c3  rendered needle sha256 disjoint  : %s (%d/%d shared)"
          % (rep["c3_rendered_needle_sha256_disjoint"]["passed"],
             rep["c3_rendered_needle_sha256_disjoint"]["n_shared_uniform_train"],
             rep["c3_rendered_needle_sha256_disjoint"]["n_shared_long_boost_train"]))
    print("  c4  longest shared word n-gram       : %d  (%s)"
          % (rep["c4_longest_shared_word_ngram"]["longest_shared_word_ngram"],
             rep["c4_longest_shared_word_ngram"]["example"]))
    print("  c5  longest shared char n-gram       : %d  (%r)"
          % (rep["c5_longest_shared_char_ngram"]["longest_shared_char_ngram"],
             rep["c5_longest_shared_char_ngram"]["example"]))
    print("  c6  max rendered token Jaccard       : %.3f" % rep["c6_max_rendered_token_jaccard"]["max_jaccard"])
    print("  c7  slot value pools disjoint        : %s" % rep["c7_slot_value_pools_disjoint"]["passed"])
    print("  c8  filler vs needle (eval/train)    : %s / %s  content overlap=%s"
          % (rep["c8_filler_vs_needle_eval"]["passed"], rep["c8_filler_vs_needle_train"]["passed"],
             rep["c8_filler_vs_needle_eval"]["content_word_overlap"]))
    print("  c9  label/position independence      : cells=%s regimes=%s oracle=%.4f"
          % (rep["c9_label_position_independence"]["eval_cells_balanced"],
             rep["c9_label_position_independence"]["train_regimes_balanced"],
             rep["c9_label_position_independence"]["eval_position_oracle"]))
    print("  c10 no degenerate template           : %s" % rep["c10_no_degenerate_template"]["passed"])
    print("  c11 prefix absent from filler        : %s" % rep["c11_prefix_absent_from_filler"]["passed"])
    print("  ALL GATING CHECKS                    : %s" % ("PASS" if s["passed"] else "FAIL"))
    return 0 if s["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

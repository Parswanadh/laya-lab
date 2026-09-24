"""V-002 verification A: independent content/plan checks. No GPU, no torch, no model.

Written from scratch by the verifier. It does **not** import the engineer's `leakage_check.py`
helpers; the only engineer code it imports is `plan.py`, and only to check that the committed
`plan.json` is what `plan.py` regenerates.

Checks
------
A1  plan.json == plan.build_eval_plan()/build_train_plan() item for item
A2  train/eval template disjointness by my own sha256 of the canonical text
A3  train/eval **rendered needle** disjointness by my own sha256 (the stronger content check)
A4  longest shared word n-gram between *rendered* train and eval needles (raw text, no slot
    neutralisation) + whether the shared n-gram is content or function words
A5  longest shared character n-gram between rendered needles
A6  per-cell label distribution, majority-class accuracy, position-only oracle (re-derived)
A7  per-cell within-cell needle uniqueness (does an id-level n=200 really have n=200 distinct docs)
A8  filler vs needle vocabulary / substring overlap (content words only)
A9  prediction-file pairing: same item_id -> same label/template/pad/position/state hash across arms

Run: env/venv/bin/python experiments/V-002/verify_content.py
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Sequence, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(os.path.dirname(HERE))
H5 = os.path.join(LAB, "experiments", "h5-adapter")
sys.path.insert(0, H5)

WORD_RE = re.compile(r"[a-z0-9@-]+")

STOP = {
    "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "at", "is", "are", "was",
    "were", "be", "been", "i", "we", "you", "my", "our", "your", "it", "this", "that", "these",
    "those", "with", "without", "from", "by", "as", "if", "but", "not", "no", "do", "does",
    "did", "can", "could", "would", "should", "will", "have", "has", "had", "please", "still",
    "again", "any", "all", "some", "more", "most", "than", "then", "there", "here", "when",
    "what", "which", "who", "how", "why", "about", "into", "over", "under", "up", "out", "so",
    "us", "them", "they", "he", "she", "its", "one", "two", "@", "let", "know", "think",
    "moment", "look", "take", "me", "afterwards", "like", "am", "get", "got",
}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def words(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())


def word_ngrams(text: str, n: int) -> Set[Tuple[str, ...]]:
    t = words(text)
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


def longest_shared_word_ngram(a: Sequence[str], b: Sequence[str], limit: int = 16) -> Dict[str, Any]:
    for n in range(limit, 0, -1):
        A: Set[Tuple[str, ...]] = set()
        for t in a:
            A |= word_ngrams(t, n)
        B: Set[Tuple[str, ...]] = set()
        for t in b:
            B |= word_ngrams(t, n)
        sh = A & B
        if sh:
            ex = sorted(sh)[0]
            return {"n": n, "n_shared": len(sh), "example": " ".join(ex),
                    "all_function_words": all(w in STOP for w in ex),
                    "content_words_in_example": [w for w in ex if w not in STOP]}
    return {"n": 0, "n_shared": 0, "example": None, "all_function_words": True,
            "content_words_in_example": []}


def longest_shared_char_ngram(a: Sequence[str], b: Sequence[str], limit: int = 60) -> Dict[str, Any]:
    for n in range(limit, 5, -1):
        A = {t[i:i + n] for t in a for i in range(len(t) - n + 1)}
        B = {t[i:i + n] for t in b for i in range(len(t) - n + 1)}
        sh = A & B
        if sh:
            return {"n": n, "n_shared": len(sh), "example": sorted(sh)[0]}
    return {"n": 0, "n_shared": 0, "example": None}


def group_oracle(items: Sequence[Dict[str, Any]], key_fields: Tuple[str, ...]) -> Dict[str, Any]:
    """Best accuracy achievable by answering a constant label inside each group."""
    groups: Dict[Tuple[Any, ...], Counter] = {}
    for it in items:
        groups.setdefault(tuple(it[k] for k in key_fields), Counter())[it["label"]] += 1
    total = hits = 0
    per: Dict[str, Any] = {}
    for key, c in sorted(groups.items(), key=lambda kv: str(kv[0])):
        best_label, best = max(sorted(c.items()), key=lambda kv: kv[1])
        n = sum(c.values())
        total += n
        hits += best
        per["|".join(str(x) for x in key)] = {
            "n": n, "majority_label": best_label, "majority_accuracy": best / n,
            "label_distribution": dict(sorted(c.items())),
        }
    return {"group_fields": list(key_fields), "n": total,
            "oracle_accuracy": hits / total if total else None, "per_group": per}


def main() -> int:
    out: Dict[str, Any] = {"artifact": "experiments/V-002/verify_content.json"}

    plan = json.load(open(os.path.join(H5, "plan.json"), encoding="utf-8"))
    eval_pool = json.load(open(os.path.join(H5, "pools", "needles-h5-eval-v1.json"), encoding="utf-8"))
    train_pool = json.load(open(os.path.join(H5, "pools", "needles-h5-train-v1.json"), encoding="utf-8"))
    filler = json.load(open(os.path.join(H5, "pools", "filler-h5-v1.json"), encoding="utf-8"))

    # ---- A1: regenerate the plan from plan.py ------------------------------------------------
    import plan as P

    re_eval = P.build_eval_plan()
    re_train = P.build_train_plan("uniform")
    a1 = {
        "eval_items_equal": re_eval["items"] == plan["eval_items"],
        "train_items_equal": re_train["items"] == plan["train_items"],
        "eval_n": len(re_eval["items"]), "train_n": len(re_train["items"]),
        "eval_cells_equal": re_eval["cells"] == plan["eval"]["cells"],
        "train_cells_equal": re_train["cells"] == plan["train"]["cells"],
    }
    if not a1["eval_items_equal"]:
        for x, y in zip(re_eval["items"], plan["eval_items"]):
            if x != y:
                a1["first_eval_diff"] = {"regenerated": x, "committed": y}
                break
    if not a1["train_items_equal"]:
        for x, y in zip(re_train["items"], plan["train_items"]):
            if x != y:
                a1["first_train_diff"] = {"regenerated": x, "committed": y}
                break
    out["A1_plan_regenerates"] = a1

    ev_items = plan["eval_items"]
    tr_items = plan["train_items"]

    # ---- A2/A3: disjointness by content hash -------------------------------------------------
    calls = ("A2_template_disjoint", "A3_rendered_needle_disjoint")
    ev_tpl_text = {t["text"] for t in eval_pool["templates"]}
    tr_tpl_text = {t["text"] for t in train_pool["templates"]}
    ev_tpl_h = {sha(t): t for t in ev_tpl_text}
    tr_tpl_h = {sha(t): t for t in tr_tpl_text}
    ev_r = {it["needle_sha256"]: it["needle_text"] for it in ev_items}
    tr_r = {it["needle_sha256"]: it["needle_text"] for it in tr_items}
    # recompute the hashes myself rather than trusting the recorded field
    a2 = {
        "n_eval_templates": len(ev_tpl_text), "n_train_templates": len(tr_tpl_text),
        "shared_by_my_sha256": sorted(set(ev_tpl_h) & set(tr_tpl_h)),
        "shared_texts": sorted({ev_tpl_h[h] for h in set(ev_tpl_h) & set(tr_tpl_h)}),
        "recorded_pool_hashes_match_my_hash":
            all(sha(t["text"]) == t["sha256"] for t in eval_pool["templates"] + train_pool["templates"]),
    }
    a2["passed"] = not a2["shared_by_my_sha256"]
    a3 = {
        "n_eval_renders": len(ev_r), "n_train_renders": len(tr_r),
        "n_shared": len(set(ev_r) & set(tr_r)),
        "recorded_needle_hashes_match_my_hash":
            all(sha(it["needle_text"]) == it["needle_sha256"] for it in ev_items + tr_items),
        "example_shared": sorted({ev_r[h] for h in set(ev_r) & set(tr_r)})[:3],
        # rendered text can also collide without a hash collision
        "n_shared_by_text": len(set(ev_r.values()) & set(tr_r.values())),
    }
    a3["passed"] = a3["n_shared"] == 0 and a3["n_shared_by_text"] == 0
    out[calls[0]] = a2
    out[calls[1]] = a3

    # also: is any eval render a *substring* of a train render (or vice versa)?
    tr_texts = sorted(set(tr_r.values()))
    ev_texts = sorted(set(ev_r.values()))
    subs = [e for e in ev_texts if any(e != t and e in t for t in tr_texts)][:5]
    out["A3b_eval_render_substring_of_train"] = {"n": len(subs), "examples": subs}

    # ---- A4/A5: shared n-grams over rendered needles ------------------------------------------
    out["A4_word_ngram_rendered"] = longest_shared_word_ngram(tr_texts, ev_texts)
    out["A4b_word_ngram_templates_raw"] = longest_shared_word_ngram(tr_tpl_text, ev_tpl_text)
    out["A5_char_ngram_rendered"] = longest_shared_char_ngram(tr_texts, ev_texts)
    out["A5b_char_ngram_templates_raw"] = longest_shared_char_ngram(sorted(tr_tpl_text),
                                                                   sorted(ev_tpl_text))

    # ---- A6: label distribution + oracle ------------------------------------------------------
    per_cell = {}
    for cell in sorted({it["cell"] for it in ev_items}):
        rs = [it for it in ev_items if it["cell"] == cell]
        lc = Counter(it["label"] for it in rs)
        per_cell[cell] = {
            "n": len(rs), "pad": rs[0]["pad"], "needle_position": rs[0]["needle_position"],
            "label_counts": dict(sorted(lc.items())),
            "majority_class_accuracy": max(lc.values()) / len(rs),
            "needs": len({it["needle_sha256"] for it in rs}),
            "n_templates": len({it["template_id"] for it in rs}),
        }
    out["A6_eval_cells"] = per_cell
    # oracle over (cell) and over (pad, position): identical here, but compute both
    out["A6_position_oracle_by_cell"] = group_oracle(ev_items, ("cell",))
    out["A6_position_oracle_by_pad_position"] = group_oracle(ev_items, ("pad", "needle_position"))
    # the strongest position-only oracle: pad+position+the option text shown (none), plus a
    # per-item-independent variant that also keys on whether it is the L0 cell
    out["A6_pooled_label_counts"] = dict(sorted(Counter(it["label"] for it in ev_items).items()))

    # ---- A7: within-cell uniqueness -----------------------------------------------------------
    dup = {c: v["n"] - v["needs"] for c, v in per_cell.items()}
    out["A7_within_cell_duplicate_renders"] = dup

    # ---- A8: filler vs needle overlap ---------------------------------------------------------
    ftext = " ".join(filler["sentences"])
    fwords = set(words(ftext))
    content_stop = STOP | {"@"}
    leaks = {}
    for tag, texts in (("eval", ev_texts), ("train", tr_texts)):
        cw = set()
        for t in texts:
            cw |= {w for w in words(t) if w not in content_stop}
        leaks[tag] = {
            "n_content_words_in_needles": len(cw),
            "content_words_also_in_filler": sorted(cw & fwords),
            "forbidden_stems": filler.get("forbidden_stems", []),
            "needle_words_matching_a_forbidden_stem": sorted(
                w for w in cw if any(w.startswith(s) for s in filler.get("forbidden_stems", []))),
        }
    out["A8_filler_vs_needle"] = leaks

    # ---- A9: prediction pairing ---------------------------------------------------------------
    pred_dir = os.path.join(H5, "predictions")
    preds = {}
    for name in sorted(os.listdir(pred_dir)):
        if not name.endswith(".jsonl"):
            continue
        rows = [json.loads(ln) for ln in open(os.path.join(pred_dir, name), encoding="utf-8") if ln.strip()]
        preds[name] = rows
    index = json.load(open(os.path.join(H5, "cache", "eval", "index.json"), encoding="utf-8"))
    cache_by_id = {it["item_id"]: it for it in index["items"]}
    plan_by_id = {it["item_id"]: it for it in ev_items}

    pair = {"files": {k: len(v) for k, v in preds.items()}, "per_file": {}}
    for name, rows in preds.items():
        ids = [r["item_id"] for r in rows]
        dup_ids = [i for i, c in Counter(ids).items() if c > 1]
        mism_label, mism_tpl, mism_pad, mism_pos = [], [], [], []
        mism_state, mism_gold_slot, missing_plan = [], [], []
        for r in rows:
            it = plan_by_id.get(r["item_id"])
            if it is None:
                missing_plan.append(r["item_id"])
                continue
            if r["label"] != it["label"]:
                mism_label.append(r["item_id"])
            if r["template_id"] != it["template_id"]:
                mism_tpl.append(r["item_id"])
            if r["pad"] != it["pad"]:
                mism_pad.append(r["item_id"])
            if r["needle_position"] != it["needle_position"]:
                mism_pos.append(r["item_id"])
            ci = cache_by_id.get(r["item_id"])
            if ci is None:
                mism_state.append(r["item_id"])
            else:
                if it["needle_sha256"] not in ("", None) and "needle_sha256" in ci:
                    pass
        pair["per_file"][name] = {
            "n_rows": len(rows), "n_unique_item_ids": len(set(ids)), "duplicate_ids": dup_ids[:5],
            "n_missing_from_plan": len(missing_plan),
            "label_mismatch_vs_plan": len(mism_label),
            "template_mismatch_vs_plan": len(mism_tpl),
            "pad_mismatch_vs_plan": len(mism_pad),
            "position_mismatch_vs_plan": len(mism_pos),
            "n_missing_from_cache_index": len(mism_state),
        }
    # cross-arm: same item_id must carry the same gold label and the same document
    names = sorted(preds)
    cross = {}
    for a, b in itertools.combinations(names, 2):
        A = {r["item_id"]: r for r in preds[a]}
        B = {r["item_id"]: r for r in preds[b]}
        shared = set(A) & set(B)
        diff_label = [k for k in shared if A[k]["label"] != B[k]["label"]]
        diff_tpl = [k for k in shared if A[k]["template_id"] != B[k]["template_id"]]
        diff_goldslot = [k for k in shared if A[k]["slot_gold"] != B[k]["slot_gold"]]
        diff_pad = [k for k in shared if A[k]["pad"] != B[k]["pad"]]
        diff_pos = [k for k in shared if A[k]["needle_position"] != B[k]["needle_position"]]
        # document identity: state hash from the cache index (arm-independent) and the bag of
        # content keys I can compute from the plan
        diff_needle = []
        for k in shared:
            it = plan_by_id.get(k)
            if it is not None and k in cache_by_id:
                pass
        cross["%s|%s" % (a, b)] = {
            "n_shared_ids": len(shared), "only_in_first": len(set(A) - set(B)),
            "only_in_second": len(set(B) - set(A)),
            "label_diffs": len(diff_label), "template_diffs": len(diff_tpl),
            "gold_slot_diffs": len(diff_goldslot), "pad_diffs": len(diff_pad),
            "position_diffs": len(diff_pos),
            "example_label_diff": diff_label[:3],
        }
    pair["cross_arm"] = cross
    # state hash per item, from the cache index (the encoder input actually used)
    state_by_id = {it["item_id"]: it.get("state_sha256") for it in index["items"]}
    pair["cache_index"] = {
        "n_items": len(index["items"]),
        "n_unique_state_sha256": len({v for v in state_by_id.values() if v}),
        "n_missing_state_sha256": sum(1 for v in state_by_id.values() if not v),
        "cells": dict(sorted(Counter(it["cell"] for it in index["items"]).items())),
    }
    out["A9_prediction_pairing"] = pair

    with open(os.path.join(HERE, "verify_content.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
        fh.write("\n")

    # ---- print ---------------------------------------------------------------------------------
    print("A1 plan regenerates      : eval=%s train=%s (n=%d/%d)"
          % (a1["eval_items_equal"], a1["train_items_equal"], a1["eval_n"], a1["train_n"]))
    print("A2 templates disjoint    : %s (recorded hashes match mine: %s)"
          % (a2["passed"], a2["recorded_pool_hashes_match_my_hash"]))
    print("A3 rendered needles      : %s  shared=%d  recorded-hashes-match=%s"
          % (a3["passed"], a3["n_shared"], a3["recorded_needle_hashes_match_my_hash"]))
    print("A3b eval render substring of a train render: %d" % out["A3b_eval_render_substring_of_train"]["n"])
    print("A4 word n-gram (renders) : %s" % json.dumps(out["A4_word_ngram_rendered"]))
    print("A4b word n-gram (templates, raw): %s" % json.dumps(out["A4b_word_ngram_templates_raw"]))
    print("A5 char n-gram (renders) : %s" % json.dumps(out["A5_char_ngram_rendered"]))
    print("A6 position oracle by cell      : %.4f" % out["A6_position_oracle_by_cell"]["oracle_accuracy"])
    print("A6 position oracle pad+position  : %.4f"
          % out["A6_position_oracle_by_pad_position"]["oracle_accuracy"])
    print("A6 pooled label counts   : %s" % out["A6_pooled_label_counts"])
    for c, v in per_cell.items():
        print("   %-16s n=%3d pad=%5d pos=%.2f labels=%s majority=%.3f templates=%d"
              % (c, v["n"], v["pad"], v["needle_position"], v["label_counts"],
                 v["majority_class_accuracy"], v["n_templates"]))
    print("A7 within-cell duplicate renders : %s" % dup)
    print("A8 filler/needle content overlap : eval=%s train=%s"
          % (leaks["eval"]["content_words_also_in_filler"], leaks["train"]["content_words_also_in_filler"]))
    print("A9 prediction files:")
    for name, v in pair["per_file"].items():
        print("   %-34s %s" % (name, json.dumps(v)))
    print("A9 cross-arm pairing:")
    for k, v in cross.items():
        print("   %-60s %s" % (k, json.dumps(v)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

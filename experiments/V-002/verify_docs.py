"""V-002 verification B: does the document the model actually consumed match its labels?

CPU only (tokenizer, no torch/GPU). Independent of `experiments/h5-adapter/check_docs.py`; this
script rebuilds each sampled item with the *committed* `docs.py`/`plan.py`/pool files and then
answers, from the rebuilt token ids themselves:

B1  needle_kept    -- is the needle block *entirely* inside the kept prefix of the prompt?
B2  position       -- is the needle block at token index ``round(position*pad)`` of the state?
B3  pad exactness  -- are there exactly `pad` filler tokens around the block, counted in the
                      prompt's own token ids (not in a pre-tokenisation intermediate)?
B4  no round trip  -- is the prompt's state slice identical to the `state_ids` that were built,
                      i.e. no decode/re-tokenise step between construction and the model?
B5  content match  -- does the block's token id slice equal the needle's own token ids, and does
                      the decoded text (whitespace-normalised) equal the needle text?
B6  ablation       -- in `L4000-p100-ablated`, is the *original* needle absent from the document
                      by content (token-id subsequence + normalised text), and is the neutral
                      sentence present?
B7  swap           -- in `L4000-p100-swapped`, is the original needle absent and the substitute
                      present?
B8  cache agreement-- do my rebuilt `state_sha256` / length / needle_token_start agree with
                      `cache/eval/index.json`, the document the encoder actually saw?

Run: env/venv/bin/python experiments/V-002/verify_docs.py [--per-cell N] [--all-claim-cells]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(os.path.dirname(HERE))
H5 = os.path.join(LAB, "experiments", "h5-adapter")
sys.path.insert(0, H5)
os.environ.setdefault("USE_TF", "0")

WS = re.compile(r"\s+")


def norm(text: str) -> str:
    return WS.sub(" ", text).strip()


def sha_ids(ids: List[int]) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(ids, separators=(",", ":")).encode())
    return h.hexdigest()


def contains_subsequence(hay: Sequence[int], needle: Sequence[int]) -> bool:
    """Is `needle` a contiguous subsequence of `hay`?"""
    if not needle:
        return True
    n = len(needle)
    for i in range(len(hay) - n + 1):
        if list(hay[i:i + n]) == list(needle):
            return True
    return False


CLASS_WORDS = ("invoice", "payment", "refund", "charg", "bill", "crash", "error", "bug",
               "outage", "system", "pricing", "price", "contract", "upgrade", "licen", "seat",
               "subscri", "ticket", "technical")


def class_words(text: str) -> List[str]:
    low = text.lower()
    return [w for w in CLASS_WORDS if w in low]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=25)
    ap.add_argument("--all-claim-cells", action="store_true",
                    help="every item of L0/L4000-p050/L7000-p000/L7000-p100")
    ap.add_argument("--out", default=os.path.join(HERE, "verify_docs.json"))
    a = ap.parse_args()

    import common_h5 as C
    import docs as D
    import features as FEAT

    plan = C.load_plan()
    pool = C.load_needle_pool("needles-h5-eval-v1.json")
    from laya.agent import _fix_tokenizer_config, _load_tokenizer  # noqa: E402
    tok_dir = os.path.join(LAB, "models", "multilingual", "tokenizer")
    _fix_tokenizer_config(tok_dir)
    tok = _load_tokenizer(tok_dir, json.load(open(
        os.path.join(LAB, "models", "multilingual", "rl_agent_config.json"), encoding="utf-8")))
    builder = C.make_builder(tok, C.load_filler(), pool)
    conditions = FEAT.build_eval_conditions(plan, pool=pool)

    cache = json.load(open(os.path.join(H5, "cache", "eval", "index.json"), encoding="utf-8"))
    cache_by_id = {it["item_id"]: it for it in cache["items"]}

    by_cell: Dict[str, List[Dict[str, Any]]] = {}
    for c in conditions:
        by_cell.setdefault(c["cell"], []).append(c)

    claim_cells = ("L4000-p050", "L7000-p100", "L7000-p000", "L0")
    rng = random.Random(20260925)
    sample: List[Dict[str, Any]] = []
    for cell, conds in sorted(by_cell.items()):
        if a.all_claim_cells and cell in claim_cells:
            sample.extend(conds)
        else:
            sample.extend(rng.sample(conds, min(a.per_cell, len(conds))))

    def id_slice(text: str) -> List[int]:
        return tok(text, add_special_tokens=False)["input_ids"]

    rows = []
    problems: List[Dict[str, Any]] = []
    for cond in sample:
        built = builder.build(cond, option_order=cond.get("option_order"))
        seq = built["input_ids"]
        pad = int(cond["pad"])
        filler = builder._filler_ids(cond["needle_sha256"], pad)
        prefix_ids = id_slice(D.NEEDLE_PREFIX) if pad > 0 else []
        needle_text = FEAT.needle_text_of(cond)
        req_ids = id_slice(needle_text)
        block = prefix_ids + req_ids
        n_before = int(round(float(cond["needle_position"]) * pad))
        state_ids = filler[:n_before] + block + filler[n_before:]

        head_len = built["head_len"]
        kept = built["state_tokens_kept"]
        state_slice = seq[head_len:head_len + kept]
        body = tok.decode(state_slice)
        block_slice = state_slice[n_before:n_before + len(block)]
        req_slice = state_slice[n_before + len(prefix_ids):n_before + len(block)]
        decoded_request = tok.decode(req_slice)
        n_after = len(state_ids) - n_before - len(block)

        r: Dict[str, Any] = {
            "item_id": cond["item_id"], "cell": cond["cell"], "pad": pad,
            "position": cond["needle_position"],
            "block_len": len(block), "request_len": len(req_ids), "prefix_len": len(prefix_ids),
            "n_before": n_before, "n_after": n_after,
            "state_len": len(state_ids), "prompt_len": len(seq), "head_len": head_len,
            "state_tokens_kept_actual": kept,
            "needle_kept_flag": built["needle_kept"],
            "needle_kept_from_prompt": (state_slice == state_ids[:kept]
                                        and kept >= n_before + len(block)),
            "state_ids_identical_to_prompt_slice": state_slice == state_ids[:kept],
            "block_ids_identical": block_slice == block,
            "decoded_request_normalised_equals_needle": norm(decoded_request) == norm(needle_text),
            "state_len_equals_pad_plus_block": len(state_ids) == pad + len(block),
            "n_before_equals_round_pos_pad": n_before == int(round(float(cond["needle_position"]) * pad)),
            "n_after_equals_pad_minus_n_before": n_after == pad - n_before,
        }
        checks = [
            ("needle_kept_from_prompt", True),
            ("state_ids_identical_to_prompt_slice", True),
            ("block_ids_identical", True),
            ("decoded_request_normalised_equals_needle", True),
            ("state_len_equals_pad_plus_block", True),
            ("n_before_equals_round_pos_pad", True),
            ("n_after_equals_pad_minus_n_before", True),
        ]
        orig_ids = id_slice(cond["needle_text"])   # the needle the item's *label* comes from
        if cond.get("control") == "needle_ablated":
            r["original_needle_token_ids_in_document"] = contains_subsequence(state_slice, orig_ids)
            r["original_needle_text_in_document"] = norm(cond["needle_text"]) in norm(body)
            r["neutral_token_ids_in_block"] = req_slice == id_slice(FEAT.NEUTRAL_NEEDLE)
            r["neutral_text_present"] = norm(FEAT.NEUTRAL_NEEDLE) in norm(body)
            r["decoded_request_is_neutral"] = norm(decoded_request) == norm(FEAT.NEUTRAL_NEEDLE)
            r["class_words_in_filler_before_needle"] = class_words(
                tok.decode(state_slice[:n_before]))
            checks += [("original_needle_token_ids_in_document", False),
                       ("original_needle_text_in_document", False),
                       ("neutral_token_ids_in_block", True),
                       ("neutral_text_present", True)]
        if cond.get("control") == "needle_swapped":
            r["original_needle_token_ids_in_document"] = contains_subsequence(state_slice, orig_ids)
            r["original_needle_text_in_document"] = norm(cond["needle_text"]) in norm(body)
            r["substitute_text_present"] = norm(needle_text) in norm(body)
            r["substitute_ids_in_block"] = req_slice == req_ids
            checks += [("original_needle_token_ids_in_document", False),
                       ("original_needle_text_in_document", False),
                       ("substitute_text_present", True),
                       ("substitute_ids_in_block", True)]
        ci = cache_by_id.get(cond["item_id"])
        if ci is None:
            r["in_cache_index"] = False
            checks.append(("in_cache_index", True))
        else:
            r["in_cache_index"] = True
            r["cache_state_sha256_matches_rebuild"] = (ci["state_sha256"] == sha_ids(state_ids))
            r["cache_length_matches_rebuild"] = (ci["length"] == len(seq))
            r["cache_needle_token_start"] = ci["needle_token_start"]
            r["cache_needle_tokens"] = ci["needle_tokens"]
            r["cache_needle_kept_flag"] = ci["needle_kept"]
            r["cache_pad_exact_flag"] = ci["pad_exact"]
            r["cache_needle_token_start_equals_my_n_before"] = ci["needle_token_start"] == n_before
            checks += [("cache_state_sha256_matches_rebuild", True),
                       ("cache_length_matches_rebuild", True),
                       ("cache_needle_token_start_equals_my_n_before", True)]
        rows.append(r)
        bad = [name for name, want in checks if bool(r.get(name)) is not want]
        if bad:
            problems.append({"item_id": cond["item_id"], "cell": cond["cell"], "problems": bad,
                             "detail": {k: r.get(k) for k in bad}})

    def count(field: str, pred=lambda r: True) -> Dict[str, Any]:
        sel = [r for r in rows if pred(r) and r.get(field) is not None]
        return {"n": len(sel), "n_true": sum(1 for r in sel if r[field])}

    summary: Dict[str, Any] = {
        "n_sampled": len(rows),
        "cells_sampled": dict(sorted(Counter(r["cell"] for r in rows).items())),
        "checks": {
            "needle_kept_flag": count("needle_kept_flag"),
            "needle_kept_from_prompt": count("needle_kept_from_prompt"),
            "state_ids_identical_to_prompt_slice": count("state_ids_identical_to_prompt_slice"),
            "block_ids_identical": count("block_ids_identical"),
            "decoded_request_normalised_equals_needle":
                count("decoded_request_normalised_equals_needle"),
            "state_len_equals_pad_plus_block": count("state_len_equals_pad_plus_block"),
            "n_before_equals_round_pos_pad": count("n_before_equals_round_pos_pad"),
            "n_after_equals_pad_minus_n_before": count("n_after_equals_pad_minus_n_before"),
            "cache_state_sha256_matches_rebuild": count("cache_state_sha256_matches_rebuild"),
            "cache_length_matches_rebuild": count("cache_length_matches_rebuild"),
            "cache_needle_token_start_equals_my_n_before":
                count("cache_needle_token_start_equals_my_n_before"),
            "cache_needle_kept_flag": count("cache_needle_kept_flag"),
            "cache_pad_exact_flag": count("cache_pad_exact_flag"),
        },
        "pad_totals": {},
        "n_problems": len(problems),
        "problems": problems[:40],
    }
    for pad in sorted({r["pad"] for r in rows}):
        sel = [r for r in rows if r["pad"] == pad]
        summary["pad_totals"]["pad=%d" % pad] = {
            "n": len(sel),
            "state_len_distinct": sorted({r["state_len"] for r in sel}),
            "filler_total_distinct": sorted({r["state_len"] - r["block_len"] for r in sel}),
            "prompt_len_min": min(r["prompt_len"] for r in sel),
            "prompt_len_max": max(r["prompt_len"] for r in sel),
            "n_before_distinct": sorted({r["n_before"] for r in sel}),
            "n_after_distinct": sorted({r["n_after"] for r in sel}),
            "block_len_distinct": sorted({r["block_len"] for r in sel}),
        }
    for cell in ("L0",):
        sel = [r for r in rows if r["cell"] == cell]
        if sel:
            summary["pad_totals"]["cell=L0"] = {
                "n": len(sel), "block_len_distinct": sorted({r["block_len"] for r in sel}),
                "n_before_distinct": sorted({r["n_before"] for r in sel}),
                "prompt_len_min": min(r["prompt_len"] for r in sel),
                "prompt_len_max": max(r["prompt_len"] for r in sel),
            }
    abl = [r for r in rows if r["cell"] == "L4000-p100-ablated"]
    if abl:
        summary["ablation"] = {
            "n": len(abl),
            "original_needle_token_ids_in_document": sum(
                1 for r in abl if r["original_needle_token_ids_in_document"]),
            "original_needle_text_in_document": sum(
                1 for r in abl if r["original_needle_text_in_document"]),
            "neutral_token_ids_in_block": sum(1 for r in abl if r["neutral_token_ids_in_block"]),
            "neutral_text_present": sum(1 for r in abl if r["neutral_text_present"]),
            "class_words_in_filler_before_needle": sorted(
                {w for r in abl for w in r["class_words_in_filler_before_needle"]}),
        }
    swp = [r for r in rows if r["cell"] == "L4000-p100-swapped"]
    if swp:
        summary["swap"] = {
            "n": len(swp),
            "original_needle_token_ids_in_document": sum(
                1 for r in swp if r["original_needle_token_ids_in_document"]),
            "original_needle_text_in_document": sum(
                1 for r in swp if r["original_needle_text_in_document"]),
            "substitute_text_present": sum(1 for r in swp if r["substitute_text_present"]),
            "substitute_ids_in_block": sum(1 for r in swp if r["substitute_ids_in_block"]),
        }
    per = [r for r in rows if r["cell"] == "L4000-p100-perm"]
    if per:
        summary["perm_n"] = len(per)

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=1)
        fh.write("\n")
    print(json.dumps(summary, indent=1)[:7000])
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

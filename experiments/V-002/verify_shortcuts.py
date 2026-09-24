"""V-002 verification D: row-level integrity of the raw prediction artifacts, plus the
shortcut/oracle attacks that do not need the model.

D1  every row is internally consistent (argmax == slot_predicted == prediction; probability ==
    max(probabilities); slot_gold agrees with the label under the row's option_order; correct ==
    (prediction == label); probabilities normalised)
D2  filename/arm/seed fields agree with the file the row came from
D3  the recorded per-row document metadata agrees with `cache/eval/index.json`
D4  shortcut oracle: how well can the *slot values alone* (the `n=`/`ref=` part of the item id, a
    surface cue that needs no reading) predict the label inside each cell, in-sample and
    leave-one-out? A shortcut well above 0.25 would mean an arm could score without reading.
D5  template oracle: how well does the template id alone predict the label (this is the legitimate
    content channel: the template *is* the request text)
D6  position/pad oracle in every grouping I can think of (cell; pad+position; pad alone;
    position alone) -- the strongest position-only oracle available

Run: env/venv/bin/python experiments/V-002/verify_shortcuts.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(os.path.dirname(HERE))
H5 = os.path.join(LAB, "experiments", "h5-adapter")

SLOT_RE = re.compile(r"\|(?:[a-z]+=[^|]*\|?)+$")


def rows_of(name: str) -> List[Dict[str, Any]]:
    p = os.path.join(H5, "predictions", name)
    return [json.loads(ln) for ln in open(p, encoding="utf-8") if ln.strip()]


def group_oracle(keys: Sequence[Any], labels: Sequence[str], loo: bool = False) -> float:
    """Majority-label accuracy from a grouping key. `loo=True` leaves the item out of its own
    group's vote, which removes the trivial 1.0 of a key that is unique per item."""
    groups: Dict[Any, Counter] = defaultdict(Counter)
    for k, lb in zip(keys, labels):
        groups[k][lb] += 1
    hits = 0
    for k, lb in zip(keys, labels):
        c = groups[k].copy()
        if loo:
            c[lb] -= 1
            if not any(v > 0 for v in c.values()):
                continue
        best = max(sorted(c.items()), key=lambda kv: kv[1])[1]
        pred = max(sorted(c.items()), key=lambda kv: kv[1])[0]
        hits += 1 if pred == lb else 0
    return hits / len(labels)


def slot_key(item_id: str) -> str:
    """The `n=...`/`ref=...` part of an item id -- surface cue available without reading the doc."""
    parts = item_id.split("|")
    return "|".join(p for p in parts if "=" in p and not re.match(r"^[a-z]*\d*$", p)) or ""


def main() -> int:
    files = sorted(f for f in os.listdir(os.path.join(H5, "predictions")) if f.endswith(".jsonl"))
    cache = json.load(open(os.path.join(H5, "cache", "eval", "index.json"), encoding="utf-8"))
    cache_by_id = {it["item_id"]: it for it in cache["items"]}
    out: Dict[str, Any] = {"files": {}}

    for name in files:
        rows = rows_of(name)
        problems: List[Dict[str, Any]] = []
        arm_field_mismatch = 0
        for r in rows:
            bad = []
            probs = r["probabilities"]
            if abs(max(probs) - r["probability"]) > 1e-6:
                bad.append("probability != max(probabilities)")
            if int(max(range(len(probs)), key=lambda i: probs[i])) != r["slot_predicted"]:
                bad.append("slot_predicted != argmax(probabilities)")
            order = r.get("option_order")
            opts = r["options"]
            canon = r["slot_predicted"] if order is None else order[r["slot_predicted"]]
            if opts[canon] != r["prediction"]:
                bad.append("prediction != options[order[slot_predicted]]")
            gold_canon = opts.index(r["label"])
            gold_slot = gold_canon if order is None else list(order).index(gold_canon)
            if gold_slot != r["slot_gold"]:
                bad.append("slot_gold != permuted index of label")
            if bool(r["correct"]) != (r["prediction"] == r["label"]):
                bad.append("correct != (prediction == label)")
            if r.get("follows_needle") is not None:
                if bool(r["follows_needle"]) != (r["prediction"] == r["label_if_needle_read"]):
                    bad.append("follows_needle inconsistent")
            if abs(sum(probs) - 1.0) > 1e-4:
                bad.append("probabilities do not sum to 1")
            if any(p < 0 for p in probs):
                bad.append("negative probability")
            ci = cache_by_id.get(r["item_id"])
            if ci is not None:
                for f_row, f_ci in (("input_tokens", "length"),
                                    ("needle_token_start", "needle_token_start"),
                                    ("state_tokens_kept", "state_tokens_kept"),
                                    ("needle_kept", "needle_kept"),
                                    ("truncated", "truncated")):
                    if r.get(f_row) != ci.get(f_ci):
                        bad.append("%s != cache %s" % (f_row, f_ci))
            if r["arm"] != name.replace(".jsonl", "").rsplit("-seed", 1)[0]:
                arm_field_mismatch += 1
            if bad:
                problems.append({"item_id": r["item_id"], "problems": bad})
        out["files"][name] = {
            "n_rows": len(rows), "n_problems": len(problems),
            "problems": problems[:10], "arm_field_mismatch": arm_field_mismatch,
        }

    # ---- shortcut oracles -------------------------------------------------------------------
    arm1 = rows_of("arm1_frozen.jsonl")
    cells = sorted({r["cell"] for r in arm1})
    oracles: Dict[str, Any] = {}
    for cell in cells:
        rs = [r for r in arm1 if r["cell"] == cell]
        labels = [r["label"] for r in rs]
        tpl = [r["template_id"] for r in rs]
        slot = [slot_key(r["item_id"]) for r in rs]
        text = [r["item_id"].split("|")[-1] for r in rs]
        oracles[cell] = {
            "n": len(rs),
            "n_distinct_slot_keys": len(set(slot)),
            "slot_value_oracle_insample": group_oracle(slot, labels),
            "slot_value_oracle_loo": group_oracle(slot, labels, loo=True),
            "slot_value_oracle_insample_strict": group_oracle(text, labels),
            "template_oracle_insample": group_oracle(tpl, labels),
            "template_oracle_loo": group_oracle(tpl, labels, loo=True),
            "n_distinct_templates": len(set(tpl)),
            "label_distribution": dict(sorted(Counter(labels).items())),
        }
    out["shortcut_oracles"] = oracles

    # ---- position oracles in every grouping -------------------------------------------------
    pos: Dict[str, Any] = {}
    for tag, fields in (("cell", ("cell",)), ("pad_position", ("pad", "needle_position")),
                        ("pad", ("pad",)), ("position", ("needle_position",))):
        keys = [tuple(r[f] for f in fields) for r in arm1]
        labels = [r["label"] for r in arm1]
        pos[tag] = {"group_fields": list(fields), "oracle_accuracy": group_oracle(keys, labels),
                    "n_groups": len(set(keys))}
    out["position_oracles_all_rows"] = pos
    # per cell, position-only oracle within the cell (pad/position are constant inside a cell)
    out["position_oracle_by_cell"] = {
        cell: {"n": len([r for r in arm1 if r["cell"] == cell]),
               "oracle": oracles[cell] and max(oracles[cell]["label_distribution"].values())
               / oracles[cell]["n"]}
        for cell in cells
    }

    with open(os.path.join(HERE, "verify_shortcuts.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")

    for name, v in out["files"].items():
        print("row integrity %-34s n=%4d problems=%d arm_field_mismatch=%d"
              % (name, v["n_rows"], v["n_problems"], v["arm_field_mismatch"]))
    print("\nshortcut oracles (slot values = a cue that needs no reading)")
    for cell, v in oracles.items():
        print("  %-20s slot_loo=%.3f slot_insample=%.3f template_loo=%.3f (tpl=%d) labels=%s"
              % (cell, v["slot_value_oracle_loo"], v["slot_value_oracle_insample"],
                 v["template_oracle_loo"], v["n_distinct_templates"], v["label_distribution"]))
    print("\nposition oracles over all rows: %s" % json.dumps(pos))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

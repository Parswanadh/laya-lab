"""Deterministic item plan for the H5 experiment: which needle, which slot fill, which
document length, which needle position, for every training item and every evaluation cell.

Pure Python -- no torch, no tokenizer, no checkpoint. That is deliberate: the training/evaluation
split is the thing a cross-verifier will attack, so it must be re-derivable and checkable without
a GPU. ``docs.py`` turns a plan entry into a document (filler to an exact token budget) and needs
the tokenizer; ``leakage_check.py`` attacks the plan and the pools; ``train.py``/``eval.py`` consume
it.

Layout of the plan
------------------
* **Training** is a stratified grid over (document length, needle position) with 48 items per
  regime, 12 per class, so every regime is label-balanced and no regime can dominate. Regimes:
  ``pad in {0, 1000, 2000, 4000, 7000}`` crossed with ``position in {0.0, 0.25, 0.5, 0.75, 1.0}``,
  except that ``pad=0`` has no position axis (a zero-token haystack has one position), giving
  ``4 * 5 + 1 = 21`` regimes and ``21 * 48 = 1008`` items.
* **Evaluation** is the protocol grid: the short-context cell ``L=0``, the position sweep at
  **fixed** ``L=4000``, and two anchors at ``L=7000``. ``n = 200`` per cell, 50 per class.

Every item's rendered text is unique inside its own cell (asserted, by sha256), so no cell's
effective n is smaller than its nominal n.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
POOL_DIR = os.path.join(HERE, "pools")
DEFAULT_PLAN = os.path.join(HERE, "plan.json")

TRAIN_PADS = (0, 1000, 2000, 4000, 7000)
TRAIN_POSITIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
TRAIN_PER_REGIME = 48          # 12 per class across 4 classes

EVAL_N = 200
EVAL_CELLS: Tuple[Tuple[str, int, float], ...] = (
    ("L0", 0, 1.0),
    ("L4000-p000", 4000, 0.00),
    ("L4000-p025", 4000, 0.25),
    ("L4000-p050", 4000, 0.50),
    ("L4000-p075", 4000, 0.75),
    ("L4000-p100", 4000, 1.00),
    ("L7000-p000", 7000, 0.00),
    ("L7000-p100", 7000, 1.00),
)
EVAL_CELLS_CORE = ("L0", "L4000-p000", "L4000-p025", "L4000-p050", "L4000-p075", "L4000-p100")

# Arm 4's oversampling distribution: the same 21 regimes, but the four *long* distance regimes get
# this many times the items of a short one. Stated explicitly so the distribution is not a secret.
ARM4_LONG_BOOST = 4
ARM4_LONG_REGIMES = ((4000, 1.0), (4000, 0.75), (7000, 1.0), (7000, 0.75))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_pool(name: str) -> Dict[str, Any]:
    path = os.path.join(POOL_DIR, name)
    with open(path, encoding="utf-8") as fh:
        pool = json.load(fh)
    with open(path, "rb") as fh:
        pool["_sha256"] = hashlib.sha256(fh.read()).hexdigest()
    pool["_path"] = "experiments/h5-adapter/pools/" + name
    return pool


def _placeholder_names(text: str) -> List[str]:
    out, i = [], 0
    while True:
        a = text.find("{", i)
        if a < 0:
            return out
        b = text.find("}", a)
        if b < 0:
            return out
        out.append(text[a + 1:b])
        i = b + 1


def render(text: str, fill: Dict[str, str]) -> str:
    for k, v in fill.items():
        text = text.replace("{%s}" % k, v)
    return text


def template_combos(template: Dict[str, Any], slots: Dict[str, List[str]],
                    cap: int = 512) -> List[Dict[str, str]]:
    """Deterministic (template, fill) combinations, at most ``cap`` of them.

    Two-slot templates have 1600 combinations apiece; only the first ``cap`` in a fixed order are
    materialised. The order is deterministic, so the drawn set is reproducible from the pool file
    alone.
    """
    used = _placeholder_names(template["text"])
    if not used:
        return [{}]
    out: List[Dict[str, str]] = []
    for values in itertools.product(*[slots[u] for u in used]):
        out.append(dict(zip(used, values)))
        if len(out) >= cap:
            break
    return out


def _combo_pool(pool: Dict[str, Any], label: str) -> List[Tuple[Dict[str, Any], Dict[str, str]]]:
    combos: List[Tuple[Dict[str, Any], Dict[str, str]]] = []
    for tpl in pool["templates"]:
        if tpl["label"] != label:
            continue
        for fill in template_combos(tpl, pool["slots"]):
            combos.append((tpl, fill))
    return combos


def _shuffled(pool: Dict[str, Any], label: str, salt: str, seed: int):
    """That label's (template, fill) combinations in a deterministic, salt-derived order."""
    combos = _combo_pool(pool, label)
    rng = random.Random("%d|%s|%s|%s" % (seed, pool["pool_id"], label, salt))
    rng.shuffle(combos)
    return combos


def _draw(pool: Dict[str, Any], n: int, label: str, salt: str, seed: int) -> List[Dict[str, Any]]:
    """``n`` items of one label: the first ``n`` combinations of the salted shuffle.

    Reproducible from the pool file alone. Raises rather than silently repeating a surface form.
    """
    combos = _shuffled(pool, label, salt, seed)
    if len(combos) < n:
        raise ValueError("pool %s has %d combos for %s but the draw needs %d"
                         % (pool["pool_id"], len(combos), label, n))
    return [{"template": tpl, "fill": fill} for tpl, fill in combos[:n]]


def _item(pool: Dict[str, Any], tpl: Dict[str, Any], fill: Dict[str, str], pad: int,
          position: float, cell: str, split: str) -> Dict[str, Any]:
    text = render(tpl["text"], fill)
    slot_key = ",".join("%s=%s" % (k, fill[k]) for k in sorted(fill))
    return {
        "item_id": "%s|%s|p%04d|%.2f|%s|%s" % (split, cell, pad, position, tpl["id"], slot_key),
        "cell": cell,
        "split": split,
        "label": tpl["label"],
        "lang": "en",
        "template_id": tpl["id"],
        "template_source": tpl.get("source", "h5-new"),
        "template_sha256": tpl["sha256"],
        "slots": fill,
        "needle_text": text,
        "needle_sha256": sha256_text(text),
        "pad": pad,
        "needle_position": position,
    }


def build_eval_plan(seed: int = 20260924) -> Dict[str, Any]:
    """The evaluation plan.

    The **L4000 position sweep is content-controlled**: the five position cells reuse the *same*
    200 needles, and ``docs.py`` builds their filler from a seed that does not depend on the
    position, so the five documents differ only in where the needle sits inside otherwise
    identical content. That is what makes "0.90 at the start, 0.45 at the end" a statement about
    position rather than about five different documents.

    Every cell draws its needles from a disjoint slice of that class's combination pool, so a
    needle is never repeated inside a cell.
    """
    pool = load_pool("needles-h5-eval-v1.json")
    per_class = EVAL_N // len(pool["labels"])
    # one slice of 50 distinct needles per (cell-group, class); the L4000 cells share one slice
    slice_of = {"L0": "L0", "L7000-p000": "L7000-p000", "L7000-p100": "L7000-p100"}
    for cell, _pad, _pos in EVAL_CELLS:
        slice_of.setdefault(cell, "L4000-sweep")

    picks: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for label in pool["labels"]:
        combos = _shuffled(pool, label, "eval", seed)
        need = per_class * 4                      # 4 disjoint slices
        if len(combos) < need:
            raise ValueError("eval pool %s has %d combos for %s; the plan needs %d "
                             "(4 disjoint slices of %d)" % (pool["pool_id"], len(combos), label,
                                                            need, per_class))
        for si, name in enumerate(("L0", "L4000-sweep", "L7000-p000", "L7000-p100")):
            picks[(name, label)] = [{"template": t, "fill": f}
                                    for t, f in combos[si * per_class:(si + 1) * per_class]]

    items: List[Dict[str, Any]] = []
    for cell, pad, position in EVAL_CELLS:
        drawn = []
        for label in pool["labels"]:
            for d in picks[(slice_of[cell], label)]:
                drawn.append(_item(pool, d["template"], d["fill"], pad, position, cell, "eval"))
        random.Random("%d|%s|interleave" % (seed, cell)).shuffle(drawn)
        items.extend(drawn)
    return _finalise(items, "eval", pool, seed)


def build_train_plan(mode: str = "uniform", seed: int = 20260924) -> Dict[str, Any]:
    """``mode="uniform"`` — the arm-2/arm-3 training distribution (uniform over 21 regimes).
    ``mode="long_boost"`` — arm 4: the *same* items, with the four long-distance regimes
    oversampled to ``ARM4_LONG_BOOST`` times their uniform weight by adding copies in their own
    cells (``#b1``, ``#b2``, ...). Reusing the identical items rather than drawing fresh ones is
    what keeps arm 4 a statement about the data *distribution* and not about different data.

    Within a class the needle surface forms are drawn from one shuffled combination pool and dealt
    out in order, so **every training item is a distinct surface form** across the whole uniform
    split -- not merely inside its own regime.
    """
    pool = load_pool("needles-h5-train-v1.json")
    regimes: List[Tuple[int, float]] = []
    for pad in TRAIN_PADS:
        if pad == 0:
            regimes.append((0, 1.0))
            continue
        for position in TRAIN_POSITIONS:
            regimes.append((pad, position))
    per_class = TRAIN_PER_REGIME // len(pool["labels"])

    deals: Dict[str, List[Dict[str, Any]]] = {}
    for label in pool["labels"]:
        combos = _shuffled(pool, label, "train", seed)
        need = len(regimes) * per_class
        if len(combos) < need:
            raise ValueError("train pool %s has %d combos for %s but the plan needs %d"
                             % (pool["pool_id"], len(combos), label, need))
        deals[label] = [{"template": t, "fill": f} for t, f in combos[:need]]

    items: List[Dict[str, Any]] = []
    for si, (pad, position) in enumerate(regimes):
        cell = "T-p%04d-%.2f" % (pad, position)
        drawn = []
        for label in pool["labels"]:
            for d in deals[label][si * per_class:(si + 1) * per_class]:
                drawn.append(_item(pool, d["template"], d["fill"], pad, position, cell, "train"))
        random.Random("%d|%s|interleave" % (seed, cell)).shuffle(drawn)
        items.extend(drawn)

    if mode == "long_boost":
        extra: List[Dict[str, Any]] = []
        for pad, position in ARM4_LONG_REGIMES:
            base = [it for it in items
                    if it["pad"] == pad and it["needle_position"] == position]
            if not base:
                raise ValueError("no uniform items in regime (%d, %.2f)" % (pad, position))
            for k in range(1, ARM4_LONG_BOOST):
                for it in base:
                    copy = dict(it)
                    copy["cell"] = it["cell"] + "#b%d" % k
                    copy["item_id"] = it["item_id"] + "|b%d" % k
                    copy["boost_copy"] = k
                    copy["oversampled_from"] = it["cell"]
                    extra.append(copy)
        items = items + extra
    elif mode != "uniform":
        raise ValueError("unknown train mode %r" % mode)

    plan = _finalise(items, "train", pool, seed)
    plan["mode"] = mode
    plan["arm4_long_boost"] = ARM4_LONG_BOOST if mode == "long_boost" else None
    plan["oversampled_regimes"] = [list(r) for r in ARM4_LONG_REGIMES] if mode == "long_boost" else []
    # the effective sampling weight per regime: how many items of each (pad, position) the arm sees
    weights: Dict[str, int] = {}
    for it in items:
        key = "pad=%d|pos=%.2f" % (it["pad"], it["needle_position"])
        weights[key] = weights.get(key, 0) + 1
    plan["regime_weights"] = dict(sorted(weights.items()))
    return plan


def _finalise(items: List[Dict[str, Any]], split: str, pool: Dict[str, Any], seed: int) -> Dict[str, Any]:
    ids = [it["item_id"] for it in items]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})[:5]
        raise ValueError("%s plan has duplicate item_ids, e.g. %s" % (split, dupes))
    label_counts: Dict[str, int] = {}
    for it in items:
        label_counts[it["label"]] = label_counts.get(it["label"], 0) + 1
    cells: Dict[str, Any] = {}
    for it in items:
        cells.setdefault(it["cell"], []).append(it)
    cell_rows = {}
    for cell, its in sorted(cells.items()):
        needles = [it["needle_sha256"] for it in its]
        templates = sorted({it["template_id"] for it in its})
        lc: Dict[str, int] = {}
        for it in its:
            lc[it["label"]] = lc.get(it["label"], 0) + 1
        cell_rows[cell] = {
            "n": len(its),
            "pad": its[0]["pad"],
            "needle_position": its[0]["needle_position"],
            "label_counts": dict(sorted(lc.items())),
            "n_unique_needles": len(set(needles)),
            "duplicate_needles": len(its) - len(set(needles)),
            "n_templates": len(templates),
            "template_counts": {t: sum(1 for it in its if it["template_id"] == t) for t in templates},
            "balanced": max(lc.values()) - min(lc.values()) <= 1,
        }
    return {
        "split": split,
        "plan_seed": seed,
        "pool_id": pool["pool_id"],
        "pool_sha256": pool["_sha256"],
        "n": len(items),
        "label_counts": dict(sorted(label_counts.items())),
        "cells": cell_rows,
        "items": items,
    }


def validate_plan(plan: Dict[str, Any]) -> None:
    """Assertions that make the plan's promises checkable rather than asserted.

    In-cell needle uniqueness is asserted (a repeated surface form would shrink the effective n).
    Cross-cell reuse is *not* an error -- the L4000 sweep is built on it on purpose -- and is
    reported by ``cross_cell_reuse`` instead.
    """
    assert plan["n"] == len(plan["items"]), "plan n disagrees with its item list"
    labels = sorted(plan["label_counts"])
    spread = max(plan["label_counts"].values()) - min(plan["label_counts"].values())
    assert spread <= 1, "plan %s is not label-balanced: %s" % (plan["split"], plan["label_counts"])
    assert len(labels) == 4, "plan %s has %d labels, expected 4" % (plan["split"], len(labels))
    for cell, row in plan["cells"].items():
        assert row["balanced"], "cell %s is not label-balanced: %s" % (cell, row["label_counts"])
        assert row["duplicate_needles"] == 0, \
            "cell %s repeats a needle surface form %d time(s)" % (cell, row["duplicate_needles"])


def cross_cell_reuse(plan: Dict[str, Any]) -> Dict[str, Any]:
    """How many needle surface forms appear in more than one cell (the content-controlled sweep),
    and the global uniqueness of the plan's needles."""
    by_needle: Dict[str, List[str]] = {}
    for it in plan["items"]:
        by_needle.setdefault(it["needle_sha256"], []).append(it["cell"])
    multi = {h: sorted(set(c)) for h, c in by_needle.items() if len(set(c)) > 1}
    return {
        "n_items": len(plan["items"]),
        "n_unique_needles": len(by_needle),
        "n_needles_in_multiple_cells": len(multi),
        "max_cells_per_needle": max((len(c) for c in multi.values()), default=1),
        "example": list(sorted(multi.items()))[:1],
    }


def position_oracle(plan: Dict[str, Any]) -> Dict[str, Any]:
    """The position-only oracle for a plan: best accuracy from (cell, and hence pad+position)
    alone. With a label-balanced cell it is exactly 1/n_labels, and that is the number every arm
    must beat to have read anything."""
    total = hits = 0
    per_cell = {}
    for cell, row in plan["cells"].items():
        best = max(row["label_counts"].values())
        total += row["n"]
        hits += best
        per_cell[cell] = {"n": row["n"], "oracle_accuracy": best / row["n"],
                          "oracle_label": max(sorted(row["label_counts"]), key=lambda k: row["label_counts"][k])}
    return {"n": total, "oracle_accuracy": hits / total, "per_cell": per_cell}


def main() -> None:
    eval_plan = build_eval_plan()
    train_plan = build_train_plan("uniform")
    validate_plan(eval_plan)
    validate_plan(train_plan)
    out = {
        "eval": {k: v for k, v in eval_plan.items() if k != "items"},
        "train": {k: v for k, v in train_plan.items() if k != "items"},
        "train_long_boost": {k: v for k, v in build_train_plan("long_boost").items() if k != "items"},
        "eval_items": eval_plan["items"],
        "train_items": train_plan["items"],
        "train_long_boost_items": build_train_plan("long_boost")["items"],
        "position_oracle_eval": position_oracle(eval_plan),
        "position_oracle_train": position_oracle(train_plan),
        "eval_needle_reuse": cross_cell_reuse(eval_plan),
        "train_needle_reuse": cross_cell_reuse(train_plan),
    }
    with open(DEFAULT_PLAN, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, sort_keys=False)
        fh.write("\n")
    print("wrote experiments/h5-adapter/plan.json")
    print("  eval  n=%d  labels=%s" % (eval_plan["n"], eval_plan["label_counts"]))
    print("  train n=%d  labels=%s" % (train_plan["n"], train_plan["label_counts"]))
    for cell, row in sorted(eval_plan["cells"].items()):
        print("  eval cell %-12s n=%3d pad=%5d pos=%.2f templates=%2d dup=%d labels=%s"
              % (cell, row["n"], row["pad"], row["needle_position"], row["n_templates"],
                 row["duplicate_needles"], row["label_counts"]))
    print("  position-only oracle (eval) = %.4f" % out["position_oracle_eval"]["oracle_accuracy"])


if __name__ == "__main__":
    main()

"""Extend the primary cell to n=600 -- more *items*, not more seeds (orchestrator ruling, issue #11).

    # 1. plan + leakage check (CPU)
    env/venv/bin/python experiments/h5-adapter/extra_items.py --plan
    # 2. encoder features for the new items (GPU, lock held)
    env/venv/bin/python experiments/h5-adapter/extra_items.py --build-cache
    # 3. score every trained arm on them and append to its per-item JSONL (GPU, lock held)
    env/venv/bin/python experiments/h5-adapter/extra_items.py --eval --arms arm3r_residual,arm2long_shipped_init
    # 4. recompute every statistic over the enlarged cell
    env/venv/bin/python experiments/h5-adapter/stats.py

**Why.** A paired McNemar test buys power with *discordant pairs*, which scale with the number of
items, not with the number of training seeds. M-001's power analysis (ledger L-022) puts a +0.05
effect at n ≈ 312 / 626 / 1568 for q = 0.10 / 0.20 / 0.50, so n=200 resolves almost nothing and
n=600 covers q ≈ 0.20. When the seed-0 gap between arm3r and its schedule-matched control is small,
extra seeds would produce more underpowered estimates of the same 200 items; extra items are the
only thing that can settle it.

**What is held fixed.** The cell (`L7000-p100`: pad=7000, needle at position 1.0), the label balance
(100 more needles per class, so the enlarged cell is 150/class), the document construction
(`docs.H5DocBuilder`, filler seeded on the needle hash *and* the pad), the model path (frozen
encoder, cached features, the same head forward), and the leakage rules: the extra needles are new
combinations from the **evaluation** pool, disjoint from every training needle, and each new needle
gets its own haystack. The 200 existing items are untouched -- their rows stay in the JSONL and the
new rows are appended, so the cell becomes a composition of 200 original + 400 new items, which is
stated wherever the number is quoted.

**What this cannot fix.** It cannot raise the *number of trained models*: the comparison stays one
seed per arm unless extra seeds are run. The ext items measure the same two heads on more evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from typing import Any, Dict, List

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CELL = "L7000-p100"
PAD = 7000
POSITION = 1.0
N_EXTRA = 400                      # -> 600 items in the cell, 150 per class
USED_SLICE = slice(0, 200)         # what the original plan already drew per class
EXTRA_SLICE = slice(200, 300)      # 100 more per class, untouched by the original plan
CACHE_EXTRA = os.path.join(HERE, "cache", "extra_eval")
PROVENANCE = os.path.join(HERE, "extra_items_600.json")


def build_conditions() -> List[Dict[str, Any]]:
    import plan as P

    pool = P.load_pool("needles-h5-eval-v1.json")
    per_class = N_EXTRA // len(pool["labels"])
    items: List[Dict[str, Any]] = []
    provenance: Dict[str, Any] = {}
    for label in pool["labels"]:
        combos = P._shuffled(pool, label, "eval", 20260924)
        if len(combos) < EXTRA_SLICE.stop:
            raise SystemExit("eval pool has %d combos for %s; the extension needs %d"
                             % (len(combos), label, EXTRA_SLICE.stop))
        picked = combos[EXTRA_SLICE]
        if len(picked) != per_class:
            raise SystemExit("wanted %d extra %s needles, drew %d" % (per_class, label, len(picked)))
        provenance[label] = {
            "n": len(picked),
            "first_needle_sha256": [P.sha256_text(P.render(tpl["text"], fill))[:16]
                                    for tpl, fill in picked[:3]],
        }
        for tpl, fill in picked:
            items.append(P._item(pool, tpl, fill, PAD, POSITION, CELL, "eval"))
    random.Random("20260924|%s|extra-interleave" % CELL).shuffle(items)
    # item_ids must not collide with the originals: templates and fills are disjoint slices, and the
    # plan's own uniqueness check is repeated here rather than assumed
    ids = [it["item_id"] for it in items]
    if len(set(ids)) != len(ids):
        raise SystemExit("the extension produced duplicate item_ids")
    plan = P._finalise(items, "eval", pool, 20260924)
    final = plan["items"]

    # Leakage, re-derived rather than assumed: the extra needles must not appear in training or in
    # any existing evaluation item. Checked by content hash, the way `leakage_check.py` does it.
    with open(os.path.join(HERE, "plan.json"), encoding="utf-8") as fh:
        main_plan = json.load(fh)
    extra_h = {it["needle_sha256"] for it in final}
    train_h = {it["needle_sha256"] for it in main_plan.get("train_items", [])}
    eval_h = {it["needle_sha256"] for it in main_plan.get("eval_items", [])}
    if extra_h & train_h or extra_h & eval_h:
        raise SystemExit("LEAKAGE: %d extra needles collide with training, %d with existing eval"
                         % (len(extra_h & train_h), len(extra_h & eval_h)))
    out = {
        "artifact": "experiments/h5-adapter/extra_items_600.json",
        "what": "400 extra L7000-p100 eval items, so the primary cell is 600 items / 150 per class",
        "why": ("paired McNemar power scales with discordant pairs (items), not with training seeds; "
                "M-001/L-022 puts +0.05 at n ~ 312-1568 depending on q"),
        "cell": CELL, "pad": PAD, "needle_position": POSITION, "n_extra": len(final),
        "per_class": per_class,
        "slice_used_by_original_plan": [USED_SLICE.start, USED_SLICE.stop],
        "slice_used_by_extension": [EXTRA_SLICE.start, EXTRA_SLICE.stop],
        "pool": "experiments/h5-adapter/pools/needles-h5-eval-v1.json",
        "pool_sha256": hashlib.sha256(
            open(os.path.join(HERE, "pools", "needles-h5-eval-v1.json"), "rb").read()).hexdigest(),
        "leakage": ("extra needles come from the evaluation pool only and are disjoint from every "
                    "training needle; each new needle gets its own haystack (filler is seeded on the "
                    "needle hash and the pad), so no document is reused"),
        "label_counts": {lb: sum(1 for it in final if it["label"] == lb) for lb in sorted(
            {it["label"] for it in final})},
        "leakage_check": {
            "checked": "content hash of every rendered needle",
            "n_extra_needle_hashes": len(extra_h),
            "n_train_needle_hashes": len(train_h),
            "n_existing_eval_needle_hashes": len(eval_h),
            "overlap_with_training": len(extra_h & train_h),
            "overlap_with_existing_eval": len(extra_h & eval_h),
        },
        "provenance": provenance,
        "cell_row": plan["cells"][CELL],
        "generated_by": "experiments/h5-adapter/extra_items.py --plan",
    }
    with open(PROVENANCE, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    return final


def load_conditions() -> List[Dict[str, Any]]:
    with open(PROVENANCE, encoding="utf-8") as fh:
        meta = json.load(fh)
    conds = build_conditions()
    if len(conds) != meta["n_extra"]:
        raise SystemExit("rebuild produced %d items, provenance says %d"
                         % (len(conds), meta["n_extra"]))
    return conds


def build_cache(conditions: List[Dict[str, Any]], device_name: str = "cuda") -> str:
    import common_h5 as C
    import features as FEAT

    device = __import__("torch").device(device_name)
    agent = C.load_agent(device_name)
    builder = C.make_builder(agent.tok, C.load_filler(), C.load_needle_pool("needles-h5-eval-v1.json"))
    lengths = [builder.estimate_length(c) for c in conditions]
    total = sum(lengths)
    print("extra items: n=%d tokens=%d (%.1f GB fp16)" % (len(conditions), total,
                                                          total * 768 * 2 / 1e9), flush=True)
    path = FEAT.build_cache(conditions, "eval", CACHE_EXTRA, agent.model, builder, device)
    print("wrote cache %s" % os.path.relpath(path, os.path.dirname(HERE)), flush=True)
    return path


def eval_arms(arms: List[str], seeds: List[int]) -> None:
    import torch

    import arms as A
    import common_h5 as C
    import eval as E
    import features as FEAT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conditions = load_conditions()
    store = FEAT.FeatureStore(CACHE_EXTRA)
    have = {it["item_id"] for it in store.items}
    missing = [c["item_id"] for c in conditions if c["item_id"] not in have]
    if missing:
        raise SystemExit("%d of %d extra items are not in the cache, e.g. %s"
                         % (len(missing), len(conditions), missing[:2]))
    agent = C.load_agent(str(device))
    for arm in arms:
        for seed in seeds:
            path = os.path.join(HERE, "predictions", "%s-seed%d.jsonl" % (arm, seed))
            with open(path, encoding="utf-8") as fh:
                existing = {json.loads(line)["item_id"] for line in fh if line.strip()}
            if any(c["item_id"] in existing for c in conditions):
                raise SystemExit("%s already contains extra items -- refusing to append twice" % path)
            model = A.build_arm_model(agent.model, arm, seed).to(device)
            A.freeze_for_training(model)
            E.load_trained(model, arm, seed, device)
            rows = E.evaluate_from_cache(model, store, conditions, device, arm, seed)
            with open(path, "a", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            acc = sum(1 for r in rows if r["correct"]) / len(rows)
            by_label: Dict[str, List[int]] = {}
            for r in rows:
                by_label.setdefault(r["label"], []).append(int(r["correct"]))
            print("  %s seed%d: appended %d rows to %s  (extra-item accuracy %.3f; %s)"
                  % (arm, seed, len(rows), os.path.basename(path), acc,
                     " ".join("%s=%.3f" % (k, sum(v) / len(v)) for k, v in sorted(by_label.items()))),
                  flush=True)
            del model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="build the 400 extra conditions (CPU)")
    ap.add_argument("--build-cache", action="store_true", help="encoder features for them (GPU)")
    ap.add_argument("--eval", action="store_true", help="score trained arms and append JSONL (GPU)")
    ap.add_argument("--arms", default="arm3r_residual,arm2long_shipped_init")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    t0 = time.time()
    if a.plan:
        conds = build_conditions()
        print("built %d extra %s items (%s); wrote %s"
              % (len(conds), CELL, {lb: sum(1 for c in conds if c["label"] == lb)
                                    for lb in sorted({c["label"] for c in conds})},
                 os.path.relpath(PROVENANCE, os.path.dirname(HERE))), flush=True)
    if a.build_cache:
        build_cache(load_conditions(), a.device)
    if a.eval:
        eval_arms([s for s in a.arms.split(",") if s], [int(s) for s in a.seeds.split(",") if s != ""])
    if not (a.plan or a.build_cache or a.eval):
        ap.print_help()
        return 1
    print("done in %.0f s" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

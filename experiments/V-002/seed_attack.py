"""V-002 verification F: the seed attack on arm2.

Arm 2 exists at seed 0 only in the committed artifacts, so the ledger claim is a single-seed
result unless a second seed is run. This script consumes the seed-1/seed-2 re-runs produced by
`run_seed_attack.sh` and reports:

* per-cell accuracy for arm1 and for arm2 at each available seed;
* the seed spread of arm2's accuracy per cell (max - min over seeds), which protocol.md s6 requires
  a gain to exceed;
* the arm2-vs-arm1 paired McNemar at *each* seed, and whether the sign of the effect agrees;
* the spread of the *effect* (arm2 - arm1) across seeds, and whether the seed-0 effect is inside it.

Run: env/venv/bin/python experiments/V-002/seed_attack.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(os.path.dirname(HERE))
H5 = os.path.join(LAB, "experiments", "h5-adapter")
PROTOCOL_CELLS = ("L0", "L4000-p000", "L4000-p025", "L4000-p050", "L4000-p075",
                  "L4000-p100", "L7000-p000", "L7000-p100")
KEY_CELLS = ("L7000-p100", "L4000-p050", "L0")


def rows(path: str) -> List[Dict[str, Any]]:
    return [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]


def mcnemar(b: int, c: int) -> Dict[str, Any]:
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "p": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return {"b": b, "c": c, "n_discordant": n, "p": min(1.0, 2 * tail)}


def main() -> int:
    arm1 = rows(os.path.join(H5, "predictions", "arm1_frozen.jsonl"))
    a1 = {r["item_id"]: r for r in arm1}
    seeds: Dict[int, List[Dict[str, Any]]] = {}
    candidates = {
        0: os.path.join(H5, "predictions", "arm2_shipped_init-seed0.jsonl"),
        1: os.path.join(HERE, "repro_arm2_shipped_init-seed1.jsonl"),
        2: os.path.join(HERE, "repro_arm2_shipped_init-seed2.jsonl"),
    }
    for seed, p in candidates.items():
        if os.path.exists(p):
            seeds[seed] = rows(p)
    out: Dict[str, Any] = {"seeds_available": sorted(seeds), "per_seed_cell": {}, "per_cell": {}}

    for seed, rs in seeds.items():
        by_cell: Dict[str, List[Dict[str, Any]]] = {}
        for r in rs:
            by_cell.setdefault(r["cell"], []).append(r)
        out["per_seed_cell"]["seed%d" % seed] = {
            c: {"n": len(v), "accuracy": sum(1 for r in v if r["correct"]) / len(v)}
            for c, v in sorted(by_cell.items())
        }
    for cell in PROTOCOL_CELLS:
        accs = {}
        for seed, rs in seeds.items():
            v = [r for r in rs if r["cell"] == cell]
            accs["seed%d" % seed] = sum(1 for r in v if r["correct"]) / len(v)
        v1 = [r for r in arm1 if r["cell"] == cell]
        acc1 = sum(1 for r in v1 if r["correct"]) / len(v1)
        effects = {}
        mc = {}
        for seed, rs in seeds.items():
            v = [r for r in rs if r["cell"] == cell]
            pairs = [(r, a1[r["item_id"]]) for r in v if r["item_id"] in a1]
            b = sum(1 for x, y in pairs if x["correct"] and not y["correct"])
            c = sum(1 for x, y in pairs if (not x["correct"]) and y["correct"])
            mc["seed%d" % seed] = mcnemar(b, c)
            effects["seed%d" % seed] = 100.0 * (
                sum(1 for x, _ in pairs if x["correct"]) / len(pairs) - acc1)
        eff_vals = list(effects.values())
        out["per_cell"][cell] = {
            "arm1_accuracy": acc1, "arm2_accuracy_per_seed": accs,
            "arm2_seed_spread_pp": 100.0 * (max(accs.values()) - min(accs.values())),
            "arm2_minus_arm1_pp_per_seed": effects,
            "effect_spread_pp": max(eff_vals) - min(eff_vals),
            "mcnemar_per_seed": mc,
            "sign_consistent": all(e > 0 for e in eff_vals) or all(e < 0 for e in eff_vals),
        }

    if len(seeds) < 2:
        out["note"] = ("only one seed available: the claim is single-seed and no seed spread can "
                       "be reported")
    with open(os.path.join(HERE, "seed_attack.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")

    print("seeds available: %s" % out["seeds_available"])
    for cell in PROTOCOL_CELLS:
        v = out["per_cell"][cell]
        star = " <= key" if cell in KEY_CELLS else ""
        print("  %-14s arm1=%.3f arm2=%s spread=%.2fpp effect=%s p=%s%s"
              % (cell, v["arm1_accuracy"],
                 {k: round(x, 3) for k, x in v["arm2_accuracy_per_seed"].items()},
                 v["arm2_seed_spread_pp"],
                 {k: round(x, 1) for k, x in v["arm2_minus_arm1_pp_per_seed"].items()},
                 {k: "%.3g" % x["p"] for k, x in v["mcnemar_per_seed"].items()}, star))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

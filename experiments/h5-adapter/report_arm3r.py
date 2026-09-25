"""Render the arm-3r report tables from the raw artifacts, so the finding quotes files, not memory.

    env/venv/bin/python experiments/h5-adapter/report_arm3r.py     # no GPU, no model

Reads `summary.json` (recomputed from the per-item JSONL by `stats.py`), the `runs/*/training.json`
records and `step0_identity.json`, and writes `experiments/h5-adapter/arm3r_report.md`. Nothing here
computes a statistic of its own: every number is copied out of an artifact that a verifier can
regenerate, together with the path it came from.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(LAB, "experiments", "harness"))

import metrics as M  # noqa: E402  (the program's canonical statistics, as stats.py uses)
OUT = os.path.join(HERE, "arm3r_report.md")
CELLS = ["L0", "L4000-p000", "L4000-p025", "L4000-p050", "L4000-p075", "L4000-p100",
         "L7000-p000", "L7000-p100"]
ARMS = ["arm1_frozen", "arm2_shipped_init", "arm2rerun_shipped_init", "arm2long_shipped_init",
        "arm3_xattn", "arm3r_residual", "arm3r_residual_ablated"]


PAIRS = [
    ("arm3r_residual", "arm2_shipped_init"),
    ("arm3r_residual", "arm2rerun_shipped_init"),
    ("arm2rerun_shipped_init", "arm2_shipped_init"),
    ("arm2rerun_shipped_init", "arm2long_shipped_init"),
    ("arm3r_residual", "arm2long_shipped_init"),
    ("arm3r_residual_ablated", "arm2_shipped_init"),
    ("arm3r_residual", "arm3r_residual_ablated"),
    ("arm2long_shipped_init", "arm2_shipped_init"),
    ("arm2_shipped_init", "arm1_frozen"),
]


def rows_of(pred_dir: str, arm: str) -> Dict[str, List[Dict[str, Any]]]:
    """cell -> rows, for whichever seed file exists (seed 0 for the trained arms)."""
    path = os.path.join(pred_dir, "%s.jsonl" % arm)
    if not os.path.exists(path):
        cands = sorted(glob.glob(os.path.join(pred_dir, "%s-seed*.jsonl" % arm)))
        path = cands[0] if cands else None
    if path is None:
        return {}
    by_cell: Dict[str, List[Dict[str, Any]]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                by_cell.setdefault(r["cell"], []).append(r)
    return by_cell


def paired(a: Sequence[Dict[str, Any]], b: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    ca, cb = M.pair_rows(a, b, key_fields=("item_id",))
    if not ca:
        return None
    out = M.mcnemar_exact(ca, cb)
    out["accuracy_a"] = sum(ca) / len(ca)
    out["accuracy_b"] = sum(cb) / len(cb)
    out["effect_size_pp"] = 100.0 * (out["accuracy_a"] - out["accuracy_b"])
    return out


def load(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def acc(summary: Dict[str, Any], arm: str, cell: str) -> Optional[Dict[str, Any]]:
    per = summary["per_arm_seed_cell"]
    key = "seed0" if "seed0" in per.get(arm, {}) else next(iter(per.get(arm, {}) or {}), None)
    if key is None or cell not in per[arm][key]:
        return None
    return per[arm][key][cell]


def main() -> int:
    s = load(os.path.join(HERE, "summary.json"))
    step0 = load(os.path.join(HERE, "step0_identity.json"))
    probe = load(os.path.join(HERE, "probe_arm3r.json"))
    if s is None:
        print("no summary.json -- run stats.py first")
        return 1
    L: List[str] = []
    add = L.append

    add("# arm-3r report (regenerated from raw artifacts)\n")
    add("* `experiments/h5-adapter/summary.json` — every accuracy, CI and paired test below.\n"
        "* `experiments/h5-adapter/step0_identity.json` — the step-0 identity assertion.\n"
        "* `experiments/h5-adapter/probe_arm3r.json` — the branch learning-rate probe.\n"
        "* `experiments/h5-adapter/runs/<arm>/seed<k>/training.json` — the per-epoch curves.\n")
    add("\nRegenerate: `env/venv/bin/python experiments/h5-adapter/report_arm3r.py`\n")

    add("\n## Per-cell accuracy, n=200, label-balanced (oracle beside every number)\n")
    hdr = "| cell | oracle | " + " | ".join(ARMS) + " |"
    add(hdr)
    add("|" + "---|" * (len(ARMS) + 2))
    for cell in CELLS:
        orc = (s["position_only_oracle"].get(cell) or {}).get("oracle_accuracy")
        vals = []
        for arm in ARMS:
            a = acc(s, arm, cell)
            vals.append("%.3f [%.3f, %.3f]" % (a["accuracy"], a["accuracy_ci95_low"],
                                               a["accuracy_ci95_high"]) if a else "—")
        add("| %s | %s | %s |" % (cell, "%.4f" % orc if orc is not None else "—",
                                  " | ".join(vals)))

    add("\n## Paired McNemar, recomputed here from the raw per-item JSONL\n")
    add("Holm column: `summary.json`'s `comparisons` entry for the same pair where it exists "
        "(the correction is over that file's family), `—` where the pair is not in the family.\n")
    add("| candidate | baseline | cell | acc cand | acc base | Δ pp | wrong→right | right→wrong | p exact | p Holm |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    holm = {(c["candidate"], c["baseline"], c["cell"]): c["p_holm"] for c in s["comparisons"]}
    for cand, base in PAIRS:
        ca, cb = rows_of(os.path.join(HERE, "predictions"), cand), rows_of(
            os.path.join(HERE, "predictions"), base)
        for cell in CELLS:
            if cell not in ca or cell not in cb:
                continue
            st = paired(ca[cell], cb[cell])
            if st is None:
                continue
            ph = holm.get((cand, base, cell))
            add("| %s | %s | %s | %.3f | %.3f | %+.1f | %d | %d | %.3g | %s |"
                % (cand, base, cell, st["accuracy_a"], st["accuracy_b"], st["effect_size_pp"],
                   st["a_wrong_b_right"], st["a_right_b_wrong"], st["p_value_two_sided_exact"],
                   ("%.3g" % ph) if ph is not None else "—"))

    add("\n## Convergence (per-epoch curves, from the training records)\n")
    for arm in ("arm2long_shipped_init", "arm3r_residual"):
        rec = load(os.path.join(HERE, "runs", arm, "seed0", "training.json"))
        if rec is None:
            add("\n`%s`: no training record.\n" % arm)
            continue
        h = rec["history"]
        add("\n### %s — %d epochs, lr=%g, lr_cross=%s, %.0f s, %d steps\n"
            % (arm, rec["epochs"], rec["lr"], rec.get("lr_cross"), rec["train_seconds"], rec["steps"]))
        add("| epoch | train loss | train acc | lr | branch \\|Δlogit\\| | \\|W_out\\| | s |")
        add("|---|---|---|---|---|---|---|")
        for r in h:
            add("| %d | %.4f | %.4f | %.2e | %s | %s | %.0f |"
                % (r["epoch"], r["train_loss"], r["train_accuracy"], r["lr"],
                   ("%.3g" % r["branch_logit_contribution_max"])
                   if r.get("branch_logit_contribution_max") is not None else "—",
                   ("%.3g" % r["out_proj_weight_fro"]) if r.get("out_proj_weight_fro") else "—",
                   r["seconds"]))
        k = min(5, len(h) - 1)
        add("\nLast %d epochs: loss %.4f → %.4f (Δ %.4f), train acc %.4f → %.4f. "
            "First→last: loss %.4f → %.4f.\n"
            % (k, h[-k - 1]["train_loss"], h[-1]["train_loss"],
               h[-1]["train_loss"] - h[-k - 1]["train_loss"], h[-k - 1]["train_accuracy"],
               h[-1]["train_accuracy"], h[0]["train_loss"], h[-1]["train_loss"]))

    add("\n## Step-0 identity and liveness\n")
    if step0 is None:
        add("`step0_identity.json`: missing.\n")
    else:
        for c in step0["checks"]:
            add("* **%s** — n=%d, bitwise-identical logits: `%s`, max |Δlogit| %s, items with a "
                "different prediction: **%d**"
                % (c["check"], c["n_items"], c["bitwise_identical_logits"],
                   c["max_abs_logit_delta"], c["items_with_different_prediction"]))
        t = step0["trainability"]
        add("* trainability: out_proj weight grad sum %s; inner-branch max grad sum at step 0 %s "
            "(%d tensors, zero by construction while W=0)"
            % (t["out_proj_weight_grad_sum"], t["max_inner_grad_sum_at_step0"], t["n_inner_tensors"]))
        add("* verdict: **%s**\n" % step0["verdict"])

    add("\n## Branch learning-rate probe\n")
    if probe is None:
        add("`probe_arm3r.json`: missing.\n")
    else:
        add("| cross lr | loss per epoch | acc per epoch | branch \\|Δlogit\\| per epoch | \\|W_out\\| per epoch |")
        add("|---|---|---|---|---|")
        for r in probe["configs"]:
            add("| %g | %s | %s | %s | %s |"
                % (r["lr_cross"], " ".join("%.4f" % v for v in r["loss_per_epoch"]),
                   " ".join("%.3f" % v for v in r["acc_per_epoch"]),
                   " ".join("%.3g" % (v or 0.0) for v in r["branch_contribution_per_epoch"]),
                   " ".join("%.3g" % (v or 0.0) for v in r["out_proj_weight_fro_per_epoch"])))

    add("\n## Verdict cells as `stats.py` computes them (oracle included)\n")
    add("```json")
    add(json.dumps(s["verdict"], indent=1, sort_keys=True))
    add("```\n")
    add("* oracle: `summary.json` → `position_only_oracle`; n=200 per cell, 50 per class. Any arm "
        "that does not beat it has learned nothing about content and is reported in those words.\n")
    add("* seed spread: `summary.json` → `seed_spread`.\n")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote %s (%d lines)" % (os.path.relpath(OUT, os.path.dirname(HERE)), len(L)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the caches, run every arm, evaluate every arm, and finalise the manifest.

    flock -n /home/parshu/projects/contri/laya-lab/.gpu.lock -c \
      "env/venv/bin/python experiments/h5-adapter/run_all.py" \
      || echo "GPU busy -- another agent holds the lock"

One job for the whole experiment so the lock is taken once and the GPU is never shared. Stages are
idempotent and individually skippable (`--only`, `--skip`), so a killed run resumes where it
stopped instead of re-encoding features that are already on disk.

The manifest is written **before** any heavy work and rewritten at the end, so a killed run still
has provenance. It records: environment and library versions, both git revisions, the pool hashes,
the plan's own summary (train length/position distribution, per-cell n and label counts), the
parameter accounting, and the exact command.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(os.path.dirname(HERE))
LOG = os.path.join(HERE, "run.log")
MANIFEST = os.path.join(HERE, "manifest.json")

# Ordered most-valuable-first: if the GPU lock is lost or the window closes, whatever ran is the
# part of the experiment that mattered most. `latency` is last because the accuracy result does not
# depend on it, and `arm2_step0` (the cache-fidelity bridge) is cheap and sits beside `arm1`.
ALL_STAGES = ("manifest", "pilot", "cache_eval", "arm1", "cache_train", "train", "eval",
              "arm2_step0", "stats", "latency")


def log(msg: str) -> None:
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(cmd: List[str]) -> int:
    log("run: " + " ".join(cmd))
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=LAB)
    log("  -> exit %d in %.1fs" % (rc, time.time() - t0))
    if rc != 0:
        raise SystemExit("stage failed: %s" % " ".join(cmd))
    return rc


def write_manifest(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    import common_h5 as C
    import arms as A

    plan = C.load_plan()
    man: Dict[str, Any] = {
        "run_id": "h5-adapter",
        "issue": "Parswanadh/laya-lab#6",
        "hypothesis": "H5 -- the aggregation path, not the encoder, is where the headroom is",
        "artifact_dir": "experiments/h5-adapter",
        "exact_command": ("flock -n /home/parshu/projects/contri/laya-lab/.gpu.lock -c "
                          "'env/venv/bin/python experiments/h5-adapter/run_all.py'"),
        "protocol": "protocol.md (FROZEN)",
        "task": "synthetic needle-in-haystack decision; every claim is labelled synthetic",
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "git": C.git_state(),
        "pool_hashes": C.pool_hashes(),
        "plan": {
            "path": "experiments/h5-adapter/plan.json",
            "plan_seed": plan["eval"]["plan_seed"],
            "eval_n": plan["eval"]["n"],
            "train_n": plan["train"]["n"],
            "train_long_boost_n": plan["train_long_boost"]["n"],
            "eval_cells": {k: {"n": v["n"], "pad": v["pad"], "needle_position": v["needle_position"],
                               "label_counts": v["label_counts"], "n_templates": v["n_templates"],
                               "duplicate_needles": v["duplicate_needles"]}
                           for k, v in plan["eval"]["cells"].items()},
            "position_only_oracle_eval": plan["position_oracle_eval"],
            "position_only_oracle_train": plan["position_oracle_train"],
            "train_length_distribution": plan["train"]["regime_weights"],
            "train_long_boost_distribution": plan["train_long_boost"]["regime_weights"],
            "train_long_boost_regimes": plan["train_long_boost"]["oversampled_regimes"],
            "eval_needle_reuse": plan["eval_needle_reuse"],
            "train_needle_reuse": plan["train_needle_reuse"],
            "train_cell_row_counts": {k: v["n"] for k, v in plan["train"]["cells"].items()},
        },
        "leakage": {
            "artifact": "experiments/h5-adapter/leakage_report.json",
            "reproduce": "env/venv/bin/python experiments/h5-adapter/leakage_check.py",
            "gating_checks_passed": None,  # filled in below from the report on disk
        },
        "arms": {k: v for k, v in A.ARMS.items()},
        "parameter_matching_analytic": A.analytic_parameter_table(),
        "device": None,
    }
    lr = os.path.join(HERE, "leakage_report.json")
    if os.path.exists(lr):
        with open(lr, encoding="utf-8") as fh:
            rep = json.load(fh)
        man["leakage"]["gating_checks_passed"] = rep["summary"]["passed"]
        man["leakage"]["failed"] = rep["summary"]["failed"]
        man["leakage"]["longest_shared_word_ngram"] = \
            rep["c4_longest_shared_word_ngram"]["longest_shared_word_ngram"]
        man["leakage"]["max_rendered_token_jaccard"] = \
            rep["c6_max_rendered_token_jaccard"]["max_jaccard"]
    try:
        import torch
        man["environment"]["torch"] = torch.__version__
        import transformers
        man["environment"]["transformers"] = transformers.__version__
        if torch.cuda.is_available():
            man["device"] = {
                "gpu": torch.cuda.get_device_name(0),
                "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
                "cuda": torch.version.cuda,
            }
    except Exception as e:  # pragma: no cover
        man["environment"]["torch_probe_error"] = str(e)
    if extra:
        man.update(extra)
    if os.path.exists(MANIFEST):
        with open(MANIFEST, encoding="utf-8") as fh:
            old = json.load(fh)
        for k in ("stages_completed", "stats", "latency", "training", "run_started", "run_finished"):
            if k in old and (not extra or k not in extra):
                man[k] = old[k]
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
        fh.write("\n")
    return man


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated stages to run")
    ap.add_argument("--skip", default="", help="comma-separated stages to skip")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--latency-seeds", default="0,1,2")
    a = ap.parse_args()
    stages = [s for s in (a.only.split(",") if a.only else ALL_STAGES)
              if s and s not in set(a.skip.split(","))]
    seeds = [int(x) for x in a.seeds.split(",") if x != ""]
    py = os.path.join(LAB, "env", "venv", "bin", "python")
    log("=== h5-adapter run: stages=%s seeds=%s ===" % (stages, seeds))
    write_manifest({"run_started": time.strftime("%Y-%m-%dT%H:%M:%S")})
    completed: List[str] = []
    training_summary: Dict[str, Any] = {}
    t0 = time.time()

    for stage in stages:
        log("--- stage %s" % stage)
        if stage == "manifest":
            write_manifest()
        elif stage == "pilot":
            run([py, "experiments/h5-adapter/pilot.py"])
        elif stage == "cache_eval":
            run([py, "experiments/h5-adapter/build_cache.py", "--split", "eval"])
        elif stage == "cache_train":
            run([py, "experiments/h5-adapter/build_cache.py", "--split", "train"])
        elif stage == "arm1":
            run([py, "experiments/h5-adapter/eval.py", "--arm", "arm1_frozen", "--latency"])
        elif stage == "arm2_step0":
            run([py, "experiments/h5-adapter/eval.py", "--arm", "arm2_step0"])
        elif stage == "train":
            import arms as A
            for arm in A.TRAINED_ARMS:
                for seed in seeds:
                    out = subprocess.run(
                        [py, "experiments/h5-adapter/train.py", "--arm", arm, "--seed", str(seed),
                         "--epochs", str(a.epochs)], cwd=LAB, capture_output=True, text=True)
                    sys.stdout.write(out.stdout)
                    sys.stderr.write(out.stderr)
                    if out.returncode != 0:
                        raise SystemExit("training failed for %s seed %d" % (arm, seed))
                    tp = os.path.join(HERE, "runs", arm, "seed%d" % seed, "training.json")
                    if os.path.exists(tp):
                        with open(tp, encoding="utf-8") as fh:
                            training_summary["%s|seed%d" % (arm, seed)] = json.load(fh)
        elif stage == "eval":
            import arms as A
            for arm in A.TRAINED_ARMS:
                for seed in seeds:
                    run([py, "experiments/h5-adapter/eval.py", "--arm", arm, "--seed", str(seed)])
        elif stage == "latency":
            run([py, "experiments/h5-adapter/latency.py", "--gpu-lock-held",
                 "--seeds", a.latency_seeds])
        elif stage == "stats":
            run([py, "experiments/h5-adapter/stats.py"])
        else:
            raise SystemExit("unknown stage %r" % stage)
        completed.append(stage)
        write_manifest({"stages_completed": completed})

    extra: Dict[str, Any] = {"stages_completed": completed,
                             "run_finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                             "total_seconds": round(time.time() - t0, 1)}
    if training_summary:
        extra["training"] = {
            k: {"train_seconds": v["train_seconds"], "n_train_items": v["n_train_items"],
                "steps": v["steps"], "peak_vram_gb": v["peak_vram_gb"],
                "final_train_loss": v["history"][-1]["train_loss"] if v["history"] else None,
                "final_train_accuracy": v["history"][-1]["train_accuracy"] if v["history"] else None,
                "trainable_parameters": v["parameter_accounting"]["trainable_parameters"],
                "head_state_dict_path": v["head_state_dict_path"]}
            for k, v in training_summary.items()}
    for name, key in (("pilot.json", "pilot"), ("latency.json", "latency"),
                      ("summary.json", "stats")):
        p = os.path.join(HERE, name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                extra[key] = json.load(fh)
    write_manifest(extra)
    log("=== done in %.1fs: %s" % (time.time() - t0, completed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

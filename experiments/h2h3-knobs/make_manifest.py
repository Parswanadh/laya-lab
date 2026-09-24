"""E-002 manifest — environment, git state, seeds, exact commands, pool hashes, per-cell telemetry.

    env/venv/bin/python experiments/h2h3-knobs/make_manifest.py

Re-runnable: reads the raw artifacts that exist and refreshes experiments/h2h3-knobs/manifest.json.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(LAB, "experiments", "harness"))


def sh(cmd, cwd=None):
    try:
        return subprocess.run(cmd, cwd=cwd or LAB, shell=isinstance(cmd, str),
                              capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # pragma: no cover
        return "ERROR: %s" % e


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    import torch
    import transformers
    import pools

    man = {
        "run_id": "h2h3-knobs",
        "issue": "Parswanadh/laya-lab#4",
        "owner": "engineer",
        "finding": "findings/E-002.md",
        "purpose": "training-free architectural knobs on the loaded encoder: P-a sliding-window "
                   "width (config.local_attention) and P-b all-layers-global; no training, "
                   "no fine-tuning, nothing under fork/laya/ modified",
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "nvidia_driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
            "host": platform.node(),
            "cpu_count": os.cpu_count(),
            "USE_TF": os.environ.get("USE_TF", "0"),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF",
                                                      "expandable_segments:True"),
        },
        "git": {
            "laya_lab_sha": sh("git rev-parse HEAD"),
            "laya_lab_branch": sh("git rev-parse --abbrev-ref HEAD"),
            "laya_lab_dirty": bool(sh("git status --porcelain")),
            "fork_submodule": sh("git submodule status"),
            "laya_version": sh("git -C fork describe --tags --always") or None,
        },
        "model": {
            "path": "models/multilingual",
            "encoder_config_sha256": sha256_file(os.path.join(LAB, "models/multilingual/encoder/config.json")),
            "rl_agent_config_sha256": sha256_file(os.path.join(LAB, "models/multilingual/rl_agent_config.json")),
            "encoder_config": json.load(open(os.path.join(LAB, "models/multilingual/encoder/config.json"))),
        },
        "harness": {
            "files": {os.path.basename(p): sha256_file(p)
                      for p in sorted(glob.glob(os.path.join(HERE, "*.py")))},
            "doc_construction_primary": "experiments/orch-diagnostic/run.py::place imported as a "
                                        "module and called with frac=1.0 (P1 probe C rule); the "
                                        "imported REQUESTS/FILLER/QUESTIONS objects are asserted "
                                        "equal to the harness pool's",
            "doc_construction_control": "upstream_rule: FILLER*round(pad/per_rep) + "
                                        "'\\n\\nActual request: ' (fork/research/scripts/bench_long_context.py)",
            "liveness_probe": "transformers.models.modernbert.modeling_modernbert."
                              "MODERNBERT_ATTENTION_FUNCTION['sdpa'] wrapped; per layer it records "
                              "the mask handed to F.scaled_dot_product_attention and decodes the max "
                              "attended distance for the last query row; an arm whose observed mask "
                              "does not match its spec raises and is not scored",
            "stats": "paired McNemar exact (binomial) + paired percentile bootstrap CI of the "
                     "accuracy delta, 10000 resamples, seed 20260924, in analyse.py",
        },
        "pools": {},
        "arms": {},
        "commands": {},
        "cells": [],
    }

    for name in ("upstream_multilingual", "balanced-v1", "filler-v1"):
        try:
            p = pools.load_pool(name)
            man["pools"][name] = {"path": p["_path"], "sha256": p["_sha256"],
                                  "pool_id": p["pool_id"], "kind": p["kind"]}
        except Exception as e:  # pragma: no cover
            man["pools"][name] = {"error": str(e)}

    # stage commands actually used (the flock + systemd-run wrapper is the machine rule)
    for stage, pool in (("stage1", "upstream_multilingual"), ("stage2", "upstream_multilingual"),
                        ("stage3", "upstream_multilingual"), ("stage4", "balanced-v1")):
        man["commands"][stage] = (
            "flock -n /home/parshu/projects/contri/laya-lab/.gpu.lock -c "
            "'systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 "
            "env/venv/bin/python experiments/h2h3-knobs/run.py --stage %s'" % stage)

    for path in sorted(glob.glob(os.path.join(HERE, "raw", "*.summary.json"))):
        with open(path, encoding="utf-8") as fh:
            s = json.load(fh)
        arm_key = "%s|%s|%s" % (s["arm"], s["construction"], s["pool"])
        man["arms"][arm_key] = {
            "arm_spec": s["arm_spec"], "dtype": s["dtype"], "n": s["n"], "seed": s["seed"],
            "limit": s["limit"], "pool_sha256": s["pool_sha256"],
            "config_before": s["config_before"], "config_after_patch": s["config_after_patch"],
            "summary": os.path.relpath(path, LAB),
        }
        for c in s["cells"]:
            man["cells"].append({
                "arm": s["arm"], "construction": s["construction"], "pool": s["pool"],
                "n": c["n"], "seed": s["seed"], "dtype": s["dtype"], "limit": c["limit"],
                "pad": c["pad"], "accuracy": c["accuracy"], "correct": c["correct"],
                "majority_class_accuracy": s["majority_class_accuracy"],
                "median_latency_s": c["median_latency_s"], "p95_latency_s": c["p95_latency_s"],
                "peak_vram_alloc_mib": c["peak_vram_alloc_mib"],
                "peak_vram_reserved_mib": c["peak_vram_reserved_mib"],
                "truncated_items": c["truncated_items"],
                "liveness": c["liveness"],
            })

    out = os.path.join(HERE, "manifest.json")
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, out)
    print("wrote manifest.json: %d arms, %d cells" % (len(man["arms"]), len(man["cells"])))


if __name__ == "__main__":
    main()

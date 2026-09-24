"""Shared model/data loading for the H5 arms.

One process, one model resident at a time (AGENTS.md). The encoder is loaded once and every arm
builds around the same object.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(os.path.dirname(HERE))
FORK = os.path.join(LAB, "worktrees", "h5")
sys.path.insert(0, HERE)
sys.path.insert(0, FORK)

LABELS = ("billing", "technical", "sales", "other")
MAX_LEN = 8192
HEAD_MAX_LEN = 256
CACHE_DIR = os.path.join(HERE, "cache")


def load_agent(device: str = "cuda"):
    """The shipped multilingual agent, exactly as the fork ships it."""
    import laya
    agent = laya.load(os.path.join(LAB, "models", "multilingual"), device=device)
    return agent


def load_plan(path: Optional[str] = None) -> Dict[str, Any]:
    with open(path or os.path.join(HERE, "plan.json"), encoding="utf-8") as fh:
        return json.load(fh)


def load_filler() -> Dict[str, Any]:
    with open(os.path.join(HERE, "pools", "filler-h5-v1.json"), encoding="utf-8") as fh:
        return json.load(fh)


def load_needle_pool(name: str) -> Dict[str, Any]:
    with open(os.path.join(HERE, "pools", name), encoding="utf-8") as fh:
        return json.load(fh)


def internal_question(pool: Dict[str, Any]) -> Dict[str, Any]:
    """The library's own external->internal normalisation, so the prompt is the shipped one."""
    from laya.agent import Agent
    qid = next(iter(pool["question"]))
    return {qid: Agent._to_internal(pool["question"][qid])}


def make_builder(tok, filler_pool: Dict[str, Any], pool: Dict[str, Any], seed: int = 20260924):
    import docs as D
    internal = internal_question(pool)
    qid = next(iter(internal))
    return D.H5DocBuilder(tok, filler_pool, internal[qid], max_len=MAX_LEN,
                          head_max_len=HEAD_MAX_LEN, seed=seed)


def git_state() -> Dict[str, Any]:
    import subprocess

    def run(args, cwd):
        try:
            return subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception as e:  # pragma: no cover - only when git is unavailable
            return "unavailable: %s" % e

    return {
        "lab_head": run(["git", "rev-parse", "HEAD"], LAB),
        "lab_status_porcelain": run(["git", "status", "--porcelain"], LAB),
        "fork_worktree_head": run(["git", "rev-parse", "HEAD"], FORK),
        "fork_branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"], FORK),
        "fork_status_porcelain": run(["git", "status", "--porcelain"], FORK),
    }


def environment(device) -> Dict[str, Any]:
    import platform

    import torch
    try:
        import transformers
        tf_version = transformers.__version__
    except Exception:  # pragma: no cover
        tf_version = "unknown"
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": tf_version,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "USE_TF": os.environ.get("USE_TF"),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_total_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        info["cuda_version"] = torch.version.cuda
    return info


def pool_hashes() -> Dict[str, str]:
    import hashlib
    out = {}
    for name in ("needles-h5-eval-v1.json", "needles-h5-train-v1.json", "filler-h5-v1.json"):
        p = os.path.join(HERE, "pools", name)
        with open(p, "rb") as fh:
            out["experiments/h5-adapter/pools/" + name] = hashlib.sha256(fh.read()).hexdigest()
    return out


def checked_train_items(plan: Dict[str, Any], mode: str) -> List[Dict[str, Any]]:
    if mode == "uniform":
        return plan["train_items"]
    if mode == "long_boost":
        return plan["train_long_boost_items"]
    raise ValueError("unknown train mode %r" % mode)

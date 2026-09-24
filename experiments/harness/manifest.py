"""Run manifest: environment, git state, versions, pools, cells, command, artifacts.

The manifest is written *before* the sweep starts and rewritten when it finishes, so a crashed or
killed run still leaves a full provenance record next to whatever partial JSONL it produced.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = "laya-lab.eval-manifest.v1"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def _git(repo: Path, *args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", str(repo)] + list(args), capture_output=True,
                             text=True, timeout=30)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_state(repo: Path) -> Dict[str, Any]:
    if not (repo / ".git").exists() and _git(repo, "rev-parse", "--git-dir") is None:
        return {"repo": str(repo), "available": False}
    dirty = _git(repo, "status", "--porcelain") or ""
    files = [ln for ln in dirty.splitlines() if ln.strip()]
    return {
        "repo": str(repo),
        "available": True,
        "sha": _git(repo, "rev-parse", "HEAD"),
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(files),
        "dirty_file_count": len(files),
        "dirty_files_sample": files[:20],
        "last_commit": _git(repo, "log", "-1", "--format=%H %ad %s", "--date=iso-strict"),
        "remotes": _git(repo, "remote", "-v"),
    }


def file_provenance(path: Path, hash_contents: bool = True) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.exists():
        st = path.stat()
        out["bytes"] = st.st_size
        if hash_contents:
            from pools import sha256_file
            t0 = time.perf_counter()
            out["sha256"] = sha256_file(path)
            out["hash_seconds"] = round(time.perf_counter() - t0, 3)
    return out


def memory_state() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, v = line.partition(":")
            info[k.strip()] = int(v.strip().split()[0]) * 1024
        out["mem_total_bytes"] = info.get("MemTotal")
        out["mem_available_bytes"] = info.get("MemAvailable")
        out["swap_total_bytes"] = info.get("SwapTotal")
        out["swap_free_bytes"] = info.get("SwapFree")
        out["mem_available_gib"] = round((info.get("MemAvailable") or 0) / 2**30, 2)
    except Exception as e:
        out["error"] = str(e)
    return out


def gpu_state() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            out["gpu_name"] = torch.cuda.get_device_name(0)
            out["gpu_mem_free_bytes"] = int(free)
            out["gpu_mem_total_bytes"] = int(total)
            out["gpu_mem_free_gib"] = round(free / 2**30, 2)
            out["gpu_compute_capability"] = list(torch.cuda.get_device_capability(0))
            out["cuda_version"] = torch.version.cuda
            out["cudnn_version"] = torch.backends.cudnn.version()
        else:
            out["cuda_available"] = False
    except Exception as e:
        out["error"] = str(e)
    return out


def collect_versions() -> Dict[str, Any]:
    def v(mod: str) -> Optional[str]:
        try:
            return getattr(__import__(mod), "__version__", None)
        except Exception:
            return None
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "torch": v("torch"),
        "transformers": v("transformers"),
        "numpy": v("numpy"),
        "safetensors": v("safetensors"),
        "huggingface_hub": v("huggingface_hub"),
        "tokenizers": v("tokenizers"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "hostname": socket.gethostname(),
        "env_USE_TF": os.environ.get("USE_TF"),
        "env_CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "env_PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "env_PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF"),
    }


def command_string(argv: List[str]) -> str:
    env = " ".join("%s=%s" % (k, os.environ[k]) for k in
                   ("USE_TF", "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF")
                   if os.environ.get(k))
    return ("cd %s && %s%s %s" % (os.getcwd(), (env + " ") if env else "",
                                  sys.executable, " ".join(argv)))


def build_manifest(run_id: str, argv: List[str], lab: Path, fork: Path, config: Dict[str, Any],
                   backend_info: Dict[str, Any], items_info: Dict[str, Any],
                   pool_files: Dict[str, Dict[str, Any]], leakage: Dict[str, Any],
                   model_files: List[Path], cells: List[Dict[str, Any]],
                   preflight: Dict[str, Any], gpu_lock: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "run_id": run_id,
        "seed": config.get("seed"),
        "n_per_cell": config.get("n"),
        "languages": config.get("languages"),
        "status": "running",
        "started_utc": utc_now(),
        "finished_utc": None,
        "wall_seconds": None,
        "command": command_string(argv),
        "argv": argv,
        "cwd": os.getcwd(),
        "host": {"versions": collect_versions(), "memory_at_launch": memory_state(),
                 "gpu_at_launch": gpu_state()},
        "gpu_lock": gpu_lock,
        "preflight": preflight,
        "git": {"lab": git_state(lab), "fork": git_state(fork)},
        "model": {"checkpoint_dir": config.get("checkpoint"),
                  "files": [file_provenance(p) for p in model_files],
                  "backend": backend_info},
        "pools": {"files": pool_files, "leakage_check": leakage,
                  "shipped_questions": config.get("questions"),
                  "internal_question_source": config.get("internal_question_source")},
        "items": items_info,
        "protocol": {
            "task": ("needle-in-haystack decision: one support request placed at a given fraction "
                     "through a document of unrelated filler, answered with the shipped "
                     "department question"),
            "filler_mode": config.get("filler_mode"),
            "needle_prefix_mode": config.get("needle_prefix_mode"),
            "warmup": config.get("warmup"),
            "per_item_call": ("one Agent.predict(state, questions, max_len=..., head_max_len=...) "
                              "per item, matching fork/research/scripts/bench_long_context.py; "
                              "latency therefore includes tokenization/truncation plus one forward"),
            "paired_design": ("every cell runs the same item ids; the document for an item depends "
                              "only on (seed, pool, item, pad, position), never on max_len, so "
                              "cells at different max_len are paired per item for McNemar"),
            "determinism": ("items, documents and labels are functions of (pool, n, seed, pad, "
                            "position); latency is not deterministic"),
        },
        "cells": cells,
        "artifacts": {
            "predictions_jsonl": str(Path(config["out_dir"]) / "predictions.jsonl"),
            "summary_json": str(Path(config["out_dir"]) / "summary.json"),
            "run_log": str(Path(config["out_dir"]) / "run.log"),
            "manifest_json": str(Path(config["out_dir"]) / "manifest.json"),
        },
    }

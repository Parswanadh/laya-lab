"""V-001 instrumented reproduction: nondeterminism spread + per-item predictions.

Not a modification of the scripts under verification -- it rebuilds the same cells with the same
constants (extracted by AST from the originals) and additionally records what those scripts throw
away: the per-item prediction for every cell, the dtype in force, and the real model input length
(`usage.input_tokens`).

    env/venv/bin/python experiments/V-001/repeat.py
"""
import ast
import json
import os
import statistics
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

LAB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(LAB, "fork"))

import laya  # noqa: E402
import torch  # noqa: E402

DIAG = os.path.join(LAB, "experiments", "orch-diagnostic", "run.py")
BASE = os.path.join(LAB, "experiments", "orch-baseline", "run.py")


def load_source(path):
    src = open(path).read()
    ns = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in ("place", "filler_ids"):
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in ("REQUESTS", "FILLER", "QUESTIONS"):
                    ns[t.id] = ast.literal_eval(node.value)
    return ns


def run_cell(agent, tok, cases, max_len, dtype, questions):
    if dtype is not None:
        agent.dtype = dtype
    per_item, lat = [], []
    for text, gold in cases:
        t0 = time.time()
        r = agent.predict({"text": text}, questions, max_len=max_len)
        lat.append(time.time() - t0)
        ans = r["answers"]["department"]
        per_item.append({"gold": gold, "pred": ans["choice"],
                         "p_gold": round(float(ans["probabilities"].get(gold, 0.0)), 6),
                         "pred_p": round(float(ans["probabilities"][ans["choice"]]), 6),
                         "input_tokens": int(r.get("usage", {}).get("input_tokens", -1)),
                         "max_len": max_len,
                         "dtype": str(agent.dtype)})
    acc = sum(1 for i in per_item if i["pred"] == i["gold"]) / max(1, len(per_item))
    preds = [i["pred"] for i in per_item]
    return {"accuracy": round(acc, 4), "n": len(per_item),
            "median_latency_s": round(statistics.median(lat), 4) if lat else None,
            "distinct_predictions": sorted(set(preds)),
            "pred_histogram": {p: preds.count(p) for p in sorted(set(preds))},
            "per_item": per_item}


def main():
    diag, base = load_source(DIAG), load_source(BASE)
    questions = base["QUESTIONS"]
    REQUESTS, FILLER = diag["REQUESTS"], diag["FILLER"]
    assert REQUESTS == base["REQUESTS"] and FILLER == base["FILLER"]
    place, filler_ids = diag["place"], diag["filler_ids"]

    agent = laya.load(os.path.join(LAB, "models", "multilingual"), device="cuda")
    tok = agent.tok
    out = {"gpu": torch.cuda.get_device_name(0),
           "dtype_at_load": str(agent.dtype),
           "cfg_max_len": agent.cfg.get("max_len"),
           "torch": torch.__version__,
           "cells": {}, "repeats": {}}

    fid = tok(FILLER, add_special_tokens=False)["input_ids"]

    # ---- 1. baseline cells, with the per-item detail the baseline script does not keep ------
    # dtype is whatever the agent loaded with: the baseline never touches agent.dtype.
    for limit in (1024, 8192):
        for pad in (0, 1000, 2000, 4000, 7000):
            body = (fid * (pad // max(1, len(fid)) + 1))[:pad]
            body_text = tok.decode(body)
            cases = [(body_text + " " + req, gold) for req, gold in REQUESTS]
            key = "baseline_limit%d_pad%d" % (limit, pad)
            out["cells"][key] = run_cell(agent, tok, cases, limit, None, questions)

    # ---- 2. nondeterminism: the same cell, twice, in one process, and once more in a --------
    #         second process (the second pass is driven by gpu_repro.sh and diffed later).
    key = "baseline_limit8192_pad7000"
    body70 = (fid * (7000 // max(1, len(fid)) + 1))[:7000]
    cases70 = [(tok.decode(body70) + " " + req, gold) for req, gold in REQUESTS]
    out["repeats"]["baseline_8192_7000_run2"] = run_cell(agent, tok, cases70, 8192, None, questions)
    out["repeats"]["baseline_8192_7000_run3"] = run_cell(agent, tok, cases70, 8192, None, questions)

    key = "baseline_limit1024_pad1000"
    body10 = (fid * (1000 // max(1, len(fid)) + 1))[:1000]
    cases10 = [(tok.decode(body10) + " " + req, gold) for req, gold in REQUESTS]
    out["repeats"]["baseline_1024_1000_run2"] = run_cell(agent, tok, cases10, 1024, None, questions)

    # ---- 3. the diagnostic's fp32 cells, repeated, to measure the fp32 spread ----------------
    cases_d = [(place(tok, 4000, 1.0, req), gold) for req, gold in REQUESTS]
    out["repeats"]["diag_C4000_fp32_run1"] = run_cell(agent, tok, cases_d, 8192, torch.float32, questions)
    out["repeats"]["diag_C4000_fp32_run2"] = run_cell(agent, tok, cases_d, 8192, torch.float32, questions)
    cases_b = [(place(tok, 4000, 0.5, req), gold) for req, gold in REQUESTS]
    out["repeats"]["diag_B0.5_fp32_run1"] = run_cell(agent, tok, cases_b, 8192, torch.float32, questions)
    out["repeats"]["diag_B0.5_fp32_run2"] = run_cell(agent, tok, cases_b, 8192, torch.float32, questions)

    d = os.path.join(LAB, "experiments", "V-001")
    with open(os.path.join(d, "repeat.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("dtype at load: %s  cfg max_len: %s" % (out["dtype_at_load"], out["cfg_max_len"]))
    for k, v in out["cells"].items():
        print("%-28s acc=%.3f  preds=%s  input_tok=%s" % (
            k, v["accuracy"], v["pred_histogram"],
            sorted({i["input_tokens"] for i in v["per_item"]})))
    for k, v in out["repeats"].items():
        print("%-28s acc=%.3f  preds=%s" % (k, v["accuracy"], v["pred_histogram"]))
    print("wrote experiments/V-001/repeat.json")


if __name__ == "__main__":
    main()

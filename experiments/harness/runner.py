"""Sweep orchestration: build the cells, run the model once per item per cell, write raw rows.

The raw artifact is ``predictions.jsonl`` — one line per (cell, item) — written and flushed as the
sweep proceeds. ``summary.json`` and ``manifest.json`` are rewritten after every cell so a kill at
any point leaves a consistent, readable state. Nothing is aggregated away: the JSONL is the source
of truth for every number in ``summary.json``.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backends import LayaBackend, StubBackend
from builder import DocBuilder, doc_row_fields
import checks
from checks import mechanism_checks, report, structural_checks
from manifest import build_manifest, memory_state, gpu_state, utc_now, write_json
from metrics import cell_summary, position_only_oracle
import pools as pools_mod


class Logger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.fh.write(msg + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def acquire_gpu_lock(path: Path, enabled: bool) -> Tuple[Optional[Any], Dict[str, Any]]:
    if not enabled:
        return None, {"held": False, "reason": "not requested (cpu/dry-run path)"}
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None, {"held": False, "reason": "another process holds the lock", "path": str(path)}
    fh.seek(0, os.SEEK_END)
    fh.write("%s pid=%d %s\n" % (utc_now(), os.getpid(), " ".join(sys.argv)))
    fh.flush()
    return fh, {"held": True, "path": str(path), "pid": os.getpid()}


def resolve_cells(pads: Sequence[int], positions: Sequence[float], max_lens: Sequence[Any],
                  head_max_lens: Sequence[Any], shipped_max_len: int,
                  shipped_head_max_len: int) -> List[Dict[str, Any]]:
    cells = []
    for pad in pads:
        # with no filler every needle position is the same document; one cell, flagged
        pos_list = [1.0] if pad == 0 else list(positions)
        degenerate = pad == 0
        for pos in pos_list:
            for ml in max_lens:
                for hml in head_max_lens:
                    ml_eff = shipped_max_len if ml == "default" else int(ml)
                    hml_eff = shipped_head_max_len if hml == "default" else int(hml)
                    ml_tag = "default" if ml == "default" else str(int(ml))
                    hml_tag = "default" if hml == "default" else str(int(hml))
                    cells.append({
                        "cell_key": "pad%d|pos%.2f|ml%s|hml%s" % (pad, pos, ml_tag, hml_tag),
                        "pad_tokens": int(pad),
                        "needle_position": float(pos),
                        "position_requested": float(pos),
                        "position_degenerate": degenerate,
                        "max_len_requested": ml_tag,
                        "max_len_effective": int(ml_eff),
                        "head_max_len_requested": hml_tag,
                        "head_max_len_effective": int(hml_eff),
                    })
    return cells


def make_row(run_id: str, cell: Dict[str, Any], item: Dict[str, Any], doc: Dict[str, Any],
             diag: Dict[str, Any], answer: Dict[str, Any], latency_s: float,
             backend: Any, seed: int, warmup_call: bool) -> Dict[str, Any]:
    probs = dict(answer["probabilities"])
    options = list(probs.keys())
    row = {
        "run_id": run_id,
        "schema": "laya-lab.eval-row.v1",
        "cell": cell["cell_key"],
        "item_index": item["item_index"],
        "item_id": item["item_id"],
        "label": item["label"],
        "lang": item["lang"],
        "template_id": item["template_id"],
        "slots": item["slots"],
        "prediction": answer["choice"],
        "probabilities": probs,
        "prob_vector": [float(probs[o]) for o in options],
        "options": options,
        "correct": bool(answer["choice"] == item["label"]),
        "confidence": answer.get("confidence"),
        "answer_confidence": answer.get("answer_confidence"),
        "act_probability": (answer.get("action") or {}).get("act_probability"),
        "pad_tokens": cell["pad_tokens"],
        "needle_position": cell["needle_position"],
        "position_degenerate": cell["position_degenerate"],
        "max_len_requested": cell["max_len_requested"],
        "max_len_effective": cell["max_len_effective"],
        "head_max_len_requested": cell["head_max_len_requested"],
        "head_max_len_effective": cell["head_max_len_effective"],
        "latency_s": round(float(latency_s), 6),
        "warmup_call": bool(warmup_call),
        "seed": seed,
        "device": str(getattr(backend, "device", "?")),
        "checkpoint": getattr(backend, "checkpoint", "stub"),
        "backend": backend.name,
    }
    row.update(doc_row_fields(doc))
    row.update(diag)
    return row


def run_sweep(cfg: Dict[str, Any]) -> int:
    lab = Path(cfg["lab"]).resolve()
    fork = Path(cfg["fork"]).resolve()
    if str(fork) not in sys.path:
        sys.path.insert(0, str(fork))
    os.environ.setdefault("USE_TF", "0")

    out_dir = Path(cfg["out_dir"])
    if not out_dir.is_absolute():
        out_dir = lab / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "run.log")
    t_start = time.time()
    log("=== %s ===" % cfg["run_id"])
    log("command: %s" % " ".join(sys.argv))
    log("out: %s" % out_dir)

    lock_fh, lock_info = (None, {"held": False, "reason": "cpu/dry-run"})
    if not cfg["dry_run"] and str(cfg["device"]).startswith("cuda"):
        lock_fh, lock_info = acquire_gpu_lock(Path(cfg["gpu_lock_path"]), cfg.get("use_gpu_lock", True))
        if not lock_info["held"]:
            log("GPU lock refused: %s -- do non-GPU work and retry (not queueing a second job)"
                % lock_info)
            log.close()
            return 2
        log("gpu lock: %s" % lock_info)

    # ---- preflight ----------------------------------------------------------
    mem = memory_state()
    gpu = gpu_state()
    preflight = {"mem_available_gib": mem.get("mem_available_gib"),
                 "gpu_free_gib": gpu.get("gpu_mem_free_gib"),
                 "min_available_gib_required": cfg["min_available_gib"],
                 "gpu_min_free_gib_required": cfg["min_gpu_free_gib"],
                 "allow_low_mem": cfg.get("allow_low_mem", False)}
    if not cfg["dry_run"]:
        why = None
        if (mem.get("mem_available_gib") or 0) < cfg["min_available_gib"]:
            why = "host RAM available %.2f GiB < %.2f GiB" % (mem.get("mem_available_gib") or 0,
                                                              cfg["min_available_gib"])
        if str(cfg["device"]).startswith("cuda") and (gpu.get("gpu_mem_free_gib") or 0) < cfg["min_gpu_free_gib"]:
            why = why or "GPU free %.2f GiB < %.2f GiB" % (gpu.get("gpu_mem_free_gib") or 0,
                                                           cfg["min_gpu_free_gib"])
        if why and not cfg.get("allow_low_mem"):
            log("PREFLIGHT REFUSED: %s. Free memory or pass --allow-low-mem." % why)
            log.close()
            return 3
        preflight["note"] = ("host RAM at launch %.2f GiB available; launch is capped by "
                             "systemd-run MemoryMax=%s when the documented launcher is used"
                             % (mem.get("mem_available_gib") or 0, cfg.get("memory_cap", "4G")))
    log("preflight: %s" % json.dumps(preflight, ensure_ascii=False))

    # ---- pools, items, cells ------------------------------------------------
    needle_pool = pools_mod.load_pool(cfg["pool"])
    filler_pool = pools_mod.load_pool(cfg["filler_pool"])
    if needle_pool.get("kind") == "upstream" and cfg["filler_pool"] != "filler-v1":
        log("note: upstream pool composes its own filler unit; --filler-pool is only used "
            "for the leakage check")
    leakage = pools_mod.leakage_report(needle_pool, filler_pool)
    if not leakage["passed"]:
        log("LEAKAGE CHECK FAILED: %s" % json.dumps(leakage, ensure_ascii=False)[:1000])
        log.close()
        return 4
    log("leakage check passed: %s" % json.dumps(
        {k: leakage[k] for k in ("needle_vocab_size", "filler_vocab_size", "content_word_overlap",
                                 "stem_hits")}, ensure_ascii=False))

    items_info = pools_mod.build_items(needle_pool, cfg["n"], cfg["seed"],
                                       languages=cfg.get("languages"))
    items = items_info["items"]
    log("items: n=%d label_counts=%s balance_ok=%s langs=%s"
        % (items_info["n"], items_info["label_counts"], items_info["balance_ok"],
           items_info["lang_counts"]))

    iq = pools_mod.internal_question(needle_pool)
    questions = needle_pool["question"]
    qid = next(iter(questions))
    if len(questions) != 1:
        raise ValueError("the harness answers exactly one question per item; pool has %d"
                         % len(questions))

    # ---- backend ------------------------------------------------------------
    shipped_max = None
    shipped_head = None
    if cfg["dry_run"]:
        backend: Any = StubBackend(seed=cfg["seed"], max_len=cfg["stub_max_len"],
                                   head_max_len=cfg["stub_head_max_len"])
        log("backend: stub (dry run) shipped max_len=%d head_max_len=%d"
            % (backend.cfg["max_len"], backend.cfg["head_max_len"]))
    else:
        backend = LayaBackend(cfg["checkpoint"], device=cfg["device"], subfolder=cfg.get("subfolder"))
        log("backend: laya %s device=%s dtype=%s amp=%s params=%s load=%.1fs"
            % (cfg["checkpoint"], backend.device, backend.dtype, backend.amp_enabled,
               backend.n_params, backend.load_seconds))
    shipped_max, shipped_head = backend.shipped_defaults()
    log("shipped defaults: max_len=%d head_max_len=%d" % (shipped_max, shipped_head))

    cells = resolve_cells(cfg["pads"], cfg["positions"], cfg["max_lens"], cfg["head_max_lens"],
                          shipped_max, shipped_head)
    log("cells: %d -> %s" % (len(cells), ", ".join(c["cell_key"] for c in cells)))

    builder = DocBuilder(backend.tok, needle_pool, filler_pool, iq["questions"])

    model_files = []
    if not cfg["dry_run"]:
        ckpt = Path(cfg["checkpoint"])
        for name in ("model.safetensors", "rl_agent_config.json", "encoder/config.json",
                     "tokenizer/tokenizer.json"):
            model_files.append(ckpt / name)

    manifest = build_manifest(
        run_id=cfg["run_id"], argv=list(sys.argv), lab=lab, fork=fork,
        config={"checkpoint": cfg["checkpoint"], "out_dir": str(out_dir), "seed": cfg["seed"],
                "n": cfg["n"], "languages": cfg.get("languages"),
                "questions": questions, "internal_question_source": iq["source"],
                "filler_mode": needle_pool.get("kind"), "needle_prefix_mode": needle_pool.get("needle_prefix_mode", "upstream_auto"),
                "warmup": "one untimed call on the first item of every cell, plus one global call",
                },
        backend_info=backend.info(), items_info={k: v for k, v in items_info.items() if k != "items"},
        pool_files={"needle": {k: needle_pool[k] for k in ("pool_id", "_path", "_sha256", "_bytes", "kind")},
                    "filler": {k: filler_pool[k] for k in ("pool_id", "_path", "_sha256", "_bytes", "kind")}},
        leakage=leakage, model_files=model_files, cells=cells, preflight=preflight,
        gpu_lock=lock_info)
    manifest["items"]["item_table"] = [
        {k: it[k] for k in ("item_index", "item_id", "label", "lang", "template_id", "slots", "text")}
        for it in items]
    manifest["dry_run"] = bool(cfg["dry_run"])
    if needle_pool.get("kind") == "upstream":
        manifest["pools"]["upstream_transcription_check"] = pools_mod.verify_upstream_transcription(
            needle_pool, fork / "research" / "scripts" / "bench_long_context.py")
    manifest["protocol"] = {
        "protocol_md": "protocol.md (frozen, 2026-09-24)",
        "task": "synthetic needle-in-haystack decision; NOT evidence about real long documents (protocol section 10)",
        "label_balanced": items_info["balance_ok"],
        "label_counts": items_info["label_counts"],
        "n_per_cell": cfg["n"],
        "protocol_n_requirement": ("protocol section 3 requires n>=200 for a number quoted as a "
                                   "result; this run is n=%d and every table must carry that n"
                                   % cfg["n"]),
        "L_axis": sorted({int(p) for p in cfg["pads"]}),
        "L_cells_skipped": ("L=8192/16000/32000 from protocol section 2 are not run: the baseline "
                            "question is the shipped 1024 limit versus 8192, and the encoder's "
                            "max_position_embeddings is 8192 so longer L is an arm-side question, "
                            "not a baseline one"),
        "position_axis": sorted({float(p) for p in cfg["positions"]}),
        "max_len_axis": [str(x) for x in cfg["max_lens"]],
        "head_max_len_axis": [str(x) for x in cfg["head_max_lens"]],
        "languages": cfg.get("languages") or "all pool languages",
        "warmup": ("3 discarded forward passes after load, then one discarded pass per cell before "
                   "the first timed item (protocol section 8 asks for >=3 before any timing)"),
        "gpu_lock": lock_info,
        "gpu_lock_held_for_timings": bool(lock_info.get("held")),
        "torch_deterministic_algorithms": False,
        "torch_deterministic_note": ("not enabled; instead the sweep re-runs the first cell and "
                                     "records repeat_prediction_agreement, which is the property "
                                     "that matters here"),
        "raw_per_item_predictions": "predictions.jsonl (one row per cell per item)",
        "baselines_reported": ["random_over_options", "random_over_gold_labels", "majority_class",
                               "position_only_oracle"],
        "baselines_unavailable": {
            "predict_long_PR_363": "not present in fork @ %s" % (
                __import__("manifest").git_state(fork).get("sha")),
            "library_versions_and_git": "host.versions + git block",
        },
    }
    manifest["harness_files"] = {
        name: {"sha256": pools_mod.sha256_file(Path(__file__).resolve().parent / name)}
        for name in ("run.py", "runner.py", "builder.py", "pools.py", "metrics.py", "backends.py",
                     "checks.py", "manifest.py", "presets.json")}
    summary = {"run_id": cfg["run_id"], "schema": "laya-lab.eval-summary.v1",
               "pool": needle_pool["pool_id"], "seed": cfg["seed"], "n_per_cell": cfg["n"],
               "cells": [], "notes": []}
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "summary.json", summary)

    # ---- global warm-up (protocol section 8: >=3 discarded passes) -----------
    try:
        for w in range(3):
            st = items[w % len(items)]["text"]
            backend.predict(st, questions, max_len=shipped_max, head_max_len=shipped_head)
        log("global warm-up done (3 discarded passes at shipped max_len=%d)" % shipped_max)
    except Exception as e:
        log("global warm-up failed (continuing): %r" % e)

    # ---- sweep --------------------------------------------------------------
    n_rows = 0
    try:
        with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as fh:
            for pad in cfg["pads"]:
                pos_list = [1.0] if pad == 0 else list(cfg["positions"])
                for pos in pos_list:
                    docs = [builder.build(it, pad, pos, cfg["seed"]) for it in items]
                    for cell in [c for c in cells if c["pad_tokens"] == pad
                                 and c["needle_position"] == pos]:
                        t_cell = time.time()
                        cell_rows = []
                        for i, item in enumerate(items):
                            doc = docs[i]
                            diag = builder.diagnose(doc, cell["max_len_effective"],
                                                    cell["head_max_len_effective"])
                            if i == 0:
                                backend.predict(doc["state"], questions,
                                                max_len=None if cell["max_len_requested"] == "default" else cell["max_len_effective"],
                                                head_max_len=None if cell["head_max_len_requested"] == "default" else cell["head_max_len_effective"])
                            out = backend.predict(
                                doc["state"], questions,
                                max_len=None if cell["max_len_requested"] == "default" else cell["max_len_effective"],
                                head_max_len=None if cell["head_max_len_requested"] == "default" else cell["head_max_len_effective"])
                            answer = out["result"]["answers"][qid]
                            row = make_row(cfg["run_id"], cell, item, doc, diag, answer,
                                           out["latency_s"], backend, cfg["seed"], i == 0)
                            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                            cell_rows.append(row)
                            n_rows += 1
                        fh.flush()
                        cs = cell_summary(cell_rows, n_boot=cfg["n_boot"], seed=cfg["seed"])
                        entry = dict(cell)
                        entry["cell_wall_seconds"] = round(time.time() - t_cell, 3)
                        entry.update(cs)
                        summary["cells"].append(entry)
                        write_json(out_dir / "summary.json", summary)
                        log("pad %5d  pos %.2f  max_len %-7s  acc %3d/%3d (%.3f)  ci95 [%.2f, %.2f]  "
                            "med %.3fs  tokens %s  kept %.3f  request_kept %.2f  modal %s (%.2f)"
                            % (pad, pos, cell["max_len_requested"], cs["correct"], cs["n"],
                               cs["accuracy"], cs["accuracy_ci95_low"], cs["accuracy_ci95_high"],
                               cs["median_latency_s"], cs["median_input_tokens"],
                               cs["median_state_tokens_kept"] / max(1, cs["median_state_tokens_full"]),
                               cs["request_kept_fraction"] or 0.0, cs["modal_prediction"],
                               cs["modal_prediction_share"]))
            fh.flush()
    except BaseException:
        manifest["status"] = "failed"
        manifest["failure"] = traceback.format_exc()
        manifest["finished_utc"] = utc_now()
        manifest["wall_seconds"] = round(time.time() - t_start, 3)
        manifest["rows_written"] = n_rows
        write_json(out_dir / "manifest.json", manifest)
        log("FAILED after %d rows:\n%s" % (n_rows, manifest["failure"]))
        log.close()
        raise

    # ---- read the raw artifact back; every number below comes from it --------
    all_rows: List[Dict[str, Any]] = []
    with open(out_dir / "predictions.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                all_rows.append(json.loads(line))

    # ---- protocol-required reference numbers, from the same raw rows ---------
    summary["baselines"] = {
        "random_over_options": 1.0 / len(all_rows[0]["options"]) if all_rows else None,
        "random_over_gold_labels": 1.0 / len({r["label"] for r in all_rows}) if all_rows else None,
        "majority_class_accuracy_per_cell": {c["cell_key"]: c["majority_class_accuracy"]
                                             for c in summary["cells"]},
        "position_only_oracle": position_only_oracle(all_rows),
    }

    # ---- repeat check: is inference reproducible on this box? ---------------
    if cfg.get("repeat_check_items", 0):
        first = cells[0]
        subset = list(items)[: cfg["repeat_check_items"]]
        agree = 0
        max_delta = 0.0
        prior = {r["item_id"]: r for r in all_rows if r["cell"] == first["cell_key"]}
        for it in subset:
            doc = builder.build(it, first["pad_tokens"], first["needle_position"], cfg["seed"])
            out = backend.predict(doc["state"], questions,
                                  max_len=None if first["max_len_requested"] == "default" else first["max_len_effective"],
                                  head_max_len=None if first["head_max_len_requested"] == "default" else first["head_max_len_effective"])
            ans = out["result"]["answers"][qid]
            prev = prior.get(it["item_id"])
            if prev is None:
                continue
            agree += 1 if ans["choice"] == prev["prediction"] else 0
            for k, v in ans["probabilities"].items():
                max_delta = max(max_delta, abs(float(v) - float(prev["probabilities"][k])))
        n_cmp = len(subset)
        summary["repeat_check"] = {
            "cell": first["cell_key"], "items": n_cmp,
            "prediction_agreement": (agree / n_cmp) if n_cmp else None,
            "max_abs_probability_delta": max_delta,
            "note": "same process, same seeds, same inputs, GPU lock held for both passes",
        }
        log("repeat check: %d/%d identical predictions, max |dp| = %.6f"
            % (agree, n_cmp, max_delta))

    try:
        import torch
        if torch.cuda.is_available():
            summary["peak_vram_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            summary["peak_vram_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    except Exception:
        pass

    manifest["status"] = "finished"
    manifest["finished_utc"] = utc_now()
    manifest["wall_seconds"] = round(time.time() - t_start, 3)
    manifest["rows_written"] = n_rows
    write_json(out_dir / "manifest.json", manifest)

    # ---- checks -------------------------------------------------------------
    log("")
    log("--- structural checks (from the published artifacts) ---")
    ok_struct = report(structural_checks(out_dir, manifest, summary), log=log)
    ok_mech = True
    if cfg.get("mechanism_checks", True):
        log("--- mechanism checks (rebuild from seed) ---")
        ok_mech = report(mechanism_checks(builder, items, cells, cfg["seed"], all_rows), log=log)
    summary["notes"].append("structural_checks_passed=%s" % ok_struct)
    summary["notes"].append("mechanism_checks_passed=%s" % ok_mech)
    summary["manifest_status"] = manifest["status"]
    summary["wall_seconds"] = manifest["wall_seconds"]
    write_json(out_dir / "summary.json", summary)

    log("")
    log("wrote %s (%d rows) in %.1fs" % (out_dir / "predictions.jsonl", n_rows,
                                         time.time() - t_start))
    log("checks: structural=%s mechanism=%s" % (ok_struct, ok_mech))
    try:
        backend.close()
    except Exception:
        pass
    if lock_fh is not None:
        lock_fh.close()
    log.close()
    if cfg.get("fail_on_check", True) and not (ok_struct and ok_mech):
        return 5
    return 0


def print_table(out_dir: Path) -> None:
    summary = json.loads((Path(out_dir) / "summary.json").read_text(encoding="utf-8"))
    print("| pad | pos | max_len | n | correct | acc | ci95 | median s | median kept tok | modal pred (share) | request_kept |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in summary["cells"]:
        print("| %d | %.2f | %s | %d | %d | %.3f | [%.2f, %.2f] | %.3f | %s | %s (%.2f) | %.2f |"
              % (c["pad_tokens"], c["needle_position"], c["max_len_requested"], c["n"], c["correct"],
                 c["accuracy"], c["accuracy_ci95_low"], c["accuracy_ci95_high"],
                 c["median_latency_s"], c["median_state_tokens_kept"], c["modal_prediction"],
                 c["modal_prediction_share"], c["request_kept_fraction"] or 0.0))


def recheck(run_dir: Path) -> int:
    """Re-run every check on a finished run, from its artifacts only. No model, no GPU, no lock.

    Reads the manifest for the pools, seed, item table and cell plan, rebuilds the documents with
    the checkpoint tokenizer (tokenizer only -- the encoder is never constructed), and re-derives
    every structural and mechanism fact. This is what a verifier can run on any run directory.
    """
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    log = Logger(run_dir / "recheck.log")
    log("=== recheck %s ===" % run_dir)
    ok_struct = report(structural_checks(run_dir, manifest, summary), log=log)

    if manifest.get("dry_run"):
        from backends import StubTokenizer
        tok = StubTokenizer()
    else:
        fork = (manifest.get("git", {}).get("fork", {}) or {}).get("repo")
        if fork and fork not in sys.path:
            sys.path.insert(0, fork)
        from laya.agent import _load_tokenizer
        ckpt = Path(manifest["model"]["checkpoint_dir"])
        cfg = json.loads((ckpt / "rl_agent_config.json").read_text(encoding="utf-8"))
        tok = _load_tokenizer(str(ckpt / "tokenizer"), cfg)
    needle_pool = pools_mod.load_pool(manifest["pools"]["files"]["needle"]["_path"])
    filler_pool = pools_mod.load_pool(manifest["pools"]["files"]["filler"]["_path"])
    iq = pools_mod.internal_question(needle_pool)
    builder = DocBuilder(tok, needle_pool, filler_pool, iq["questions"])
    rows = checks.load_rows(run_dir)
    ok_mech = report(mechanism_checks(builder, items=manifest["items"]["item_table"],
                                     cells=manifest["cells"],
                                     seed=manifest.get("seed") or summary["seed"], rows=rows),
                     log=log)
    log("checks: structural=%s mechanism=%s" % (ok_struct, ok_mech))
    log.close()
    return 0 if (ok_struct and ok_mech) else 5

#!/usr/bin/env python
"""One-command entry point for the laya-lab long-context evaluation harness.

    # what the run would do, without a checkpoint
    env/venv/bin/python experiments/harness/run.py --preset baseline-repro --plan

    # pipeline self-test: stub tokenizer + tiny random model, no GPU, no checkpoint
    env/venv/bin/python experiments/harness/run.py --preset baseline-repro --dry-run

    # the real thing (takes the GPU lock itself)
    env/venv/bin/python experiments/harness/run.py --preset baseline-repro \
        --device cuda --checkpoint models/multilingual

    # re-check an existing run from its published artifacts only
    env/venv/bin/python experiments/harness/run.py --check-only experiments/baseline-repro
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
os.environ.setdefault("USE_TF", "0")

LAB = HERE.parents[1]

PRESETS_PATH = HERE / "presets.json"

DEFAULTS = {
    "preset": None,
    "out": None,
    "pool": "balanced-v1",
    "filler_pool": "filler-v1",
    "pads": [0, 1000, 2000, 4000, 7000],
    "positions": [1.0],
    "max_lens": ["default", 8192],
    "head_max_lens": ["default"],
    "n": 20,
    "seed": 20240924,
    "device": "cuda",
    "checkpoint": str(LAB / "models" / "multilingual"),
    "subfolder": None,
    "dry_run": False,
    "n_boot": 10000,
    "allow_low_mem": False,
    "use_gpu_lock": True,
    "gpu_lock_path": str(LAB / ".gpu.lock"),
    "min_available_gib": 2.0,
    "min_gpu_free_gib": 2.0,
    "stub_max_len": 64,
    "stub_head_max_len": 24,
    "memory_cap": os.environ.get("LAYA_MEMORY_CAP", "not recorded"),
    "languages": None,
    "repeat_check_items": 20,
}


def _ints(text) -> list:
    return [int(x) for x in str(text).split(",") if str(x).strip() != ""]


def _floats(text) -> list:
    return [float(x) for x in str(text).split(",") if str(x).strip() != ""]


def _lens(text) -> list:
    out = []
    for x in str(text).split(","):
        x = x.strip()
        if not x:
            continue
        out.append("default" if x == "default" else int(x))
    return out


def load_presets() -> dict:
    return json.loads(PRESETS_PATH.read_text(encoding="utf-8"))


def build_config(args) -> dict:
    presets = load_presets()
    cfg = dict(DEFAULTS)
    if args.preset:
        if args.preset not in presets:
            raise SystemExit("unknown preset %r; have %s" % (args.preset, sorted(presets)))
        preset = presets[args.preset]
        for k, v in preset.items():
            if k == "out" and v is not None and not Path(v).is_absolute():
                v = str(LAB / v)
            cfg[k] = v
        cfg["preset"] = args.preset
    # explicit CLI flags win over the preset
    for key in ("out", "pool", "filler_pool", "n", "seed", "device", "checkpoint", "subfolder",
                "n_boot", "stub_max_len", "stub_head_max_len", "repeat_check_items"):
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    for key, conv in (("pads", _ints), ("positions", _floats), ("max_lens", _lens),
                      ("head_max_lens", _lens)):
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = conv(val)
    if args.languages is not None:
        cfg["languages"] = [x.strip().lower() for x in args.languages.split(",") if x.strip()]
    if args.dry_run:
        cfg["dry_run"] = True
    if args.allow_low_mem:
        cfg["allow_low_mem"] = True
    if args.no_gpu_lock:
        cfg["use_gpu_lock"] = False
    if getattr(args, "no_mechanism_checks", False):
        cfg["mechanism_checks"] = False
    if getattr(args, "no_fail_on_check", False):
        cfg["fail_on_check"] = False
    if cfg["dry_run"]:
        cfg["device"] = "cpu"
        cfg["use_gpu_lock"] = False
    if cfg["out"] is None:
        cfg["out"] = str(LAB / "experiments" / (args.preset or "adhoc"))
    cfg["out_dir"] = cfg["out"]
    cfg["lab"] = str(LAB)
    cfg["fork"] = str(LAB / "fork")
    cfg["memory_cap"] = DEFAULTS["memory_cap"]
    cfg["run_id"] = "%s|%s|n%d|seed%d%s" % (args.preset or "adhoc", cfg["pool"], cfg["n"],
                                           cfg["seed"], ("|" + args.tag) if args.tag else "")
    return cfg


def shipped_defaults_from_config(checkpoint: str) -> tuple:
    p = Path(checkpoint) / "rl_agent_config.json"
    if not p.exists():
        return 512, 192
    cfg = json.loads(p.read_text(encoding="utf-8"))
    return int(cfg.get("max_len", 512)), int(cfg.get("head_max_len", 192))


def main(argv=None) -> int:
    presets = load_presets()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(presets))
    ap.add_argument("--out", help="run directory (default: experiments/<preset>)")
    ap.add_argument("--pool", help="needle pool id or path")
    ap.add_argument("--filler-pool", dest="filler_pool", help="filler pool id or path")
    ap.add_argument("--pads", help="comma list of filler token budgets, e.g. 0,1000,2000")
    ap.add_argument("--positions", help="comma list of needle positions in [0,1], e.g. 0.0,0.5,1.0")
    ap.add_argument("--max-lens", dest="max_lens", help="comma list of max_len values; 'default' = shipped")
    ap.add_argument("--head-max-lens", dest="head_max_lens", help="comma list of head_max_len values; 'default' = shipped")
    ap.add_argument("--n", type=int, help="items per cell")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device", help="cuda | cpu | mps")
    ap.add_argument("--checkpoint")
    ap.add_argument("--subfolder")
    ap.add_argument("--dry-run", action="store_true", help="stub tokenizer + tiny random model")
    ap.add_argument("--languages", help="comma list of languages to draw items from (default: all)")
    ap.add_argument("--repeat-check-items", dest="repeat_check_items", type=int,
                    help="re-run this many items of the first cell and record agreement (0 disables)")
    ap.add_argument("--tag", default="", help="suffix for the run id")
    ap.add_argument("--n-boot", dest="n_boot", type=int, help="bootstrap resamples (default 10000)")
    ap.add_argument("--allow-low-mem", action="store_true")
    ap.add_argument("--no-gpu-lock", action="store_true")
    ap.add_argument("--no-mechanism-checks", action="store_true")
    ap.add_argument("--no-fail-on-check", action="store_true")
    ap.add_argument("--stub-max-len", dest="stub_max_len", type=int)
    ap.add_argument("--stub-head-max-len", dest="stub_head_max_len", type=int)
    ap.add_argument("--plan", action="store_true", help="print the resolved plan and exit")
    ap.add_argument("--check-only", metavar="RUN_DIR", help="run the structural checks on a finished run")
    ap.add_argument("--recheck", metavar="RUN_DIR",
                    help="re-run structural AND mechanism checks on a finished run (no model, no GPU)")
    ap.add_argument("--table", metavar="RUN_DIR", help="print the summary table of a finished run")
    args = ap.parse_args(argv)

    if args.table:
        import runner
        runner.print_table(Path(args.table))
        return 0

    if args.recheck:
        import runner
        return runner.recheck(Path(args.recheck))

    if args.check_only:
        import checks
        ok = checks.report(checks.structural_checks(Path(args.check_only)))
        return 0 if ok else 5

    cfg = build_config(args)

    if args.plan:
        import pools as pools_mod
        from runner import resolve_cells
        needle_pool = pools_mod.load_pool(cfg["pool"])
        filler_pool = pools_mod.load_pool(cfg["filler_pool"])
        leak = pools_mod.leakage_report(needle_pool, filler_pool)
        items_info = pools_mod.build_items(needle_pool, cfg["n"], cfg["seed"],
                                           languages=cfg.get("languages"))
        smax, shead = shipped_defaults_from_config(cfg["checkpoint"])
        cells = resolve_cells(cfg["pads"], cfg["positions"], cfg["max_lens"], cfg["head_max_lens"],
                              smax, shead)
        print("run_id      : %s" % cfg["run_id"])
        print("out         : %s" % cfg["out_dir"])
        print("pool        : %s (%s)" % (needle_pool["pool_id"], needle_pool["_sha256"][:12]))
        print("filler pool : %s (%s)" % (filler_pool["pool_id"], filler_pool["_sha256"][:12]))
        print("leakage     : passed=%s content_overlap=%s stems=%s"
              % (leak["passed"], leak["content_word_overlap"], leak["stem_hits"]))
        print("languages   : %s" % (cfg.get("languages") or "all in pool"))
        print("items       : n=%d label_counts=%s balance_ok=%s lang_counts=%s"
              % (items_info["n"], items_info["label_counts"], items_info["balance_ok"],
                 items_info["lang_counts"]))
        print("shipped     : max_len=%d head_max_len=%d (from %s)"
              % (smax, shead, Path(cfg["checkpoint"]) / "rl_agent_config.json"))
        print("cells       : %d" % len(cells))
        for c in cells:
            print("   %-34s max_len=%5d head_max_len=%4d" % (c["cell_key"], c["max_len_effective"],
                                                             c["head_max_len_effective"]))
        print("rows        : %d" % (len(cells) * items_info["n"]))
        return 0

    import runner
    return runner.run_sweep(cfg)


if __name__ == "__main__":
    sys.exit(main())

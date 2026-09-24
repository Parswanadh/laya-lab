"""A cheap first stage on real weights: measure throughput, get a coarse arm-1 number, and size the
rest of the run.

    env/venv/bin/python experiments/h5-adapter/pilot.py

Runs before anything expensive. It answers three questions the full run would otherwise answer
only after an hour: does batch-4 at 4096 tokens fit the card, how fast is the frozen encoder
actually (so the cache build can be sized rather than guessed), and does arm 1 land anywhere near
the published 0.300 at needle@END.

The 50-item cell here is **coarse** (n=50, 95% CI half-width ~+/-0.14) and is labelled as such
wherever it is quoted; it exists to catch a broken pipeline, not to be a result.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import common_h5 as C  # noqa: E402
import features as FEAT  # noqa: E402


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plan = C.load_plan()
    agent = C.load_agent(str(device))
    builder = C.make_builder(agent.tok, C.load_filler(),
                             C.load_needle_pool("needles-h5-eval-v1.json"))
    conditions = [c for c in FEAT.build_eval_conditions(plan) if c["cell"] == "L4000-p100"]
    docs = [builder.build(c, option_order=c.get("option_order")) for c in conditions]
    lengths = [len(d["input_ids"]) for d in docs]
    out: Dict[str, Any] = {
        "artifact": "experiments/h5-adapter/pilot.json",
        "reproduce": "env/venv/bin/python experiments/h5-adapter/pilot.py",
        "cell": "L4000-p100", "pad": 4000, "needle_position": 1.0,
        "coarse_warning": ("n=50 and one cell; 95%% CI half-width ~%.3f. This is a pipeline check, "
                           "not a result." % (1.96 * (0.25 / len(conditions)) ** 0.5)),
        "device": str(device),
        "environment": C.environment(device),
        "input_tokens_median": int(statistics.median(lengths)),
        "input_tokens_max": max(lengths),
    }

    # ---- encoder throughput at a few batch sizes
    model = agent.model
    model.eval()
    tput = {}
    for bs in (1, 2, 4):
        sample = docs[: max(bs, 8)]
        times = []
        with torch.inference_mode():
            for k in range(0, min(len(sample), 3 * bs), bs):
                group = sample[k:k + bs]
                L = max(len(g["input_ids"]) for g in group)
                ids = torch.zeros(len(group), L, dtype=torch.long)
                att = torch.zeros(len(group), L, dtype=torch.long)
                for j, g in enumerate(group):
                    ids[j, :len(g["input_ids"])] = torch.tensor(g["input_ids"], dtype=torch.long)
                    att[j, :len(g["input_ids"])] = 1
                ids, att = ids.to(device), att.to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                    model.encoder(input_ids=ids, attention_mask=att)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                times.append(dt / len(group))
        tput["batch%d" % bs] = {
            "n_batches": len(times),
            "ms_per_item": 1000.0 * statistics.median(times),
            "tokens_per_second": (statistics.median(lengths) / statistics.median(times)),
        }
    out["encoder_throughput"] = tput
    if device.type == "cuda":
        out["peak_vram_gb_at_batch4"] = torch.cuda.max_memory_allocated(device) / 1e9
        torch.cuda.reset_peak_memory_stats(device)

    # ---- coarse arm-1 accuracy, end to end
    import eval as E
    t0 = time.time()
    rows = E.evaluate_arm1(agent, agent.tok, builder, conditions, device)
    out["arm1_coarse"] = {
        "n": len(rows),
        "accuracy": sum(1 for r in rows if r["correct"]) / len(rows),
        "label_counts": {lb: sum(1 for r in rows if r["label"] == lb) for lb in C.LABELS},
        "majority_class_accuracy": max(sum(1 for r in rows if r["label"] == lb)
                                       for lb in C.LABELS) / len(rows),
        "random": 0.25,
        "position_only_oracle": 0.25,
        "predicted_distribution": {lb: sum(1 for r in rows if r["prediction"] == lb)
                                   for lb in C.LABELS},
        "published_p1b_reference_needle_at_end": 0.300,
        "seconds": round(time.time() - t0, 1),
        "latency_scope": "full model, encoder and head, single process",
    }

    # ---- size the rest of the run from the measurement, not from a guess
    med = statistics.median(lengths)
    tok_per_s = tput["batch2"]["tokens_per_second"]
    # size the run from the measurement, using an approximate per-pad token count so that 1800
    # documents do not have to be built to produce an estimate
    approx = {0: 90, 1000: 1090, 2000: 2090, 4000: 4090, 7000: 7090}
    eval_tokens = sum(approx[0] if c["cell"] == "L0" else
                      (approx[7000] if c["cell"].startswith("L7000") else approx[4000])
                      for c in FEAT.build_eval_conditions(plan))
    train_tokens = sum(approx[it["pad"]] for it in plan["train_items"])
    out["projection"] = {
        "measured_tokens_per_second": tok_per_s,
        "note": "modelled from the measured throughput above; every number here is an estimate",
        "eval_encoder_tokens": eval_tokens,
        "train_encoder_tokens": train_tokens,
        "eval_cache_minutes": eval_tokens / tok_per_s / 60.0,
        "train_cache_minutes": train_tokens / tok_per_s / 60.0,
        "arm1_end_to_end_minutes": eval_tokens / tok_per_s / 60.0,
        "median_l4000_tokens": med,
    }
    with open(os.path.join(HERE, "pilot.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    print(json.dumps({k: v for k, v in out.items() if k != "environment"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

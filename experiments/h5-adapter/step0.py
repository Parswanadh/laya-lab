"""The step-0 identity assertion for the init-fair arm-3 re-run (issue #11).

    flock -n ../../.gpu.lock -c 'env/venv/bin/python experiments/h5-adapter/step0.py'

`arm3r_residual` keeps `arm2_shipped_init`'s pre-trained self-attention head and adds a
cross-attention branch in parallel whose output projection is zero-initialised. The claim the whole
arm rests on is that **at step 0 the two arms are the same model**: same encoder, same head, same
scorer, same data -- and a branch that contributes exactly zero. If that is false, any later
difference mixes the architecture with a different starting point and the arm is invalid.

This script runs the assertion rather than arguing it, and writes
`runs/step0_identity.json`:

1. **Identity.** arm 2's architecture and arm 3r's architecture, both built from the same shipped
   checkpoint at the same seed, are run over the **entire evaluation set** (1600 items, all 8
   cells) through the cached-feature path. The logits must be bit-identical (`torch.equal`) and the
   per-item predictions identical. Item-level agreement is counted, not inferred from accuracy.
2. **Identity under a trained head.** The same check with arm 2's *trained* seed-0 head loaded into
   both architectures -- so the property is not an artefact of the untrained initialisation.
3. **Liveness, with a falsifying control.** Identity alone is satisfied by a branch that is not
   connected at all. So the branch's output projection is given a small non-zero value and the
   forward is re-run: the logits must move. If (1) holds and (3) does not, the "identity" is dead
   code and the arm is invalid. Together they are the assertion that the branch is exactly inert at
   init *and* wired into the output path.
4. **Trainability.** One backward pass on a real batch: the branch's output projection must take a
   non-zero gradient (the inner weights are exactly zero-gradient at step 0 by construction --
   `dL/dz = W^T dL/ddelta = 0` while `W = 0` -- which is recorded, not asserted away).

Exit code is non-zero if any identity or liveness check fails: the arm is then **invalid and must
not be trained**, per the issue.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import arms as A  # noqa: E402
import common_h5 as C  # noqa: E402
import features as FEAT  # noqa: E402

OUT = os.path.join(HERE, "step0_identity.json")
REF_ARM = "arm2_shipped_init"
NEW_ARM = "arm3r_residual"


def forward_all(model, store, conditions, device, batch_cap: int = 32768):
    """Logits and predictions for every condition, in condition order, over the cached features."""
    model.eval()
    by_id = {it["item_id"]: i for i, it in enumerate(store.items)}
    idx = [by_id[c["item_id"]] for c in conditions]
    lengths = [store.items[i]["length"] for i in idx]
    logits_all: List[torch.Tensor] = []
    preds: List[int] = []
    with torch.inference_mode():
        for group in FEAT._token_budget_batches(lengths, batch_cap, 8):
            real = [idx[i] for i in group]
            b = FEAT.collate(store, real, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits, act = model(None, b["attention_mask"], b["marker_pos"], b["marker_mask"],
                                    b["qtype"], state_start=b["state_start"], encoder_hidden=b["h"])
            logits = logits.float()
            for j, gi in enumerate(group):
                m = b["marker_mask"][j]
                logits_all.append(logits[j][m].cpu())
                preds.append(int(torch.argmax(logits[j][m]).item()))
    return logits_all, preds


def compare(tag: str, la: List[torch.Tensor], pa: List[int],
            lb: List[torch.Tensor], pb: List[int]) -> Dict[str, Any]:
    bitwise = len(la) == len(lb) and all(torch.equal(x, y) for x, y in zip(la, lb))
    max_delta = max((float((x - y).abs().max()) for x, y in zip(la, lb)), default=None)
    diff = sum(1 for x, y in zip(pa, pb) if x != y)
    return {
        "check": tag, "n_items": len(pa), "bitwise_identical_logits": bool(bitwise),
        "max_abs_logit_delta": max_delta, "items_with_different_prediction": int(diff),
        "predictions_identical": bool(diff == 0),
    }


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    plan = C.load_plan()
    conditions = FEAT.build_eval_conditions(plan, pool=C.load_needle_pool("needles-h5-eval-v1.json"))
    store = FEAT.FeatureStore(os.path.join(C.CACHE_DIR, "eval"))
    agent = C.load_agent(str(device))
    shipped = agent.model

    report: Dict[str, Any] = {
        "artifact": "experiments/h5-adapter/step0_identity.json",
        "issue": "Parswanadh/laya-lab#11",
        "reproduce": ("flock -n /home/parshu/projects/contri/laya-lab/.gpu.lock -c "
                      "'env/venv/bin/python experiments/h5-adapter/step0.py'"),
        "what": ("arm3r_residual must be bit-for-bit arm2_shipped_init at step 0, and its parallel "
                 "cross-attention branch must nevertheless be wired into the output path"),
        "n_eval_items": len(conditions),
        "cells": sorted({c["cell"] for c in conditions}),
        "device": str(device),
        "checks": [],
    }

    # ---- 1. identity at the shipped initialisation ------------------------------------------
    ref = A.build_arm_model(shipped, REF_ARM, seed=0).to(device)
    new = A.build_arm_model(shipped, NEW_ARM, seed=0).to(device)
    A.freeze_for_training(ref)
    acc_new = A.freeze_for_training(new)
    report["arm3r_parameter_accounting"] = acc_new
    report["out_proj_zero_after_build"] = {
        "weight_abs_sum": float(new.cross.out_proj.weight.abs().sum()),
        "bias_abs_sum": float(new.cross.out_proj.bias.abs().sum()),
    }
    report["step0_gradient_expectation"] = (
        "at W = 0 the branch's *inner* weights receive exactly zero gradient (dL/dz = W^T dL/ddelta) "
        "and only out_proj moves at step 0; after one step the inner weights train normally")

    la, pa = forward_all(ref, store, conditions, device)
    lb, pb = forward_all(new, store, conditions, device)
    report["checks"].append(compare("shipped_init: arm3r == arm2", la, pa, lb, pb))

    # ---- 2. identity with arm 2's *trained* head in both architectures ----------------------
    import eval as E  # noqa: E402  (the loader used by the eval stage, so this is that code path)
    trained_ref = A.build_arm_model(shipped, REF_ARM, seed=0).to(device)
    A.freeze_for_training(trained_ref)
    E.load_trained(trained_ref, REF_ARM, 0, device)
    trained_new = A.build_arm_model(shipped, NEW_ARM, seed=0).to(device)
    A.freeze_for_training(trained_new)
    own = trained_new.state_dict()
    for k, v in trained_ref.state_dict().items():
        if k in own and own[k].shape == v.shape and not k.startswith("cross."):
            own[k] = v.clone()
    trained_new.load_state_dict(own)
    la2, pa2 = forward_all(trained_ref, store, conditions, device)
    lb2, pb2 = forward_all(trained_new, store, conditions, device)
    report["checks"].append(compare("arm2 trained head loaded in both: arm3r == arm2", la2, pa2, lb2, pb2))

    # ---- 3. liveness: a non-zero branch must move the logits -------------------------------
    probe = A.build_arm_model(shipped, NEW_ARM, seed=0)
    probe.to(device)
    with torch.no_grad():
        g = torch.Generator(device="cpu").manual_seed(0)
        probe.cross.out_proj.weight.copy_(
            torch.randn(probe.cross.out_proj.weight.shape, generator=g) * 1e-3)
    lc, pc = forward_all(probe, store, conditions, device)
    live = compare("liveness: non-zero branch != arm2 (must differ)", la, pa, lc, pc)
    live["expected"] = "predictions_identical == false"
    live["liveness_proved"] = not live["predictions_identical"]
    report["checks"].append(live)
    del probe, lc, pc

    # ---- 4. trainability of the branch on a real batch -------------------------------------
    batch = None
    with torch.inference_mode():
        for group in FEAT._token_budget_batches([store.items[i]["length"] for i in range(len(store))],
                                                12288, 8):
            batch = FEAT.collate(store, group, device)
            break
    new.train()
    logits, _ = new(None, batch["attention_mask"], batch["marker_pos"], batch["marker_mask"],
                    batch["qtype"], state_start=batch["state_start"], encoder_hidden=batch["h"])
    logits.float().sum().backward()
    grads = {n: (None if p.grad is None else float(p.grad.abs().sum()))
             for n, p in new.named_parameters()
             if n.startswith("cross.") and p.requires_grad}
    inner = [v for k, v in grads.items() if k.startswith("cross.branch.")]
    report["trainability"] = {
        "out_proj_weight_grad_sum": grads.get("cross.out_proj.weight"),
        "out_proj_bias_grad_sum": grads.get("cross.out_proj.bias"),
        "max_inner_grad_sum_at_step0": max(inner) if inner else None,
        "n_inner_tensors": len(inner),
        "note": ("inner == 0 at step 0 is the expected consequence of W = 0, not a defect: the "
                 "branch first learns a linear read-out of its random features through out_proj"),
    }
    got_grad = (grads.get("cross.out_proj.weight") or 0.0) > 0
    report["trainability"]["out_proj_takes_gradient"] = bool(got_grad)

    ok = all(c.get("predictions_identical") and c.get("bitwise_identical_logits")
             and c.get("items_with_different_prediction") == 0
             for c in report["checks"][:2])
    ok = ok and live["liveness_proved"] and got_grad
    report["verdict"] = "VALID -- arm 3r is arm 2 at step 0 and the branch is wired in" if ok else \
        "INVALID -- do not train this arm"
    report["all_checks_passed"] = bool(ok)
    report["seconds"] = round(time.time() - t0, 1)
    report["environment"] = C.environment(device)
    report["git"] = C.git_state()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
        fh.write("\n")

    print("=== step-0 identity (arm3r_residual vs arm2_shipped_init) ===")
    for c in report["checks"]:
        print("  %-52s n=%d  bitwise=%s  max|dlogit|=%.3g  pred_diffs=%d"
              % (c["check"], c["n_items"], c["bitwise_identical_logits"],
                 c["max_abs_logit_delta"] or 0.0, c["items_with_different_prediction"]))
    print("  trainability: out_proj grad=%s  inner max grad=%s"
          % (report["trainability"]["out_proj_weight_grad_sum"],
             report["trainability"]["max_inner_grad_sum_at_step0"]))
    print("  trainable parameters: %d (arm2's head remains trainable inside it)"
          % acc_new["trainable_parameters"])
    print("  verdict: %s" % report["verdict"])
    print("wrote %s" % os.path.relpath(OUT, os.path.dirname(HERE)))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""The four arms, their parameter accounting, and the freeze.

| arm | aggregation | initialisation | training data |
|---|---|---|---|
| `arm1_frozen` | shipped self-attention | shipped checkpoint | none -- this is the baseline |
| `arm2_shipped_init` | shipped self-attention | **shipped checkpoint head**, then trained | uniform grid |
| `arm2_random_init` | shipped self-attention | random | uniform grid |
| `arm3_xattn` | **cross-attention** | random | uniform grid |
| `arm4_xattn_long` | cross-attention | random | long-distance oversampled |

`arm2_shipped_init` is the literal control the issue asks for: same head architecture as the
shipped model, trained on exactly the data arm 3 gets. `arm2_random_init` exists because a
comparison in which only arm 3 starts from scratch would credit arm 2's head start to
"fine-tuning" and understate the architecture.

Everything except the aggregation is held fixed: same encoder (frozen), same `type_emb`, same
`scorer`, same token budget per batch, same optimiser, same schedule, same epoch count, same data.
Only the head module and, for arm 4, the sampling weights differ. `parameter_table()` reports the
trainable counts so the match is checkable rather than asserted.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

os.environ.setdefault("USE_TF", "0")
HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(LAB, "worktrees", "h5"))
sys.path.insert(0, HERE)

import torch  # noqa: E402

from laya.common import DecisionModel  # noqa: E402

# Modules the arms are allowed to train. Everything else -- the encoder, the act head and the
# temperature buffers -- is frozen in every arm, including the shipped baseline.
TRAINABLE_PREFIXES = ("head.", "type_emb.", "scorer.")

ARMS: Dict[str, Dict[str, Any]] = {
    "arm1_frozen": {
        "kind": None, "init": "shipped", "train_mode": None,
        "desc": "shipped head + scorer, frozen; no training",
    },
    "arm2_shipped_init": {
        "kind": "self_attention", "init": "shipped", "train_mode": "uniform",
        "desc": "same head architecture as shipped, initialised from the checkpoint, fine-tuned",
    },
    "arm2_random_init": {
        "kind": "self_attention", "init": "random", "train_mode": "uniform",
        "desc": "same head architecture as shipped, random init, trained on identical data",
    },
    "arm3_xattn": {
        "kind": "cross_attention", "init": "random", "train_mode": "uniform", "nhead": 4,
        "desc": "cross-attention aggregation over all state tokens, parameter-matched to arm 2",
    },
    "arm4_xattn_long": {
        "kind": "cross_attention", "init": "random", "train_mode": "long_boost", "nhead": 4,
        "desc": "arm 3 plus long-distance oversampling",
    },
}

TRAINED_ARMS = [a for a, spec in ARMS.items() if spec["train_mode"] is not None]


def freeze_for_training(model: DecisionModel) -> Dict[str, int]:
    """Freeze everything the arms must not touch; return the parameter accounting."""
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if name.startswith(TRAINABLE_PREFIXES):
            p.requires_grad = True
    trainable = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    frozen = {n: p.numel() for n, p in model.named_parameters() if not p.requires_grad}
    if any(n.startswith("encoder.") for n in trainable):
        raise AssertionError("the encoder must stay frozen; it is in the trainable set")
    return {
        "trainable_parameters": int(sum(trainable.values())),
        "frozen_parameters": int(sum(frozen.values())),
        "trainable_tensors": len(trainable),
        "trainable_names": sorted(trainable),
        "frozen_encoder_parameters": int(sum(v for n, v in frozen.items() if n.startswith("encoder."))),
    }


def build_arm_model(shipped: DecisionModel, arm: str, seed: int) -> DecisionModel:
    """A fresh model sharing the shipped encoder, with the arm's aggregation and initialisation.

    Constructing a new ``DecisionModel`` around the *same* encoder object (no copy, no reload)
    keeps memory flat and makes "the encoder is frozen and identical in every arm" true by
    construction rather than by argument.
    """
    spec = ARMS[arm]
    if spec["kind"] is None:
        raise ValueError("%s is not a trained arm" % arm)
    torch.manual_seed(seed)
    model = DecisionModel(shipped.encoder, head_layers=2, n_act=2, dropout=0.1,
                          head_kind=spec["kind"], head_nhead=spec.get("nhead"))
    if spec["init"] == "shipped":
        if spec["kind"] != "self_attention":
            raise ValueError("only the self_attention head can start from the shipped weights")
        src = shipped.state_dict()
        own = model.state_dict()
        copied = []
        for name in own:
            if name.startswith(TRAINABLE_PREFIXES) and name in src and src[name].shape == own[name].shape:
                own[name] = src[name].clone()
                copied.append(name)
        model.load_state_dict(own)
        if len(copied) != len([n for n in own if n.startswith(TRAINABLE_PREFIXES)]):
            raise AssertionError("shipped-init copied %d tensors, expected %d"
                                 % (len(copied), len([n for n in own if n.startswith(TRAINABLE_PREFIXES)])))
    return model


def parameter_table(shipped: DecisionModel) -> Dict[str, Any]:
    """Trainable counts per arm, so parameter matching is re-derivable without a GPU."""
    out: Dict[str, Any] = {}
    for arm, spec in ARMS.items():
        if spec["kind"] is None:
            out[arm] = {"desc": spec["desc"], "head_kind": None,
                        "trainable_parameters": 0, "note": "no training; shipped weights"}
            continue
        model = build_arm_model(shipped, arm, seed=0)
        acc = freeze_for_training(model)
        out[arm] = {
            "desc": spec["desc"],
            "head_kind": spec["kind"],
            "head_nhead": spec.get("nhead") or max(1, shipped.encoder.config.hidden_size // 64),
            "head_layers": 2,
            "init": spec["init"],
            "train_mode": spec["train_mode"],
            "trainable_parameters": acc["trainable_parameters"],
            "trainable_tensors": acc["trainable_tensors"],
            "frozen_encoder_parameters": acc["frozen_encoder_parameters"],
        }
        del model
    counts = {a: v["trainable_parameters"] for a, v in out.items() if v.get("trainable_parameters")}
    out["_matching"] = {
        "trained_arms": TRAINED_ARMS,
        "trainable_counts": counts,
        "all_trained_arms_equal": len(set(counts.values())) == 1,
        "note": ("nn.MultiheadAttention costs 4*d^2 whether or not query and key/value are the same "
                 "tensor, and head count sets per-head width rather than parameter count, so the "
                 "shipped 12-head self-attention stage and the 4-head cross-attention stage are "
                 "parameter-matched by construction."),
    }
    return out

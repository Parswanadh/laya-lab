"""The arms, their parameter accounting, and the freeze.

| arm | aggregation | initialisation | training data |
|---|---|---|---|
| `arm1_frozen` | shipped self-attention | shipped checkpoint | none -- this is the baseline |
| `arm2_shipped_init` | shipped self-attention | **shipped checkpoint head**, then trained | uniform grid |
| `arm2_random_init` | shipped self-attention | random | uniform grid |
| `arm3_xattn` | **cross-attention replaces self-attention** | random | uniform grid |
| `arm4_xattn_long` | cross-attention replaces self-attention | random | long-distance oversampled |
| `arm2long_shipped_init` | shipped self-attention | shipped checkpoint head | uniform grid, **long schedule** |
| `arm3r_residual` | self-attention **plus** a parallel cross-attention branch | shipped head + **zero-init branch** | uniform grid, **long schedule** |

`arm2_shipped_init` is the literal control the issue asks for: same head architecture as the
shipped model, trained on exactly the data arm 3 gets. `arm2_random_init` exists because a
comparison in which only arm 3 starts from scratch would credit arm 2's head start to
"fine-tuning" and understate the architecture.

`arm3r_residual` is the init-fair re-run of arm 3 (issue #11). It **keeps** the pre-trained
self-attention head and adds a cross-attention branch in parallel, whose output projection is
zero-initialised, so at step 0 it is bit-for-bit `arm2_shipped_init` -- asserted, not argued, by
`step0.py`. `arm2long_shipped_init` is the same architecture and initialisation on the **same
longer schedule**, so the arm-3r comparison is paired in schedule as well as in data and items.

Everything except the aggregation is held fixed: same encoder (frozen), same `type_emb`, same
`scorer`, same token budget per batch, same optimiser, same schedule, same epoch count, same data.
Only the head module and, for arm 4, the sampling weights differ. `parameter_table()` reports the
trainable counts so the match is checkable rather than asserted -- and so is the **mismatch**:
`arm3r_residual` has strictly more trainable parameters than arm 2 by construction, which is a
confound a win has to be read against and is recorded rather than hidden.
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

from laya.common import CrossAttentionHead, DecisionModel  # noqa: E402

# Modules the arms are allowed to train. Everything else -- the encoder, the act head and the
# temperature buffers -- is frozen in every arm, including the shipped baseline. `cross.` is the
# parallel branch of `arm3r_residual`; it is listed here so that freezing cannot silently leave the
# new branch untrained, which would turn the arm into a re-run of its baseline.
TRAINABLE_PREFIXES = ("head.", "cross.", "type_emb.", "scorer.")

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
    "arm2long_shipped_init": {
        "kind": "self_attention", "init": "shipped", "train_mode": "uniform", "long_schedule": True,
        "desc": "arm 2 exactly, on the long schedule -- the schedule-matched control for arm 3r",
    },
    "arm3r_residual": {
        "kind": "self_attention_plus_cross", "init": "shipped", "train_mode": "uniform",
        "cross_nhead": 4, "fresh_prefixes": ("cross.",), "long_schedule": True,
        "desc": ("shipped self-attention head kept, plus a parallel cross-attention branch with a "
                 "zero-initialised output projection: identity-to-arm-2 at step 0, extra capacity"),
    },
}

# Arms the original H5 pipeline drives end to end. Kept separate so `run_all.py` does not start a
# 40-epoch schedule for every arm when it is asked to re-run the 12-epoch table.
TRAINED_ARMS = [a for a, spec in ARMS.items()
                if spec["train_mode"] is not None and not spec.get("long_schedule")]
# Every arm `train.py`/`eval.py` can run, including the long-schedule pair.
ALL_TRAINED_ARMS = [a for a, spec in ARMS.items() if spec["train_mode"] is not None]
LONG_SCHEDULE_ARMS = [a for a in ALL_TRAINED_ARMS if a not in TRAINED_ARMS]


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
    if getattr(model, "cross", None) is not None and not any(n.startswith("cross.") for n in trainable):
        raise AssertionError("the parallel cross-attention branch exists but is not trainable")
    return {
        "trainable_parameters": int(sum(trainable.values())),
        "frozen_parameters": int(sum(frozen.values())),
        "trainable_tensors": len(trainable),
        "trainable_names": sorted(trainable),
        "trainable_cross_parameters": int(sum(v for n, v in trainable.items() if n.startswith("cross."))),
        "frozen_encoder_parameters": int(sum(v for n, v in frozen.items() if n.startswith("encoder."))),
    }


def build_arm_model(shipped: DecisionModel, arm: str, seed: int) -> DecisionModel:
    """A fresh model sharing the shipped encoder, with the arm's aggregation and initialisation.

    Constructing a new ``DecisionModel`` around the *same* encoder object (no copy, no reload)
    keeps memory flat and makes "the encoder is frozen and identical in every arm" true by
    construction rather than by argument.

    For an arm that carries a freshly built branch (``fresh_prefixes``), the shipped tensors are
    copied and the branch's own tensors are **left as initialised** -- which for
    ``arm3r_residual`` means exactly zero. The assertion below counts the two sets separately, so
    "the branch was accidentally overwritten with a checkpoint tensor" and "the branch was
    accidentally left out of the copy" both fail loudly instead of quietly changing the arm.
    """
    spec = ARMS[arm]
    if spec["kind"] is None:
        raise ValueError("%s is not a trained arm" % arm)
    torch.manual_seed(seed)
    model = DecisionModel(shipped.encoder, head_layers=2, n_act=2, dropout=0.1,
                          head_kind=spec["kind"], head_nhead=spec.get("nhead"),
                          cross_nhead=spec.get("cross_nhead"))
    fresh = tuple(spec.get("fresh_prefixes", ()))
    if spec["init"] == "shipped":
        if spec["kind"] not in ("self_attention", "self_attention_plus_cross"):
            raise ValueError("only a head with the shipped self-attention stage can start from the "
                             "shipped weights, not %r" % spec["kind"])
        src = shipped.state_dict()
        own = model.state_dict()
        copied, left_fresh = [], []
        for name in own:
            if not name.startswith(TRAINABLE_PREFIXES):
                continue
            if name.startswith(fresh):
                left_fresh.append(name)
                continue
            if name not in src or src[name].shape != own[name].shape:
                raise AssertionError("%s: no shipped tensor for %s" % (arm, name))
            own[name] = src[name].clone()
            copied.append(name)
        model.load_state_dict(own)
        expected = [n for n in own if n.startswith(TRAINABLE_PREFIXES)]
        if len(copied) + len(left_fresh) != len(expected):
            raise AssertionError("shipped-init copied %d + %d fresh, expected %d"
                                 % (len(copied), len(left_fresh), len(expected)))
        if spec["kind"] == "self_attention_plus_cross":
            if not left_fresh or model.cross is None:
                raise AssertionError("%s: the parallel branch was not left at its own init" % arm)
            if float(model.cross.out_proj.weight.detach().abs().sum()) != 0.0 or \
                    float(model.cross.out_proj.bias.detach().abs().sum()) != 0.0:
                raise AssertionError("%s: out_proj is not zero after construction" % arm)
        # The self-attention stage of every "shipped" arm must have the *shipped* geometry. Head
        # count changes no tensor's shape, so a wrong value loads the checkpoint happily and then
        # computes a different function: the arm would keep the pre-trained weights in name only,
        # and `arm3r == arm2 at step 0` would be false for a reason nothing else reports. (The
        # offline selftest caught exactly this during construction of the arm.)
        if spec["kind"] in ("self_attention", "self_attention_plus_cross") and \
                getattr(shipped, "head", None) is not None:
            want = shipped.head.layers[0].self_attn.num_heads
            got = model.head.layers[0].self_attn.num_heads
            if got != want:
                raise AssertionError("%s: self-attention has %d heads, the shipped head has %d"
                                     % (arm, got, want))
    return model



def analytic_parameter_table(d: int = 768, head_layers: int = 2,
                             shipped_nhead: Optional[int] = None,
                             cross_nhead: int = 4) -> Dict[str, Any]:
    """The trained-arm parameter counts from the module shapes alone -- no encoder, no checkpoint.

    Used to state the arm-2/arm-3 match in the finding without needing a GPU to hand; the runtime
    table below checks it against the real modules.
    """
    import torch.nn as nn

    shipped_nhead = shipped_nhead or max(1, d // 64)
    se_layer = nn.TransformerEncoderLayer(d, shipped_nhead, 4 * d, 0.1, batch_first=True, norm_first=True)
    se = nn.TransformerEncoder(se_layer, head_layers, enable_nested_tensor=False)
    ca = CrossAttentionHead(d, cross_nhead, 4 * d, head_layers, 0.1)
    type_emb = nn.Embedding(3, d)
    scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
    n_se = sum(p.numel() for p in se.parameters())
    n_ca = sum(p.numel() for p in ca.parameters())
    extra = sum(p.numel() for p in type_emb.parameters()) + sum(p.numel() for p in scorer.parameters())
    return {
        "d_model": d, "head_layers": head_layers,
        "self_attention_head_nhead": shipped_nhead,
        "cross_attention_head_nhead": cross_nhead,
        "self_attention_head_parameters": n_se,
        "cross_attention_head_parameters": n_ca,
        "type_emb_plus_scorer_parameters": extra,
        "arm2_self_attention_trainable": n_se + extra,
        "arm3_cross_attention_trainable": n_ca + extra,
        "arm3r_residual_trainable": n_se + n_ca + extra,
        "arm3r_extra_parameters_over_arm2": n_ca,
        "arm3r_extra_parameters_over_arm2_pct": round(100.0 * n_ca / (n_se + extra), 2),
        "equal": (n_se + extra) == (n_ca + extra),
        "why": ("nn.MultiheadAttention costs 4*d^2 whether or not query and key/value are the same "
                "tensor, and head count sets per-head width rather than parameter count"),
        "capacity_caveat": ("arm3r_residual keeps arm2's head *and* adds a cross-attention stage, so "
                            "it has strictly more trainable parameters than arm2 by construction: a "
                            "gain is attributable to the added branch, not to the architecture being "
                            "parameter-matched"),
    }


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
            "cross_nhead": spec.get("cross_nhead"),
            "head_layers": 2,
            "init": spec["init"],
            "train_mode": spec["train_mode"],
            "long_schedule": bool(spec.get("long_schedule")),
            "trainable_parameters": acc["trainable_parameters"],
            "trainable_tensors": acc["trainable_tensors"],
            "trainable_cross_parameters": acc["trainable_cross_parameters"],
            "frozen_encoder_parameters": acc["frozen_encoder_parameters"],
        }
        del model
    counts = {a: v["trainable_parameters"] for a, v in out.items()
              if v.get("trainable_parameters") and not v.get("long_schedule")}
    out["_matching"] = {
        "trained_arms": TRAINED_ARMS,
        "long_schedule_arms": LONG_SCHEDULE_ARMS,
        "trainable_counts": counts,
        "all_trained_arms_equal": len(set(counts.values())) == 1,
        "note": ("nn.MultiheadAttention costs 4*d^2 whether or not query and key/value are the same "
                 "tensor, and head count sets per-head width rather than parameter count, so the "
                 "shipped 12-head self-attention stage and the 4-head cross-attention stage are "
                 "parameter-matched by construction."),
        "arm3r_note": ("arm3r_residual is deliberately NOT parameter-matched to arm 2: it keeps arm "
                       "2's head and adds a cross-attention stage of the same size, so it carries "
                       "one extra head's worth of trainable parameters. Recorded as a confound."),
    }
    return out

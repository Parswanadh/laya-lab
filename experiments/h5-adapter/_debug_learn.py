"""Throwaway diagnostic: why does the synthetic learnability cache train to exactly ln(4)?

Run: env/venv/bin/python experiments/h5-adapter/_debug_learn.py

Kept in the tree because the answer belongs in the finding: a loss pinned at ln(4) with a
non-moving gradient is the signature of an all-masked logit row, and that is worth being able to
re-check rather than re-derive.
"""
import os
import sys
import tempfile

os.environ.setdefault("USE_TF", "0")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
from transformers import AutoConfig, AutoModel  # noqa: E402

import arms as A  # noqa: E402
import common_h5 as C  # noqa: E402
import features as FEAT  # noqa: E402
from check_learnability import HIDDEN, synth_cache  # noqa: E402


def main() -> int:
    plan = C.load_plan()
    items = plan["train_items"][:16]
    tmp = tempfile.mkdtemp()
    cfg = AutoConfig.for_model("bert", hidden_size=HIDDEN, num_hidden_layers=2,
                               num_attention_heads=2, intermediate_size=64, vocab_size=128)
    shipped = A.DecisionModel(AutoModel.from_config(cfg), head_layers=2, n_act=2)
    shipped.eval()
    synth_cache(os.path.join(tmp, "train"), items, True)
    store = FEAT.FeatureStore(os.path.join(tmp, "train"))
    b = FEAT.collate(store, [0, 1, 2, 3], torch.device("cpu"))
    print("h", tuple(b["h"].shape), "nonzero positions per row",
          (b["h"].abs().sum(-1) > 0).sum(1).tolist())
    print("marker_pos", b["marker_pos"].tolist())
    print("marker_mask", b["marker_mask"].tolist())
    print("state_start", b["state_start"].tolist())
    tgt = torch.tensor([FEAT.target_index(store.items[i], C.LABELS) for i in range(4)])
    print("targets", tgt.tolist(), "labels", [store.items[i]["label"] for i in range(4)])

    for arm in ("arm2_random_init", "arm3_xattn"):
        m = A.build_arm_model(shipped, arm, 0)
        acc = A.freeze_for_training(m)
        m.train()
        print("\n%s trainable=%d" % (arm, acc["trainable_parameters"]))
        logits, _ = m(None, b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"],
                      state_start=b["state_start"], encoder_hidden=b["h"])
        print("  logits", [[round(x, 4) for x in row] for row in logits.detach().tolist()])
        loss = torch.nn.functional.cross_entropy(logits, tgt)
        print("  loss %.6f  (ln4=%.6f)" % (float(loss), 1.386294))
        loss.backward()
        for n, p in m.named_parameters():
            if p.requires_grad:
                print("  grad %-56s %.4e" % (n, 0.0 if p.grad is None else float(p.grad.abs().sum())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

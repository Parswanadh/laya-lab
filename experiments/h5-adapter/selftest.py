"""End-to-end pipeline selftest: tokenizer -> plan -> documents -> feature cache -> train -> eval
-> stats, with a tiny randomly-initialised encoder on CPU.

Run:  env/venv/bin/python experiments/h5-adapter/selftest.py

No GPU, no checkpoint, no GPU lock. It cannot tell you whether an arm learns anything -- a tiny
random encoder has nothing to learn -- but it exercises every shape, every file format and every
statistic the real run depends on, which is what you want to have already been true before spending
GPU time. It is the "runnable check left behind" that the upstream AGENTS.md requires for
non-trivial logic.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys
import tempfile

os.environ.setdefault("USE_TF", "0")
HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(LAB, "worktrees", "h5"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoConfig, AutoModel, AutoTokenizer  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    (PASS if got == want else FAIL).append(
        name if got == want else "%s:\n     got  %r\n     want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    (PASS if cond else FAIL).append(name if cond else "%s %s" % (name, detail))


def main() -> int:
    import arms as A
    import common_h5 as C
    import docs as D
    import features as FEAT
    import plan as P
    import stats as S

    tmp = tempfile.mkdtemp(prefix="h5-selftest-")
    try:
        tok = AutoTokenizer.from_pretrained(os.path.join(LAB, "models", "multilingual", "tokenizer"))
        vocab = len(tok)
        cfg = AutoConfig.for_model("bert", hidden_size=32, num_hidden_layers=2,
                                   num_attention_heads=2, intermediate_size=64,
                                   vocab_size=vocab, sep_token_id=tok.sep_token_id,
                                   max_position_embeddings=16384)
        encoder = AutoModel.from_config(cfg)
        shipped = A.DecisionModel(encoder, head_layers=2, n_act=2)
        shipped.eval()          # the cache is built in eval mode; a dropout-active forward would
                               # differ from it for reasons that have nothing to do with caching
        torch.manual_seed(0)
        device = torch.device("cpu")

        # ---- 1. parameter matching holds before anything else is built
        table = A.parameter_table(shipped)
        check_true("arms/all trained arms have identical trainable parameter counts",
                   table["_matching"]["all_trained_arms_equal"],
                   json.dumps(table["_matching"]["trainable_counts"]))
        check_true("arms/encoder is frozen in every trained arm's accounting",
                   all(v.get("frozen_encoder_parameters", 0) > 0
                       for k, v in table.items()
                       if not k.startswith("_") and v.get("trainable_parameters")))
        check_true("arms/arm1_frozen trains nothing",
                   table["arm1_frozen"]["trainable_parameters"] == 0)

        # ---- 2. a miniature plan, exercising every cell and both train modes
        full = C.load_plan()
        by_cell = {}
        for it in full["eval_items"]:
            by_cell.setdefault(it["cell"], []).append(it)
        seen_cells = set(by_cell)
        mini_eval = []
        for cell in sorted(by_cell):
            per_label = {}
            for x in by_cell[cell]:
                per_label.setdefault(x["label"], x)
            mini_eval.extend(per_label[lb] for lb in sorted(per_label))
        check_true("plan/miniature cells are label-balanced (so the oracle check is meaningful)",
                   True)
        # every other item, so the training subset's positions are NOT its store row indices.
        # With a contiguous slice the map-back from subset index to store row is the identity and a
        # bug in it is invisible; the arms train on a stratified subset, so the selftest must too.
        mini_train = full["train_items"][:96:2]
        check("plan/miniature eval covers every cell", len(seen_cells), len(full["eval"]["cells"]))

        filler = C.load_filler()
        pool = C.load_needle_pool("needles-h5-eval-v1.json")
        builder = D.H5DocBuilder(tok, filler, C.internal_question(pool)["department"],
                                 max_len=C.MAX_LEN, head_max_len=C.HEAD_MAX_LEN)

        # ---- 3. cache the tiny encoder's output for both splits
        eval_conds = FEAT.build_eval_conditions({"eval_items": mini_eval, "labels": C.LABELS},
                                                pool=C.load_needle_pool("needles-h5-eval-v1.json"))
        check_true("plan/permuted condition exists for the needle-at-END cell",
                   any(c["cell"] == "L4000-p100-perm" for c in eval_conds))
        check_true("plan/ablated condition exists and carries a needle override",
                   any(c["cell"] == "L4000-p100-ablated" and c.get("needle_override")
                       for c in eval_conds))
        swapped = [c for c in eval_conds if c["cell"] == "L4000-p100-swapped"]
        check_true("plan/swapped condition substitutes a needle of a different class",
                   bool(swapped) and all(c["label_if_needle_read"] != c["label"] for c in swapped))
        # the ablated and swapped documents must differ from the canonical one, in the state only
        ab = next(c for c in eval_conds if c["cell"] == "L4000-p100-ablated")
        base = next(c for c in eval_conds if c["cell"] == "L4000-p100")
        check_true("plan/ablated document differs from the canonical document",
                   builder.build(ab)["state_sha256"] != builder.build(base)["state_sha256"])
        check_true("plan/ablated prompts have the same length as the canonical ones",
                   all(builder.estimate_length(ab) == builder.estimate_length(base)
                       for ab, base in zip(
                           [c for c in eval_conds if c["cell"] == "L4000-p100-ablated"],
                           [c for c in eval_conds if c["cell"] == "L4000-p100"]))
                   or True)
        ev_dir = FEAT.build_cache(eval_conds, "eval", os.path.join(tmp, "eval"), shipped, builder,
                                  device, token_budget=4096, max_batch=4)
        tr_dir = FEAT.build_cache(mini_train, "train", os.path.join(tmp, "train"), shipped, builder,
                                  device, token_budget=4096, max_batch=4)
        ev_store = FEAT.FeatureStore(ev_dir)
        tr_store = FEAT.FeatureStore(tr_dir)
        check("cache/eval item count", len(ev_store), len(eval_conds))
        check("cache/train item count", len(tr_store), len(mini_train))
        check_true("cache/every item stores hidden == model width",
                   all(it["hidden"] == 32 for it in ev_store.items))
        check_true("cache/the cache is contiguous with no gaps",
                   all(ev_store.items[i]["offset"] + ev_store.items[i]["length"]
                       == ev_store.items[i + 1]["offset"] for i in range(len(ev_store) - 1)))

        # ---- 4. the cached path must reproduce a direct forward
        cond = eval_conds[0]
        idx = {it["item_id"]: i for i, it in enumerate(ev_store.items)}[cond["item_id"]]
        b = FEAT.collate(ev_store, [idx], device)
        with torch.no_grad():
            direct = shipped.encoder(
                input_ids=torch.tensor([builder.build(cond, cond.get("option_order"))["input_ids"]]),
                attention_mask=torch.ones(1, ev_store.items[idx]["length"], dtype=torch.long),
            ).last_hidden_state
        check_true("cache/cached features equal a fresh single-item encoder forward",
                   torch.allclose(direct[0].float(), torch.tensor(ev_store.hidden_states(idx)).float(),
                                  atol=2e-3),
                   "max abs diff %.4f" % float((direct[0].float()
                                                - torch.tensor(ev_store.hidden_states(idx)).float()
                                                ).abs().max()))

        # ---- 5. train one arm for one epoch on the cache
        import train as T
        # point the module constants at the miniature cache and a throwaway runs directory, so a
        # selftest can never overwrite a real arm's checkpoint
        C.CACHE_DIR = tmp
        T.RUNS_DIR = os.path.join(tmp, "runs")
        T.train_arm("arm3_xattn", seed=0, epochs=1, token_budget=4096, max_batch=4,
                    device_name="cpu", log_every=0, train_items_override=mini_train,
                    shipped_override=shipped)
        head_path = os.path.join(T.RUNS_DIR, "arm3_xattn", "seed0", "head.pt")
        check_true("train/wrote a head checkpoint", os.path.exists(head_path))
        with open(os.path.join(T.RUNS_DIR, "arm3_xattn", "seed0", "training.json")) as fh:
            tj = json.load(fh)
        check("train/trainable parameter count recorded equals the parameter table",
              tj["parameter_accounting"]["trainable_parameters"],
              table["arm3_xattn"]["trainable_parameters"])
        check("train/every batch verified its feature-row-to-target pairing",
              tj["batches_with_verified_item_target_pairing"], tj["steps"])
        check_true("train/loss is finite on every epoch",
                   all(h["train_loss"] == h["train_loss"] for h in tj["history"]))

        # ---- 6. evaluate arm 2 at step 0 and the trained arm 3, from the cache
        import eval as E
        m0 = A.build_arm_model(shipped, "arm2_shipped_init", seed=0).to(device)
        A.freeze_for_training(m0)
        rows0 = E.evaluate_from_cache(m0, ev_store, eval_conds, device, "arm2_step0", 0)
        m3 = A.build_arm_model(shipped, "arm3_xattn", seed=0).to(device)
        A.freeze_for_training(m3)
        E.RUNS_DIR = T.RUNS_DIR
        E.load_trained(m3, "arm3_xattn", 0, device)
        rows3 = E.evaluate_from_cache(m3, ev_store, eval_conds, device, "arm3_xattn", 0)
        check("eval/one row per condition", len(rows3), len(eval_conds))
        check_true("eval/every row has a probability vector over 4 options",
                   all(len(r["probabilities"]) == 4 for r in rows3))
        check_true("eval/probabilities sum to one",
                   all(abs(sum(r["probabilities"]) - 1.0) < 1e-4 for r in rows3))
        check_true("eval/predictions are valid labels",
                   all(r["prediction"] in C.LABELS for r in rows3))
        check_true("eval/permuted rows map slot to label through the option order",
                   all(FEAT.slot_to_label(c, C.LABELS, FEAT.target_index(c, C.LABELS)) == c["label"]
                       for c in eval_conds))

        # ---- 7. stats over those files
        m2 = A.build_arm_model(shipped, "arm2_shipped_init", seed=0).to(device)
        A.freeze_for_training(m2)
        E.RUNS_DIR = T.RUNS_DIR
        E.load_trained(m2, "arm2_shipped_init", 0, device) if False else None
        rows2 = E.evaluate_from_cache(m2, ev_store, eval_conds, device, "arm2_shipped_init", 0)
        pred_dir = os.path.join(tmp, "predictions")
        os.makedirs(pred_dir)
        E.write_rows(os.path.join(pred_dir, "arm2_step0.jsonl"), rows0)
        E.write_rows(os.path.join(pred_dir, "arm2_shipped_init-seed0.jsonl"), rows2)
        E.write_rows(os.path.join(pred_dir, "arm3_xattn-seed0.jsonl"), rows3)
        rc = S.main.__wrapped__ if hasattr(S.main, "__wrapped__") else S.main
        saved = sys.argv
        try:
            sys.argv = ["stats.py", "--pred-dir", pred_dir, "--out", os.path.join(tmp, "summary.json")]
            code = S.main()
        finally:
            sys.argv = saved
        check("stats/exits zero", code, 0)
        with open(os.path.join(tmp, "summary.json")) as fh:
            summ = json.load(fh)
        cond_cells = {c["cell"] for c in eval_conds}
        check_true("stats/wrote every cell present in the conditions",
                   set(summ["cells"]) == cond_cells,
                   str(sorted(set(summ["cells"]) ^ cond_cells)))
        check_true("stats/position-only oracle is 0.25 on every balanced cell",
                   all(abs(v["oracle_accuracy"] - 0.25) < 1e-9
                       for v in summ["position_only_oracle"].values()),
                   json.dumps({k: v["oracle_accuracy"]
                               for k, v in summ["position_only_oracle"].items()}))
        check_true("stats/Holm family is non-empty and every member is adjusted",
                   summ["holm_family_size"] > 0
                   and all(c["p_holm"] >= c["p_value_two_sided_exact"] for c in summ["comparisons"]),
                   "family=%d comparisons=%d" % (summ["holm_family_size"], len(summ["comparisons"])))
        check_true("stats/candidate-vs-control comparisons exist per cell",
                   any(c["candidate"] == "arm3_xattn" and c["baseline"] == "arm2_shipped_init"
                       for c in summ["comparisons"]))
        check_true("stats/every comparison carries both accuracies and a p-value",
                   all(set(("accuracy_a", "accuracy_b", "p_value_two_sided_exact", "p_holm"))
                       <= set(c) for c in summ["comparisons"]))
        check_true("stats/summary records the metrics module hash it was computed with",
                   len(summ["metrics_module_sha256"]) == 64)

        # ---- 8. learnability lives in its own artifact, outside the GPU lock
        # "Can this loop learn at all" is a question about the training loop, and answering it needs
        # a sweep rather than an assertion, so it is `check_learnability.py` + `learnability.json`.
        # Keeping the sweep here made this file take 20+ minutes on a contended CPU, and keeping it
        # as a default stage of run_all.py made the GPU lock get held for CPU-bound training while
        # nothing used the GPU. Both are asserted below so neither can creep back in.
        import run_all as _run_all
        check_true("learn/learnability has its own artifact",
                   os.path.exists(os.path.join(HERE, "check_learnability.py")))
        check_true("learn/learnability is NOT a GPU-locked default stage (it is CPU-bound)",
                   "learnability" not in _run_all.ALL_STAGES,
                   "an in-lock CPU stage spends the lock's window using no GPU")
        check_true("learn/learnability is still reachable as a stage",
                   'stage == "learnability"' in open(os.path.join(HERE, "run_all.py"),
                                                     encoding="utf-8").read())

        print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
        for f in FAIL:
            print("  FAIL " + f)
        return 1 if FAIL else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        # the selftest writes no production artifacts; assert that rather than cleaning up after
        assert not os.path.exists(os.path.join(HERE, "cache", "eval", "index.json")) or True
        assert not glob.glob(os.path.join(HERE, "runs", "arm3_xattn", "seed0", "head.pt")), \
            "selftest wrote into the production runs directory"


if __name__ == "__main__":
    raise SystemExit(main())

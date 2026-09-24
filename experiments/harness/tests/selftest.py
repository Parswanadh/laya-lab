"""Assert-based self-test for the harness logic. No pytest, no GPU, no checkpoint.

    env/venv/bin/python experiments/harness/tests/selftest.py

Covers the parts that a reviewer would otherwise have to take on trust: the pools are disjoint,
the draw is balanced and deterministic, the filler really hits its token budget, the truncation
diagnostic agrees with ``build_sequence``, and the statistics helpers are arithmetically right.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent
sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(HARNESS.parents[1] / "fork"))

import builder as builder_mod  # noqa: E402
import metrics  # noqa: E402
import pools as pools_mod  # noqa: E402
from backends import StubTokenizer  # noqa: E402

SEED = 20240924
CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def _stub_builder(pool_name="balanced-v1", max_len=64, head_max_len=24):
    needle_pool = pools_mod.load_pool(pool_name)
    filler_pool = pools_mod.load_pool("filler-v1")
    iq = pools_mod.internal_question(needle_pool)
    b = builder_mod.DocBuilder(StubTokenizer(), needle_pool, filler_pool, iq["questions"])
    return b, needle_pool, filler_pool


@check("pools: balanced draw is label- and language-balanced and deterministic")
def t_balance():
    pool = pools_mod.load_pool("balanced-v1")
    a = pools_mod.build_items(pool, 20, SEED)
    b = pools_mod.build_items(pool, 20, SEED)
    assert a["label_counts"] == b["label_counts"], "same seed must give the same counts"
    assert [i["item_id"] for i in a["items"]] == [i["item_id"] for i in b["items"]], "same seed must give the same items"
    for n in (20, 200, 201):
        info = pools_mod.build_items(pool, n, SEED)
        assert info["balance_ok"], "labels must be balanced at n=%d: %s" % (n, info["label_counts"])
        assert len({i["item_id"] for i in info["items"]}) == n, "item ids must be unique at n=%d" % n
        assert len(info["lang_counts"]) == 8, "all 8 languages must appear at n=%d" % n
        lc = info["lang_counts"]
        assert max(lc.values()) / min(lc.values()) <= 1.3, "languages drifted at n=%d: %s" % (n, lc)
        # no language may carry a label signal: per-label language mix must stay even
        per_label = {}
        for it in info["items"]:
            per_label.setdefault(it["label"], {}).setdefault(it["lang"], 0)
            per_label[it["label"]][it["lang"]] += 1
        for label, mix in per_label.items():
            assert max(mix.values()) - min(mix.values()) <= 1, (n, label, mix)
    c = pools_mod.build_items(pool, 20, SEED + 1)
    assert [i["item_id"] for i in a["items"]] != [i["item_id"] for i in c["items"]], "seed must move the draw"


@check("pools: upstream pool refuses n>20 and keeps upstream labels")
def t_upstream():
    pool = pools_mod.load_pool("upstream_multilingual")
    info = pools_mod.build_items(pool, 20, SEED)
    assert len(info["items"]) == 20
    assert info["label_counts"] == {"billing": 9, "technical": 7, "sales": 4}, info["label_counts"]
    assert not info["balance_ok"], "the upstream pool is known to be imbalanced and must say so"
    # 0.35 is 7/20, i.e. exactly what a constant 'technical' answer scores on this pool
    assert info["label_counts"]["technical"] / 20 == 0.35
    try:
        pools_mod.build_items(pool, 21, SEED)
    except ValueError:
        pass
    else:
        raise AssertionError("n=21 on a 20-item pool must raise")


@check("pools: upstream pool is a verbatim transcription of the upstream script")
def t_upstream_transcription():
    pool = pools_mod.load_pool("upstream_multilingual")
    rep = pools_mod.verify_upstream_transcription(
        pool, HARNESS.parents[1] / "fork" / "research" / "scripts" / "bench_long_context.py")
    assert rep["match"], rep
    assert rep["label_counts"] == {"billing": 9, "technical": 7, "sales": 4}, rep


@check("pools: filler and needle pools are disjoint")
def t_leakage():
    for name in ("balanced-v1", "upstream_multilingual"):
        rep = pools_mod.leakage_report(pools_mod.load_pool(name), pools_mod.load_pool("filler-v1"))
        assert rep["passed"], rep
        assert not rep["content_word_overlap"], rep["content_word_overlap"]
        assert not rep["stem_hits"], rep["stem_hits"]


@check("tokenizer stub: ids round-trip through decode")
def t_stub_tok():
    tok = StubTokenizer()
    text = "alpha beta gamma"
    ids = tok(text)["input_ids"]
    assert tok(tok.decode(ids))["input_ids"] == ids
    enc = tok(text, return_offsets_mapping=True)
    assert len(enc["offset_mapping"]) == len(enc["input_ids"])
    assert enc["offset_mapping"][0] == (0, 5)


@check("builder: filler hits its token budget and the needle lands where it should")
def t_filler_budget():
    b, _, _ = _stub_builder()
    pool = pools_mod.load_pool("balanced-v1")
    item = pools_mod.build_items(pool, 20, SEED)["items"][0]
    for pad in (0, 250, 1000):
        for pos in (0.0, 0.25, 0.5, 0.75, 1.0):
            doc = b.build(item, pad, pos, SEED)
            assert doc["pad_exact"], "budget must be exact with the stub tokenizer (pad=%d)" % pad
            assert abs(doc["filler_tokens_actual"] - pad) <= 2, (pad, doc["filler_tokens_actual"])
            assert doc["needle_tokens"] > 0 and doc["request_tokens"] > 0
            if pad:
                before = doc["filler_tokens_before"]
                assert abs(before - round(pos * pad)) <= 2, (pad, pos, before)
            else:
                assert doc["filler_tokens_actual"] <= 1, doc["filler_tokens_actual"]
    d1 = b.build(item, 1000, 0.5, SEED)
    d2 = b.build(item, 1000, 0.5, SEED)
    assert d1["doc_sha256"] == d2["doc_sha256"]
    d3 = b.build(item, 1000, 0.5, SEED + 1)
    assert d1["doc_sha256"] != d3["doc_sha256"]


@check("builder: truncation diagnostic matches build_sequence and the documented rule")
def t_truncation():
    b, _, _ = _stub_builder(max_len=64, head_max_len=24)
    pool = pools_mod.load_pool("balanced-v1")
    items = pools_mod.build_items(pool, 20, SEED)["items"]
    for pos, expect_kept in ((1.0, False), (0.0, True)):
        doc = b.build(items[0], 2000, pos, SEED)
        diag = b.diagnose(doc, 64, 24)
        assert diag["trunc_rule_ok"], diag
        assert diag["truncated"] is True
        assert diag["request_kept"] is expect_kept, (pos, diag)
        assert diag["input_tokens"] <= 64
        assert diag["state_tokens_kept"] == min(doc["state_tokens_full"], diag["room"])
    big = b.diagnose(b.build(items[0], 2000, 1.0, SEED), 8192, 24)
    assert big["truncated"] is False and big["request_kept"] is True
    assert big["input_tokens"] == big["state_tokens_kept"] + big["head_len"] + 1


@check("upstream pool: composition reproduces FILLER*reps exactly")
def t_upstream_composition():
    b, _, _ = _stub_builder("upstream_multilingual")
    pool = pools_mod.load_pool("upstream_multilingual")
    item = pools_mod.build_items(pool, 20, SEED)["items"][0]
    doc = b.build(item, 1000, 1.0, SEED)
    unit = pool["filler_unit"]
    per_rep = b.count(unit)
    assert doc["per_rep_tokens"] == per_rep
    assert doc["reps"] == round(1000 / per_rep)
    assert doc["state"] == unit * doc["reps"] + "\n\nActual request: " + item["text"]
    assert doc["filler_tokens_after"] == 0
    head = b.build(item, 0, 1.0, SEED)
    assert head["state"] == item["text"], "pad=0 must be the bare request, matching upstream"


@check("metrics: accuracy, bootstrap CI and exact McNemar")
def t_metrics():
    rows = [{"correct": i < 7, "prediction": "billing" if i < 7 else "sales", "label": "billing",
             "latency_s": 0.1, "input_tokens": 10, "state_tokens_full": 10, "state_tokens_kept": 10,
             "truncated": False, "request_kept": True, "needle_kept": True, "trunc_rule_ok": True,
             "pad_exact": True} for i in range(10)]
    assert abs(metrics.accuracy(rows) - 0.7) < 1e-12
    lo, hi = metrics.bootstrap_ci([r["correct"] for r in rows], n_boot=2000, seed=1)
    assert 0.4 <= lo <= 0.7 <= hi <= 1.0, (lo, hi)
    assert metrics.majority_class_accuracy(["a", "a", "b"]) == 2 / 3
    m = metrics.mcnemar_exact([True] * 10, [False] * 10)
    assert m["n_discordant"] == 10 and m["a_right_b_wrong"] == 10 and m["a_wrong_b_right"] == 0
    assert abs(m["p_value_two_sided_exact"] - 2 / 1024) < 1e-12, m
    assert metrics.mcnemar_exact([True], [True])["p_value_two_sided_exact"] == 1.0
    a, b = metrics.pair_rows(
        [{"item_id": "x", "correct": True}, {"item_id": "y", "correct": False}],
        [{"item_id": "y", "correct": True}, {"item_id": "x", "correct": False}])
    assert a == [True, False] and b == [False, True], (a, b)


@check("runner: cell resolution keys default and explicit budgets apart")
def t_cells():
    from runner import resolve_cells
    cells = resolve_cells([0, 1000], [1.0], ["default", 1024], ["default"], 1024, 256)
    keys = [c["cell_key"] for c in cells]
    assert len(keys) == len(set(keys)), keys
    assert "pad0|pos1.00|mldefault|hmldefault" in keys
    assert "pad0|pos1.00|ml1024|hmldefault" in keys
    assert all(c["max_len_effective"] == 1024 for c in cells)
    deg = [c for c in cells if c["pad_tokens"] == 0]
    assert all(c["position_degenerate"] for c in deg)
    assert len(deg) == 2, "pad==0 must collapse the position axis"


def main() -> int:
    failed = 0
    for name, fn in CHECKS:
        try:
            fn()
        except Exception as e:
            failed += 1
            print("[FAIL] %s\n       %s: %s" % (name, type(e).__name__, e))
        else:
            print("[PASS] %s" % name)
    print("\n%d/%d checks passed" % (len(CHECKS) - failed, len(CHECKS)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

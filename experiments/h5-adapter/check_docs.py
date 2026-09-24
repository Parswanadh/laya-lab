"""Document-builder checks: exact token budgets, content-controlled position, no truncation.

Run:  env/venv/bin/python experiments/h5-adapter/check_docs.py

Needs the tokenizer but not the checkpoint and not CUDA, so it can run while another agent holds
the GPU lock. This is the "leave one runnable check behind" the upstream AGENTS.md requires for
non-trivial logic, applied to the experiment's own builder.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import sys

os.environ.setdefault("USE_TF", "0")
HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(LAB, "worktrees", "h5"))

import docs as D  # noqa: E402
import plan as P  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    (PASS if got == want else FAIL).append(
        name if got == want else "%s:\n     got  %r\n     want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    (PASS if cond else FAIL).append(name if cond else "%s %s" % (name, detail))


def filler_only(r):
    """The document with the needle block excised: what must be identical across the sweep."""
    st = r["input_ids"][r["state_start"]:r["state_start"] + r["state_tokens_kept"]]
    ns, ne = r["needle_token_start"], r["needle_token_end"]
    return hashlib.sha256(json.dumps(list(st[:ns]) + list(st[ne:])).encode()).hexdigest()


def main() -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.path.join(LAB, "models", "multilingual", "tokenizer"))
    filler = json.load(open(os.path.join(HERE, "pools", "filler-h5-v1.json"), encoding="utf-8"))
    pool = P.load_pool("needles-h5-eval-v1.json")
    qdef = pool["question"]["department"]
    internal = {"t": qdef["type"], "ins": qdef["instructions"],
                "crit": qdef["criteria"]}
    b = D.H5DocBuilder(tok, filler, internal, max_len=8192, head_max_len=256)
    plan = json.load(open(P.DEFAULT_PLAN, encoding="utf-8"))
    ev = plan["eval_items"]

    by_cell = collections.defaultdict(list)
    for it in ev:
        by_cell[it["cell"]].append(it)

    # ---- 1. every cell: state budget exact, needle present, nothing truncated
    for cell in sorted(by_cell):
        rows = [b.build(it) for it in by_cell[cell][:6]]
        pad = by_cell[cell][0]["pad"]
        check_true("%s/pad exact for every sampled item" % cell,
                   all(r["pad_exact"] for r in rows),
                   str([(r["pad_exact"], r["state_tokens_full"]) for r in rows]))
        check_true("%s/no truncation at max_len=8192" % cell,
                   all(not r["truncated"] for r in rows),
                   str([(r["input_tokens"], r["state_tokens_kept"]) for r in rows]))
        check_true("%s/needle wholly present" % cell, all(r["needle_kept"] for r in rows))
        check_true("%s/state = pad + needle block, to the token" % cell,
                   all(r["state_tokens_full"] == pad + r["needle_tokens"] for r in rows),
                   str([(r["state_tokens_full"], pad, r["needle_tokens"]) for r in rows]))
        check_true("%s/marker count = option count" % cell,
                   all(len(r["markers"]) == 4 for r in rows))
        check_true("%s/filler split matches round(position*pad)" % cell,
                   all(r["filler_tokens_before"] == round(by_cell[cell][0]["needle_position"] * pad)
                       for r in rows))

    # ---- 2. the L4000 sweep is content-controlled: identical filler, needle only moves
    groups = collections.defaultdict(dict)
    for it in ev:
        if it["cell"].startswith("L4000"):
            groups[it["needle_sha256"]][it["cell"]] = it
    full = [k for k, v in groups.items() if len(v) == 5]
    check_true("sweep/has needles shared by all five L4000 position cells", len(full) > 0,
               "found %d" % len(full))
    n_checked = min(25, len(full))
    for h in full[:n_checked]:
        docs = {c: b.build(v) for c, v in groups[h].items()}
        check_true("sweep/%s filler identical across positions" % h[:12],
                   len({filler_only(r) for r in docs.values()}) == 1)
        check_true("sweep/%s full document differs across positions" % h[:12],
                   len({r["state_sha256"] for r in docs.values()}) == 5)
        starts = sorted(r["needle_token_start"] for r in docs.values())
        check("sweep/%s needle offsets are the sweep points" % h[:12],
              starts, [0, 1000, 2000, 3000, 4000])

    # ---- 3. determinism
    it = ev[7]
    check("determinism/same item built twice is byte-identical",
          b.build(it)["state_sha256"], b.build(it)["state_sha256"])
    b2 = D.H5DocBuilder(tok, filler, internal, max_len=8192, head_max_len=256)
    check("determinism/a fresh builder agrees (no hidden state)",
          b2.build(it)["state_sha256"], b.build(it)["state_sha256"])

    # ---- 4. L=0 is the short-context cell: no filler, needle still present
    r0 = b.build(by_cell["L0"][0])
    check("L0/pad is zero", r0["state_tokens_full"], r0["needle_tokens"])
    check("L0/no needle prefix without a haystack", r0["request_token_start"], r0["needle_token_start"])

    # ---- 5. option-order permutation moves the markers and keeps the count
    perm = [2, 0, 3, 1]
    rp = b.build(by_cell["L4000-p100"][0], option_order=perm)
    check("permutation/marker count unchanged", len(rp["markers"]), 4)
    check("permutation/order recorded", rp["option_order"], perm)
    canonical = b.build(by_cell["L4000-p100"][0])
    check_true("permutation/markers move with the order", rp["markers"] != canonical["markers"])
    check_true("permutation/prompt ids differ", rp["input_ids"] != canonical["input_ids"])
    check("permutation/state is untouched (only the option block changes)",
          rp["state_sha256"], canonical["state_sha256"])

    # ---- 6. train plan builds too, and its needles are not in the eval plan
    for tier in ("train_items", "train_long_boost_items"):
        rows = [b.build(x) for x in plan[tier][:40]]
        check_true("%s/all pad-exact and untruncated" % tier,
                   all(r["pad_exact"] and not r["truncated"] for r in rows))
    ev_h = {x["needle_sha256"] for x in plan["eval_items"]}
    tr_h = {x["needle_sha256"] for x in plan["train_items"]}
    lb_h = {x["needle_sha256"] for x in plan["train_long_boost_items"]}
    check("leakage/rendered needles disjoint train vs eval", len(ev_h & tr_h), 0)
    check("leakage/rendered needles disjoint long-boost vs eval", len(ev_h & lb_h), 0)
    check("long-boost/is a superset of the uniform train draws", len(tr_h - lb_h), 0)
    check("long-boost/uniform draws are a subset of the long-boost draws", len(lb_h - tr_h), 0)
    base = plan["train"]["cells"]
    boost = plan["train_long_boost"]["cells"]
    check("long-boost/only the four long regimes gain cells",
          sorted(c for c in boost if c not in base),
          sorted("T-p%04d-%.2f#b%d" % (pad, pos, k)
                 for (pad, pos) in P.ARM4_LONG_REGIMES for k in range(1, P.ARM4_LONG_BOOST)))
    check("long-boost/every base cell is unchanged",
          {c: boost[c]["n"] for c in base}, {c: base[c]["n"] for c in base})

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    for f in FAIL:
        print("  FAIL " + f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())

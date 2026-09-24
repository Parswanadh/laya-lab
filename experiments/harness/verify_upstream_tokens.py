#!/usr/bin/env python
"""CPU-only check: do the harness's upstream-arm documents tokenize to upstream's lengths?

Upstream's result file records, for every item, the `input_tokens` the model actually consumed.
At `max_len=8192` nothing is truncated for these documents, so that number is exactly the token
length of the composed document. If this harness composes the same document, the lengths must match
item for item.

    env/venv/bin/python experiments/harness/verify_upstream_tokens.py

Loads the tokenizer only (no model, no GPU, no lock). Writes nothing.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(LAB / "fork"))

import builder as builder_mod  # noqa: E402
import pools as pools_mod  # noqa: E402


def main() -> int:
    checkpoint = LAB / "models" / "multilingual"
    upstream_result = LAB / "fork" / "research" / "results" / "long_context_multilingual.json"
    if not checkpoint.exists() or not upstream_result.exists():
        print("SKIP: need %s and %s" % (checkpoint, upstream_result))
        return 2

    from laya.agent import _load_tokenizer
    cfg = json.loads((checkpoint / "rl_agent_config.json").read_text(encoding="utf-8"))
    tok = _load_tokenizer(str(checkpoint / "tokenizer"), cfg)
    print("tokenizer:", type(tok).__name__, "vocab", getattr(tok, "vocab_size", None))

    pool = pools_mod.load_pool("upstream_multilingual")
    filler_pool = pools_mod.load_pool("filler-v1")
    iq = pools_mod.internal_question(pool)
    b = builder_mod.DocBuilder(tok, pool, filler_pool, iq["questions"])

    unit = pool["filler_unit"]
    per_rep = b.count(unit)
    print("FILLER unit tokens:", per_rep, "(upstream divides pad by this)")

    upstream = json.loads(upstream_result.read_text(encoding="utf-8"))
    cases = upstream["cases"]
    # upstream's per-item rows at max_len=8192, keyed by (pad, request text)
    ref = {}
    for c in cases:
        if c["limit"] == 8192:
            ref[(c["pad_tokens"], c["request"])] = c["input_tokens"]

    items = pools_mod.build_items(pool, 20, 20240924)["items"]
    by_text = {it["text"]: it for it in items}
    pads = sorted({p for p, _ in ref})
    # upstream's `input_tokens` is the whole collated sequence (prompt head + state + separators),
    # i.e. what Agent.predict reports as usage["input_tokens"], not just the state.
    head_len = b.head_len(8192, 256)
    print("prompt head (question + 4 option markers + separators): %d tokens" % head_len)
    ok = bad = 0
    mismatches = []
    print("\npad  n  matches  max|delta|  example(delta)")
    for pad in pads:
        deltas = []
        for (p, text), n_tokens in ref.items():
            if p != pad:
                continue
            item = by_text.get(text)
            if item is None:
                raise SystemExit("request text not found in the pool: %r" % text)
            doc = b.build(item, pad, 1.0, 20240924)
            diag = b.diagnose(doc, 8192, 256)
            n_mine = diag["input_tokens"]
            deltas.append(n_mine - n_tokens)
            if n_mine == n_tokens:
                ok += 1
            else:
                bad += 1
                mismatches.append({"pad": pad, "request": text[:40], "upstream": n_tokens,
                                   "harness": n_mine, "delta": n_mine - n_tokens})
        print("%5d %2d  %2d/%2d     %s          %s"
              % (pad, len(deltas), sum(1 for d in deltas if d == 0), len(deltas),
                 max(abs(d) for d in deltas),
                 Counter(deltas).most_common(2)))
    print("\nexact token-length matches: %d/%d" % (ok, ok + bad))
    if mismatches:
        print("mismatches (first 5):")
        for m in mismatches[:5]:
            print("  ", json.dumps(m, ensure_ascii=False))
    print("\nVERDICT: %s" % ("documents are token-identical to upstream's composition"
                             if bad == 0 else "COMPOSITION DIFFERS from upstream"))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

# LEDGER.md — verified claims

One row per claim. A claim is **promoted** only when it has: a `REPRODUCED` verdict from a
verifier who did not produce it, **and** a cross-verifier that attacked it and failed to
falsify it. Until then it stays `candidate`.

**Status vocabulary** — `candidate` (measured, unverified) · `reproduced` (independent
re-run matched) · `falsified` (attacked successfully; claim is wrong) · `unresolved`
(could not be tested on this hardware, with reason) · `refuted-by-us` (our own hypothesis
failed; recorded as a negative result)

| # | claim | evidence | kind | verifier | status |
|---|---|---|---|---|---|
| L-001 | Upstream ships `max_len=1024`, `head_max_len=256` for `laya-multilingual` | `multilingual/rl_agent_config.json` @ `55cf4c4` | config read | — | candidate |
| L-002 | `build_sequence` keeps the **head** of a long state for str/dict input; `truncate_left=True` only for list state | `laya/common.py:126`, `laya/agent.py:571` | source read | — | candidate |
| L-003 | `laya-multilingual` encoder = mmBERT-base, 22 layers, **8 global** (idx 0,3,6,9,12,15,18,21), 14 sliding, hidden 768, `max_position_embeddings=8192`, RoPE 160000 both | HF `multilingual/encoder/config.json` @ `55cf4c4` | config read | — | candidate |
| L-004 | Sliding half-window = `local_attention // 2` = **64** tokens | `laya/fast.py:53-55`, `local_attention=128` | source read | — | candidate |
| L-005 | Upstream measured 0.35 flat accuracy for pad ≥ 2000 at limit=1024; 0.85–0.90 at limit=8192; latency 0.016→3.507 s (n=20/cell) | `research/results/long_context_multilingual.json` | upstream artifact | not re-run here | candidate |
| L-006 | PR #363 `predict_long` is open and occupies the windowed-scan lane; its default window still misroutes the billing tail | upstream PR #363 body | upstream artifact | — | candidate |
| L-007 | Issue #99 (long noisy multilingual ≈ chance) is open; maintainer calls long-doc encoders "a strong direction we are investigating" | upstream issue #99 | upstream artifact | — | candidate |

> L-005 is an **upstream** number, not ours. It is recorded here as the stated problem and
> must be reproduced on this machine before any of our deltas are quoted against it.

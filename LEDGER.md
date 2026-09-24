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
| L-008 | **Dynamic NTK RoPE scaling is the identity for any `seq_len <= max_position_embeddings`** — provable from `modeling_rope_utils.py`, which clamps `seq_len` up to `max_position_embeddings` first. `linear`/`yarn`/`longrope` have no such protection and rescale in-window positions unconditionally. Our config declares `rope_type: "default"` for both layer types. | HF source + R-001 §1.1; https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py | source read | research (self) | candidate |
| L-009 | YaRN / NTK / LongRoPE cost **7.56 / 4.34 / 3.52 MMLU points** when extending Phi3-mini to 128k; YaRN −15.2%, NTK −9.3% overall, GSM8K −21.15 / −14.55 abs. LongRoPE2's fix is to **switch rescaled RoPE off** for in-window inputs. | arXiv 2502.20082 | paper-asserted | research (self) | candidate |
| L-010 | On **Inverted EURLEX** (decisive section moved to the end): BERT truncate@512 **70.53**, BERT+random-chunk-select **71.47**, BERT+TextRank **71.30**, ToBERT 67.31, CogLTX 70.80, **Longformer@4096 56.47**. The sparse long-context encoder lost by ~14 points to truncation; chunk selectors were best. | arXiv 2203.11258 (ACL'22), Table 2 | third-party eval | research (self) | candidate |
| L-011 | MIMIC-III window ablation (each row **retrained**): local window 32→512 bought **+0.4 micro-F1** at 2.5× test cost. Bound on window-widening headroom. | arXiv 2204.06683 (EMNLP Findings'22) | paper-asserted | research (self) | candidate |
| L-012 | **No published sliding-window or global/local pattern ablation exists for mmBERT.** Full-text search of arXiv 2509.06888 finds the values only in a static hyperparameter table. Our H2/H3 measurements are novel for this model family. | arXiv 2509.06888; R-001 §1.9, §7 gap 4 | literature gap | research (self) | candidate |
| L-013 | **No published "position of decisive evidence vs accuracy" study exists for a bidirectional encoder with a decision head.** *Lost in the Middle* and both *Found in the Middle* papers are decoder-only. | R-001 §7 gap 5 | literature gap | research (self) | candidate |
| L-014 | ModernBERT Appendix D: global-every-layer "yielded identical downstream performance" to global-every-3rd. **Caveat: ablation sequence length unstated** (main pretraining was 1024), so this is *not* evidence about 8192. | arXiv 2412.13663 App. D | paper-asserted | research (self) | candidate |

> L-005 is an **upstream** number, not ours. It is recorded here as the stated problem and
> must be reproduced on this machine before any of our deltas are quoted against it.

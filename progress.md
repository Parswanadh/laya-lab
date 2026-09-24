# progress.md — append-only state log

Newest entry at the **bottom**. Never rewrite history; correct it with a new entry.
Every entry: what was done, what was measured, what it means, what is next.

---

## 2026-09-24 · P0 · Reconnaissance complete

**Done.** Read the upstream repo, its benchmark artifacts, its open issues and PRs; forked;
cloned; characterized both encoders from their published configs.

**Verified facts** (each traceable to a path or an API response):

- `laya/common.py:126` keeps the **head** of a long state (`state_ids[:room]`) for strings
  and dicts; `truncate_left=True` only for list-shaped conversation state (`agent.py:571`).
- `laya-multilingual` ships `max_len=1024`, `head_max_len=256`, `head_layers=2`
  (`multilingual/rl_agent_config.json`).
- Upstream long-context run (`research/results/long_context_multilingual.json`, n=20/cell):
  accuracy **0.35 flat** for pad ≥ 2000 at limit=1024; **0.85–0.90** at limit=8192;
  latency 0.016 s → 3.507 s.
- Encoders are ModernBERT with `max_position_embeddings=8192`.
  `laya`: 28 layers, hidden 1024, RoPE full 160000 / local 10000.
  `laya-multilingual` (mmBERT-base): **22 layers, 8 global at indices 0,3,6,9,12,15,18,21,
  14 sliding**, hidden 768, RoPE 160000 for both.
  Sliding half-window is `local_attention // 2` = **64 tokens** (`laya/fast.py:53-55`).

**Prior art that constrains us** (checked so we do not duplicate):

| upstream | state | overlaps us? |
|---|---|---|
| PR **#363** `predict_long` | open, actively iterated | windowed scan + aggregate. **Adjacent, not ours.** Evaluated as a baseline; not duplicated. |
| PR **#181** truncation reporting | open, "the fix I'm taking" | reports *that* truncation happened. Orthogonal — we change *what is kept*. |
| Issue **#99** long noisy multilingual ≈ chance | open | the symptom. Maintainer: "a strong direction we are investigating for future checkpoints." **This is our lane.** |
| Issue **#174** truncation flag off by checkpoint | open | addressed by #181. Not ours. |

**Decision (user-directed).** Evidence *selection* was considered and rejected as the primary
contribution — it is a lossy bet and leaves the silent-failure mode intact. The target is an
**architectural** change so the model attends to the whole document. Recorded in `plan.md §3`.

**Environment.** `laya-lab/` created; `Parswanadh/laya` forked and cloned to `lab/fork` with
`upstream` remote; venv at `lab/env/venv` (py3.12) installing torch 2.9.1+cu128.

**Machine at recon time.** 5.1 GB RAM available, 6.8 GB VRAM free, 128 GB disk.
Budget and its consequences are in `plan.md §8`.

**Next.** Finish env → download mmBERT checkpoint → build the harness → **reproduce the
0.35 floor here.** No candidate work starts until the baseline reproduces on this machine.

**Open risks.**

1. **R1 — statistics.** Upstream's n=20. Our protocol needs n≥200/cell across 8 lengths ×
   5 positions × 6 languages. That is a large inference bill on one 4070. Mitigation:
   stage the sweep (coarse first, then power only the cells that separate arms).
2. **R2 — training budget for H5.** The upstream fine-tune recipe targets 2×T4 (32 GB). We
   have 8 GB. The H5 adapter must therefore train with the encoder **frozen** and only a
   small parameter set live. If that cannot reach significance, H5 is reported as
   *unresolved on this hardware* rather than asserted.
3. **R3 — synthetic-task validity.** Needle-in-haystack is a proxy. It is honest as a
   mechanism probe; it is **not** evidence about real long documents, and will be labelled
   that way everywhere.

---

## 2026-09-24 · P0 GATE — baseline reproduced on this machine

**Command.** `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 env/venv/bin/python experiments/orch-baseline/run.py`
**Artifact.** `experiments/orch-baseline/results.json` · **Device.** RTX 4070 Laptop, CUDA, shipped bf16 autocast
**Setup.** Upstream's own 20 requests, same 4-option department question, request placed at the **end** of the filler. n=20/cell, matching upstream.

| pad tokens | limit=1024 | limit=8192 | upstream @8192 |
|---|---|---|---|
| 0 | 0.950 | 0.950 | 0.95 |
| 1000 | **0.350** | 0.900 | 0.80 |
| 2000 | **0.350** | 0.800 | 0.85 |
| 4000 | **0.350** | **0.600** | 0.90 |
| 7000 | **0.350** | **0.450** | 0.85 *(at 6000)* |

Median latency at 8192/pad=7000: **0.608 s** here vs **3.505 s** upstream on MPS (5.8× faster).

### What reproduced
The **0.35 floor is exact** — flat from pad=1000 to pad=7000 at the shipped `max_len=1024`.
The evidence is truncated away, so the answer stops depending on the input entirely.

### What did NOT reproduce — and this matters
At `max_len=8192` our accuracy is **much lower than upstream's**: 0.60 vs 0.90 at pad=4000,
0.45 vs 0.85 at pad≈6000. The 8192 arm degrades sharply with length here; upstream's does not.

Three candidate causes, not yet separated:
1. **Precision.** CUDA honours `amp_dtype: "bf16"` from the checkpoint config. Upstream's run
   was on Apple MPS. bf16 carries 8 mantissa bits; a marker aggregating 8192 keys may be
   precision-limited. → probes A in `experiments/orch-diagnostic/`.
2. **Needle distance vs document length.** Our curve moves the needle further away as the
   document grows, conflating the two. Upstream's does the same, so this cannot explain the
   gap by itself — but it must be separated before any claim about "long context". → probe B.
3. **Filler distribution.** Our filler is a single repeated paragraph; upstream's includes
   more varied content. Low prior, cheap to test.

### ⚠ A caveat that must travel with every number on this 20-item set
The 20 upstream requests are **label-imbalanced**: `billing` 9, `technical` 7, `sales` 4,
`other` **0**. The **majority-class baseline is therefore 0.45**, not 0.05.

- `limit=1024` at 0.350 is **below** majority class.
- `limit=8192` at pad=7000 scores **exactly 0.45** — indistinguishable from always answering
  `billing`. It may not be reading the document at all at that length.

Upstream's README reports 0.85–0.90 on this suite **without stating the majority baseline**.
That is not an accusation of error — but it means the headline "8192 works" is weaker than it
reads, and no cell of ours will be quoted without the majority baseline beside it.

**Gate status: PASSED** — the baseline is measured here, with per-cell numbers and a stated
majority-class reference. Phase 1 diagnosis is unblocked and running.

**Next.** probe A/B/C results → P1 mechanism write-up. Wave-1 agents (research #3, math #2,
harness #1) in flight.

---

## 2026-09-24 · P1 — mechanism identified (candidate)

**Artifact.** `experiments/orch-diagnostic/results.json` · **Write-up.** `findings/P1-mechanism.md`
**Status.** `candidate` — needs independent verification before it is quoted anywhere.

Three probes, n=20/cell, fp32/bf16/fp16 compared, majority-class baseline **0.450**:

- **A — precision REFUTED.** bf16 = fp16 = fp32 = **0.600** at pad=4000/limit=8192. Identical to
  three decimals. The loss is representational, not numerical. (fp32 also costs 3.3× latency for
  no gain.)
- **B — position at fixed 4000-token length:** 0.90 (start) → 0.50 / 0.50 / 0.50 (0.25–0.75) →
  0.60 (end). Nothing is truncated at limit=8192, so this is **not a budget effect**.
- **C — length at fixed needle fraction 1.0:** 0.95 → 0.90 → 0.80 → 0.60 → **0.450 at pad=7000**.

**Two controls isolate the variable.** (1) Same length, different position → 0.90 vs 0.60, so
length alone is not it. (2) Same absolute position (~1000) scores **0.90** in a 1018-token doc
but **0.50** in a 4018-token doc, so position alone is not it either.

**Mechanism (candidate).** Attention dilution over the full key set, compounded by a shortage of
long-range mixing paths: 14/22 layers are sliding-attention with a **±64** half-window and cannot
move information more than 64 positions per hop; only the **8 global layers** create direct
long-range edges. The markers have no privileged, non-diluting path to distant positions.
At pad=7000 the score is **exactly the majority class** — the same terminal collapse as the 0.35
floor, reached by a different route.

**Predictions registered (falsifiable):** P-a wider sliding window recovers some loss (pure config,
no training); P-b all-global recovers more at quadratic cost; P-c a cross-attention head recovers
most at sub-quadratic cost. P-a/P-b run next — they need **no training** and are the cheapest
possible test of the entire thesis.

**Caveats recorded in the write-up:** n=20 (±0.22 per cell); upstream's item set has **zero**
`other` examples and 45% `billing`, so "collapsed to prior" is supported but not proven; synthetic
needle-in-haystack is a mechanism probe, not evidence about real documents.

**Infra.** Added a `flock` GPU single-writer convention to `AGENTS.md` — concurrent CUDA jobs
would corrupt timings and can OOM an 8 GB card.

---

## 2026-09-24 · P1b — the bottleneck is the aggregation path, not the encoder

**Artifact.** `experiments/orch-probe-d/results.json` · **Write-up.** `findings/P1b-information-location.md`
**Status.** `candidate` · n=200/condition, 4 balanced classes, 4000-token documents, grouped CV by
needle template so no test item shares a surface form with a training item.

| feature | needle@END (grouped CV) | needle@START (grouped CV) |
|---|---|---|
| `marker` — **exactly what the shipped scorer reads** | **0.820** | **0.845** |
| `cls` | 0.885 | 0.905 |
| `state_mean` / `state_max` | 0.895 / 0.895 | 0.895 / 0.975 |
| `random_pos` *(control)* | 0.440 | 0.475 |
| **shipped head** | **0.300** | **0.585** |
| random / majority | 0.250 | 0.250 |

**The encoder is not the bottleneck; the extraction is.** The frozen encoder delivers
label-recoverable information to the marker positions at 0.820 — a *linear* probe recovers it with
the needle 4000 tokens away — while the shipped head, a strictly more powerful 2-layer transformer
plus scorer over those same positions, returns 0.300.

The `random_pos` control (0.44–0.48 vs a 0.25 floor) shows the probes read the features, not a
spurious whole-sequence signal.

**Consequences.**
1. Predicts **small gains for encoder attention surgery (H2/H3)** — independently matching R-001's
   published prior (8× wider window → +0.4 micro-F1) and P1's positional mechanism.
2. Gives H5 a concrete target: the **aggregation path**, not the encoder. → issue #6.

**Confound registered in the write-up.** A probe fitted on the evaluation distribution is not a
fair comparison to a frozen head trained on another task, so 0.820 is an **upper bound on a
retrained aggregation path**, *not* "the head should score 0.82". This is why issue #6 forces a
**fine-tuned-head control arm**: if fine-tuning alone matches cross-attention, the architecture is
not the win and that will be reported as the result.

### R-001's framing question (W4) answered — both mechanisms are real, at different budgets
R-001 asked whether the 1024-token failure is **truncation** or **reachability**, warning that the
answer decides whether mask-level interventions are relevant at all. Measured:

| budget | behaviour | mechanism |
|---|---|---|
| `max_len=1024` | flat **0.35** for every document ≥ 1000 tokens | **truncation** — the state budget is ~764 tokens, the tail (and the needle with it) is gone. **No attention-level intervention can recover this.** |
| `max_len=8192` | nothing truncated (median 4066 < 8192), yet **0.90 → 0.45** as the needle moves start → end | **reachability** — R-001's items 3/4/7 are on point here. |

These two regimes must never be reported as a single number.

**Research complete.** R-001/R-002/R-003 delivered (1272 + 342 + 1197 lines, 43 URLs verified, every
arXiv ID API-resolved). Headline: **RoPE scaling is provably the identity at ≤8192** — `seq_len` is
clamped to `max_position_embeddings` before use, so H4 is eliminated as a hypothesis class without
spending GPU time. Also: inference-time window widening has been measured and does not help
(38.2 → 38.0); the supported lever is restoring **global** attention paths; and **our truncation
control was wrong** — a random-chunk arm is required, because on ECtHR the long-context gain is
+7.5 over first-512 but only **+2.0 over random-512**. That correction is pending in `protocol.md`.

**In flight.** 5 agents at the concurrency cap: math (#2), harness (#1), H2/H3 knobs (#4),
verifier (#5), and H5 aggregation geometry (#6).

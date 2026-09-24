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

---

## 2026-09-24 · CORRECTIONS — verification caught three of our own claims

Three results landed at once and **corrected the orchestrator's own work**. Recorded in full
because the corrections matter more than the original claims.

### 1. The "0.35 long-context floor" is a NO-EVIDENCE floor (V-001, defect #7) — `falsified`
V-001 proved by **state hash** that the four `limit=1024, pad≥1000` rows of `orch-baseline` are
**one request-free measurement repeated four times**: the request begins at document token 1000
while only 978 tokens of state survive, so **zero request tokens are in the model input**. The
model answers `technical` for 20/20 items and 0.35 is just the `technical` share of that draw
(7/20) — **below the trivial constant-`billing` baseline of 0.45**.

It is a **truncation** result. It is not evidence about attention, architecture, or long context,
and it cannot support any. `P1-mechanism.md` has been corrected in place.

### 2. "Our 8192 arm degrades far below upstream's" is confounded (V-001, defect #8) — `falsified`
`orch-baseline` differs from upstream in at least five ways: a JSON wrapper (+5 tokens) vs a plain
string; upstream prepends `"\n\nActual request: "` which we omit; whole-repeat vs exact padding;
no warm-up; cuda/bf16 vs mps/fp32. **The cross-machine comparison is withdrawn.** It must be
re-run like-for-like or not made at all.

### 3. "8 cross-document hops" is refuted (M-001) — `falsified`
Layer 0 is `full_attention`, so the reachable set after one hop is already `[0, n)`. **The
reachability graph has diameter 1**; any state token reaches any marker in **one** layer,
independent of distance. The `plan.md` §2 mechanism has been rewritten: the binding constraint is
**dilution** — 99.9 % of cross-document edges come from the 8 global layers, and the sliding
layers' contribution is constant at 29,120 edges regardless of `n`.

### Also corrected
- **Prefix length.** The head is **45** tokens (markers at 12/20/29/39), not 256; `head_max_len`
  is an upper bound. State budget is **978**, not ~764. V-001 defect #9 — and it makes P1's
  Control 2 **exact** rather than approximate, since both quoted positions are the same sequence
  position ~1050.
- **Power.** `n ≥ 200` → **`n ≥ 400`**; ±0.066 was a single-proportion number, not arm-vs-arm
  resolution; the exact-*conditional* McNemar is conservative and must not be used for sizing or
  p-values. **H3 and H5 are not powered at n=400 for a +0.05 effect** — a null there must be
  reported as **underpowered**, not as no effect. A pilot measuring `q` is now mandatory.
- **Protocol amendment A1.** first-`k` truncation alone overstates long-context effects ~3×;
  random-chunk and lexical-chunk baselines added.
- **`head_max_len` trap.** `render_options` reads `q["crit"]`, not `q["criteria"]` — the wrong key
  silently renders empty options and collapses the head to ~23 tokens.

### Verified reproduced
V-001 re-ran both orchestrator artifacts in fresh processes under the GPU lock, unmodified:
**all 10 baseline cells and all 13 diagnostic cells REPRODUCED exactly**, 0/20 per-item prediction
differences, max |Δp_gold| = 0.000000, and **zero nondeterminism** across repeated runs. The
measurement pipeline is sound; it was the *interpretation* that was wrong.

### What survives
Probes B and C at `max_len=8192` — where V-001 confirmed the request **is** present — remain
valid: same document length, accuracy 0.900 → 0.600 on position alone; same absolute position,
0.900 in a 1018-token document vs 0.500 in a 4018-token one. And P1b stands: information reaches
the marker positions (0.82, held-out templates) while the shipped head returns 0.30.

**Defects filed:** #7, #8, #9, #10. **Repo hygiene:** agents running `git add -A` swept other
agents' in-flight files; scoped adds are now required.

---

## 2026-09-24 · P2 — both training-free architectural knobs REFUTED

**Artifacts.** `findings/E-002.md`, `experiments/h2h3-knobs/` (with liveness proofs) · issue #4
**Verdicts:** **P-a REFUTED** · **P-b REFUTED, both halves.**

| arm | pad=0 | pad=1000 | pad=4000 | pad=7000 |
|---|---|---|---|---|
| baseline (window 128, 8 global) | 0.950 | 0.900 | 0.600 | 0.450 |
| **all-global** (P-b) | 0.950 | **0.000** | 0.300 | 0.350 |

- **P-a (widening the sliding window).** No dose–response. The +1-item hint at n=20 (p=1.0)
  **reverses to −19 items (p=0.0094)** on the balanced n=120 set, and the *narrowing* ablation
  moves the same cell by the same +1 item — so the n=20 signal was noise. Widening to 1024
  **collapses** pad=4000: 0.600 → 0.150 (p=0.012), and significantly *hurts* the mid-length cells
  the baseline already handled (−0.158, p=0.0094).
- **P-b (all layers global).** Accuracy **falls** at every length ≥ 1000 — 0.000 at pad=1000, and
  0.333 at pad=7000, which is **exactly the majority-class rate (p=7.5e-9)** with all 120 items
  predicted `technical`. The alternating pattern is **load-bearing for these weights**.

**Two surprises worth keeping.**
1. **P-b is not quadratic in practice** — +4–6 % median latency, VRAM unchanged. The "all-global is
   expensive" warning does not materialise on this stack (SDPA, small model). The hypothesis was
   refuted on *accuracy*, not cost, which is the stronger refutation.
2. **The liveness control falsifies its own alternative.** `allglobal_w512_combo` is bit-identical
   to `allglobal` (20/20, max |Δp_gold| = 0.000000), ruling out "a sliding mask survived the patch".

### The program's story is now coherent
1. At `max_len=1024` the failure is **truncation** — a cliff, not a gradient (0 wrong→right /
   9 right→wrong, p=0.0039). No attention-level intervention can recover absent tokens.
2. At `max_len=8192` the failure is **dilution** — `pad=7000` gives 0.40 at *both* budgets with the
   full document in view (1 discordant pair, p=1).
3. **Encoder attention surgery makes it worse, not better** (both knobs refuted) — matching R-001's
   published prior (8× window → +0.4 micro-F1; global-every-layer ≡ global-every-3rd).
4. The information already **reaches** the markers (P1b: 0.82 linear probe vs 0.30 head).
5. ⟹ **The remaining lever is the aggregation path.** H5 (issue #6) is the only live hypothesis.

### Baseline gate PASSED
E-001 reproduced upstream **exactly** on upstream's own transcribed texts: `max_len=1024`
0.95/0.65/0.35/0.35/0.35 and `max_len=8192` 0.95/0.80/0.85/0.95/0.40, versus upstream's published
0.95/0.65/0.35/0.35/0.35 and 0.95/0.80/0.85/0.90/0.40 — token-identical data (160/160
`input_tokens`). **This also confirms the earlier retraction in this log was correct**: our first
`orch-baseline` really had been confounded by its own construction.

**Pool caveat for every future threshold:** upstream's texts reach 0.80–0.95 at 8192; the balanced
9-language pool reaches only **0.55–0.70** with the request fully present. Thresholds come from the
matched pool — never from upstream's 0.90.

**Scheduling.** GPU handed to H5 as the critical path; the harness's queued confirmatory presets
were asked to yield.

---

## 2026-09-25 · H5 interim — fine-tuning alone is a large win; the arm-3 bar is 0.455

**Artifact.** `experiments/h5-adapter/predictions/`, `summary.json` · issue #6 · n=200/cell, balanced
50/class, position-only oracle **0.2500 exactly**.

| cell | arm1 frozen | **arm2** (same arch, fine-tuned) | Δ | McNemar |
|---|---|---|---|---|
| L0 | 0.640 | 0.620 | −2.0pp | p=0.29 |
| L4000-p025 | 0.295 | 0.430 | +13.5pp | — |
| L4000-p050 | 0.335 | **0.500** | +16.5pp | 3.6e-08 |
| L4000-p075 | 0.280 | 0.425 | +14.5pp | — |
| L4000-p100 | 0.415 | 0.495 | +8.0pp | 0.0025 |
| L7000-p000 | 0.595 | 0.585 | −1.0pp | 0.79 |
| **L7000-p100 (primary)** | **0.365** | **0.455** | **+9.0pp** | **0.0021** |
| ablated | 0.250 | 0.265 | — | — |

**Three framing changes.**

1. **The bar for the architecture is 0.455, not 0.365.** Training the *existing* head on the task
   recovers +9 to +16.5pp exactly where the headroom is, and changes nothing at the easy cells
   (L0 p=0.29, L7000-p000 p=0.79). So a cross-attention arm that merely matches 0.455 means
   **"the fix is adaptation, not design"** — the outcome issue #6 pre-registered as a real
   possibility. Arm 3 must *beat* 0.455 for the architectural claim to stand.
2. **The distance curve is U-shaped, not monotone** (0.580 / 0.295 / 0.335 / 0.280 / 0.415 at
   pad=4000). P1b sampled only the two endpoints and read them as a monotone decline. Corrected in
   `P1-mechanism.md` and ledger L-035. The *position dependence* conclusion survives; the implied
   monotonicity does not.
3. **The ablated control collapses to a constant `technical`** (200/200), so its 0.250 is a
   balanced-pool artefact, not "reading nothing, guessing evenly". arm1's above-chance needle@END
   score comes from a **minority** of items over a strong `technical` prior.

**Cost / honesty.** ~18 min per self-attention arm at 12 epochs (4848 steps). arm2's train loss was
**still falling at epoch 12** (1.414 → 1.259, train acc 0.383), so 12 epochs is **not converged** —
recorded as a limitation, not hidden.

**Disk incident.** The filesystem hit **100 % full** mid-run (1.6 GB free). I freed space without
touching other projects' data; the H5 agent cleared ~10 GB of re-creatable caches. Training
survived. Head checkpoints are now gitignored and deleted after eval; raw per-item JSONL is kept.

---

## 2026-09-25 · H5 arm3 — FAILED TO TRAIN. H5 unresolved, not refuted.

**Artifact.** `experiments/h5-adapter/predictions/arm3_xattn-seed0.jsonl` · n=200/cell · oracle 0.2500

| cell | arm1 frozen | **arm2** fine-tuned | arm3 cross-attn | Δ a2→a3 | p |
|---|---|---|---|---|---|
| **L0** | 0.640 | **0.620** | **0.225** | **−0.395** | 2.1e-18 |
| L4000-p000 | 0.580 | 0.590 | 0.270 | −0.320 | 4.6e-16 |
| L4000-p025 | 0.295 | 0.430 | 0.230 | −0.200 | 8.6e-06 |
| L4000-p050 | 0.335 | **0.500** | 0.250 | −0.250 | 1.4e-08 |
| L4000-p100 | 0.415 | 0.495 | 0.255 | −0.240 | 4.5e-09 |
| **L7000-p100** | 0.365 | **0.455** | **0.240** | **−0.215** | 1.8e-06 |

**Arm 3 scores below chance at L0** — a 64-token sequence with no long-range problem whatsoever.
A module that fails *there* has not learned the task, so this cannot be read as evidence against
cross-attention.

**Root cause, from the committed training record.** arm2 trains `head.layers.*.self_attn.*`;
arm3 trains `head.layers.*.cross_attn.*` — a **randomly initialised** module replacing a
**pre-trained** one, with the identical recipe (lr 5e-4, 12 epochs, 4848 steps, same warmup).
Arm 2 starts from the shipped solution; arm 3 starts from noise. **The comparison was unfair by
construction**, and neither arm was converged (arm2 train loss still falling at epoch 12:
1.414 → 1.259, train acc 0.383).

**What stands: arm 2 is the strong baseline.** Fine-tuning the shipped head on the task (encoder
frozen) buys **+9.0pp at the primary cell (p=0.0021)** and **+16.5pp mid-document (p=3.6e-08)**,
with **no change at the easy cells** — exactly the headroom-only signature the mechanism predicts.
That is a real, significant, pre-registered intervention.

**Next (the loop).** Re-run arm 3 with an init-fair design:
1. **Keep the pre-trained self-attention and add cross-attention in parallel**, with the
   cross-attention `out_proj` **zero-initialised**, so at step 0 arm3 ≡ arm2 *exactly*. Any gain
   is then attributable to the architecture, not to initialisation.
2. **Longer schedule / convergence check** — 12 epochs is provably not enough.
3. Report against **0.455**, with the position-only oracle (0.250) beside it.

---

## 2026-09-25 · The baseline is frozen, and it is validated end to end

**New file: `BASELINE.md`** — the single frozen reference for what "the baseline" means. Every
number carries its `n`, its CI, its artifact path and its reproduce command. `AGENTS.md` now points
there first so agents stop re-deriving it.

**End-to-end validation run just now** (`env/venv/bin/python experiments/harness/run.py --recheck`):

| run | result |
|---|---|
| `baseline-power-n200` | **18/18 checks PASS** — 2000 rows, 10 cells, probabilities sum to 1, prediction is argmax, truncation matches the documented rule, **items paired across `max_len`**, **document independent of `max_len`**, label balance 67/66/67 (max share 0.335), deterministic for seed, seed changes documents, doc hashes rebuild from seed |
| `baseline-repro-upstream` | checks PASS |
| `position-sweep` | checks PASS |
| `baseline-repro` | checks PASS |

The `document_independent_of_max_len` check is the load-bearing one: it proves the two budget arms
see the **same document**, so the 1024-vs-8192 comparison is a budget effect and not a
construction artefact.

**Strong baseline, restated.** The shipped checkpoint at n=200/cell, balanced: 0.840 at pad=0
falling to **0.420** at pad=7000 with `max_len=8192`, and flat at **0.335** (= majority rate) for
every pad ≥ 1000 at the shipped `max_len=1024`. Oracle 0.335, random 0.250.

**Strongest verified intervention: arm2.** Fine-tune the shipped decision head, encoder frozen —
**+9.0pp at the primary cell (p=0.0021)**, **+16.5pp mid-document (p=3.6e-08)**, and **no change at
the easy cells**. Caveats recorded: single seed, not converged at 12 epochs.

**Loop status.** Two agents running: `H5b` (issue #11) re-runs the cross-attention arm **init-fair**
— cross-attention added *in parallel* with `out_proj` **zero-initialised**, so at step 0 arm3 ≡ arm2
bit-for-bit and any gain is attributable to the architecture; `V-002` (issue #12) independently
verifies the pipeline end to end and the arm2 claim from raw JSONL.

---

## 2026-09-25 · V-002 — independent verification: everything reproduced, one promotion caveat

**Artifacts.** `findings/V-002.md`, `experiments/V-002/` · issue #12 · verifier did not produce any of it.

### Reproduced, bit-for-bit
- **End-to-end re-run** of arm1 and arm2 from pinned revision `54cff08`: **2200/2200 items, 0
  prediction mismatches, max |Δp| = 0.0.**
- **Split disjointness** re-established by the verifier's own content hashing: 0/96 shared
  templates, 0 shared rendered needles across 800 eval × 1008 train, 0 shared slot values. Longest
  shared word n-gram = 4, and it is pure function words.
- **All 2200 eval documents rebuilt at token level**: `needle_token_start == round(p·pad)`, pad
  exact after re-tokenisation, `state_sha256` matches an independent rebuild. Position-only oracle
  exactly **0.2500** in every grouping. The ablated cell is verified needle-free **by token-id
  subsequence**, not by a flag.
- **arm2 statistics**: +9.00pp at L7000-p100 (exact McNemar p=2.102e-03), +16.50pp at L4000-p050
  (p=3.609e-08), L0 −2.00pp (p=0.2891). **Holm: all 5 positive cells survive** (m=8).
  **The pairing is load-bearing** — the wrong independent test gives p=0.067 at the primary cell
  instead of 0.0021.

### ⚠ The caveat that governs how this may be quoted
**arm1/arm2 bootstrap CIs OVERLAP at both headline cells**: L7000-p100 [0.300, 0.435] vs
[0.385, 0.525]; L4000-p100 [0.345, 0.485] vs [0.425, 0.565]. **Only `L4000-p050` and `L4000-p075`
satisfy BOTH prongs** (non-overlapping CIs *and* a surviving paired test). At the primary cell the
effect satisfies the **paired prong only**, so under our own `roles/README.md` promotion rule it is
**not yet a promoted baseline**.

**arm2 is also single-seed** (seed 0), so protocol §6's "gain must exceed the seed spread" is not
evidenced. Seeds 1–2 retrains are queued.

### What the attacks found
- Position-only oracle is 0.2500 in every cell; slot-value shortcut oracle is at chance.
- No *scored* cell collapses to a single answer (arm1 max modal share 0.96, arm2 0.73); only the
  ablated cell is 1.00.
- **arm1's above-oracle mid-document score is essentially an "always technical" prior** — technical
  accuracy 0.96–1.00, **`other` accuracy 0.00**.
- **arm2's gain is a genuine content gain but partial**: it concentrates in billing (+17..+25
  discordant) and sales (+5..+15) with a small technical loss (−3..−7); its swapped-needle
  follows-content accuracy rises to 0.415 from arm1's 0.275 (chance).

### Defects filed (verifier fixed nothing)
- **#13** `leakage_report.json` c6 is stale (true max Jaccard 0.3158 vs reported 0.2222; gate is
  0.60, so no verdict changes).
- **#14** H5 artifacts depend on an **uncommitted fork worktree**, and `laya/common.py` in it was
  **edited by another agent during verification** (a0425683 → 134bc965). Runs still matched
  bit-for-bit, but the revision is **unpinned** — must be pinned before any PR.
- **#15** the L-038 numbers have **no committed generator** (`stats.py`'s `CANDIDATE_FAMILY`
  excludes arm2), and `stats.py`'s `PRIMARY_CELL` is `L4000-p100` whereas the pre-registered
  primary cell is `L7000-p100`.

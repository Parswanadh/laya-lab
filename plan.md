# plan.md — Long-Context Decision Engine for Laya

**Objective.** Make Laya's non-autoregressive decision engine genuinely consume long
documents (8k–32k tokens) — not by routing around the evidence, but by changing how the
model attends to it.

**Target repo.** `NandhaKishorM/laya` (upstream) · fork `Parswanadh/laya` · lab `Parswanadh/laya-lab`

**Status.** Phase 0 — environment & baseline reproduction.

---

## 1. The measured problem

Laya's `build_sequence` (`laya/common.py:126`) truncates a long state to the remaining
token budget by keeping the **head** of the document:

```python
st = state_ids[max(0, len(state_ids) - room):] if truncate_left else state_ids[:room]
```

`truncate_left` is `False` for strings and dicts. Everything past `room` tokens is dropped
silently and irrecoverably. The head of the prompt is the option list, so the document
budget at the shipped default is:

| checkpoint | `max_len` | `head_max_len` | state budget |
|---|---|---|---|
| `laya` (english) | 512 | 192 | ~320 tok |
| `laya-multilingual` | 1024 | 256 | ~768 tok |

### Upstream's own measurement (`research/results/long_context_multilingual.json`)

20 multilingual requests placed after `pad` tokens of filler, `laya-multilingual`:

| pad tokens | acc @ limit=1024 (shipped) | acc @ limit=8192 | latency @8192 |
|---|---|---|---|
| 0 | 0.95 | 0.95 | 0.016 s |
| 1000 | 0.65 | 0.80 | 0.21 s |
| 2000 | **0.35** | 0.85 | 0.50 s |
| 4000 | **0.35** | 0.90 | 1.71 s |
| 7000 | **0.35** | — | — |

Two facts fall out:

1. **Accuracy flatlines at 0.35** for every document ≥ 2000 tokens. It is not degrading
   gracefully — the deciding span is *gone*, so the answer is constant. 0.35 is the
   model's answer-frequency prior on this 4-option question, not reading comprehension.
2. **Raising `max_len` to 8192 works** (0.85–0.90) but costs 17× the latency (0.2 s → 3.5 s).

> ⚠ **n = 20 per cell.** No confidence interval is reported. A 0.35→0.85 shift survives
> that sample size, but per-cell comparisons between close candidates do not. Fixing the
> statistical power of this evaluation is itself part of the contribution.

## 2. Why it fails — architecture, not just budget

Both encoders are **ModernBERT** with native 8192 RoPE positions:

| | `laya` (english) | `laya-multilingual` |
|---|---|---|
| encoder | ModernBERT (28 L) | mmBERT-base (22 L) |
| hidden | 1024 | 768 |
| `max_position_embeddings` | 8192 | 8192 |
| `local_attention` | 128 | 128 |
| sliding half-window | `local_attention // 2` = **64** | **64** |
| layer types | alternating local/global | 8 global, 14 sliding |
| global layer indices | every 3rd | 0, 3, 6, 9, 12, 15, 18, 21 |
| RoPE theta (full / local) | 160000 / 10000 | 160000 / 160000 |

The encoder is **not** the limit — it was trained at 8192. The prompt layout is:

```
[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 … [SEP] <state> [SEP]
       └───── head, 45 tok measured ─────┘        └──── the document ────┘
```

> ⚠ **CORRECTED (M-001, 2026-09-24).** Two claims that stood here are **refuted**. They are
> recorded rather than deleted, because both were load-bearing and both were wrong:
>
> 1. **"The head is ~256 tokens."** `head_max_len` is an **upper bound, not the head length**.
>    Probed with the shipped tokenizer and question, the option markers land at positions
>    `[12, 20, 29, 39]` and the state begins at **~45**, not ~256. The shipped state budget at
>    `max_len=1024` is therefore **978 tokens, not ~764**.
> 2. **"The marker must cross the document through 8 cross-document hops."** **False.** Layer 0
>    is `full_attention`, so the set reachable from a marker after **one** hop is already
>    `[0, n)`. **The reachability graph has diameter 1.** The minimum number of layers for a
>    nonzero attention path from any state token to any marker is **1, independent of distance**,
>    verified by exact simulation at `n=8192`. Any argument of the form "the marker cannot reach
>    position 7000" is wrong, and no experiment should be built whose only informative outcome
>    requires it to be true.

**What actually binds is dilution, not reachability.** Every state token is reachable from every
marker, but the cross-document *edge budget* is overwhelmingly concentrated: at `n=8192` there
are 24,015,296 (head-query × state-key) edges, **99.9 % of them from the 8 global layers** and
0.1 % from the 14 sliding ones — and the sliding contribution is **constant** at 29,120 edges,
independent of `n`. As the document grows, each additional filler token is another competitor for
a fixed pool of long-range attention mass. A token 7000 positions away is first *read* at layer
12, leaving only 4 of the 8 global layers to mix it (M-001's mixing budget).

This is the mechanism our own measurements support: information **arrives** at the marker
positions (P1b: 0.82 linear probe on held-out templates) while the shipped head returns 0.30. The
constraint is **how much attention mass the evidence can command**, not whether a path exists.

**This is the lever.** The cost is concentrated in the quadratic global layers, and so is the
long-range mixing capacity. An architectural intervention has to break that coupling.


## 3. Why not "just select the right window"

Considered and rejected as the primary contribution. A selector is a **lossy bet**: if it
ranks the wrong span first, the evidence is gone with no signal that anything was lost —
the same silent-failure mode as today's truncation, just with better odds. It also inherits
the routing problem it was meant to solve (a cheap scorer must be right *before* the model
that is actually good at the task has run).

Open upstream PR **#363** (`predict_long`) already occupies the windowed-scan lane, and its
own evidence shows the default window still fails (billing tail → `other`; recovers only at
`window=128`). We do not duplicate it; where a selector is useful it is evaluated as a
**composable front-end**, not as the answer.

## 4. Hypotheses

Each is falsifiable, and each has a pre-registered decision rule.

| id | hypothesis | decision rule | cost to test |
|---|---|---|---|
| **H1** | The 0.35 floor is pure truncation; a correct evidence-placement policy recovers most of the 8192 accuracy at ~1× cost | If head+tail beats head-only by ≥ 0.20 acc at pad=4000, H1 holds; else the bottleneck is capacity, not placement | cheap, no training |
| **H2** | Long-range mixing is bottlenecked by the **8 global layers**, not by the encoder's training | If forcing all layers global at inference raises acc ≥ 0.10 over the same budget, H2 holds | cheap-ish, inference only |
| **H3** | The ±64 sliding window is too narrow to propagate evidence, and widening it recovers accuracy without retraining | Sweep half-window ∈ {64,128,256,512}; a monotone gain ≥ 0.05 at equal budget supports H3 | cheap, inference only |
| **H4** | RoPE scaling (YaRN / NTK-ABF) buys little, because the encoder is natively 8192-trained and the failure is not extrapolation | If YaRN@16384 gains < 0.05 over native@8192 on a needle sweep, H4 holds (**expected negative result**) | cheap, config only |
| **H5** | A **state-encoder + cross-attention head** — document encoded at full resolution, option markers cross-attend to *every* state position — beats flat concatenation at equal or lower FLOPs | If the trained adapter beats the best training-free arm by ≥ 0.05 acc at pad=4000 with non-overlapping bootstrap CIs, H5 holds | **expensive**, needs training |

H1–H4 are training-free and can be settled quickly. H5 is the architectural claim and the
real prize; it is attempted only after H1–H4 establish what the baseline actually is.

## 5. Phases and gates

```
P0  Environment + harness + reproduced baseline          GATE: numbers reproduce here, with CIs
P1  Diagnosis: needle sweep, receptive-field math,     GATE: a written mechanism, not a guess
    per-layer attention analysis
P2  Training-free architectural knobs (H1–H4)          GATE: best arm identified, CIs reported
P3  Architectural fix: state-encoder + cross-attn       GATE: beats best P2 arm, verified
    head adapter (H5), trained on frozen encoder
P4  Independent + adversarial verification              GATE: cross-verifier fails to falsify
P5  PR to upstream + evidence package                   GATE: CI green, PR-clean diff
```

**The loop.** Each phase's verified best becomes the new baseline for the next. A new
baseline is only promoted when it clears the previous one with **non-overlapping bootstrap
CIs** and survives the cross-verifier. Nothing is promoted on a point estimate.

## 6. Evaluation protocol (frozen before any candidate is run)

Defined in `protocol.md` and hashed into every experiment manifest. Summary:

- **Task.** Needle-in-haystack decision: a labelled request with a known answer embedded at
  position `p` in a document of length `L` of domain-matched filler.
- **Sweep.** `L ∈ {0, 1k, 2k, 4k, 7k, 8k, 16k, 32k}` × `p ∈ {0.0, 0.25, 0.5, 0.75, 1.0}`.
- **Languages.** en, es, hi, ja, ar, zh — chosen to span scripts and tokenizer fertility.
- **n ≥ 400 per cell** (upstream used 20). *Corrected by M-001; the original said 200.* At n=400
  the single-proportion 95 % CI half-width at p̂=0.5 is **±0.049** (Wald) / **±0.050** (exact
  Clopper–Pearson). At n=200 it is ±0.069/±0.071.
- **Metrics.** accuracy, macro-F1, ECE, p50/p95 latency, peak VRAM, token throughput.
- **Statistics.** 10k-resample bootstrap CIs; **McNemar's paired test** for candidate-vs-
  baseline (paired by item); Holm correction within a **pre-registered small primary family**.

  > ⚠ **CORRECTED (M-001) — the original power argument was wrong in two ways.**
  >
  > 1. **±0.066 at n≥200 is a *single-proportion* number, not arm-vs-arm resolution.** It says
  >    nothing about whether two arms differ.
  > 2. **The exact-*conditional* McNemar test is conservative** — its power stays in 0.054–0.089
  >    for every (n, δ, q) out to n=1600. **Do not size the study on it and do not quote its
  >    p-values.** Use the exact *unconditional* test.
  >
  > Power is set by **q = P(the two arms disagree on an item)**, not by the effect size δ alone.
  > At δ=0.10, n=400 gives ≥0.90 power for every q ≤ 0.50. But **at δ=0.05, n=400 gives 0.89 at
  > q=0.10, 0.62 at q=0.20, and 0.48 at q=0.30** — so **H3 and H5, both of which use a +0.05
  > threshold, are NOT powered at n=400.** Normal-approximation design numbers: δ=0.05 needs
  > n = 312 / 626 / 1568 for q = 0.10 / 0.20 / 0.50; δ=0.10 needs 77 / 155 / 391.
  >
  > **Consequences, binding on every arm:**
  > - **A pilot that measures `q` in the tail cells (d ≥ 2048) from ~100 items is mandatory**
  >   before sizing the main run. Realized `q` is reported with every comparison.
  > - **A null result on a +0.05 arm at n=400 must be reported as UNDERPOWERED, not as "no
  >   effect".** This is the single easiest way for this program to publish a false negative.
  > - Multiplicity: 46 cells at α=0.05 gives P(≥1 false positive) = 0.72 and inflates the required
  >   n by 2.49× under Bonferroni. Pre-register a **small primary family** (M-001 suggests
  >   d ∈ {64, 2048, 4096, end} × n ∈ {2048, 8192} = 8 comparisons), Holm within it only, and
  >   label everything else exploratory.

- **Leakage guard.** Filler and needle drawn from disjoint pools; needle templates held out
  from any training set; positions and labels balanced so position or majority priors
  cannot beat chance.
- **Baselines that must be reported.** random, majority-class, *position-only oracle*,
  upstream head-truncation at 1024, upstream head-truncation at 8192, `predict_long` (#363)
  where available.

A candidate that does not beat the **position-only oracle** has learned nothing about
content, and the result is reported as such.

## 7. Team and roles

Charters live in `roles/`. Each agent receives **only its own charter** plus an explicit
input/output contract — never the orchestration transcript. That is the context-poisoning
control: an agent that has seen the hypothesis cannot independently verify it, and an agent
that has seen a failed arm tends to avoid reporting it.

| role | owns | must not |
|---|---|---|
| `research` | prior art with citations; what is already known | run experiments or assert results |
| `math` | the formal model, the FLOPs/receptive-field analysis, the sweep design | touch the fork |
| `engineer` | harness, manifests, implementations | grade its own work |
| `verifier` | reproduces a claim from the raw artifact, independently | fix the code it verifies |
| `cross-verifier` | **tries to falsify** — leakage, priors, seeds, statistics | accept a claim it did not attack |
| `scribe` (orchestrator) | `plan.md`, `progress.md`, `LEDGER.md` | — |

**Concurrency: 3–4 agents, hard cap 5.** Machine budget is the limit, not ambition — see §8.

## 8. Machine budget

Measured on this box, not assumed:

```
22 cores · 15 GB RAM · RTX 4070 Laptop 8188 MiB · ~128 GB free disk
```

At the time of writing: **~5.1 GB RAM available**, **~6.8 GB VRAM free**. One agent worker
is ~0.7 GB. Therefore:

- **≤ 4 concurrent agents** (~2.8 GB) with **no** GPU job running simultaneously.
- **Exactly one GPU job at a time** (`CUDA_VISIBLE_DEVICES=0`, `expandable_segments:True`).
- Heavy runs go through `systemd-run --user --scope -p MemoryMax=… -p MemorySwapMax=0`,
  never unbounded. Unbounded `torch.load` has OOM-killed this box 31 times in one session.
- Checkpoints load with `low_cpu_mem_usage=True`; only mmBERT-base (644 MB) is resident by
  default. ModernBERT-large (843 MB) loads alone, never beside it.
- Every long run is `setsid nohup` with the log **inside the lab repo**, never `/tmp`.

Compute is scheduled as a **queue**, not a fan-out: agents write specs and analysis in
parallel (cheap); the GPU stage is serialized through one runner.

## 9. Deliverables

1. `protocol.md` — frozen evaluation, with the statistical power argument.
2. A properly-powered long-context benchmark (`experiments/`), reproducible from manifests.
3. Diagnosis write-up: the mechanism, with per-layer evidence.
4. The architectural intervention, with measured before/after and a negative-results section
   covering H4 and any dead arms.
5. `LEDGER.md` — one row per claim: claim, artifact, who verified, status.
6. A PR-clean branch on `Parswanadh/laya` and a PR to upstream, diff limited to the change.
7. Reproducibility: a single runnable check left behind, per upstream `AGENTS.md`.

## 10. Non-negotiables

- **No number in any deliverable without an artifact path** that regenerates it.
- **Measured vs modelled is tagged explicitly.** Analytic FLOPs are labelled *modelled*.
- **Refuted claims are corrected in place**, not deleted — the negative results are part of
  the contribution.
- **A claim is not promoted until a verifier who did not produce it reproduces it** from the
  raw artifact, and a cross-verifier fails to break it.
- Upstream `AGENTS.md` rules bind all work: no hosted-service dependency, conventional
  commits, no drive-by reformatting, one logical change per commit, `tests/test_hooks_api.py`
  updated with any public API change, CI gates green.

---

# REVISION 2 — after R-001 (prior art) and P1 (mechanism)

R-001 is a 637-line prior-art review with 24 cited sources. It changed the strategy. Both
revisions are recorded here rather than edited into §4, so the reasoning stays auditable.

## What prior art settles before we spend compute

| our hypothesis | prior art says | action |
|---|---|---|
| **H4** RoPE scaling (YaRN/NTK/ABF) helps at ≤8192 | **Ruled out on definitional grounds.** Dynamic scaling is *defined* as identity below the trained length; a fixed factor inside the window causes "a flat reduction of performance" (YaRN authors). YaRN/NTK/LongRoPE cost **3.5–7.6 MMLU points** applied at short lengths on Phi3-mini; LongRoPE2's contribution is *undoing* that damage and it **switches rescaled RoPE off** for in-window inputs. | Run anyway — it is a one-line config change and a *confirmed* negative is citable. Budget: one run. |
| **H3** widening the sliding window helps | **Headroom is tiny and measured.** The one published window ablation on long-doc classification (MIMIC-III, 5 window sizes, each *retrained*): 32→512 tokens of window bought **+0.4 micro-F1** for 2.5× test cost. | Still run: the **inference-time** version of this question appears genuinely unmeasured for any architecture (R-001 §7 gap 2). But if we see a *large* gain, suspect a measurement artefact first. |
| **H2** making all layers global helps | **Two independent negative results.** ModernBERT: global-every-layer "yielded identical downstream performance" to global-every-3rd. Longformer: "adding more tokens using global attention does not improve F1". | Run as the cheap falsification arm. Low prior. |
| — | **mmBERT has never had a sliding-window or global/local ablation published at all** (R-001 §1.9, §7 gap 4). | Our H2/H3 measurements are **novel for this model family**, whatever they show. |

**H4 is now a predicted-negative.** If we run it and it is negative, that is a *confirmation of
prior art on a new model family*, which is worth reporting — and it costs one config line.

## The finding that reshaped the objective

R-001 §4.1 surfaced the closest published analogue to our failure. **Inverted EURLEX** is
constructed by moving a document's decisive section — "the first two sections (header, recitals)
carry the most relevant information" — **to the end**, "particularly challenging for models that
focus only on the first 512 tokens."

Results on it (third-party evaluation, ACL 2022):

| model | Inverted EURLEX |
|---|---|
| BERT, truncate @512 | 70.53 |
| BERT + TextRank selection | 71.30 |
| BERT + **random** chunk selection | **71.47** |
| Longformer @4096 (sparse long-context attention) | **56.47** |
| ToBERT (hierarchical) | 67.31 |
| CogLTX | 70.80 |

**The long-context sparse-attention encoder lost by ~14 points to plain truncation, and the best
models were chunk-*selectors* — including one that selects chunks at random.**

This is the single most important input to our design, and it cuts against the obvious
"just attend further" instinct:

- Simply widening or globalising attention is **not** a promising direction. The literature has
  already run that experiment in a neighbouring setting and it lost.
- What wins is **non-diluted focus**: choose a small number of positions and let the decision
  read them at full strength.
- But selection is lossy, and a wrong selection destroys the evidence silently — the failure mode
  we set out to remove.

### The synthesis this forces

The objective is **not** "attend to more positions". It is:

> read **every** position, but let the decision head **choose what to weight** without the
> softmax mass being spread thin across the whole document.

That is precisely what a **learned query cross-attending over all encoded state positions**
does. It keeps selection's focus (the query learns where to look) without selection's lossiness
(every position is available, and the mechanism is differentiable end-to-end).

**This reframes H5 from "a bigger attention window" into the actual contribution.**

## Revised H5 — Decoupled State Encoder + Cross-Attention Decision Head

```
Stage 1  state encoder (FROZEN mmBERT)
         D (n tokens) → H_D ∈ R^(n×768)          full bidirectional context, native 8192
                                                  chunked w/ overlap beyond 8192
Stage 2  decision tokens (small transformer, the existing head)
         [CLS] type instr [SEP] [MASK] o1 … [MASK] ok [SEP] → H_M ∈ R^(m×768),  m ≈ 256
Stage 3  CROSS-ATTENTION  (NEW — the architectural change, ~2 layers, 4 heads)
         Q = W_q H_M ;  K = W_k H_D ;  V = W_v H_D
         H_M ← H_M + MHA(Q, K, V) + FFN(·)
Stage 4  scoring (UNCHANGED) — gather at marker positions → scorer → logits
```

**Why it addresses the measured mechanism.** In the flat layout a marker reaches a distant state
token only through the **8 global layers**, and every filler token competes for its attention
mass. Here every state token is a K/V pair, so each marker reaches **all n positions in one hop**,
and the query is **trained for this task** rather than being a by-product of 22 self-attention
layers. Focus without lossiness.

**Marginal cost.** Stage 1 dominates and is unchanged. Stage 3 is `m·n·d ≈ 256 × 8192 × 768 ≈
1.6 GFLOP` per layer — on the order of **1–2 % of the encoder's cost**. Sub-quadratic in the
decision path. *(modelled, to be measured.)*

**Training.** Stage 1 frozen; gradients flow through Stage 3 only (~3 M params). Fits the 8 GB
card at 4096 tokens with batch 2–4 and the encoder under `no_grad`. Feasible here.

**Honest risks, registered before the run.**
1. Training data will be **synthetic** needle-in-haystack. Transfer to real documents is
   **unproven** and will be labelled as such everywhere.
2. The adapter could learn a **position heuristic** instead of reading content. The
   cross-verifier's **position-only oracle** attack exists precisely to catch this, and it is
   mandatory on this arm.
3. R-001's own warning: absence of prior art is not support. Tight falsification budget.

## The second contribution — a benchmark that does not exist

R-001 §7 gap 5: **"No 'position of decisive evidence vs accuracy' study for encoders with a
decision head."** *Lost in the Middle* and both *Found in the Middle* papers are decoder-only.

Our P1 diagnostic already produces this curve. Done properly — balanced labels, `n ≥ 200`,
length held fixed, distance swept, with random/majority/position-oracle baselines and McNemar —
it is a **standalone contribution to the encoder literature**, independent of whether H5 wins.

This matters for the PR strategy: even if the architectural arm does not beat the baseline, a
correctly-powered position-sensitivity benchmark for decision encoders is a mergeable artifact.
That de-risks the whole program.

## Consequence for the PR

The upstream contribution becomes:

1. a **position-sensitivity benchmark harness** for decision encoders (the gap in the literature),
2. the **measured mechanism** with the controls that isolated it, and
3. the cross-attention adapter **if and only if** it clears the frozen baseline with
   non-overlapping CIs, a surviving McNemar test, independent reproduction, and a failed
   falsification attempt.

Item 1 and 2 are deliverable regardless of item 3.

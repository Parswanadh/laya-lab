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
       └──────────── head, ~256 tok ────────┘         └── the document ──┘
```

The option markers sit at the **start**; the evidence sits **after** them. For a marker to
be influenced by a state token, the information must cross the sequence, and in this
architecture it can only do so in the **8 global layers**. The 14 sliding layers bridge
±64 tokens. So the marker's final representation aggregates an 8192-token document through
8 cross-document hops, while the O(n²) cost of those 8 global layers is what makes
`max_len=8192` 17× slower.

**This is the lever.** The cost is concentrated in a handful of quadratic layers; the
long-range mixing capacity is concentrated in the same handful. That tension is what an
architectural intervention has to resolve.

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
- **n ≥ 200 per cell** (upstream used 20). Power: ±0.066 at 95% for a proportion near 0.5.
- **Metrics.** accuracy, macro-F1, ECE, p50/p95 latency, peak VRAM, token throughput.
- **Statistics.** 10k-resample bootstrap CIs; **McNemar's paired test** for candidate-vs-
  baseline (paired by item, so it is far more sensitive than comparing two CIs); Holm
  correction across the sweep.
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

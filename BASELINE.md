# BASELINE.md — the frozen reference

**Status: FROZEN 2026-09-25.** This is the strong baseline the program measures against. Every
number here is `measured` on this machine, with its `n`, its artifact path and its reproduce
command. Nothing in this file is modelled or inherited from upstream without being labelled.

Machine: RTX 4070 Laptop (8188 MiB) · 22 cores · 15 GB RAM · torch 2.9.1+cu128 · transformers 4.57.6
Checkpoint: `convaiinnovations/laya` `multilingual/` @ `55cf4c4` (mmBERT-base, 22 layers, 8 global,
`max_position_embeddings=8192`, `local_attention=128`)
Shipped config: `max_len=1024`, `head_max_len=256` (**an upper bound** — the real head is **45** tokens)

---

## 1. Headline: the reference table (n = 200/cell, label-balanced)

`experiments/baseline-power-n200/` · **all 18 structural + mechanism checks PASS** (`run.py --recheck`)
Majority class = position-only oracle = **0.335** · random over 4 options = **0.250**

| pad tokens | `max_len=1024` (shipped) acc [95 % CI] | `max_len=8192` acc [95 % CI] | paired McNemar |
|---|---|---|---|
| 0 | **0.840** [0.785, 0.890] | **0.840** [0.785, 0.890] | p = 1 |
| 1000 | **0.325** [0.260, 0.390] | **0.705** [0.640, 0.765] | 3.8e-18 |
| 2000 | **0.335** [0.270, 0.400] | **0.670** [0.605, 0.735] | 2.2e-18 |
| 4000 | **0.330** [0.270, 0.395] | **0.495** [0.425, 0.565] | 1.0e-08 |
| 7000 | **0.335** [0.270, 0.400] | **0.420** [0.355, 0.490] | 7.6e-05 |

**All four `pad ≥ 1000` comparisons survive Holm–Bonferroni.** CIs on the two budgets do not
overlap at pads 1000 and 2000.

### The two failure regimes — never report them as one number
| regime | what happens | mechanism |
|---|---|---|
| `max_len=1024`, pad ≥ 1000 | accuracy sits at **0.325–0.335**, i.e. exactly the majority rate. `request_tokens_kept = 0.00` in every collapsed cell; the model emits one constant label (modal share 0.98–0.99). | **Truncation.** The head is 45 tokens, so the state window is 978. The request is simply not in the input. **No attention-level intervention can recover absent tokens.** |
| `max_len=8192` | accuracy still falls **0.840 → 0.420** as the document grows, with the whole request present (`request_tokens_kept = 1.000`). | **Dilution.** Length effect at 8192, paired: 1000→2000 p=0.435 (flat); **2000→4000 p=1.2e-06 (real drop)**; 4000→7000 p=0.032 (does not survive Holm). |

**`pad=7000 @ max_len=8192` = 0.420 is the primary cell**: truncation is excluded by construction,
so it is where an aggregation-level intervention can actually show an effect. Oracle 0.335.

## 2. Upstream-faithful reproduction (n = 20/cell)

`experiments/baseline-repro-upstream/` · **checks PASS** · transcribed from upstream's own
`bench_long_context.py`, verified **token-identical**, 160/160 `input_tokens`.

| pad | `max_len=1024` upstream → ours | `max_len=8192` upstream → ours |
|---|---|---|
| 0 | 0.95 → **0.95** | 0.95 → **0.95** |
| 1000 | 0.65 → **0.65** | 0.80 → **0.80** |
| 2000 | 0.35 → **0.35** | 0.85 → **0.85** |
| 4000 | 0.35 → **0.35** | 0.90 → **0.95** |
| 7000 | 0.35 → **0.35** | 0.40 → **0.40** |

Every default-`max_len` cell exact; 8192 matches 4/5 exactly, the fifth one item apart.
**The baseline gate is satisfied.**

> ⚠ **The quoted `0.85–0.90` ceiling is pool-dependent.** On upstream's texts 8192 reaches
> 0.80–0.95; on the balanced 9-language pool it reaches only **0.495–0.84**. **Set thresholds from
> the matched pool, never from upstream's 0.90.**

## 3. Position sweep at fixed length (`experiments/position-sweep/`, n = 200/cell)

At `pad=4000`, `max_len=8192`, needle swept through the document: **the curve is U-shaped, not
monotone** — arm1 at pad=4000 scores **0.580 / 0.295 / 0.335 / 0.280 / 0.415** for positions
0.00/0.25/0.50/0.75/1.00. Interior cells sit **below** the end. *(Two-point samples cannot
distinguish "decays with distance" from "high at both edges, low between" — this corrected an
earlier reading of ours.)*

At `max_len=1024` it is a **cliff, not a gradient**: 0 wrong→right / 9 right→wrong, p=0.0039.

## 3b. Programme headline: **the fix is adaptation, not design**

Four architectural interventions were tested and all failed — RoPE scaling (identity in-window by
construction), sliding-window widening, all-layers-global, and a cross-attention aggregation head.
The last is the most informative: built **init-fair** so it starts bitwise-identical to the fine-tuned
baseline, it trains, it is harmless at short context, it beats the position-only oracle at every cell
— and its own **ablation shows the added attention does not carry the decisions**. The one
intervention that works is **fine-tuning the shipped head**.

This is a **mechanistic negative**, not an inconclusive one, and it is the pre-registered outcome.

## 4. The strongest verified intervention

**arm2 — fine-tune the shipped decision head, encoder frozen.** `experiments/h5-adapter/`
n=200/cell, balanced, paired McNemar against the frozen head:

| cell | arm1 frozen | **arm2 fine-tuned** | Δ | p |
|---|---|---|---|---|
| L0 | 0.640 | 0.620 | −2.0pp | 0.29 |
| L4000-p025 | 0.295 | 0.430 | +13.5pp | — |
| L4000-p050 | 0.335 | **0.500** | **+16.5pp** | 3.6e-08 |
| L4000-p075 | 0.280 | 0.425 | +14.5pp | — |
| L4000-p100 | 0.415 | 0.495 | +8.0pp | 0.0025 |
| L7000-p000 | 0.595 | 0.585 | −1.0pp | 0.79 |
| **L7000-p100 (primary)** | 0.365 | **0.455** | **+9.0pp** | **0.0021** |

The gain appears **only where the headroom is** and is absent at the easy cells — the signature the
mechanism predicts. *Caveats: single seed (0); 12 epochs is provably not converged (train loss still
falling, 1.414 → 1.259); control `arm2_random_init` not yet run.*

## 5. Refuted — do not spend compute here again

| intervention | verdict | decisive evidence |
|---|---|---|
| **RoPE scaling** (YaRN / NTK / ABF / PI) | **Ruled out on definitional grounds** | HF clamps `seq_len` up to `max_position_embeddings` *before use*, so `dynamic` is the **identity** at ≤8192 for any factor; the other branches rescale in-window positions unconditionally. YaRN/NTK/LongRoPE cost **3.5–7.6 MMLU points** applied at short lengths. |
| **Widening the sliding window** | **REFUTED** | No dose–response; the n=20 hint **reverses to −0.158 (p=0.0094)** at n=120; the *narrowing* ablation moves the same cell by the same amount. Widening to 1024 **collapses** pad=4000 (0.600 → 0.150). |
| **All layers global** | **REFUTED, both halves** | **0.000 at pad=1000**; 0.333 at pad=7000 = exactly the majority rate (p=7.5e-9), all 120 items `technical`. Cost is only **+4–6 %** latency — measured dense masks mean the "quadratic" warning never materialised. |
| **Cross-attention head (arm3, 1st attempt)** | **UNRESOLVED — the arm did not train** | 0.225–0.290 **at every cell including L0**. It replaced *pre-trained* self-attention with *randomly initialised* cross-attention under the identical recipe. Not a fair test. |
| **Cross-attention head (arm3r, init-fair re-run)** | **NO SIGNIFICANT GAIN — and the branch is INFERENCE-INERT** | Init-fair by construction: **bitwise identical to arm2 at step 0** (`max_abs_logit_delta = 0.0`, 2200 items), 40 epochs, probed branch LR. Primary cell **0.485 vs arm2long 0.410 (+7.5pp, p=0.086, Holm 1.0)** and vs arm2 0.455 (**+3.0pp, p=0.512**, CIs overlap). **The decisive control:** the same trained checkpoint with `out_proj` re-zeroed *after* training differs by **0/200 at L0, 3/200 at the primary cell, ≤5/200 anywhere** — the branch is live in training and **unused at inference**. The two Holm-surviving cells are won by the **ablated twin** by the same margin ⇒ **training-trajectory effects, not architecture**. |
| **Precision (bf16 vs fp16 vs fp32)** | **REFUTED** | bf16 = fp16 = fp32 = 0.600 to three decimals at pad=4000/limit=8192. fp32 also costs 3.3× latency for nothing. |

## 6. Reproduce

```bash
cd /home/parshu/projects/contri/laya-lab
# artifact-only re-validation (tokenizer only, no GPU, ~1 min)
env/venv/bin/python experiments/harness/run.py --recheck experiments/baseline-power-n200
# re-derive every table and paired test from raw JSONL, no model/GPU
env/venv/bin/python experiments/harness/tables.py experiments/baseline-power-n200
```

All GPU runs took `flock -n .gpu.lock`; logs are inside the repo, never `/tmp`.

## 7. Verified vs candidate

| | status |
|---|---|
| The n=20 upstream-faithful reproduction | `reproduced` — token-identical, cell-for-cell |
| The n=200 powered table | **structural + mechanism checks PASS**; independent per-item verification in flight |
| arm2's gain | `candidate` — single seed, not converged, no independent verification yet |
| The 1024 = truncation mechanism | `reproduced` — `request_tokens_kept = 0.00`, constant answers, state-hash proofs |
| The 8192 = dilution mechanism | `candidate` — supported by the paired length effect at 8192 and the H5 position sweep |
| The U-shape of the position curve | `candidate` — n=200, one item set |
| All refutations in §5 | `reproduced` with liveness assertions and falsifying controls |

## 8. Known weaknesses, stated

1. **arm2 is single-seed and not converged.** No seed spread is measured; a null or a gain smaller
   than the seed spread would not be interpretable.
2. **n=20 tables are ±0.22** and none of their balanced-pool budget comparisons survives Holm. Quote
   §1, not §2, for any claim.
3. **Synthetic task.** Needle-in-haystack with repetitive filler is a *mechanism probe*. It is **not**
   evidence about real long documents; repetition likely exaggerates dilution relative to natural text.
4. **Construction sensitivity is as large as the effects being chased.** Upstream's document rule
   scores 0.850 at pad=4000 vs 0.600 for the exact-token rule. **Every claim names its construction.**
5. **Only one label-draw seed** (20240924) for the powered run. Whole-run repeatability is verified
   identical (200/200 rows), but seed spread across draws is not measured.

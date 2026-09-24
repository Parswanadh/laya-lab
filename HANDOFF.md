# HANDOFF.md — complete state, 2026-09-25

**Read this after `AGENTS.md` and `BASELINE.md`.** `BASELINE.md` is the numbers; this is the state.

Repo: **`Parswanadh/laya-lab`** (public) · Fork: **`Parswanadh/laya`** · Upstream: `NandhaKishorM/laya`
Fork branch under test: **`exp/h5-adapter` @ `99020ba`** (2 commits ahead of `upstream/main`)

---

## 1. What was asked and what was delivered

The goal was to make Laya's non-autoregressive decision engine work at **high context**, by changing
the architecture rather than routing around the evidence, and to contribute the result upstream.

## 2. The baseline is reproduced, frozen and validated

`BASELINE.md` is the single reference. Headline, **n = 200/cell, label-balanced**, majority = oracle
= 0.335:

| pad | `max_len=1024` (shipped) | `max_len=8192` |
|---|---|---|
| 0 | 0.840 | 0.840 |
| 1000 | 0.325 | 0.705 |
| 4000 | 0.330 | 0.495 |
| 7000 | 0.335 | **0.420** ← primary cell |

**Reproduced upstream exactly** on upstream's own transcribed texts (token-identical, 160/160
`input_tokens`): 0.95/0.65/0.35/0.35/0.35 at the shipped default, matching all five pads.

**Validated end to end**: `run.py --recheck` passes **18/18 structural + mechanism checks** on the
powered run, including the load-bearing `document_independent_of_max_len` and
`items_paired_across_max_len`.

**Two distinct failure regimes** — never to be reported as one number:
- `max_len=1024`, pad ≥ 1000 → **truncation**. `request_tokens_kept = 0.00`; the request is not in
  the input. No attention-level intervention can recover absent tokens.
- `max_len=8192` → **dilution**. The whole request is present (`request_tokens_kept = 1.000`) and
  accuracy still falls 0.840 → 0.420 with length (2000→4000 paired p=1.2e-06).

## 3. What the program proved — including three of our own claims being wrong

| finding | status |
|---|---|
| **RoPE scaling** (YaRN/NTK/ABF/PI) | **Ruled out by construction** — HF clamps `seq_len` to `max_position_embeddings` before use, so `dynamic` is the identity in-window for any factor; other branches rescale in-window positions unconditionally. |
| **Sliding-window widening** | **REFUTED** with liveness proofs — n=20 hint reverses to **−0.158 (p=0.0094)** at n=120; the *narrowing* ablation moves the cell equally; w1024 collapses pad=4000 to 0.150. |
| **All-layers-global** | **REFUTED both halves** — **0.000 at pad=1000**; 0.333 at pad=7000 = exactly the majority rate (p=7.5e-9), 120/120 predicted `technical`. Cost only +4–6 % (dense masks mean the "quadratic" warning never materialised). |
| **bf16 vs fp16 vs fp32** | **REFUTED** — identical to three decimals. |
| **"8 cross-document hops"** (ours) | **REFUTED** by M-001 — layer 0 is full attention, so the reachability graph has **diameter 1**. What binds is *dilution*, not reachability. |
| **"The 0.35 floor is a long-context result"** (ours) | **REFUTED** by V-001 — the four `limit=1024` rows are **one request-free measurement repeated**; 0.35 is just the `technical` share of the draw, *below* the trivial 0.45 baseline. |
| **"Our 8192 arm degrades far below upstream's"** (ours) | **REFUTED** by V-001 — our construction was confounded five ways. Withdrawn; the like-for-like re-run matches upstream. |
| **The distance curve is monotone** (ours) | **REFUTED** — it is **U-shaped** (0.580 / 0.295 / 0.335 / 0.280 / 0.415 at pad=4000). Two endpoints cannot distinguish a decay from edge effects. |

**Where the information dies: not the encoder.** A linear probe on the *marker positions* — exactly
what the shipped scorer reads — recovers the label at **0.820** on **held-out templates** with the
needle 4000 tokens away, while the shipped head returns **0.300**. A `random_pos` control sits at
0.44 (chance floor 0.25), so the probe reads the features, not a global artifact.

## 4. The strongest verified intervention

**arm2 — fine-tune the shipped decision head, encoder frozen.** Verified **bit-for-bit** by an
independent re-run (2200/2200 items, 0 mismatches, max |Δp| = 0.0):

| cell | arm1 frozen | arm2 seed 0 | arm2 seed 1 | seed spread | Δ vs arm1 |
|---|---|---|---|---|---|
| **L7000-p100** | 0.365 | 0.455 | 0.505 | 5.0pp | **+9.0 / +14.0pp** |
| **L4000-p050** | 0.335 | 0.500 | 0.590 | 9.0pp | **+16.5 / +25.5pp** |
| L0 *(control)* | 0.640 | 0.620 | 0.635 | 1.5pp | −2.0 / −0.5pp |

The gain appears **only where the headroom is** and is absent at the easy cells — the signature the
mechanism predicts. **The gain exceeds the seed spread at both headline cells.**

⚠ **Not yet a promoted baseline.** At the two headline cells the arm1/arm2 bootstrap CIs **overlap**;
only `L4000-p050`/`p075` satisfy *both* the non-overlap prong and the paired test. Only **2 seeds**
against the protocol's ≥3.

## 5. In flight when this was written

**`H5b` (issue #11) — the init-fair cross-attention re-run**, holding the GPU lock and running.
The first attempt replaced *pre-trained* self-attention with *randomly initialised* cross-attention
under the identical recipe, and collapsed to 0.225–0.290 **at every cell including L0** — a training
failure, not a verdict. The re-run adds cross-attention **in parallel** with a **zero-initialised
`out_proj`**, so at step 0 arm3 ≡ arm2 bit-for-bit and any gain is attributable to the architecture.
Pre-registered decision rule in `findings/E-004.md`. **The bar is 0.455.** If arm3r ≈ 0.455 the honest
conclusion is *"the fix is adaptation, not design"* — pre-registered as a real outcome.

**Also queued:** `arm2long_shipped_init` (schedule-matched control) and a third arm2 seed.

## 6. The PR

Branch `exp/h5-adapter` @ `99020ba`, **all gates green**:

```
laya/common.py                     | 210 +++++++++++++++++--
tests/test_cross_attention_head.py | 286 ++++++++++++++++++++++
```
| gate | result |
|---|---|
| upstream `ruff` line | **All checks passed** |
| `compileall laya/ tests/` | **OK** |
| `tests/test_cross_attention_head.py` | **31 passed, 0 failed** |
| `test_hooks_api` / `test_router` / `test_criteria` / `test_hooks` | **4/4 PASS** |

The new head is **selectable**, so upstream's default path is untouched.

**Do not open the PR until arm3r resolves.** If the architecture does not beat 0.455, the mergeable
contribution is the **benchmark + negative results** (a position-sensitivity harness for decision
encoders is a genuine gap in the literature — no such study exists for an encoder with a decision
head), not the head.

## 7. Defects still open

| # | defect | impact |
|---|---|---|
| #13 | `leakage_report.json` c6 stale (true max Jaccard 0.3158 vs reported 0.2222; gate 0.60) | no verdict changes |
| #14 | H5 artifacts depended on an **uncommitted worktree** | **materially resolved** — branch now committed and pinned at `99020ba` |
| #15 | L-038 numbers have **no committed generator**; `stats.py` `PRIMARY_CELL` = `L4000-p100` disagrees with the pre-registered `L7000-p100` | must be fixed before the PR |
| L-043 | arm2 has **2 seeds, protocol wants ≥3** | narrowed, not closed |

## 8. Infrastructure lessons (worth keeping)

1. **An agent that dies holding `.gpu.lock` strands it permanently.** V-002's orphans blocked H5b
   until swept. Check `fuser .gpu.lock` and `/proc/<pid>/cmdline` for PIDs whose agent is gone.
2. **`git add -A` causes cross-agent contamination.** Scope adds to your own paths.
3. **The disk hit 100 % full** and killed a run. Keep large intermediates off disk; checkpoints are
   regenerable and must be deleted after eval; raw per-item JSONL is the artifact to keep.
4. **Two-point samples cannot establish curve shape.** That error cost us one corrected claim.
5. **A control that scores at chance on a *balanced* cell may still be collapsing to a constant.**
   Report the modal-prediction share, not just the accuracy.

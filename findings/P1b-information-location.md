# P1b — Where the information dies: the encoder, or the head?

**Status:** `candidate` — measured here, not yet independently verified.
**Artifact.** `experiments/orch-probe-d/results.json` · **Script.** `experiments/orch-probe-d/run.py`
**Device.** RTX 4070, CUDA, fp32 · **n = 200 per condition** (50/class, 4 balanced classes)
**Condition.** Document = 4000 filler tokens; needle at the **end** (the failing regime) or the
**start**. Median sequence length **4066 tokens**, scored at `max_len=8192`.

This experiment decides whether an architectural change to the **head** can work at all, or
whether the encoder is the thing that must change.

## Method

Run the frozen encoder once per item. Extract four candidate feature sets and fit a **closed-form
linear probe** (ridge on one-hot targets, 5-fold CV) to each:

| feature | what it is |
|---|---|
| `marker` | mean-pool of encoder output at the option `[MASK]` marker positions — **exactly what the shipped `scorer` reads** |
| `cls` | encoder output at position 0 |
| `state_mean` / `state_max` | mean / max pool over all state positions |
| `random_pos` | **control** — same number of positions drawn at random |

Two CV modes: random folds, and **grouped folds by needle template**, so no test item shares a
surface form with any training item. The grouped column is the only one that means anything; the
random-fold column is reported to show how much a probe can cheat without grouping.

## Result

| feature | needle@END random-CV | **needle@END grouped** | needle@START random-CV | **needle@START grouped** |
|---|---|---|---|---|
| `marker` | 1.000 | **0.820** | 1.000 | **0.845** |
| `cls` | 1.000 | 0.885 | 1.000 | 0.905 |
| `state_mean` | 1.000 | 0.895 | 1.000 | 0.895 |
| `state_max` | 1.000 | 0.895 | 1.000 | 0.975 |
| `random_pos` *(control)* | 0.550 | **0.440** | 0.545 | **0.475** |
| **shipped head** | — | **0.300** | — | **0.585** |
| random / majority | 0.250 | 0.250 | 0.250 | 0.250 |

**The control behaves.** `random_pos` sits at 0.44–0.48 against a 0.25 chance floor — a modest
lift, consistent with a small amount of global topic signal leaking into any pooled vector, and
nowhere near the 0.82–0.90 of the real features. The probes are reading the features, not a
spurious whole-sequence signal.

**The load-bearing number is the pair in bold.** The `marker` representation — the one the shipped
head actually consumes — carries the answer at **0.820** accuracy on **unseen templates**, while
the shipped head reading those same positions returns **0.300**.

## Reading

**The encoder is not the bottleneck. The extraction is.**

The frozen encoder delivers label-recoverable information to the marker positions at 82 % on
held-out templates, with the needle 4000 tokens away. A *linear* probe recovers it. The shipped
head — a strictly more powerful 2-layer transformer plus an MLP scorer over the same positions —
returns 30 %.

This is consistent with everything else measured so far:

- It explains why P1's failure is positional rather than length-driven: the information *arrives*;
  what degrades with distance is the head's ability to act on it.
- It predicts that **encoder-side attention surgery (H2/H3) has little to win**, matching R-001's
  prior art independently: an 8× wider window bought +0.4 micro-F1 in the one published ablation,
  and ModernBERT reports global-every-layer ≡ global-every-3rd.
- It gives the architectural intervention (H5) a concrete target: **the aggregation path**, not
  the encoder.

## The confound, stated plainly

**A probe fitted on the evaluation distribution is not a fair like-for-like comparison against a
frozen head trained on a different task.** The probe is trained on these 4 classes with these
labels; the shipped head has never seen this task. So 0.82 is **not** "the head should score 0.82".

What the number legitimately establishes is an **upper bound on what a retrained aggregation path
could reach**, and therefore that retraining the aggregation path is where the headroom is. It
does not establish that cross-attention specifically beats a simpler retrained head.

**That distinction is the whole design of the next experiment.** H5 must be run against a
**fine-tuned baseline head**, not against the frozen one:

| arm | what it isolates |
|---|---|
| **1. frozen head** (0.300 measured) | the shipped baseline |
| **2. fine-tuned head**, same architecture, same data, encoder frozen | *is fine-tuning alone sufficient?* — **the control that stops H5 claiming credit for training** |
| **3. cross-attention head** (H5), matched parameter budget | *does the architecture add anything over arm 2?* |

If arm 2 ≈ arm 3, the architecture is **not** the win and the honest conclusion is that the fix
is adaptation, not design. That outcome is pre-registered here as a real possibility.

## Answer to R-001's framing question (W4)

R-001 asked whether the 1024-token failure is **truncation** (evidence outside the processed input)
or **reachability** (evidence inside the input but unreachable by the marker), warning that the
answer determines whether mask-level interventions are relevant at all.

**Both are real, and they occur at different budgets.** This is measured, not inferred:

| budget | what happens | mechanism |
|---|---|---|
| `max_len=1024` | accuracy **flat 0.35** for every document ≥ 1000 tokens | **truncation.** The state budget is ~764 tokens, so a 4000-token document loses its tail entirely — including the needle. No mask-level intervention can help this. |
| `max_len=8192` | nothing truncated (median 4066 < 8192), yet accuracy falls **0.90 → 0.45** as the needle moves from the start to the end | **reachability.** The evidence is inside the processed input and still not usable. |

The P1 probe-B/C controls establish the second row: at a fixed 4000-token document the answer
changes with position alone (0.90 at the start, 0.60 at the end) while nothing is truncated.

**So R-001's items 3, 4 and 7 are on point** — for the `max_len=8192` regime. And for the
`max_len=1024` regime the honest conclusion is that **no attention-level intervention can recover
it**, because the tokens are not there. Those two regimes must never be reported as one number.

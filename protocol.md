# protocol.md — FROZEN evaluation protocol

**Status: FROZEN as of 2026-09-24.** Changing this file to make an arm look better is the one
unforgivable act in this program. If the protocol is wrong, open an issue, record the defect,
and version the change as `protocol-v2` — running both on every arm so the change is visible.

This protocol is what makes our numbers comparable to upstream's and to each other.

---

## 1. Task

**Needle-in-haystack decision.** A short labelled request with a known answer is embedded at
position `p` inside a document of length `L` of domain-matched filler. The model must return
the correct label. Nothing else in the document supports any label, so a model that cannot
find the needle can only guess from its prior.

This is a **mechanism probe**, not a substitute for real long-document evaluation. It is
honest precisely because it is controlled: we know exactly where the evidence is. Every claim
derived from it must be labelled *synthetic needle-in-haystack*.

## 2. Sweeps

| axis | values | purpose |
|---|---|---|
| document length `L` (filler tokens) | 0, 1000, 2000, 4000, 7000, 8192, 16000, 32000 | length curve |
| needle position `p` (fraction) | 0.00, 0.25, 0.50, 0.75, 1.00 | distance curve at **fixed** `L` |
| `max_len` | checkpoint default, 8192, and the arm's own setting | budget effect |
| language | en, es, hi, ja, ar, zh | script + tokenizer fertility |

`L=0` is the **short-context regression cell**. Every arm reports it. An arm that gains at
`L=7000` and loses at `L=0` is not shippable, however large the gain.

Not every arm needs the full grid — but every arm must report `L=0`, `L=4000`, `L=7000`, and
must state which cells it skipped and why.

## 3. Sample size and power

`n` per cell **≥ 200** for any number that will be quoted as a result.

Justification: for a proportion near 0.5, the 95% CI half-width is ≈ `1.96·sqrt(0.25/n)`:

| n | 95% CI half-width |
|---|---|
| 20 (upstream's) | **± 0.219** |
| 50 | ± 0.139 |
| 100 | ± 0.098 |
| 200 | **± 0.069** |
| 400 | ± 0.049 |

At `n=20` a single cell cannot distinguish 0.50 from 0.72. Upstream's headline comparison
(0.35 vs 0.85) survives that, but no comparison between *close* candidates does — and close
candidates are exactly what we will be choosing between.

**Staged runs are permitted and encouraged:** a coarse sweep at `n=20` to localise the
interesting region, then `n≥200` on the cells that separate the arms. Every reported cell
states its own `n`. A coarse cell is never mixed into a fine table without the label.

## 4. Label balance

The evaluation set must be **label-balanced**. Report the label counts beside every number.

Rationale: upstream's 20-item set contains `billing` 9, `technical` 7, `sales` 4, `other` **0**
— a majority-class baseline of **0.450**. A failing model that always answers `billing` scores
0.45. Any arm reported on an imbalanced set must carry the majority-class figure next to it.

## 5. Metrics

| metric | note |
|---|---|
| accuracy | headline |
| macro-F1 | balance-robust; required whenever the set is not perfectly balanced |
| **majority-class accuracy** | **mandatory reference on every table** |
| random | `1/n_options` |
| **position-only oracle** | best achievable by answering from needle position alone |
| ECE | calibration; the shipped temperatures are unfitted, so report raw and fitted |
| p50 / p95 latency | ms, warm, single process, GPU lock held |
| peak VRAM | GB |
| tokens/sec | batched throughput where relevant |

**The position-only oracle is not optional.** If needle position correlates with label in the
generated data, a model can score above chance without reading anything. Any candidate that
does not beat the position-only oracle has learned nothing about content and must be reported
in exactly those words.

## 6. Statistics

- **Bootstrap CIs**, 10,000 resamples, percentile method, seeded. Report for every quoted number.
- **McNemar's paired test** for candidate-vs-baseline. Our comparisons are **paired by item** —
  the same items go to every arm — so McNemar is far more sensitive than comparing two
  independent CIs. Using two-proportion tests on paired data is a protocol violation.
- **Holm–Bonferroni** correction across each sweep's family of comparisons.
- **Seed spread.** Run each arm at ≥3 seeds. A gain smaller than the seed spread is not a gain.
- Report effect size, not only significance.

## 7. Baselines every arm is measured against

1. random (`1/n_options`)
2. majority class
3. position-only oracle
4. upstream head-truncation at the checkpoint default `max_len`
5. upstream head-truncation at `max_len=8192`
6. `predict_long` (upstream PR #363) where available, at its default window and at `window=128`

## 8. Determinism and hygiene

- Seeds fixed and recorded in the manifest. `torch.use_deterministic_algorithms` where it does
  not cost more than 2× — if it does, record that it was not enabled.
- **Raw per-item predictions are always written**, never only aggregates. Every statistic above
  is recomputed from them by the verifier.
- Warm-up: ≥3 discarded forward passes before any timing.
- Timings taken with the **GPU lock held**, single process. Report if the lock was not held.
- Library versions, git SHA of both the lab and the fork, GPU name, and driver version in the
  manifest.

## 9. Leakage rules

- Needle texts and filler must come from **disjoint pools**, verified by content, not by
  variable name.
- Any training data must be disjoint from every evaluation item. The cross-verifier re-derives
  the split and checks by hash.
- Needle templates used in training may not appear in evaluation in any surface form.

## 10. What may not be claimed from this protocol

- Nothing about **real** long documents. The filler is synthetic and repetitive; repetition
  likely exaggerates attention dilution relative to natural text. Any transfer claim requires
  a separate run on real long-document data with real labels.
- Nothing about a language not in the sweep.
- Nothing from a cell whose intervention was not **asserted live**. An arm whose config patch
  silently did nothing is reported as invalid, not as a null result.

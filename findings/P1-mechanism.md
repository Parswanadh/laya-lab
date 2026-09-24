# P1 — Mechanism of long-context failure (orchestrator diagnostic)

**Status:** `candidate` — measured here, **not yet independently verified**. A verifier agent must
reproduce it from `experiments/orch-diagnostic/results.json` before it is quoted as settled.

**Artifact.** `experiments/orch-diagnostic/results.json`
**Command.** `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 env/venv/bin/python experiments/orch-diagnostic/run.py`
**Device.** RTX 4070 Laptop, CUDA · checkpoint `convaiinnovations/laya` `multilingual/` @ `55cf4c4`
**n = 20 per cell** (upstream's item set) · **majority-class baseline = 0.450** (`billing` 9, `technical` 7, `sales` 4, `other` 0)

---

## Probe A — precision: REFUTED

Hypothesis: the `max_len=8192` degradation is a bf16 numerics artifact (CUDA honours
`amp_dtype: "bf16"` from the checkpoint config; upstream's published run was on Apple MPS).

| dtype | accuracy (pad=4000, limit=8192) | median latency |
|---|---|---|
| bf16 (shipped default) | **0.600** | 0.230 s |
| fp16 | **0.600** | 0.277 s |
| fp32 | **0.600** | 0.763 s |

**Identical to three decimal places.** Precision is not the cause; the loss is representational,
not numerical. fp32 also costs 3.3× the latency for nothing — a usable secondary result.

## Probe B — needle position at **fixed** document length (pad=4000, limit=8192, fp32)

| needle fraction | absolute position | accuracy | mean P(gold) |
|---|---|---|---|
| 0.00 (start) | ~256 | **0.900** | 0.794 |
| 0.25 | ~1256 | 0.500 | 0.450 |
| 0.50 | ~2256 | 0.500 | 0.415 |
| 0.75 | ~3256 | 0.500 | 0.409 |
| 1.00 (end) | ~4256 | 0.600 | 0.459 |

## Probe C — document length at **fixed** needle fraction (1.0, limit=8192, fp32)

| pad | accuracy | vs majority (0.450) | median latency |
|---|---|---|---|
| 0 | 0.950 | +0.500 | 0.024 s |
| 1000 | 0.900 | +0.450 | 0.120 s |
| 2000 | 0.800 | +0.350 | 0.275 s |
| 4000 | 0.600 | +0.150 | 0.809 s |
| 7000 | **0.450** | **0.000** | 2.125 s |

---

## The two controls that isolate the variable

**Control 1 — length is not the variable.** At a fixed 4000-token document, moving the needle from
the start to the end costs 0.90 → 0.60. Both documents are the same length and both fit entirely
inside the 8192 budget, so **nothing is truncated**. The failure is therefore not a budget effect
at all at this setting; it is positional.

**Control 2 — position alone is not the variable either.** The needle at absolute position ~1000
scores **0.900** in a 1018-token document (probe C, pad=1000) but **0.500** at the same absolute
position in a 4018-token document (probe B, frac=0.25). Same evidence, same position, same
checkpoint — the only difference is **how much text surrounds it**.

## Mechanism

Neither distance nor length alone explains the data; both act. This is consistent with
**attention dilution over the full key set**, compounded by a **shortage of long-range mixing
paths**:

- The option `[MASK]` markers sit at the **start** of the sequence; the document follows them.
- 14 of 22 layers are sliding-attention with a **±64-token** half-window. They cannot move
  information more than 64 positions per hop.
- Only the **8 global layers** (indices 0,3,6,9,12,15,18,21) create direct long-range edges.
- Every additional filler token is another key competing for the markers' attention mass, while
  the number of *useful* hops is fixed at 8.

Two consequences, both observed:

1. **Dilution.** At pad=7000 the score lands on **0.450 — exactly the majority class**. The model
   has not degraded gracefully; it has stopped reading the document and fallen back on its label
   prior. That is the same terminal behaviour as the 0.35 floor at `max_len=1024`, reached by a
   different route.
2. **Position dependence.** Evidence adjacent to the markers (frac=0.00) survives 4000 trailing
   tokens at 0.90; evidence 1000+ positions away does not.

**Both are consistent with one architectural statement:** the markers have no privileged,
non-diluting path to distant document positions. Everything must route through 8 global layers
and compete with every filler token for attention mass.

## What this predicts (falsifiable)

| prediction | test | if it fails |
|---|---|---|
| **P-a** Widening the sliding window (more mixing paths, no extra params) recovers part of the loss | set `config.local_attention` 128→512, re-run probe B | window width is not the constraint |
| **P-b** Making **all** layers global recovers more, at quadratic cost | `global_attn_every_n_layers = 1` | the 8-global-layer count is not the constraint, and the head itself is the bottleneck |
| **P-c** A cross-attention head that lets markers query **every** state position in one hop recovers the most, at sub-quadratic cost | H5 adapter | dilution is intrinsic to softmax attention over long keys and needs a different aggregation |

P-a and P-b are pure config changes requiring **no training** and are the cheapest possible test
of the whole thesis. They run next.

## Caveats that must travel with this finding

- **n = 20.** Every cell is ±0.22 at 95%. The 0.90-vs-0.50 gap is far outside that, but the
  ordering *among* the 0.50/0.50/0.50/0.60 cells is not resolvable at this n. Do not read the
  non-monotonicity at frac=0.25–1.00 as structure until n is raised.
- **Label imbalance.** `other` appears **zero** times in upstream's 20 items, and `billing` is
  45%. The 0.450 "floor" in probe C is the majority class, so "the model stopped reading" is
  supported but not proven — a balanced set is required to distinguish "collapsed to prior" from
  "reading poorly."
- **Synthetic task.** Needle-in-haystack with a single repeated filler paragraph is a *mechanism
  probe*, not evidence about real documents. The repeated filler in particular may exaggerate
  dilution relative to natural text.
- **Not yet verified.** This file is `candidate`. Independent reproduction is required before any
  of it is quoted in a PR.

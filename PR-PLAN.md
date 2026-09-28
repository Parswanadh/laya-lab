# PR-PLAN.md — where we can actually raise a PR

**Written 2026-09-25** against upstream `9d95567` (**v0.3.21**). Our fork branch point is `9fbb1eb`
(v0.3.20), which is **438 commits behind**. Everything below is cross-checked against upstream's
*current* code, not the state we forked from.

---

## 0. Reconnaissance: what changed while we worked

`git log 9fbb1eb..upstream/main` — 438 commits. The long-context work that landed is **all plumbing**:

| landed upstream | what it does |
|---|---|
| `#497`, `#577`, `#653`, `#689`, `#692` | `predict_long` — windowed scanning, its hooks, its window bounds, its usage merge |
| `#566`, `#668`, `#687`, `#691` | token budget on every surface; **report** that truncation happened; tokenize only what survives |
| `#542`, `#583`, `#530` | budgets through the CLI / Router / LangChain |
| `0.3.21` | ONNX batch+long parity, opt-in abstention |

**Nobody upstream has addressed *why* long-document accuracy collapses, or made it better.** Every
long-context commit either adds a way to *scan* more windows, or *reports* that tokens were dropped.
The accuracy question is untouched.

Two consequences for us:

1. **The lanes we identified as taken are now heavily occupied** — `predict_long` and truncation
   reporting both landed and are being actively patched. Do not go near them.
2. **Our four negative results and the position-sensitivity measurement are unobstructed**, because
   they are about accuracy and cost, not about plumbing.

---

## 1. Ranked PR candidates

### 🥇 P1 — `perf(common): sliding-attention layers do dense O(n²) work in the SDPA path`

**The strongest mergeable PR.** Upstream's identity is *"fast, local, on-device decision engine"* —
latency is the headline claim — and this is a measured latency defect with no accuracy claim attached.

**Evidence (all committed):**
- `experiments/h2h3-knobs/mask_evidence.json`: at pad=7000 **all 22 layers** — including the 14
  declared `sliding_attention` — receive a dense `(1, 1, 7069, 7069)` mask.
  **4.40 GB of dense mask per forward**, summed over layers.
- `sdpa_microbench.json` (L=7068, 12 heads, fp32): no-mask **0.03042 s** · dense ±64 sliding mask
  **0.03595 s** · dense global mask **0.03768 s**. **sliding/global = 0.954.**
- **Interpretation:** the ±64 window — the architectural feature that is supposed to make 2/3 of the
  layers cheap — buys **~5 % latency instead of a roughly linear saving**. The 14 "local" layers were
  already paying full dense attention cost while being restricted to ±64 tokens.

**Why it merges:** it is a perf claim with a benchmark, no public-API change, and it does not touch
accuracy. Upstream already ships a fast path (`laya/fast.py` + TileLang) whose kernels *do* pass a
real `window=` per layer — so the fix direction is already blessed in-tree; it is just not the
default path.

**Scope of a PR:** (a) the measurement, added as `research/scripts/bench_sliding_cost.py`; (b) a fix —
`torch.nn.attention.flex_attention` with a `BlockMask` for sliding layers, or a block-diagonal
chunked path; (c) a parity test asserting the windowed path is **numerically identical** to the dense
one. We already hold that reference: E-002 showed `w128` re-applied through the patch is
**bit-identical** to baseline (20/20 predictions, max |Δp_gold| = 0.000000), so there is a hard
correctness baseline to match.

**Risk:** implementing a windowed kernel correctly is real work. **But (a) alone is a mergeable PR** —
a benchmark documenting that `local_attention` is not delivering its cost benefit is useful on its own
and is exactly the kind of "measured delta" upstream asks for.

---

### 🥈 P2 — `research: position sensitivity of the decision head` (+ Honest-limits corrections)

**The safest high-value PR, and the one that fills a documented literature gap.**

**The gap:** upstream's `research/scripts/bench_long_context.py` varies **only `pad`** — the request
is always appended at the **end** — at **n=20**, with **no CIs and no oracle baseline**. So an upstream
reader cannot tell *"the document is long"* from *"the evidence is far from the option markers"*. Our
work shows those are different, and that the second one is the dominant effect.

**Evidence (all committed, n=200–600, balanced, position-only oracle 0.2500 on every cell):**

| needle position at **fixed** pad=4000, `max_len=8192` | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| accuracy | **0.580** | 0.295 | 0.335 | 0.280 | **0.415** |

- **The curve is U-shaped, not monotone.** A two-point sample (which is all upstream's bench gives)
  cannot distinguish a distance decay from edge effects.
- At `max_len=8192` with the **whole document present** (`request_tokens_kept = 1.000`) accuracy still
  falls **0.840 → 0.420** across pads 0→7000. Length effect at 8192, paired: 2000→4000 **p=1.2e-06**.
- At the shipped `max_len=1024` the failure is **truncation, not comprehension**: `request_tokens_kept
  = 0.00` in every collapsed cell, one constant answer (modal share 0.98–0.99), accuracy equal to that
  label's share of the pool.
- **Two regimes that upstream currently reports as one number.**

**Why it merges:** it is additive, research-only, touches no library code, fits `research/scripts/` +
`research/README.md` conventions exactly, and upstream's `AGENTS.md` explicitly asks for a runnable
check and measured before/after.

**Scope:** a `bench_position_sensitivity.py` (our harness generalised), a results JSON, a
`research/README.md` row, and a **correction to the README's Honest limits** — the claim that 8192
"recovers" the long-document case needs the position qualifier.

---

### 🥉 P3 — `docs(common): the state budget is not max_len - head_max_len`

**Near-certain merge, tiny scope, verified empirically just now.**

README (current `upstream/main`, "High-cardinality choice questions and token budgets") says the
sequence splits into an option budget and *"the remaining document/state budget (`max_len -
head_max_len`)"*. **That formula is wrong by a wide margin**, because `head_max_len` is an *upper
bound*, not the head length. `build_sequence` uses the **actual** head:
`room = max_len - len(ids) - 1`.

Measured just now with the shipped tokenizer and the shipped department question (4 options):

| checkpoint | `max_len` | `head_max_len` | README formula | **measured** | understated by |
|---|---|---|---|---|---|
| `laya-multilingual` | 1024 | 256 | 768 | **978** | **210 (27 %)** |
| `laya` (english) | 512 | 192 | 320 | **466** | **146 (46 %)** |

Measured **head = 45 tokens** (option markers land at positions 12/20/29/39 — `head_max_len` is a cap
the renderer does not fill).

**User impact:** anyone sizing a document to the documented budget silently discards 210–146 usable
tokens — i.e. they truncate earlier than the library does, for no reason. This is the same *class* of
defect as #174 (a limit derived from a cap rather than from the budget actually in force), which
upstream has already accepted as real.

**Scope:** one README paragraph plus the same sentence in `docs/`. No code change — **the code is
already correct**; only the description is wrong.

---

### P4 — `docs: four architectural fixes that do not help` (optional, foldable into P2)

Upstream's Honest-limits section is already written in this style. A short subsection recording that
**RoPE scaling** (identity in-window by construction — HF clamps `seq_len` to
`max_position_embeddings` before use), **widening the sliding window** (−0.158 at n=120; the *narrowing*
ablation moves the cell equally), and **making all layers global** (0.000 at pad=1000) were each tried
and did not help would save the next contributor the same weeks.

**Why it merges:** it is the kind of "we already checked" note the maintainer writes himself. **Why it
might not:** it is opinion-shaped rather than a measured delta, so it is best folded into P2 as the
discussion section rather than submitted alone.

---

## 2. What NOT to submit

| do not submit | why |
|---|---|
| **The cross-attention head** (`exp/h5-adapter` @ `99020ba`) | Its central claim is a **negative**. The init-fair re-run is bitwise-identical to the fine-tuned baseline at step 0, trains, is harmless at L0, beats the oracle everywhere — and **produces no significant primary-cell gain** (+2.2pp p=0.393 vs the reproduced bar; +4.33pp p=0.089, Holm 1.0 vs the schedule-matched bar, n=600). **Its own ablation settles it:** the trained checkpoint with `out_proj` re-zeroed scores **0.4717 vs 0.4717, Δ=0.0pp, p=1.0**. The added attention does not carry the decisions. |
| Anything touching `predict_long` / truncation reporting | `#497/#577/#653/#689/#692` and `#566/#668/#687/#691` occupy those lanes and are under active iteration. |
| A calibration/ECE PR | `laya-multilingual` still ships unfitted temperatures, but that is upstream's known, documented state — not a defect we found. |

## 3. Rebase requirement — do this before any PR

Our branch touches `laya/common.py`, which changed substantially in the 438 commits
(`predict_long`, budget plumbing, truncation reporting all landed there). **`exp/h5-adapter` will not
apply cleanly.** Any submitted branch must be **re-created from `upstream/main`** rather than rebased,
because:
- the cross-attention code is not being submitted (P1–P4 touch `research/`, `docs/` and the README);
- `tests/test_cross_attention_head.py` would be dead weight in a PR about perf or docs.

**So: none of P1–P4 depends on our fork branch.** They are all additive — a research script, a results
file, a README paragraph. That is deliberate: it makes them rebase-proof.

## 4. Recommended sequence

1. **P3 first** (docs, ~1 h). Smallest, verified, near-certain merge, and it puts a credible first PR
   in front of the maintainer before anything larger.
2. **P2 second** (research bench, ~1 day). The highest-value *safe* PR, and the one that carries the
   programme's real finding: the failure is **positional**, the two regimes are distinct, and the
   documented recovery at 8192 is weaker than it reads because the baseline set has no `other`
   examples (majority class 0.45, not 0.05).
3. **P1 third** (perf, split into measurement-then-fix). Land the benchmark first; offer the kernel
   fix as a follow-up once the measurement is accepted.
4. **P4** folded into P2's discussion section.

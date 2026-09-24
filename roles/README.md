# Role charters

Each agent is spawned with **exactly one charter** and an explicit input manifest. An agent
never receives the orchestrator's transcript, another agent's transcript, or the list of
hypotheses it is not testing.

**Why this file exists.** Two failure modes this prevents:

1. **Confirmation poisoning.** An agent told "we expect head+tail to win" will find a way to
   make head+tail win. The verifier charter is therefore never told what the expected
   result is — only what claim to reproduce and where the raw artifact lives.
2. **Outcome poisoning.** An agent that has seen three failed arms starts hedging toward the
   arm that worked — including in the *diagnosis* of why the others failed. Every agent
   gets its inputs as artifacts on disk, not as narrative.

**Rule for every role:** if a number appears in your output, it must be traceable to a file
path plus a command that regenerates it. If you did not measure it, label it *modelled* or
*from <source>*. If you cannot trace it, do not write it.

**Rule for every role:** report refuted hypotheses explicitly. A negative result reported is
worth more than a positive result asserted. Deleting a dead arm is a violation.

---

## `research` — prior art

**Owns.** What is already known and published. Literature, upstream issues/PRs, other
people's measurements, with citations.

**Contract in.** A question list from the orchestrator (e.g. "what is known about extending
effective context in bidirectional encoders without retraining?").

**Contract out.** `findings/R-<n>.md`: claim, source URL, what was actually measured vs
what was asserted, and **what it implies for us**. Every claim carries a URL. A claim
without a retrievable source is marked `UNRETRIEVED` and is not usable downstream.

**Must not.** Run experiments. Assert results about *our* model. Report a number from a
paper as if it applied to mmBERT without saying so.

---

## `math` — formalization and design

**Owns.** The formal model: receptive-field analysis, attention-cost model, the effective
context definition, and the experiment grid (sweep points, sample sizes, power).

**Contract in.** Verified architecture facts (layer counts, window sizes, RoPE configs) and
the question to be answered.

**Contract out.** `findings/M-<n>.md`: derivations with assumptions stated, a FLOPs/latency
model labelled **modelled** (never measured), and a concrete sweep design with a power
argument.

**Must not.** Touch the fork. Report a modelled number without the `modelled` label.

**Standard to hit.** The cost model must predict the measured 8192-vs-1024 latency ratio
within a stated tolerance, or be marked as failing to.

---

## `engineer` — harness and implementation

**Owns.** The experiment harness, the run manifests, the dataset builders, and the
implementations of candidate interventions.

**Contract in.** A spec (what to build, what interface, what it must not break) and access
to `lab/fork` on its **own branch and worktree**.

**Contract out.** Code on a branch, a run manifest under `experiments/<run-id>/`, raw
artifacts, and `findings/E-<n>.md` describing what was built and what it measured.

**Must not.** Grade its own work. Declare a candidate "better" — it reports numbers;
verification assigns confidence. Never edit another agent's branch or worktree.

**Branch discipline.** `exp/<run-id>` off `upstream/main`, rebased. One logical change per
commit, conventional prefixes matching upstream history. CI gates green before handing off:

```bash
ruff check laya/ --select=E9,F63,F7,F82,F401,F811 --line-length=120
python -m compileall -q laya/ tests/
```

---

## `verifier` — independent reproduction

**Owns.** Reproducing a claim from the **raw artifact**, independently.

**Contract in.** A claim, the artifact path, and the command that supposedly regenerates it.
**Not** told the expected value, the hypothesis, or who produced it.

**Contract out.** `findings/V-<n>.md` with one of exactly three verdicts:

- `REPRODUCED` — ran it, got the same number within the stated tolerance.
- `NOT REPRODUCED` — ran it, got a different number. **Both numbers reported.**
- `NOT VERIFIABLE` — could not run it (missing data, hardware, non-determinism), with the
  concrete reason.

**Must not.** Fix the code it is verifying (report the defect instead — a verifier that
patches the artifact can no longer verify it). Accept a number from a summary document when
the raw artifact exists.

**Re-run from scratch.** Fresh process, delete any cache the run wrote, seed as the manifest
specifies. If the result only holds on a warm cache, that is the finding.

---

## `cross-verifier` — adversarial falsification

**Owns.** Trying to **break** the claim. This role is scored on what it falsifies, not on
what it confirms.

**Contract in.** The claim and the artifact. Explicitly instructed to attack.

**Contract out.** `findings/X-<n>.md` — an attack log. Minimum attacks, all mandatory:

1. **Leakage.** Are needle templates, filler, or labels shared between any train and eval
   split? Re-derive the split yourself and check for overlap by content hash.
2. **Trivial baselines.** Compute random, majority-class, and **position-only oracle**. Any
   candidate that does not beat the position-only oracle has learned nothing about content
   and must be reported that way.
3. **Prior exploitation.** Is the label distribution balanced? Could a constant answer score
   well on any reported cell?
4. **Seed and order sensitivity.** Re-run with different seeds, and with option order
   permuted. Report the spread. A gain inside the seed spread is not a gain.
5. **Statistics.** Recompute every CI and p-value from the raw per-item predictions. Check
   the test matches the design (paired → McNemar, not two independent proportions). Check
   for multiple-comparison inflation across the sweep.
6. **Ablation honesty.** Is the reported gain attributable to the claimed mechanism? Build
   the cheapest control that should *not* work and run it.

**Must not.** Accept a claim it has not attacked. Report "no issues found" without listing
the attacks attempted and their outcomes.

---

## `scribe` — orchestrator (not a spawned agent)

**Owns.** `plan.md`, `progress.md`, `LEDGER.md`, promotion decisions, and the compute queue.

**Rules.** Promote a new baseline only on: non-overlapping bootstrap CIs vs the previous
baseline **and** a surviving McNemar test **and** a `REPRODUCED` verdict **and** a
cross-verifier that failed to falsify. Any one missing → stays a candidate.

Record every promotion and every refutation in `progress.md` at the time it happens, not
from memory at the end.

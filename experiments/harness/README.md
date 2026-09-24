# `experiments/harness/` — long-context needle-in-haystack harness

Frozen, deterministic evaluation harness for the laya-lab long-context program. It builds
needle-in-haystack decision documents, runs the shipped checkpoint over a parameter grid, and
writes **raw per-item predictions** as JSONL plus a run manifest.

Nothing here modifies `fork/`. The fork is imported read-only from
`experiments/harness/backends.py` and `experiments/harness/builder.py`.

## One command

```bash
cd /home/parshu/projects/contri/laya-lab

# 1. what a run would do, with no checkpoint and no GPU
env/venv/bin/python experiments/harness/run.py --preset baseline-repro --plan

# 2. pipeline self-test: stub tokenizer + tiny random model (no GPU, no checkpoint)
env/venv/bin/python experiments/harness/run.py --preset baseline-repro --dry-run

# 3. the real run. The harness takes the GPU lock itself (.gpu.lock, flock LOCK_EX|LOCK_NB)
#    and refuses to start if another agent holds it, or if free RAM/VRAM is below the gate.
env/venv/bin/python experiments/harness/run.py --preset baseline-repro --device cuda \
    --checkpoint models/multilingual
```

Long runs are launched detached, log inside the repo:

```bash
setsid nohup systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 \
  --unit=laya-baseline-$$ \
  env LAYA_MEMORY_CAP=4G CUDA_VISIBLE_DEVICES=0 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True USE_TF=0 \
  "$PWD/env/venv/bin/python" experiments/harness/run.py --preset baseline-repro \
      --device cuda --checkpoint models/multilingual \
  > experiments/baseline-repro/launch.log 2>&1 &
```

`USE_TF=0` is set by the entry point itself; it is repeated above so the launcher is
self-describing. The runner also writes its own `run.log` inside the run directory.

## What a run writes

| file | contents |
|---|---|
| `manifest.json` | environment, git SHA of lab + fork, library versions, GPU, pool hashes, item table, cell plan, exact command, protocol conformance notes, checksums of the harness files. Written **before** the sweep and finalised after, so a killed run still has provenance. |
| `predictions.jsonl` | **the raw artifact**: one row per (cell, item). Appended and flushed as the sweep proceeds. |
| `summary.json` | per-cell descriptive numbers recomputed from the JSONL, plus baselines. Rewritten after every cell. |
| `run.log` | progress, per-cell lines, and the check results. |

Per-item row fields (see `runner.make_row`): `cell`, `item_id`, `label`, `lang`, `template_id`,
`slots`, `prediction`, `probabilities`, `prob_vector`, `options`, `correct`, `confidence`,
`answer_confidence`, `act_probability`, `pad_tokens`, `needle_position`, `max_len_requested`,
`max_len_effective`, `head_max_len_effective`, `latency_s`, `warmup_call`, `seed`, `device`,
`checkpoint`, `state_tokens_full`, `state_tokens_kept`, `input_tokens`, `truncated`,
`needle_token_start`, `needle_tokens`, `needle_tokens_kept`, `needle_kept`, `request_tokens`,
`request_tokens_kept`, `request_kept`, `filler_tokens_before/after/actual`, `pad_exact`,
`doc_sha256`, `doc_chars`, `doc_preview_head`, `doc_preview_tail`, `trunc_rule_ok`, `span_method`.

The document text itself is **not** stored per row (`n=200` × 25 cells of 7,000-token documents
would be hundreds of MB). It is regenerated exactly: `doc_sha256` is a function of
`(pool, seed, item_id, pad, needle_position)` only, and `checks.py` rebuilds it from the manifest
to prove it.

## Parameterisation

| flag | meaning |
|---|---|
| `--pads 0,1000,2000,4000,7000` | filler tokens in the document (`0` = the short-context regression cell) |
| `--positions 0.0,0.25,0.5,0.75,1.0` | fraction of the filler that precedes the needle |
| `--max-lens default,8192` | `max_len`; `default` means *do not override*, i.e. use what the checkpoint ships |
| `--head-max-lens default,128,512` | `head_max_len`, same convention |
| `--n`, `--seed` | items per cell, and the item/document seed |
| `--languages en,zh` | restrict the draw to a subset of the pool's languages |
| `--device cuda\|cpu\|mps`, `--checkpoint`, `--subfolder` | backend |
| `--dry-run` | stub tokenizer + tiny random model, exercises the identical pipeline |
| `--repeat-check-items N` | re-run N items of the first cell and report prediction agreement |
| `--plan`, `--table RUN_DIR`, `--check-only RUN_DIR` | inspect without running anything |

Presets live in `presets.json`: `baseline-repro`, `baseline-repro-upstream`, `baseline-power`
(n=200), `position-sweep`, `head-budget`.

## Pools

* `pools/needles-balanced-v1.json` — 81 templates = 3 labels × 9 languages × 3 templates, with a
  shared numeric slot pool. Authored for this harness.
* `pools/filler-v1.json` — 30 unrelated office-logistics sentences, no digits, no support-domain
  vocabulary.
* `pools/upstream-multilingual.json` — verbatim transcription of upstream's 20 requests, its
  filler unit and its question, used only for the faithful replication arm.

`pools.leakage_report` checks filler against needle + question-criteria vocabulary: whole-word
content overlap (stopword-filtered) plus a substring check against an explicit domain-stem list.
`pools.verify_upstream_transcription` re-parses the upstream script with `ast.literal_eval` and
compares its `REQUESTS`/`FILLER`/`QUESTIONS` to the pool file. Both are recorded in the manifest.

**Item draw.** The balanced pool fixes *both* marginals exactly: every label gets `n // 3` (+1 for
the remainder) and every language gets `n // 9` (+1 for the remainder), with the joint cells
allocated so no language dominates any label. So a constant answer cannot beat ~1/3 on any cell,
and label and language cannot carry each other's signal.

## Modes

**Balanced (default).** Filler is built to an exact *token* budget with the live tokenizer
(tokenize → slice → decode → re-tokenize, re-checked), so `pad` means tokens, not characters, and
`filler_tokens_actual` is measured rather than assumed.

**Upstream.** `reps = round(pad / tokens(FILLER_UNIT))` and the unit is repeated, reproducing
upstream's composition rule exactly (`--preset baseline-repro-upstream`).

## Truncation diagnostics

For every row the harness records what happened to the sequence, computed with the library's own
`build_sequence`:

* `state_tokens_full` / `state_tokens_kept` / `input_tokens` / `room` / `head_len`
* `request_kept` — is the whole decision request still inside the model's window?
* `needle_kept`, `request_tokens_kept`
* `trunc_rule_ok` — does the observed keep-length equal `min(len(state_ids), max_len - head - 1)`,
  the rule `laya/common.py` documents for string state? A silent change in that rule flips this to
  `false` in every raw row.

## Checks

`tests/selftest.py` (no GPU, no checkpoint) covers pool balance and determinism (n = 20 … 600,
multi-seed), the language filter, pool disjointness, upstream transcription fidelity, the token
budget, the truncation rule, upstream composition, and the statistics helpers.

`checks.structural_checks` runs against a finished run's published artifacts only — including
recomputing every `summary.json` accuracy from the JSONL, and verifying that the same document hash
is used at every `max_len` (which is what makes the cells paired for McNemar).

`checks.mechanism_checks` rebuilds documents from the manifest seed and asserts the mechanism:
deterministic for a seed, different for another seed, the row's `doc_sha256` reproducible, the tail
request dropped at a small budget and kept at 8,192.

## Statistics

`metrics.py` provides accuracy, macro-F1, percentile bootstrap CIs resampling *items*, exact
two-sided McNemar for paired arms, the modal-prediction share (a collapse diagnostic), the
majority-class and random baselines, the position-only oracle, and latency/tokens-per-second
summaries. It reports numbers only — it assigns no confidence and declares no winner.

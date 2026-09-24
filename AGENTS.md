# AGENTS.md — laya-lab

Coordination repo for the **long-context decision engine** effort against
[`NandhaKishorM/laya`](https://github.com/NandhaKishorM/laya).

**Read this file first.** It is the cheap entry point. Do not read the whole repo to get
oriented — read this, then your role charter, then the one or two artifacts your task names.

---

## What this repo is

| path | what |
|---|---|
| **`BASELINE.md`** | **the frozen reference: every number we measure against, with n, CIs and reproduce commands. Read this to know what "the baseline" means.** |
| `plan.md` | the master plan: problem, hypotheses H1–H5, phases, gates, budget |
| `progress.md` | **append-only** state log, newest at the bottom. Read the last entry, not the whole file |
| `LEDGER.md` | one row per claim: claim, artifact, verifier, status |
| `protocol.md` | the **frozen** evaluation protocol. Do not change it to make an arm look better |
| `roles/README.md` | role charters. Find your role, follow only it |
| `findings/` | one file per finding: `R-*` research, `M-*` math, `E-*` engineering, `V-*` verify, `X-*` cross-verify |
| `experiments/<run-id>/` | one directory per run: `manifest.json` + raw artifacts |
| `decisions/` | ADRs — why a direction was taken or abandoned |
| `fork/` | clone of the fork `Parswanadh/laya` (upstream remote configured) |
| `env/venv` | Python 3.12 venv, torch cu128 + transformers. **Gitignored** |
| `models/` | local checkpoints. **Gitignored** |

## Working rules

**Traceability.** No number without an artifact path and a command that regenerates it.
Tag `measured` vs `modelled` explicitly. Unretrieved citations are marked `UNRETRIEVED`.

**Negative results are results.** If a hypothesis is refuted, write it up and leave it.
Never delete a dead arm, never quietly re-run until it looks better.

**Evidence over narrative.** If a summary document and a raw artifact disagree, the raw
artifact wins and the summary is corrected in place.

**One writer per file.** Each agent writes only its own `findings/<ROLE>-<n>.md` and its own
`experiments/<run-id>/`. Never edit another agent's finding.

**Git.** Commit and push as you go — small, frequent commits. Conventional prefixes:
`research:`, `math:`, `exp:`, `verify:`, `cross-verify:`, `chore:`, `docs:`.
Push to `origin` (`Parswanadh/laya-lab`) so other agents can review without re-deriving.

```bash
git add -A && git commit -m "exp: <what> " && git push -q origin main
```

**Issues are the message bus.** Cross-agent questions, disagreements, and handoffs go in
GitHub Issues on `Parswanadh/laya-lab`, not in chat. A verifier disputing an engineer's
number opens an issue and links the artifact. Reference issues as `#<n>` in findings.

## Machine budget — read before running anything

```
22 cores · 15 GB RAM · RTX 4070 Laptop 8188 MiB · ~10 GB RAM used by the desktop baseline
```

**Check `free -h` before launching. `available` is the decision input.**

| available | allowed |
|---|---|
| ≥ 6 GB | 1 heavy (4–6 GB) **or** ≤ 4 light (≤1 GB) |
| 3–6 GB | 1 heavy (≤4 GB) **or** ≤ 2 light |
| 2–3 GB | light only, one at a time |
| < 2 GB | **launch nothing** — report and wait |

- One agent worker ≈ 0.7 GB. **≤ 4 concurrent agents.**
- **Exactly one GPU job at a time.** `CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Cap memory explicitly; never run unbounded:
  `systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 <cmd>`
- Load checkpoints with `low_cpu_mem_usage=True`. Only one model resident at a time.
- **Never `torch.load` an unbounded checkpoint** — this box has taken 31 OOM kills in one
  session from exactly that.
- Long runs: `run_in_background: true`, log **inside this repo** (never `/tmp`).
- Do not poll `nvidia-smi` in a loop. Sample it.

## Layout of the fork

`fork/` tracks `Parswanadh/laya`. Work on a branch, never on `main`:

```bash
cd fork
git fetch upstream main -q
git worktree add ../worktrees/<agent> -b exp/<run-id> upstream/main
```

Use **worktrees**, not extra clones — one object store, no cross-agent interference.
`worktrees/` is gitignored.

Upstream `fork/AGENTS.md` binds all code changes: no hosted-service dependency, no
drive-by reformatting, one logical change per commit, `tests/test_hooks_api.py` updated with
any public-API change, CI gates green:

```bash
ruff check laya/ --select=E9,F63,F7,F82,F401,F811 --line-length=120
python -m compileall -q laya/ tests/
```

## Running the model

```bash
lab/env/venv/bin/python -c "
import os; os.environ.setdefault('USE_TF','0')
import sys; sys.path.insert(0,'/home/parshu/projects/contri/laya-lab/fork')
import laya
agent = laya.load('/home/parshu/projects/contri/laya-lab/models/multilingual', device='cuda')
"
```

`USE_TF=0` is required — `transformers` probes for TensorFlow at import and its abseil
runtime can deadlock on this box.

## Current state

The baseline **is reproduced and frozen** — see `BASELINE.md`. Read that file, not this
section, for numbers. Read the **last entry** of `progress.md` for what is in flight.

**Already refuted — do not spend compute re-testing:** RoPE scaling (identity in-window by
construction), sliding-window widening, all-layers-global, and bf16-vs-fp32 precision. All four
have liveness-proved, falsifying controls in `findings/E-002.md` and `BASELINE.md` §5.

**The strong intervention is arm2** — fine-tuning the shipped decision head with the encoder
frozen: +9.0pp at the primary cell (p=0.0021), +16.5pp mid-document (p=3.6e-08), and no change at
the easy cells.

## GPU is a single-writer resource — take the lock

More than one agent running a GPU job at once will silently corrupt timing measurements and
can OOM an 8 GB card. Before any CUDA job:

```bash
flock -n /home/parshu/projects/contri/laya-lab/.gpu.lock -c '<your command>' \
  || echo "GPU busy — another agent holds the lock; wait, do not queue a second job"
```

Use `flock -n` (non-blocking) so you *know* you were refused rather than silently stalling. If
refused, do non-GPU work and retry later; do not launch anyway. Record in your finding whether
the numbers came from a locked run (clean) or an unlocked one (timings may be contended).

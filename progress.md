# progress.md — append-only state log

Newest entry at the **bottom**. Never rewrite history; correct it with a new entry.
Every entry: what was done, what was measured, what it means, what is next.

---

## 2026-09-24 · P0 · Reconnaissance complete

**Done.** Read the upstream repo, its benchmark artifacts, its open issues and PRs; forked;
cloned; characterized both encoders from their published configs.

**Verified facts** (each traceable to a path or an API response):

- `laya/common.py:126` keeps the **head** of a long state (`state_ids[:room]`) for strings
  and dicts; `truncate_left=True` only for list-shaped conversation state (`agent.py:571`).
- `laya-multilingual` ships `max_len=1024`, `head_max_len=256`, `head_layers=2`
  (`multilingual/rl_agent_config.json`).
- Upstream long-context run (`research/results/long_context_multilingual.json`, n=20/cell):
  accuracy **0.35 flat** for pad ≥ 2000 at limit=1024; **0.85–0.90** at limit=8192;
  latency 0.016 s → 3.507 s.
- Encoders are ModernBERT with `max_position_embeddings=8192`.
  `laya`: 28 layers, hidden 1024, RoPE full 160000 / local 10000.
  `laya-multilingual` (mmBERT-base): **22 layers, 8 global at indices 0,3,6,9,12,15,18,21,
  14 sliding**, hidden 768, RoPE 160000 for both.
  Sliding half-window is `local_attention // 2` = **64 tokens** (`laya/fast.py:53-55`).

**Prior art that constrains us** (checked so we do not duplicate):

| upstream | state | overlaps us? |
|---|---|---|
| PR **#363** `predict_long` | open, actively iterated | windowed scan + aggregate. **Adjacent, not ours.** Evaluated as a baseline; not duplicated. |
| PR **#181** truncation reporting | open, "the fix I'm taking" | reports *that* truncation happened. Orthogonal — we change *what is kept*. |
| Issue **#99** long noisy multilingual ≈ chance | open | the symptom. Maintainer: "a strong direction we are investigating for future checkpoints." **This is our lane.** |
| Issue **#174** truncation flag off by checkpoint | open | addressed by #181. Not ours. |

**Decision (user-directed).** Evidence *selection* was considered and rejected as the primary
contribution — it is a lossy bet and leaves the silent-failure mode intact. The target is an
**architectural** change so the model attends to the whole document. Recorded in `plan.md §3`.

**Environment.** `laya-lab/` created; `Parswanadh/laya` forked and cloned to `lab/fork` with
`upstream` remote; venv at `lab/env/venv` (py3.12) installing torch 2.9.1+cu128.

**Machine at recon time.** 5.1 GB RAM available, 6.8 GB VRAM free, 128 GB disk.
Budget and its consequences are in `plan.md §8`.

**Next.** Finish env → download mmBERT checkpoint → build the harness → **reproduce the
0.35 floor here.** No candidate work starts until the baseline reproduces on this machine.

**Open risks.**

1. **R1 — statistics.** Upstream's n=20. Our protocol needs n≥200/cell across 8 lengths ×
   5 positions × 6 languages. That is a large inference bill on one 4070. Mitigation:
   stage the sweep (coarse first, then power only the cells that separate arms).
2. **R2 — training budget for H5.** The upstream fine-tune recipe targets 2×T4 (32 GB). We
   have 8 GB. The H5 adapter must therefore train with the encoder **frozen** and only a
   small parameter set live. If that cannot reach significance, H5 is reported as
   *unresolved on this hardware* rather than asserted.
3. **R3 — synthetic-task validity.** Needle-in-haystack is a proxy. It is honest as a
   mechanism probe; it is **not** evidence about real long documents, and will be labelled
   that way everywhere.

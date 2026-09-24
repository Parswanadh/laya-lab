"""E-002 harness — training-free architectural knobs on the *loaded* encoder (issue #4).

Two knobs, both applied after ``laya.load()``, no training:

  P-a  widen (or narrow) the sliding-attention window: ``config.local_attention`` 128 -> W,
       plus every sliding layer's ``attn.local_attention = (W//2, W//2)`` so the SDPA branch
       and the mask builder agree.
  P-b  make *every* layer global: ``config.global_attn_every_n_layers = 1`` (inert after
       construction) **and** ``lyr.attn.local_attention = (-1, -1)`` on all 22 layers, which is
       the attribute the SDPA path actually reads
       (``transformers/models/modernbert/modeling_modernbert.py`` in 4.57.6:
       line 412 selects ``sliding_window_mask`` iff ``local_attention != (-1, -1)``;
       line 938 builds that mask from ``self.config.local_attention // 2`` every forward).

**Liveness is asserted, not assumed.** ``MODERNBERT_ATTENTION_FUNCTION["sdpa"]`` is wrapped once;
for the first item of every cell it records, per encoder layer, the mask actually handed to
``F.scaled_dot_product_attention`` and decodes the max attended distance for the last query row.
An arm whose observed mask shape does not match its specification raises and is recorded as
``invalid`` — a silent no-op must not be able to look like a clean "no effect".

Documents are built with **P1's exact rule** (``experiments/orch-diagnostic/run.py::place`` at
needle fraction 1.0, imported from that file, not re-implemented) so the baseline arm is directly
comparable to ``findings/P1-mechanism.md`` probe C. ``--construction upstream_rule`` additionally
implements upstream's own rule (``bench_long_context.py``: ``FILLER * round(pad/per_rep)`` + the
``"\\n\\nActual request: "`` delimiter) as a construction control.

    env/venv/bin/python experiments/h2h3-knobs/run.py --stage stage1

Outputs (append-by-arm, resume-safe):
    experiments/h2h3-knobs/raw/<tag>__<arm>.jsonl          per-item predictions
    experiments/h2h3-knobs/raw/<tag>__<arm>.summary.json   per-cell aggregates + liveness
    experiments/h2h3-knobs/liveness.json                   merged liveness evidence
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(LAB, "fork"))
sys.path.insert(0, os.path.join(LAB, "experiments", "harness"))

import torch  # noqa: E402

import pools  # noqa: E402  (experiments/harness/pools.py)
import laya  # noqa: E402

MODEL_DIR = os.path.join(LAB, "models", "multilingual")
RAW_DIR = os.path.join(HERE, "raw")

# ---------------------------------------------------------------- arm specifications
# window: value written to config.local_attention and to each sliding layer's attn.local_attention
# all_global: patch every layer to (-1, -1) (true global attention)
ARM_SPECS = {
    "baseline": {"window": None, "all_global": False,
                 "expect": {"sliding": 14, "global": 8, "half": 64}},
    "w64": {"window": 64, "all_global": False,
            "expect": {"sliding": 14, "global": 8, "half": 32}},
    "w128": {"window": 128, "all_global": False,
             "expect": {"sliding": 14, "global": 8, "half": 64}},
    "w256": {"window": 256, "all_global": False,
             "expect": {"sliding": 14, "global": 8, "half": 128}},
    "w512": {"window": 512, "all_global": False,
             "expect": {"sliding": 14, "global": 8, "half": 256}},
    "w1024": {"window": 1024, "all_global": False,
              "expect": {"sliding": 14, "global": 8, "half": 512}},
    "allglobal": {"window": None, "all_global": True,
                  "expect": {"sliding": 0, "global": 22, "half": None}},
    # P-b applied together with a window setting: if "all global" is truly global, the sliding
    # mask built from config.local_attention must be bypassed on every layer and this arm must be
    # *identical* to `allglobal`. Any difference falsifies the P-b patch.
    "allglobal_w512_combo": {"window": 512, "all_global": True,
                             "expect": {"sliding": 0, "global": 22, "half": None}},
}

# stage -> list of (arm, construction, pads)
STAGES = {
    "stage1": [("baseline", "p1_exact", [0, 1000, 2000, 4000, 7000]),
               ("baseline", "upstream_rule", [0, 4000, 7000])],
    "stage2": [("w64", "p1_exact", [0, 4000, 7000]),
               ("w128", "p1_exact", [0, 4000, 7000]),
               ("w256", "p1_exact", [0, 4000, 7000]),
               ("w512", "p1_exact", [0, 1000, 2000, 4000, 7000]),
               ("w1024", "p1_exact", [0, 4000, 7000])],
    "stage3": [("allglobal", "p1_exact", [0, 1000, 2000, 4000, 7000]),
               ("allglobal_w512_combo", "p1_exact", [0, 7000])],
    "stage4": [("baseline", "p1_exact", [0, 7000]),
               ("w512", "p1_exact", [0, 7000]),
               ("allglobal", "p1_exact", [0, 7000])],
    "pilot": [("baseline", "p1_exact", [7000]),
              ("w1024", "p1_exact", [7000]),
              ("allglobal", "p1_exact", [7000])],
}

STAGE_POOL = {"pilot": "upstream_multilingual", "stage1": "upstream_multilingual",
              "stage2": "upstream_multilingual", "stage3": "upstream_multilingual",
              "stage4": "balanced-v1"}
STAGE_N = {"pilot": 5, "stage1": 20, "stage2": 20, "stage3": 20, "stage4": 120}
STAGE_SEED = {"pilot": 20260924, "stage1": 20260924, "stage2": 20260924, "stage3": 20260924,
              "stage4": 20260924}


# ---------------------------------------------------------------- P1 helpers, imported not copied
def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_p1_module():
    """Import experiments/orch-diagnostic/run.py so `place`/`FILLER`/`REQUESTS` are *the same
    objects* the baseline was measured with — not a re-implementation of them."""
    path = os.path.join(LAB, "experiments", "orch-diagnostic", "run.py")
    spec = importlib.util.spec_from_file_location("p1_diagnostic", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # module has a __main__ guard; nothing runs
    return mod


# ---------------------------------------------------------------- attention-mask probe
class AttentionProbe:
    """Wrap the SDPA attention function and record the mask each layer actually uses."""

    def __init__(self):
        self.records: list = []
        self.armed = False
        self._orig = None

    def install(self):
        import transformers.models.modernbert.modeling_modernbert as mbm
        self._orig = mbm.MODERNBERT_ATTENTION_FUNCTION["sdpa"]

        def wrapper(module, qkv, attention_mask=None, sliding_window_mask=None,
                    position_ids=None, local_attention=(-1, -1), bs=1, dim=0, **kw):
            if self.armed:
                self._record(module, qkv, attention_mask, sliding_window_mask, local_attention)
            return self._orig(module, qkv, attention_mask=attention_mask,
                              sliding_window_mask=sliding_window_mask, position_ids=position_ids,
                              local_attention=local_attention, bs=bs, dim=dim, **kw)

        mbm.MODERNBERT_ATTENTION_FUNCTION["sdpa"] = wrapper

    def _record(self, module, qkv, attention_mask, sliding_window_mask, local_attention):
        sliding = tuple(local_attention) != (-1, -1)
        eff = sliding_window_mask if sliding else attention_mask
        L = int(qkv.shape[1])
        rows = sorted({0, L // 2, L - 1})
        dec = {}
        if eff is None:
            for r in rows:
                # no mask at all == every key visible
                dec[str(r)] = {"max_dist": L - 1, "allowed_keys": L}
        else:
            t = eff
            while t.dim() > 2:      # [1,1,L,L] -> [L,L]
                t = t[0]
            if t.dim() == 1:        # [L] padding mask only
                t = t.expand(L, L)
            sub = t[rows]           # only the sampled query rows: [3, L]
            allowed = (sub > -1e30).to("cpu")
            for n, r in enumerate(rows):
                idx = torch.nonzero(allowed[n], as_tuple=False).flatten()
                if idx.numel() == 0:
                    dec[str(r)] = {"max_dist": -1, "allowed_keys": 0}
                else:
                    dec[str(r)] = {"max_dist": int((idx - r).abs().max().item()),
                                   "allowed_keys": int(idx.numel())}
        self.records.append({
            "layer_id": int(getattr(module, "layer_id", -1)),
            "branch": "sliding" if sliding else "global",
            "attn_local_attention_arg": list(local_attention),
            "L": L,
            "mask_shape": None if eff is None else list(eff.shape),
            "mask_is_4d_float": bool(eff is not None and eff.dtype.is_floating_point),
            "rows": dec,
        })

    def summarise(self, L_expected):
        """Collapse the 22 per-layer records of one forward into an assertion-ready shape."""
        last = str(L_expected - 1)
        by_branch, halves, dists = {}, set(), {}
        for r in self.records:
            by_branch[r["branch"]] = by_branch.get(r["branch"], 0) + 1
            if r["branch"] == "sliding":
                halves.add(int(r["attn_local_attention_arg"][0]))
            d = r["rows"][last]["max_dist"]
            dists[str(d)] = dists.get(str(d), 0) + 1
        return {
            "forwards_recorded": len(self.records),
            "layers_total": len(self.records),
            "by_branch": by_branch,
            "sliding_half_windows_arg": sorted(halves),
            "last_query_row": last,
            "last_row_max_dist_histogram": dict(sorted(dists.items(), key=lambda kv: int(kv[0]))),
        }

    def assert_live(self, arm, expect, L):
        """Hard gate: the *observed* mask must match the arm's specification."""
        s = self.summarise(L)
        n_sliding = s["by_branch"].get("sliding", 0)
        n_global = s["by_branch"].get("global", 0)
        if n_sliding != expect["sliding"] or n_global != expect["global"]:
            raise AssertionError(
                "arm %r NOT LIVE: observed %d sliding / %d global layers, expected %d / %d"
                % (arm, n_sliding, n_global, expect["sliding"], expect["global"]))
        if expect["half"] is not None:
            if s["sliding_half_windows_arg"] != [expect["half"]]:
                raise AssertionError(
                    "arm %r NOT LIVE: sliding half-windows observed %s, expected [%d]"
                    % (arm, s["sliding_half_windows_arg"], expect["half"]))
            want = min(expect["half"], L - 1)
            for r in self.records:
                if r["branch"] != "sliding":
                    continue
                got = r["rows"][str(L - 1)]["max_dist"]
                if got != want:
                    raise AssertionError(
                        "arm %r NOT LIVE: layer %d last-row max attended distance %d, expected %d "
                        "(half-window arg %s)" % (arm, r["layer_id"], got, want,
                                                  r["attn_local_attention_arg"]))
        else:
            for r in self.records:
                got = r["rows"][str(L - 1)]["max_dist"]
                if got != L - 1:
                    raise AssertionError(
                        "arm %r NOT LIVE: layer %d is not global (last-row max distance %d < %d)"
                        % (arm, r["layer_id"], got, L - 1))
        return s


def snapshot_config(enc):
    layers = enc.layers
    rope_eq = True
    inv0 = None
    for lyr in layers:
        inv = getattr(lyr.attn.rotary_emb, "inv_freq", None)
        if inv is None:
            continue
        if inv0 is None:
            inv0 = inv.detach().clone()
        elif not torch.equal(inv0, inv.detach()):
            rope_eq = False
    return {
        "config_local_attention": int(enc.config.local_attention),
        "config_global_attn_every_n_layers": int(enc.config.global_attn_every_n_layers),
        "num_hidden_layers": len(layers),
        "max_position_embeddings": int(enc.config.max_position_embeddings),
        "attn_impl": str(enc.config._attn_implementation),
        "layer_attn_local_attention": [
            list(lyr.attn.local_attention) for lyr in layers],
        "has_attention_type_attr": bool(hasattr(layers[0], "attention_type")),
        "has_self_attn_attr": bool(hasattr(layers[0], "self_attn")),
        "layer_attention_type": [getattr(lyr, "attention_type", None) for lyr in layers],
        "all_layers_same_rope_inv_freq": bool(rope_eq),
        "n_sliding_at_load": int(sum(1 for lyr in layers if lyr.attn.local_attention != (-1, -1))),
    }


def apply_arm(enc, spec, orig):
    """Reset to the shipped configuration, then apply this arm's patch."""
    enc.config.local_attention = orig["config_local_attention"]
    enc.config.global_attn_every_n_layers = orig["config_global_attn_every_n_layers"]
    for i, lyr in enumerate(enc.layers):
        lyr.attn.local_attention = tuple(orig["layer_attn_local_attention"][i])
        if hasattr(lyr, "attention_type") and orig["layer_attention_type"][i] is not None:
            lyr.attention_type = orig["layer_attention_type"][i]
    W = spec.get("window")
    if W is not None:
        enc.config.local_attention = int(W)
        for lyr in enc.layers:
            if lyr.attn.local_attention != (-1, -1):
                lyr.attn.local_attention = (int(W) // 2, int(W) // 2)
    if spec.get("all_global"):
        enc.config.global_attn_every_n_layers = 1
        for lyr in enc.layers:
            lyr.attn.local_attention = (-1, -1)
            if hasattr(lyr, "attention_type"):
                lyr.attention_type = "full_attention"


# ---------------------------------------------------------------- documents
def build_doc(p1, tok, construction, pad, text, filler_unit):
    if construction == "p1_exact":
        # P1's own place() at needle fraction 1.0, called from the imported module.
        return p1.place(tok, pad, 1.0, text)
    if construction == "upstream_rule":
        per_rep = max(1, len(tok(filler_unit, add_special_tokens=False)["input_ids"]))
        reps = int(round(pad / per_rep))
        return filler_unit * reps + ("\n\nActual request: " if reps else "") + text
    raise ValueError(construction)


# ---------------------------------------------------------------- scoring
def score_cell(agent, probe, items, pad, text_of, construction, arm, spec, limit,
               head_max_len, head_len, questions, warmup=True, log=print):
    probe.records = []
    torch.cuda.reset_peak_memory_stats()
    per_item, lats_cold = [], []
    for k, it in enumerate(items):
        text = text_of(it, pad)
        n_state = len(agent.tok(text, add_special_tokens=False)["input_ids"])
        probe.armed = (k == 0)          # record the masks of the first forward only
        t0 = time.perf_counter()
        r = agent.predict({"text": text}, questions, max_len=limit)
        dt = time.perf_counter() - t0
        probe.armed = False
        ans = r["answers"]["department"]
        used = int(r["usage"]["input_tokens"])
        row = {
            "arm": arm, "construction": construction, "pad": pad, "limit": limit,
            "item_id": it["item_id"], "item_index": it["item_index"], "label": it["label"],
            "lang": it["lang"], "template_id": it["template_id"],
            "gold": it["label"], "pred": ans["choice"], "correct": bool(ans["choice"] == it["label"]),
            "p_gold": round(float(ans["probabilities"].get(it["label"], 0.0)), 6),
            "pred_p": round(float(ans["probabilities"][ans["choice"]]), 6),
            "probabilities": {k2: round(float(v), 6) for k2, v in ans["probabilities"].items()},
            "state_tokens": n_state, "input_tokens": used, "head_len": head_len,
            "truncated": bool(used < head_len + n_state),
            "latency_s": round(dt, 5), "latency_excluded": bool(warmup and k == 0),
        }
        per_item.append(row)
        if not (warmup and k == 0):
            lats_cold.append(dt)
    liveness = None
    if probe.records:
        Ls = sorted({r["L"] for r in probe.records})
        if len(Ls) != 1:
            raise AssertionError("arm %r: probe saw %d distinct sequence lengths %s in one cell"
                                 % (arm, len(Ls), Ls))
        liveness = probe.summarise(Ls[0])
        probe.assert_live(arm, spec["expect"], Ls[0])
    peak_a = torch.cuda.max_memory_allocated() / 2**20
    peak_r = torch.cuda.max_memory_reserved() / 2**20
    acc = sum(1 for i in per_item if i["correct"]) / max(1, len(per_item))
    cell = {
        "arm": arm, "construction": construction, "pad": pad, "limit": limit,
        "n": len(per_item), "correct": sum(1 for i in per_item if i["correct"]),
        "accuracy": round(acc, 6),
        "mean_p_gold": round(statistics.mean(i["p_gold"] for i in per_item), 6),
        "median_latency_s": round(statistics.median(lats_cold), 4) if lats_cold else None,
        "median_latency_s_all_items": round(statistics.median([i["latency_s"] for i in per_item]), 4),
        "p95_latency_s": round(_pct(lats_cold, 95), 4) if lats_cold else None,
        "max_latency_s": round(max(lats_cold), 4) if lats_cold else None,
        "latency_n": len(lats_cold),
        "truncated_items": sum(1 for i in per_item if i["truncated"]),
        "input_tokens_max": max(i["input_tokens"] for i in per_item),
        "state_tokens_median": int(statistics.median(i["state_tokens"] for i in per_item)),
        "peak_vram_alloc_mib": round(peak_a, 1), "peak_vram_reserved_mib": round(peak_r, 1),
        "liveness": liveness,
    }
    log("    pad=%-5d acc=%.3f (%2d/%d)  median=%.3fs  p95=%.3fs  peak_vram=%.0fMiB  max_in_tok=%d"
        % (pad, acc, cell["correct"], cell["n"], cell["median_latency_s"] or -1,
           cell["p95_latency_s"] or -1, peak_r, cell["input_tokens_max"]))
    return per_item, cell


def liveness_layers(probe):
    return probe.records

def _pct(vals, q):
    if not vals:
        return float("nan")
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (q / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="pilot")
    ap.add_argument("--pool", default=None, help="override stage pool")
    ap.add_argument("--n", type=int, default=None, help="override stage n")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--limit", type=int, default=8192)
    ap.add_argument("--tag", default=None, help="output tag (default: <stage>-<pool>[-n<N>])")
    ap.add_argument("--force", action="store_true", help="re-run arms that already have a summary")
    ap.add_argument("--no-warmup", action="store_true", help="time the first item too")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    pool_name = a.pool or STAGE_POOL[a.stage]
    n_items = a.n or STAGE_N[a.stage]
    seed = a.seed if a.seed is not None else STAGE_SEED[a.stage]
    cells = STAGES[a.stage]
    tag = a.tag or ("%s-%s" % (a.stage, pool_name))
    os.makedirs(RAW_DIR, exist_ok=True)

    dt_name = a.dtype
    pool = pools.load_pool(pool_name)
    built = pools.build_items(pool, n_items, seed)
    items = built["items"]
    questions = pool["question"]
    p1 = load_p1_module()

    if pool_name == "upstream_multilingual":
        assert sorted((i["text"], i["label"]) for i in items) == \
               sorted((t, g) for t, g in p1.REQUESTS), "pool items differ from P1 REQUESTS"
        assert questions == p1.QUESTIONS, "question dict differs from P1 QUESTIONS"
    filler_unit = pool.get("filler_unit") or p1.FILLER

    major = max(built["label_counts"].values()) / len(items)
    items_sha = hashlib.sha256(json.dumps(
        [(i["item_id"], i["text"], i["label"]) for i in items], ensure_ascii=False).encode()).hexdigest()
    print("items_sha256=%s" % items_sha, flush=True)
    print("tag=%s pool=%s n=%d seed=%d dtype=%s limit=%d" % (tag, pool["pool_id"], len(items),
                                                             seed, dt_name, a.limit), flush=True)
    print("labels=%s majority=%.4f balance_ok=%s" % (built["label_counts"], major,
                                                     built["balance_ok"]), flush=True)

    logdir = os.path.join(HERE, "logs")
    os.makedirs(logdir, exist_ok=True)
    agent = laya.load(MODEL_DIR, device=a.device)
    agent.dtype = getattr(torch, dt_name)
    enc = agent.model.encoder
    orig = snapshot_config(enc)
    # hash of every document actually scored, per pad (construction-independent for p1_exact /
    # upstream_rule, so keyed by pad only)
    constructions = sorted({c for _a, c, _ps in cells})
    doc_sha = {}
    for pad in sorted({p for _a, _c, ps in cells for p in ps}):
        h = hashlib.sha256()
        for c in constructions:
            for it in items:
                h.update(build_doc(p1, agent.tok, c, pad, it["text"], filler_unit).encode())
        doc_sha[str(pad)] = h.hexdigest()
    print("doc_sha256 per pad: %s" % json.dumps(doc_sha), flush=True)
    print("shipped config: local_attention=%d global_every=%d sliding_layers=%d "
          "attn=%s rope_same_all_layers=%s attention_type_attr=%s self_attn_attr=%s"
          % (orig["config_local_attention"], orig["config_global_attn_every_n_layers"],
             orig["n_sliding_at_load"], orig["attn_impl"],
             orig["all_layers_same_rope_inv_freq"], orig["has_attention_type_attr"],
             orig["has_self_attn_attr"]), flush=True)

    # head_len for the truncation control (same method as harness/builder.py::head_len)
    from laya.common import build_sequence
    qid0 = next(iter(pool["question"]))
    q_internal = pools.internal_question(pool)["questions"][qid0]
    head_seq, _ = build_sequence(agent.tok, "", q_internal, a.limit, 256, state_ids=[])
    head_len = len(head_seq) - 1
    print("head_len=%d (question + option markers)" % head_len, flush=True)

    probe = AttentionProbe()
    probe.install()

    liveness_all = {}
    live_path = os.path.join(HERE, "liveness.json")
    if os.path.exists(live_path):
        with open(live_path) as fh:
            liveness_all = json.load(fh)

    # document cache: (construction, pad, item_id) -> text
    doc_cache = {}

    def text_of(it, pad, construction):
        key = (construction, pad, it["item_id"])
        if key not in doc_cache:
            doc_cache[key] = build_doc(p1, agent.tok, construction, pad, it["text"], filler_unit)
        return doc_cache[key]

    def cache_of(construction):
        return lambda it, pad: text_of(it, pad, construction)

    for arm, construction, pads in cells:
        spec = ARM_SPECS[arm]
        base = "%s__%s__%s" % (tag, arm, construction)
        out_jsonl = os.path.join(RAW_DIR, base + ".jsonl")
        out_sum = os.path.join(RAW_DIR, base + ".summary.json")
        want = {(construction, p) for p in pads}
        if os.path.exists(out_sum) and not a.force:
            with open(out_sum) as fh:
                prev = json.load(fh)
            have = {(c["construction"], c["pad"]) for c in prev.get("cells", [])}
            if want <= have and prev.get("dtype") == dt_name and prev.get("n") == len(items):
                print("== arm %s [%s]: already complete in %s, skipping"
                      % (arm, construction, os.path.basename(out_sum)), flush=True)
                continue
        print("== arm %-22s construction=%-14s pads=%s" % (arm, construction, pads), flush=True)
        apply_arm(enc, spec, orig)
        applied = snapshot_config(enc)
        print("   patch: config.local_attention=%d global_every=%d sliding_layers=%d"
              % (applied["config_local_attention"], applied["config_global_attn_every_n_layers"],
                 applied["n_sliding_at_load"]), flush=True)
        rows, s_cells = [], []
        for pad in pads:
            per_item, cell = score_cell(agent, probe, items, pad, cache_of(construction),
                                        construction, arm, spec, a.limit, 256, head_len,
                                        questions, warmup=not a.no_warmup)
            rows.extend(per_item)
            s_cells.append(cell)
            print("   liveness pad=%d: %s" % (pad, json.dumps(cell["liveness"])), flush=True)
        summary = {
            "tag": tag, "arm": arm, "construction": construction, "pool": pool["pool_id"],
            "pool_sha256": pool["_sha256"], "n": len(items), "seed": seed, "dtype": dt_name,
            "items_sha256": items_sha, "doc_sha256_per_pad": doc_sha,
            "harness_pools_py_sha256": _sha256_file(os.path.join(LAB, "experiments", "harness", "pools.py")),
            "harness_run_py_sha256": _sha256_file(os.path.abspath(__file__)),
            "limit": a.limit, "device": a.device,
            "agent_dtype": str(agent.dtype), "agent_amp_enabled": bool(agent.amp_enabled),
            "gpu": torch.cuda.get_device_name(0) if a.device == "cuda" else None,
            "arm_spec": spec, "cells": s_cells,
            "config_before": orig, "config_after_patch": applied,
            "label_counts": built["label_counts"], "majority_class_accuracy": round(major, 6),
            "balance_ok": built["balance_ok"], "head_len": head_len,
            "warmup_first_item_excluded": not a.no_warmup,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "argv": sys.argv,
        }
        tmp = out_sum + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(summary, fh, indent=1, ensure_ascii=False)
        os.replace(tmp, out_sum)
        tmp = out_jsonl + ".tmp"
        with open(tmp, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, out_jsonl)
        liveness_all[base] = {
            "pool": pool["pool_id"], "n": len(items), "dtype": dt_name,
            "cells": [{"pad": c["pad"], "liveness": c["liveness"]} for c in s_cells],
            "config_after_patch": applied,
        }
        print("   wrote %s (%d rows) + %s" % (os.path.basename(out_jsonl), len(rows),
                                              os.path.basename(out_sum)), flush=True)

    tmp = live_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(liveness_all, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, live_path)
    print("wrote liveness.json (%d arms)" % len(liveness_all), flush=True)
    print("python=%s torch=%s transformers=%s"
          % (platform.python_version(), torch.__version__,
             __import__("transformers").__version__), flush=True)


if __name__ == "__main__":
    main()

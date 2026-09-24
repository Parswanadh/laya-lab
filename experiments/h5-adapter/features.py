"""Cache the frozen encoder's output once, for every training item and every evaluation cell.

Why cache at all: the encoder is frozen for all four arms, so its output is arm-independent and
identical across training epochs and across seeds. Running it once turns "train four
configurations at three seeds" from 12 encoder passes per epoch into one.

Precision: the shipped checkpoint's weights are fp16 and `Agent._infer` runs under fp16 autocast,
so the cache stores fp16 and does not quantise anything the model was not already computing in.
`--fidelity` re-runs a sample single-item (unbatched) and compares it against the cached batched
values, so "batching did not change the representation" is measured rather than assumed.

Layout: one flat memmap `features.f16` of shape ``[total_tokens, hidden]`` plus an `index.json`
carrying each item's ``(offset, length, markers, state_start, ...)``. A single file avoids
thousands of small reads per epoch; the index is JSON so a verifier can read it without this code.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(LAB, "worktrees", "h5"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

QTYPE_CHOICE = 0
DEFAULT_CACHE_DIR = os.path.join(HERE, "cache")


# ---------------------------------------------------------------------------- eval conditions
def build_eval_conditions(plan: Dict[str, Any], perm_seed: int = 20260924) -> List[Dict[str, Any]]:
    """The plan's eval items, plus a per-item option-order permutation of the L4000 needle@END cell.

    The permuted entries are *the same items* -- same needles, same haystacks, same labels -- with
    the four options reordered. Reporting both is what separates "read the document" from "answer
    by option index".
    """
    import random

    out: List[Dict[str, Any]] = []
    for it in plan["eval_items"]:
        base = dict(it)
        base["option_order"] = None
        out.append(base)
        if it["cell"] == "L4000-p100":
            order = [0, 1, 2, 3]
            random.Random("%d|perm|%s" % (perm_seed, it["item_id"])).shuffle(order)
            perm = dict(it)
            perm["item_id"] = it["item_id"] + "|perm"
            perm["cell"] = it["cell"] + "-perm"
            perm["option_order"] = order
            perm["label_index_canonical"] = None
            out.append(perm)
    return out


def label_index(labels: Sequence[str], label: str) -> int:
    return list(labels).index(label)


def target_index(item: Dict[str, Any], labels: Sequence[str]) -> int:
    """The marker slot holding the gold option, given the item's option order.

    ``build_sequence`` emits markers in ``option_order`` order, so with the canonical order the
    slot is the label's own index and under a permutation it is the permuted position.
    """
    canon = label_index(labels, item["label"])
    order = item.get("option_order")
    if order is None:
        return canon
    return list(order).index(canon)


def slot_to_label(item: Dict[str, Any], labels: Sequence[str], slot: int) -> str:
    """Inverse of ``target_index``: which label a predicted marker slot denotes."""
    order = item.get("option_order")
    canon = slot if order is None else list(order)[slot]
    return list(labels)[canon]


# ---------------------------------------------------------------------------- the store
class FeatureStore:
    """Read-only view over a built feature cache."""

    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "index.json"), encoding="utf-8") as fh:
            self.index = json.load(fh)
        self.items: List[Dict[str, Any]] = self.index["items"]
        self.hidden = int(self.index["hidden"])
        self._mm: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.items)

    @property
    def mm(self) -> np.ndarray:
        if self._mm is None:
            self._mm = np.memmap(os.path.join(self.path, self.index["file"]), dtype=np.float16,
                                 mode="r", shape=(self.index["total_tokens"], self.hidden))
        return self._mm

    def hidden_states(self, i: int) -> np.ndarray:
        it = self.items[i]
        return np.asarray(self.mm[it["offset"]:it["offset"] + it["length"]])

    def ids_of(self, cells: Optional[Sequence[str]] = None, split: Optional[str] = None) -> List[int]:
        out = []
        for i, it in enumerate(self.items):
            if cells is not None and it["cell"] not in cells:
                continue
            if split is not None and it["split"] != split:
                continue
            out.append(i)
        return out


def collate(store: FeatureStore, indices: Sequence[int], device, dtype=torch.float32):
    """One batch of cached features, padded to the batch's longest sequence."""
    rows = [store.items[i] for i in indices]
    L = max(r["length"] for r in rows)
    b = len(rows)
    h = torch.zeros(b, L, store.hidden, dtype=dtype)
    att = torch.zeros(b, L, dtype=torch.long)
    kmax = max(len(r["markers"]) for r in rows)
    mpos = torch.zeros(b, kmax, dtype=torch.long)
    mmask = torch.zeros(b, kmax, dtype=torch.bool)
    start = torch.zeros(b, dtype=torch.long)
    for j, (i, r) in enumerate(zip(indices, rows)):
        n = r["length"]
        h[j, :n] = torch.from_numpy(store.hidden_states(i).astype(np.float32))
        att[j, :n] = 1
        k = len(r["markers"])
        mpos[j, :k] = torch.tensor(r["markers"], dtype=torch.long)
        mmask[j, :k] = True
        start[j] = r["state_start"]
    return {
        "h": h.to(device), "attention_mask": att.to(device),
        "marker_pos": mpos.to(device), "marker_mask": mmask.to(device),
        "state_start": start.to(device),
        "qtype": torch.zeros(b, dtype=torch.long, device=device),
        "meta": rows,
        "indices": list(indices),
    }


# ---------------------------------------------------------------------------- building
def _token_budget_batches(lengths: Sequence[int], token_budget: int, max_batch: int):
    """Group items so no batch exceeds ``token_budget`` padded tokens.

    Items are visited longest-first, which both keeps padding waste low and puts the shapes most
    likely to OOM at the front where the batch is smallest.
    """
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batch: List[int] = []
    for i in order:
        candidate = batch + [i]
        padded = len(candidate) * max(lengths[j] for j in candidate)
        if batch and (padded > token_budget or len(candidate) > max_batch):
            yield batch
            batch = [i]
        else:
            batch = candidate
    if batch:
        yield batch


def build_cache(items: List[Dict[str, Any]], split: str, out_dir: str, model, builder,
                device, token_budget: int = 8192, max_batch: int = 8,
                progress_every: int = 64) -> str:
    """Run the frozen encoder over ``items`` and write ``features.f16`` + ``index.json``."""
    os.makedirs(out_dir, exist_ok=True)
    # Lengths come from the builder's estimate so the whole document set is never materialised at
    # once; each batch is built, written and released. `build()` is asserted against the estimate
    # below, so a drifting estimate raises rather than silently mis-batching.
    lengths: List[int] = [builder.estimate_length(it) for it in items]
    t0 = time.time()
    total_tokens = int(sum(lengths))
    hidden = int(model.encoder.config.hidden_size)
    path = os.path.join(out_dir, "features.f16")
    mm = np.memmap(path, dtype=np.float16, mode="w+", shape=(total_tokens, hidden))

    offset = 0
    index_items: List[Dict[str, Any]] = []
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for bi, batch in enumerate(_token_budget_batches(lengths, token_budget, max_batch)):
            L = max(lengths[i] for i in batch)
            b = len(batch)
            built = {i: builder.build(items[i], option_order=items[i].get("option_order"))
                     for i in batch}
            for i in batch:
                if built[i]["input_tokens"] != lengths[i]:
                    raise AssertionError(
                        "builder estimate %d != built length %d for %s"
                        % (lengths[i], built[i]["input_tokens"], items[i]["item_id"]))
            ids = torch.zeros(b, L, dtype=torch.long)
            att = torch.zeros(b, L, dtype=torch.long)
            for j, i in enumerate(batch):
                seq = built[i]["input_ids"]
                ids[j, :len(seq)] = torch.tensor(seq, dtype=torch.long)
                att[j, :len(seq)] = 1
            ids, att = ids.to(device), att.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                h = model.encoder(input_ids=ids, attention_mask=att).last_hidden_state
            h = h.float().cpu().numpy().astype(np.float16)
            for j, i in enumerate(batch):
                n = lengths[i]
                mm[offset:offset + n] = h[j, :n]
                doc = built[i]
                index_items.append({
                    "item_id": item_key(items, i),
                    "split": split,
                    "cell": item_cell(items, i),
                    "label": item_label(items, i),
                    "lang": item_lang(items, i),
                    "template_id": item_template(items, i),
                    "option_order": item_option_order(items, i),
                    "pad": item_pad(items, i),
                    "needle_position": item_needle_position(items, i),
                    "offset": offset,
                    "length": n,
                    "hidden": hidden,
                    "markers": list(doc["markers"]),
                    "state_start": int(doc["state_start"]),
                    "head_len": int(doc["head_len"]),
                    "state_tokens_full": int(doc["state_tokens_full"]),
                    "state_tokens_kept": int(doc["state_tokens_kept"]),
                    "truncated": bool(doc["truncated"]),
                    "needle_tokens": int(doc["needle_tokens"]),
                    "needle_token_start": int(doc["needle_token_start"]),
                    "needle_tokens_kept": int(doc["needle_tokens_kept"]),
                    "needle_kept": bool(doc["needle_kept"]),
                    "request_kept": bool(doc["request_kept"]),
                    "pad_exact": bool(doc["pad_exact"]),
                    "state_sha256": doc["state_sha256"],
                    "filler_tokens_before": int(doc["filler_tokens_before"]),
                })
                offset += n
            del built
            if (bi + 1) % max(1, progress_every // max_batch) == 0:
                print("  [%s] %d/%d items, %d tokens, %.0fs"
                      % (split, len(index_items), len(items), offset, time.time() - t0),
                      flush=True)
    mm.flush()
    del mm
    if was_training:
        model.train()
    index = {
        "split": split,
        "file": "features.f16",
        "dtype": "float16",
        "hidden": hidden,
        "total_tokens": total_tokens,
        "n_items": len(index_items),
        "encoder_dtype_note": ("the checkpoint's weights are fp16 and Agent._infer runs under fp16 "
                              "autocast, so this cache stores exactly what the shipped path "
                              "computes"),
        "built_seconds": round(time.time() - t0, 1),
        "items": index_items,
    }
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1)
        fh.write("\n")
    print("  [%s] wrote %d items / %d tokens / %.1f GB in %.0fs"
          % (split, len(index_items), total_tokens, total_tokens * hidden * 2 / 1e9, time.time() - t0),
          flush=True)
    return out_dir


# the cache builder is generic over two item shapes (plan items and eval conditions); these
# accessors keep both working without the caller having to normalise them first
def item_key(items, i):
    return items[i]["item_id"]


def item_cell(items, i):
    return items[i]["cell"]


def item_label(items, i):
    return items[i]["label"]


def item_lang(items, i):
    return items[i].get("lang", "en")


def item_template(items, i):
    return items[i]["template_id"]


def item_option_order(items, i):
    return items[i].get("option_order")


def item_pad(items, i):
    return items[i]["pad"]


def item_needle_position(items, i):
    return items[i]["needle_position"]

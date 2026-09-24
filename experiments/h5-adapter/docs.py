"""Build H5 documents.

The only thing this module does that the harness's ``DocBuilder`` does not is control *what varies*
between the cells of the position sweep. The harness seeds its filler on ``item_id``, and the
item id carries the needle position, so moving the needle also reshuffles the haystack -- which
makes "accuracy falls with distance" a statement about five different documents.

Here the filler is built once per ``(needle, pad)`` and the needle is **inserted into the filler's
token ids** at ``round(position * pad)``. So the five L=4000 position cells are built from
byte-identical filler with the needle at a different offset, and the only thing that changes
between them is the distance the model must carry the evidence.

Working in token ids rather than text also removes the harness's decode/re-encode round trip: the
state is exactly ``pad + len(needle_block)`` tokens, so ``pad`` is exact by construction and the
position is exact to the token. ``build_sequence`` accepts ``state_ids`` directly, so the
library's own sequence construction is still what produces the prompt.
"""
from __future__ import annotations

import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(os.path.dirname(HERE))
FORK = os.path.join(LAB, "worktrees", "h5")
if FORK not in sys.path:
    sys.path.insert(0, FORK)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from laya.common import build_sequence  # noqa: E402

NEEDLE_PREFIX = "\n\nActual request: "
QTYPE_CHOICE = 0


class H5DocBuilder:
    """Deterministic, token-exact document builder for the H5 plan."""

    def __init__(self, tok, filler_pool: Dict[str, Any], internal_question: Dict[str, Any],
                 max_len: int = 8192, head_max_len: int = 256, seed: int = 20260924):
        self.tok = tok
        self.filler_sentences: List[str] = list(filler_pool["sentences"])
        self.q = internal_question
        self.max_len = int(max_len)
        self.head_max_len = int(head_max_len)
        self.seed = int(seed)
        self._head_len: Optional[int] = None
        # per (needle_sha256, pad): the filler ids, so a sweep reuses one haystack
        self._filler_cache: Dict[Tuple[str, int], List[int]] = {}

    # ---------------- pieces ----------------

    def head_len(self) -> int:
        """Tokens before the state: ``[CLS] <type> ins [SEP] <marker+option>... [SEP]``."""
        if self._head_len is None:
            seq, _markers = build_sequence(self.tok, "", self.q, self.max_len, self.head_max_len,
                                           state_ids=[])
            self._head_len = len(seq) - 1
        return self._head_len

    def _filler_ids(self, needle_sha256: str, pad: int) -> List[int]:
        """Exactly ``pad`` filler tokens, seeded on the needle and the pad **but not the position**."""
        key = (needle_sha256, pad)
        if key in self._filler_cache:
            return self._filler_cache[key]
        if pad <= 0:
            ids: List[int] = []
        else:
            rng = random.Random("%d|h5-filler|%s|%d" % (self.seed, needle_sha256, pad))
            sentences = list(self.filler_sentences)
            text = ""
            while len(self.tok(text, add_special_tokens=False)["input_ids"]) < pad:
                rng.shuffle(sentences)
                text += " ".join(sentences) + " "
            ids = self.tok(text, add_special_tokens=False)["input_ids"][:pad]
        if len(ids) != pad:
            raise AssertionError("filler is %d tokens, wanted exactly %d" % (len(ids), pad))
        self._filler_cache[key] = ids
        return ids

    # ---------------- build ----------------

    def build(self, item: Dict[str, Any], option_order: Optional[List[int]] = None) -> Dict[str, Any]:
        pad = int(item["pad"])
        position = float(item["needle_position"])
        filler = self._filler_ids(item["needle_sha256"], pad)
        n_before = int(round(position * pad))
        n_after = pad - n_before

        prefix_ids = self.tok(NEEDLE_PREFIX, add_special_tokens=False)["input_ids"] if pad > 0 else []
        request_ids = self.tok(item["needle_text"], add_special_tokens=False)["input_ids"]
        block = prefix_ids + request_ids

        state_ids = filler[:n_before] + block + filler[n_before:]
        if len(state_ids) != pad + len(block):
            raise AssertionError("state is %d tokens, expected %d" % (len(state_ids), pad + len(block)))

        state_text = self.tok.decode(state_ids)
        seq, markers = build_sequence(self.tok, state_text, self.q, self.max_len, self.head_max_len,
                                      option_order=option_order, state_ids=state_ids)
        if len(markers) != self.n_options:
            raise ValueError("question has %d options but build_sequence returned %d markers"
                             % (self.n_options, len(markers)))

        head_len = self.head_len()
        kept = max(0, len(seq) - head_len - 1)
        # The library right-truncates a string state (`state_ids[:room]`), so the needle survives
        # only if it starts inside the kept prefix. Every H5 cell is inside the 8192 budget; the
        # harness records the fact per row rather than assuming it.
        needle_start = n_before
        needle_end = n_before + len(block)
        request_start = n_before + len(prefix_ids)
        request_end = needle_end

        def kept_of(a: int, b: int) -> int:
            return max(0, min(b, kept) - a)

        return {
            "input_ids": seq,
            "markers": markers,
            "state_start": head_len,
            "head_len": head_len,
            "state_tokens_full": len(state_ids),
            "state_tokens_kept": kept,
            "input_tokens": len(seq),
            "truncated": kept < len(state_ids),
            "needle_token_start": needle_start,
            "needle_token_end": needle_end,
            "needle_tokens": len(block),
            "needle_tokens_kept": kept_of(needle_start, needle_end),
            "needle_kept": kept_of(needle_start, needle_end) == len(block),
            "request_token_start": request_start,
            "request_token_end": request_end,
            "request_tokens": len(request_ids),
            "request_tokens_kept": kept_of(request_start, request_end),
            "request_kept": request_ids and kept_of(request_start, request_end) == len(request_ids),
            "filler_tokens_before": n_before,
            "filler_tokens_after": n_after,
            "pad_exact": len(filler) == pad,
            "option_order": list(option_order) if option_order is not None else None,
            "state_sha256": _sha256_ids(state_ids),
            "doc_preview_head": state_text[:80],
            "doc_preview_tail": state_text[-80:],
            "state_text": state_text,
        }

    @property
    def n_options(self) -> int:
        return len(self.q["crit"])


def _sha256_ids(ids: List[int]) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(json.dumps(ids, separators=(",", ":")).encode())
    return h.hexdigest()

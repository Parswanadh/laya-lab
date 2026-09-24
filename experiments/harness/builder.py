"""Deterministic needle-in-haystack document construction and truncation diagnostics.

Two filler modes:

``exact_tokens``
    The filler is built to a *token* budget with the live tokenizer and recorded to the token:
    the composite is tokenized, sliced to the target, decoded back to text, and the round trip is
    re-checked. ``filler_tokens_actual`` is the observed count of the final document, not the
    request, and ``pad_exact`` says whether the round trip landed exactly.

``repeat_unit``
    Upstream's rule, transcribed: ``reps = round(pad / tokens(FILLER_UNIT))`` and the unit string
    is repeated ``reps`` times. Kept because the whole point of the upstream arm is to be able to
    say the composition follows upstream's rule exactly.

The needle sits at ``position`` fraction of the filler: ``round(position * pad)`` filler tokens
before it, the rest after.

Nothing here mutates the fork. ``build_sequence`` is called read-only, both to measure the prompt
head and to check the truncation the library actually performs against the truncation its source
documents (``keep the head for a string state``), so a silent change in that rule shows up as
``trunc_rule_ok: false`` in every raw row instead of quietly changing the numbers.

``build()`` does the max_len-independent work once per (item, pad, position) and can be cached
across the ``max_len`` cells; ``diagnose()`` adds the per-budget truncation facts.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from pools import sha256_text

NEEDLE_PREFIX = "\n\nActual request: "
JOIN_STYLE_EXACT = "newline_join"
JOIN_STYLE_CONCAT = "concat"


class DocBuilder:
    def __init__(self, tok, needle_pool: Dict[str, Any], filler_pool: Dict[str, Any],
                 internal_questions: Dict[str, Any]):
        self.tok = tok
        self.needle_pool = needle_pool
        self.filler_pool = filler_pool
        self.internal_questions = internal_questions
        self.qid = next(iter(internal_questions))
        self.q_internal = internal_questions[self.qid]
        self.filler_sentences: List[str] = list(filler_pool["sentences"])
        self.mask_token = getattr(tok, "mask_token", None)
        self.span_method = "unknown"
        self._head_len_cache: Dict[Tuple[int, int], int] = {}

    # ---------------- tokenizer helpers ----------------

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def _sentence_stream(self, rng: random.Random):
        while True:
            order = list(self.filler_sentences)
            rng.shuffle(order)
            for s in order:
                yield s

    def _filler_exact(self, target: int, rng: random.Random) -> Tuple[str, int, bool]:
        """Filler text of as close to ``target`` tokens as the tokenizer round trip allows."""
        if target <= 0:
            return "", 0, True
        stream = self._sentence_stream(rng)
        text = ""
        while self.count(text) < target + 4:
            text += next(stream) + " "
            if len(text) > 8_000_000:
                raise RuntimeError("filler construction ran away at target=%d" % target)
        ids = self.tok(text, add_special_tokens=False)["input_ids"][:target]
        text = self.tok.decode(ids)
        exact = False
        for _ in range(12):
            n = self.count(text)
            if n == target:
                exact = True
                break
            if n > target:
                text = self.tok.decode(self.tok(text, add_special_tokens=False)["input_ids"][:target])
            else:
                text = text + " " + next(stream)
                text = self.tok.decode(self.tok(text, add_special_tokens=False)["input_ids"][:target])
        return text, self.count(text), exact

    def _prefix(self, pad: int) -> str:
        mode = self.needle_pool.get("needle_prefix_mode", "upstream_auto")
        if mode == "always":
            return NEEDLE_PREFIX
        if mode == "never":
            return ""
        # "upstream_auto": upstream added the delimiter only when there was filler in front
        return NEEDLE_PREFIX if pad > 0 else ""

    # ---------------- construction ----------------

    def build(self, item: Dict[str, Any], pad: int, position: Optional[float], seed: int) -> Dict[str, Any]:
        if position is None:
            position = 1.0
        rng = random.Random("%d|%s|%s|%s|%s" % (seed, self.needle_pool["pool_id"], item["item_id"],
                                               pad, position))
        prefix = self._prefix(pad)
        extra: Dict[str, Any] = {}

        if self.needle_pool.get("kind") == "upstream":
            unit = self.needle_pool["filler_unit"]
            per_rep = max(1, self.count(unit))
            reps = int(round(pad / per_rep))
            before_reps = int(round(position * reps))
            after_reps = reps - before_reps
            left = unit * before_reps
            right = unit * after_reps
            join_style = JOIN_STYLE_CONCAT
            pad_exact: Optional[bool] = None
            extra = {"reps": reps, "per_rep_tokens": per_rep,
                     "filler_before_reps": before_reps, "filler_after_reps": after_reps}
        else:
            before_target = int(round(position * pad))
            after_target = pad - before_target
            left, _lt, left_exact = self._filler_exact(before_target, rng)
            right, _rt, right_exact = self._filler_exact(after_target, rng)
            join_style = JOIN_STYLE_EXACT
            pad_exact = bool(left_exact and right_exact)
            extra = {"filler_before_target": before_target, "filler_after_target": after_target,
                     "filler_before_exact": left_exact, "filler_after_exact": right_exact}

        pieces: List[str] = []
        sep = "\n" if join_style == JOIN_STYLE_EXACT else ""
        if left:
            pieces.append(left)
        needle_char_start = sum(len(p) for p in pieces) + (len(sep) * len(pieces))
        needle_text = prefix + item["text"]
        pieces.append(needle_text)
        if right:
            pieces.append(right)
        state = sep.join(pieces)

        needle_char_end = needle_char_start + len(needle_text)
        request_char_start = needle_char_start + len(prefix)
        request_char_end = needle_char_end

        if self.mask_token and self.mask_token in state:
            raise ValueError("document contains the tokenizer mask token %r; spans would be wrong"
                             % self.mask_token)

        state_ids, offsets, span_method = self._tokenize_with_offsets(state)
        ns, ne = self._span(offsets, state, needle_char_start, needle_char_end, span_method)
        rs, re_ = self._span(offsets, state, request_char_start, request_char_end, span_method)

        doc = {
            "state": state,
            "state_ids": state_ids,
            "doc_sha256": sha256_text(state),
            "doc_chars": len(state),
            "pad_requested": pad,
            "needle_position": position,
            "needle_prefix": prefix,
            "compose_style": join_style,
            "span_method": span_method,
            "state_tokens_full": len(state_ids),
            "needle_token_start": ns,
            "needle_token_end": ne,
            "needle_tokens": 0 if (ns is None or ne is None) else ne - ns,
            "request_token_start": rs,
            "request_token_end": re_,
            "request_tokens": 0 if (rs is None or re_ is None) else re_ - rs,
            "filler_tokens_before": 0 if ns is None else ns,
            "filler_tokens_after": 0 if (ne is None) else max(0, len(state_ids) - ne),
            "filler_tokens_actual": (len(state_ids) - (0 if (ns is None or ne is None) else ne - ns)),
            "pad_exact": pad_exact,
            "preview_head": state[:80],
            "preview_tail": state[-80:],
        }
        doc.update(extra)
        return doc

    # ---------------- tokenization + spans ----------------

    def _tokenize_with_offsets(self, state: str):
        try:
            enc = self.tok(state, add_special_tokens=False, return_offsets_mapping=True)
            offsets = enc.get("offset_mapping")
            if offsets is not None and len(offsets) == len(enc["input_ids"]):
                return enc["input_ids"], list(offsets), "offsets"
        except Exception:
            pass
        return self.tok(state, add_special_tokens=False)["input_ids"], None, "prefix_count"

    def _span(self, offsets, state: str, a: int, b: int, method: str):
        if b <= a:
            return None, None
        if offsets is not None:
            idx = [i for i, (s, e) in enumerate(offsets) if e > a and s < b]
            if not idx:
                return None, None
            return idx[0], idx[-1] + 1
        start = self.count(state[:a])
        return start, start + self.count(state[a:b])

    # ---------------- truncation diagnostics ----------------

    def head_len(self, max_len: int, head_max_len: int) -> int:
        key = (int(max_len), int(head_max_len))
        if key not in self._head_len_cache:
            from laya.common import build_sequence
            head_seq, _ = build_sequence(self.tok, "", self.q_internal, key[0], key[1], state_ids=[])
            self._head_len_cache[key] = len(head_seq) - 1
        return self._head_len_cache[key]

    def diagnose(self, doc: Dict[str, Any], max_len: int, head_max_len: int) -> Dict[str, Any]:
        """What the library's own ``build_sequence`` does to this document at this budget."""
        from laya.common import build_sequence

        state_ids = doc["state_ids"]
        head_len = self.head_len(max_len, head_max_len)
        seq, markers = build_sequence(self.tok, doc["state"], self.q_internal, max_len, head_max_len,
                                      state_ids=state_ids)
        kept = max(0, len(seq) - head_len - 1)
        room = max(0, max_len - head_len - 1)
        rule_ok = kept == min(len(state_ids), room)

        def kept_of(start: Optional[int], end: Optional[int]) -> int:
            if start is None or end is None:
                return 0
            return max(0, min(end, kept) - start)

        needle_kept = kept_of(doc["needle_token_start"], doc["needle_token_end"])
        request_kept = kept_of(doc["request_token_start"], doc["request_token_end"])
        return {
            "state_tokens_full": len(state_ids),
            "state_tokens_kept": kept,
            "input_tokens": len(seq),
            "head_len": head_len,
            "room": room,
            "truncated": kept < len(state_ids),
            "trunc_rule_ok": rule_ok,
            "needle_tokens_kept": needle_kept,
            "needle_kept": needle_kept > 0,
            "request_tokens_kept": request_kept,
            "request_kept": doc["request_tokens"] > 0 and request_kept == doc["request_tokens"],
            "markers": markers,
        }


DOC_ROW_FIELDS = (
    "doc_sha256", "doc_chars", "compose_style", "span_method",
    "state_tokens_full", "needle_token_start", "needle_token_end", "needle_tokens",
    "request_token_start", "request_token_end", "request_tokens",
    "filler_tokens_before", "filler_tokens_after", "filler_tokens_actual", "pad_exact",
    "preview_head", "preview_tail",
)


def doc_row_fields(doc: Dict[str, Any]) -> Dict[str, Any]:
    """The serializable subset of a built document (no token ids, no full text)."""
    out = {k: doc[k] for k in DOC_ROW_FIELDS if k in doc}
    for k in ("reps", "per_rep_tokens", "filler_before_reps", "filler_after_reps",
              "filler_before_target", "filler_after_target", "filler_before_exact", "filler_after_exact"):
        if k in doc:
            out[k] = doc[k]
    return out

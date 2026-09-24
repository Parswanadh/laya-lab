"""Pool loading, validation, leakage checks and deterministic item construction.

Two pool shapes are supported:

* ``kind == "balanced"``  — needle templates tagged with a label and a language, plus numeric
  slots. Items are drawn so the label distribution is balanced and the language distribution is
  balanced *within* each label.
* ``kind == "upstream"``  — a verbatim transcription of the 20 requests of
  ``fork/research/scripts/bench_long_context.py``, used only for a faithful replication of the
  upstream run. Its label distribution is unbalanced by construction (11/6/3) and that is
  recorded in the pool provenance.

Everything here is pure Python: no torch, no checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

HARNESS_DIR = Path(__file__).resolve().parent
POOL_DIR = HARNESS_DIR / "pools"

POOL_FILES = {
    "balanced-v1": "needles-balanced-v1.json",
    "filler-v1": "filler-v1.json",
    "upstream_multilingual": "upstream-multilingual.json",
}

# label order used for the round-robin balance; the *answerable* labels of the question
BALANCED_LABELS = ("billing", "technical", "sales")

# Function words and generic ordinals/quantifiers. Shared between filler and needle these are
# unavoidable and carry no label signal; the pass/fail leakage test ignores them and the raw
# overlap is still reported in full so nothing is hidden. Domain content words are NOT here --
# those are what the stem check is for.
STOPWORDS = {
    "the", "and", "for", "was", "were", "with", "without", "that", "this", "these", "those",
    "have", "has", "had", "not", "but", "you", "your", "yours", "our", "ours", "their", "theirs",
    "its", "it's", "from", "they", "them", "then", "than", "there", "here", "when", "what",
    "which", "who", "whom", "whose", "how", "why", "all", "any", "can", "could", "should",
    "would", "will", "shall", "may", "might", "must", "about", "after", "before", "again",
    "every", "each", "first", "second", "third", "fourth", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine", "ten", "twenty", "thirty", "past", "next", "last",
    "more", "most", "much", "many", "very", "just", "only", "also", "into", "onto", "over",
    "under", "between", "because", "while", "during", "against", "off", "out", "down", "same",
    "other", "others", "else", "please", "want", "wants", "need", "needs", "like", "get", "gets",
    "got", "use", "used", "uses", "make", "makes", "made", "take", "takes", "taken", "time",
    "times", "since", "still", "yet", "ever", "never", "always", "sometimes", "where", "some",
    "something", "someone", "anything", "nothing", "everything", "such", "own", "too", "now",
    "day", "days", "week", "weeks", "month", "months", "year", "years", "hour", "hours",
    "minute", "minutes", "number", "amount", "total", "left", "right", "open", "close",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_pool_path(name_or_path: str) -> Path:
    if name_or_path in POOL_FILES:
        return POOL_DIR / POOL_FILES[name_or_path]
    p = Path(name_or_path)
    if not p.is_absolute():
        p = HARNESS_DIR / p
    if not p.exists():
        raise FileNotFoundError("pool not found: %s (known: %s)" % (name_or_path, sorted(POOL_FILES)))
    return p


def load_pool(name_or_path: str) -> Dict[str, Any]:
    path = resolve_pool_path(name_or_path)
    with open(path, encoding="utf-8") as fh:
        pool = json.load(fh)
    pool["_path"] = str(path)
    pool["_sha256"] = sha256_file(path)
    pool["_bytes"] = path.stat().st_size
    return pool


def words(text: str) -> List[str]:
    """Lowercased word tokens, unicode-aware enough for the scripts used here."""
    out, cur = [], []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


def criterion_text(question: Dict[str, Any]) -> str:
    parts = []
    for qdef in question.values():
        parts.append(str(qdef.get("instructions", "")))
        crit = qdef.get("criteria")
        if isinstance(crit, dict):
            for k, v in crit.items():
                parts.append(str(k))
                parts.append(json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v)
        elif isinstance(crit, list):
            parts.extend(str(c) for c in crit)
    return " ".join(parts)


def leakage_report(needle_pool: Dict[str, Any], filler_pool: Dict[str, Any]) -> Dict[str, Any]:
    """Check that filler and needle come from disjoint pools.

    Two checks, both reported so a verifier can re-derive them:

    * whole-word overlap between the filler sentences and (needle texts + question criteria),
    * substring overlap between the filler sentences and an explicit list of support-domain
      stems, which catches a filler word that merely *contains* a cue ("billing" inside
      "billington") where whole-word matching would pass.
    """
    needle_texts = []
    if needle_pool.get("kind") == "balanced":
        needle_texts = [t["text"] for t in needle_pool["templates"]]
    else:
        needle_texts = [r["text"] for r in needle_pool["requests"]]
    needle_vocab = set()
    for text in needle_texts + [criterion_text(needle_pool["question"])]:
        needle_vocab.update(w for w in words(text) if len(w) >= 3)

    filler_vocab = set()
    for sentence in filler_pool["sentences"]:
        filler_vocab.update(w for w in words(sentence) if len(w) >= 3)

    word_overlap = sorted(needle_vocab & filler_vocab)
    content_overlap = sorted(w for w in (needle_vocab & filler_vocab) if w not in STOPWORDS)

    stems = [s.lower() for s in filler_pool.get("forbidden_stems", [])]
    stem_hits = []
    for sentence in filler_pool["sentences"]:
        low = sentence.lower()
        for stem in stems:
            if stem in low:
                stem_hits.append({"sentence": sentence, "stem": stem})

    return {
        "filler_pool": filler_pool["pool_id"],
        "needle_pool": needle_pool["pool_id"],
        "needle_vocab_size": len(needle_vocab),
        "filler_vocab_size": len(filler_vocab),
        "word_overlap_all": word_overlap,
        "content_word_overlap": content_overlap,
        "stem_hits": stem_hits,
        "passed": not content_overlap and not stem_hits,
    }


def _slot_combos(template: Dict[str, Any], slots: Dict[str, List[str]]) -> List[Dict[str, str]]:
    """Every deterministic slot fill for one template (cartesian over the slots it uses)."""
    used = [name for name in slots if "{%s}" % name in template["text"]]
    combos: List[Dict[str, str]] = [{}]
    for name in used:
        nxt = []
        for base in combos:
            for value in slots[name]:
                d = dict(base)
                d[name] = value
                nxt.append(d)
        combos = nxt
    return combos


def render_template(text: str, fill: Dict[str, str]) -> str:
    out = text
    for k, v in fill.items():
        out = out.replace("{%s}" % k, v)
    return out


def build_items(pool: Dict[str, Any], n: int, seed: int) -> Dict[str, Any]:
    """Deterministic item list. Same (pool, n, seed) -> byte-identical items, in any process."""
    if pool.get("kind") == "balanced":
        items = _balanced_items(pool, n, seed)
    else:
        items = _upstream_items(pool, n)
    order = random.Random("%d|order|%s" % (seed, pool["pool_id"]))
    order.shuffle(items)
    for i, it in enumerate(items):
        it["item_index"] = i
    label_counts, lang_counts, template_counts = {}, {}, {}
    for it in items:
        label_counts[it["label"]] = label_counts.get(it["label"], 0) + 1
        lang_counts[it["lang"]] = lang_counts.get(it["lang"], 0) + 1
        template_counts[it["template_id"]] = template_counts.get(it["template_id"], 0) + 1
    return {
        "items": items,
        "n": len(items),
        "label_counts": dict(sorted(label_counts.items())),
        "lang_counts": dict(sorted(lang_counts.items())),
        "template_counts": dict(sorted(template_counts.items())),
        "max_label_share": max(label_counts.values()) / len(items),
        "balance_ok": max(label_counts.values()) - min(label_counts.values()) <= 1,
    }


def _balanced_items(pool: Dict[str, Any], n: int, seed: int) -> List[Dict[str, Any]]:
    labels = list(pool["labels"])
    if n < len(labels):
        raise ValueError("n=%d is smaller than the %d labels" % (n, len(labels)))
    counts = {lb: n // len(labels) for lb in labels}
    for lb in labels[: n % len(labels)]:
        counts[lb] += 1

    slots = pool["slots"]
    by_label: Dict[str, List[Dict[str, Any]]] = {lb: [] for lb in labels}
    for tpl in pool["templates"]:
        by_label[tpl["label"]].append(tpl)

    items: List[Dict[str, Any]] = []
    for label in labels:
        templates = by_label[label]
        per_lang: Dict[str, List[Dict[str, Any]]] = {}
        for tpl in templates:
            per_lang.setdefault(tpl["lang"], []).append(tpl)
        # one seeded queue of (template, fill) combos per language, then round-robin the
        # languages so no language is starved at small n and none dominates at large n
        queues: Dict[str, List[Dict[str, Any]]] = {}
        for lang, tpls in per_lang.items():
            combos = []
            for tpl in tpls:
                for fill in _slot_combos(tpl, slots):
                    combos.append({"template": tpl, "fill": fill})
            random.Random("%d|%s|%s" % (seed, label, lang)).shuffle(combos)
            queues[lang] = combos
        lang_order = sorted(queues)
        random.Random("%d|%s|langs" % (seed, label)).shuffle(lang_order)
        cursor = {lang: 0 for lang in lang_order}
        quota = counts[label]
        got = 0
        while got < quota:
            progressed = False
            for lang in lang_order:
                if got >= quota:
                    break
                q = queues[lang]
                if cursor[lang] >= len(q):
                    continue
                combo = q[cursor[lang]]
                cursor[lang] += 1
                tpl, fill = combo["template"], combo["fill"]
                text = render_template(tpl["text"], fill)
                items.append({
                    "item_id": "%s|%s|%s" % (label, tpl["id"], "-".join("%s%s" % (k, fill[k]) for k in sorted(fill))),
                    "label": label,
                    "lang": lang,
                    "template_id": tpl["id"],
                    "slots": fill,
                    "text": text,
                    "split": "eval",
                })
                got += 1
                progressed = True
            if not progressed:
                raise ValueError("pool %s cannot supply %d items for label %s"
                                 % (pool["pool_id"], quota, label))
    return items


def _upstream_items(pool: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
    reqs = pool["requests"]
    if n > len(reqs):
        raise ValueError(
            "pool %s is the verbatim upstream request list (%d items); n=%d would duplicate "
            "requests. Use n<=%d, or the balanced-v1 pool to scale up."
            % (pool["pool_id"], len(reqs), n, len(reqs)))
    items = []
    for i, r in enumerate(reqs[:n]):
        items.append({
            "item_id": "upstream|%02d|%s" % (i, r["lang"]),
            "label": r["label"],
            "lang": r["lang"],
            "template_id": "upstream|%02d" % i,
            "slots": {},
            "text": r["text"],
            "split": "eval",
        })
    return items


def internal_question(pool: Dict[str, Any]) -> Dict[str, Any]:
    """The library's internal question form, taken from the library when it is importable.

    ``laya.agent.Agent._to_internal`` is the code path the model actually uses, so the harness
    prefers it and falls back to a mirror only for the stub path. The source used is recorded in
    the manifest so a verifier knows which one produced the prompt.
    """
    qdef = pool["question"]
    try:
        from laya.agent import Agent  # noqa: WPS433 (import inside function on purpose)
        return {"source": "laya.agent.Agent._to_internal",
                "questions": {qid: Agent._to_internal(d) for qid, d in qdef.items()}}
    except Exception as e:  # pragma: no cover - only on an import failure
        mirror = {}
        for qid, d in qdef.items():
            crit = d.get("criteria")
            if d["type"] == "choice" and isinstance(crit, list):
                crit = {c: None for c in crit}
            mirror[qid] = {"t": d["type"], "ins": d["instructions"], "crit": crit}
        return {"source": "harness mirror of _to_internal (laya import failed: %s)" % e,
                "questions": mirror}

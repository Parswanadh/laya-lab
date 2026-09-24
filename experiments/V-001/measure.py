"""V-001 measurement integrity: needle position, real pad token counts, leakage, majority class.

CPU + tokenizer only. No CUDA, no model, no GPU lock needed.

The prompt-building functions (`place`, `filler_ids`) and the constants are extracted from the
scripts under verification by AST, not retyped, so this measures the artifacts themselves.

    env/venv/bin/python experiments/V-001/measure.py
"""
import ast
import hashlib
import json
import os
import statistics
import sys

os.environ.setdefault("USE_TF", "0")

LAB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(LAB, "fork"))

from transformers import AutoTokenizer  # noqa: E402

DIAG = os.path.join(LAB, "experiments", "orch-diagnostic", "run.py")
BASE = os.path.join(LAB, "experiments", "orch-baseline", "run.py")


def load_source(path):
    with open(path) as f:
        src = f.read()
    tree = ast.parse(src)
    ns = {}
    picked = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("place", "filler_ids"):
            seg = ast.get_source_segment(src, node)
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
            picked[node.name] = seg.strip().splitlines()[0]
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in ("REQUESTS", "FILLER", "QUESTIONS"):
                    ns[t.id] = ast.literal_eval(node.value)
                    picked[t.id] = True
    return ns, picked


def offset_tokens(tok, text, char_start, char_end):
    """Token range [start, end) of the *whole tokens* that lie inside the character span.

    `start` is the first token whose character span begins at or after `char_start`; `end` is one
    past the last token whose character span ends at or before `char_end`. A token straddling the
    boundary belongs to neither side, which is the honest reading of "where does the needle
    start": the needle's first *whole* token.
    """
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    n = len(enc["input_ids"])
    starts = [i for i, (s, _) in enumerate(offs) if s >= char_start]
    ends = [i for i, (_, e) in enumerate(offs) if e <= char_end]
    if not starts or not ends or ends[-1] < starts[0]:
        return None, None, n
    return starts[0], ends[-1] + 1, n


def probe_cell(tok, doc, prefix_text, needle_text, cut_label=None, room=None):
    """Measure where the needle really lands in the *re-encoded* document.

    `room` (when given) is the state budget build_sequence leaves after the question head, so the
    record also says which state tokens the model actually sees.
    """
    rec = {"doc_tokens": len(tok(doc, add_special_tokens=False)["input_ids"])}
    rec["prefix_intact"] = doc.startswith(prefix_text)
    rec["needle_text_intact"] = doc[len(prefix_text):].startswith(needle_text)
    cs = len(prefix_text)
    ce = cs + len(needle_text)
    st, en, n = offset_tokens(tok, doc, cs, ce)
    rec.update({"char_start": cs, "char_end": ce,
                "needle_start_tok": st, "needle_end_tok": en, "doc_tokens_off": n})
    if st is not None:
        rec["needle_start_frac"] = round(st / n, 4)
        rec["needle_end_frac"] = round(en / n, 4)
        rec["needle_tokens"] = en - st
        if cut_label is not None:
            rec["start_minus_label"] = st - cut_label
            rec["label_frac"] = cut_label / n if n else None
    if room is not None:
        # Both scripts call agent.predict({"text": ...}), and Agent._encode_state tokenizes
        # serialize_state(state) = json.dumps({"text": doc}), i.e. a 5-token JSON wrapper around
        # the document. Measure the wrapper, not the bare document.
        wrapped = json.dumps({"text": doc}, ensure_ascii=False)
        wrapped_ids = tok(wrapped, add_special_tokens=False)["input_ids"]
        rec["wrapped_state_tokens"] = len(wrapped_ids)
        seen = rec["state_seen_tokens"] = min(len(wrapped_ids), room)
        rec["needle_fully_visible"] = bool(st is not None and en is not None and en <= seen)
        rec["state_hash"] = hashlib.sha256(json.dumps(wrapped_ids[:room]).encode()).hexdigest()[:16]
    return rec


def summ(vals):
    if not vals:
        return None
    return {"min": min(vals), "median": statistics.median(vals), "max": max(vals)}


def main():
    diag, dsrc = load_source(DIAG)
    base, bsrc = load_source(BASE)
    print("extracted from diagnostic: %s" % sorted(dsrc))
    print("extracted from baseline:   %s" % sorted(bsrc))

    REQUESTS, FILLER = diag["REQUESTS"], diag["FILLER"]
    assert REQUESTS == base["REQUESTS"], "REQUESTS differ between the two scripts"
    assert FILLER == base["FILLER"], "FILLER differs between the two scripts"

    tok = AutoTokenizer.from_pretrained(os.path.join(LAB, "models", "multilingual", "tokenizer"))
    place, filler_ids = diag["place"], diag["filler_ids"]

    out = {"tokenizer_class": type(tok).__name__, "vocab_size": tok.vocab_size}

    # ---- 0. raw tokenizer facts -------------------------------------------------------
    base_ids = tok(FILLER, add_special_tokens=False)["input_ids"]
    out["filler_base"] = {"text": FILLER, "tokens": len(base_ids),
                          "decoded_roundtrip_identical": tok.decode(base_ids) == FILLER,
                          "decoded": tok.decode(base_ids)}
    out["request_tokens"] = [{"i": i, "lang_gold": g, "tokens": len(tok(r, add_special_tokens=False)["input_ids"])}
                             for i, (r, g) in enumerate(REQUESTS)]
    out["request_token_counts"] = [r["tokens"] for r in out["request_tokens"]]

    # ---- 1. majority class, recomputed from the label list ----------------------------
    labels = [g for _, g in REQUESTS]
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    out["majority"] = {"counts": counts, "majority_label": max(counts, key=counts.get),
                       "majority_accuracy": round(max(counts.values()) / len(labels), 4),
                       "n": len(labels)}

    # ---- 2. leakage: needle text vs filler text, by content ---------------------------
    filler_norm = " ".join(FILLER.lower().split())
    leak = []
    for i, (r, g) in enumerate(REQUESTS):
        rn = " ".join(r.lower().split())
        # longest shared word n-gram between needle and filler
        rw, fw = rn.split(), filler_norm.split()
        best = 0
        for n in range(1, 6):
            fgrams = {" ".join(fw[j:j + n]) for j in range(len(fw) - n + 1)}
            for j in range(len(rw) - n + 1):
                if " ".join(rw[j:j + n]) in fgrams:
                    best = n
        leak.append({"i": i, "gold": g, "needle_is_substring_of_filler": rn in filler_norm,
                     "max_shared_word_ngram": best,
                     "uploads_token_overlap": len(set(tok(r, add_special_tokens=False)["input_ids"])
                                                  & set(base_ids))})
    out["leakage"] = leak
    out["filler_redundancy"] = {
        "base_sentence_tokens": len(base_ids),
        "distinct_sentences_in_pad_7000": 1,
        "repeats_at_pad_7000": round(7000 / len(base_ids), 1),
        "labels_present_in_filler_text": [l for l in counts if l.lower() in filler_norm],
        "filler_is_single_sentence_repeated": True,
    }

    # ---- 3. baseline document construction: is pad really pad, needle at the end? -----
    # State budget build_sequence leaves after the question head, so we can say which cells even
    # contain the needle in the model's input.
    q = base["QUESTIONS"]["department"]
    head_ids = tok("%s question: %s" % ("choice", q["instructions"]), add_special_tokens=False)["input_ids"]
    opt_ids = [tok(" " + "%s: %s" % (k, v), add_special_tokens=False)["input_ids"]
               for k, v in q["criteria"].items()]
    out["head_len_tokens"] = 1 + len(head_ids) + 1 + sum(len(o) + 1 for o in opt_ids) + 1
    room = {lim: max(0, lim - out["head_len_tokens"] - 1) for lim in (1024, 8192)}
    out["state_room_by_limit"] = {str(k): v for k, v in room.items()}

    fid = tok(FILLER, add_special_tokens=False)["input_ids"]
    out["baseline_cells"] = []
    for limit in (1024, 8192):
        for pad in (0, 1000, 2000, 4000, 7000):
            body = (fid * (pad // max(1, len(fid)) + 1))[:pad]
            body_text = tok.decode(body)
            cell = {"limit": limit, "pad": pad, "body_tokens_by_construction": len(body),
                    "body_tokens_after_roundtrip": len(tok(body_text, add_special_tokens=False)["input_ids"])
                    if body else 0, "state_room": room[limit],
                    "items": []}
            for i, (req, gold) in enumerate(REQUESTS):
                text = body_text + " " + req
                pre = body_text + " "
                rec = probe_cell(tok, text, pre, req, cut_label=pad, room=room[limit])
                rec.update({"i": i, "gold": gold})
                cell["items"].append(rec)
            cell["doc_tokens"] = summ([r["doc_tokens"] for r in cell["items"]])
            cell["doc_tokens_off"] = summ([r["doc_tokens_off"] for r in cell["items"]])
            cell["needle_start_tok"] = summ([r["needle_start_tok"] for r in cell["items"]])
            cell["needle_start_frac"] = summ([r["needle_start_frac"] for r in cell["items"]])
            cell["start_minus_label"] = summ([r["start_minus_label"] for r in cell["items"]])
            cell["items_off_label"] = [{"i": r["i"], "gold": r["gold"], "start": r["needle_start_tok"],
                                        "delta": r["start_minus_label"], "tokens": r["doc_tokens_off"]}
                                       for r in cell["items"] if r["start_minus_label"]]
            cell["all_prefix_intact"] = all(r["prefix_intact"] for r in cell["items"])
            cell["all_needle_intact"] = all(r["needle_text_intact"] for r in cell["items"])
            cell["items_needle_dropped"] = sum(1 for r in cell["items"] if not r["needle_fully_visible"])
            cell["state_hashes"] = sorted({r["state_hash"] for r in cell["items"]})
            out["baseline_cells"].append(cell)

    # ---- 4. diagnostic cells: place() at each (pad, frac) ----------------------------
    out["diagnostic_cells"] = []
    grid = [("A_dtype", 4000, 1.0), ("B_position", 4000, 0.0), ("B_position", 4000, 0.25),
            ("B_position", 4000, 0.5), ("B_position", 4000, 0.75), ("B_position", 4000, 1.0),
            ("C_length", 0, 1.0), ("C_length", 1000, 1.0), ("C_length", 2000, 1.0),
            ("C_length", 4000, 1.0), ("C_length", 7000, 1.0)]
    for probe, pad, frac in grid:
        f = filler_ids(tok, pad)
        cut = int(len(f) * frac)
        prefix_text = tok.decode(f[:cut])
        suffix_text = tok.decode(f[cut:])
        cell = {"probe": probe, "pad_label": pad, "frac_label": frac, "cut_tokens": cut,
                "items": []}
        for i, (req, gold) in enumerate(REQUESTS):
            req_ids = tok(req, add_special_tokens=False)["input_ids"]
            ids = f[:cut] + req_ids + f[cut:]
            doc = tok.decode(ids)
            needle_text = tok.decode(req_ids)
            rec = probe_cell(tok, doc, prefix_text, needle_text, cut_label=cut, room=8192 - out["head_len_tokens"] - 1)
            rec.update({"i": i, "gold": gold, "req_tokens_alone": len(req_ids),
                        "decode_of_req_ids_is_req": tok.decode(req_ids) == req,
                        "ids_len": len(ids)})
            cell["items"].append(rec)
        cell["doc_tokens_off"] = summ([r["doc_tokens_off"] for r in cell["items"]])
        cell["needle_start_tok"] = summ([r["needle_start_tok"] for r in cell["items"]])
        cell["needle_end_tok"] = summ([r["needle_end_tok"] for r in cell["items"]])
        cell["needle_start_frac"] = summ([r["needle_start_frac"] for r in cell["items"]])
        cell["needle_tokens"] = summ([r["needle_tokens"] for r in cell["items"]])
        cell["start_minus_label"] = summ([r["start_minus_label"] for r in cell["items"]])
        cell["items_off_label"] = [{"i": r["i"], "gold": r["gold"], "start": r["needle_start_tok"],
                                    "delta": r["start_minus_label"], "tokens": r["doc_tokens_off"]}
                                   for r in cell["items"] if r["start_minus_label"]]
        cell["all_prefix_intact"] = all(r["prefix_intact"] for r in cell["items"])
        cell["all_needle_intact"] = all(r["needle_text_intact"] for r in cell["items"])
        cell["items_decode_mismatch"] = [r["i"] for r in cell["items"] if not r["decode_of_req_ids_is_req"]]
        cell["pad_label_vs_true_filler_tokens"] = {
            "filler_ids_len": len(f), "prefix_len": len(f[:cut]), "suffix_len": len(f[cut:]),
            "decoded_prefix_retokenized": len(tok(prefix_text, add_special_tokens=False)["input_ids"]) if prefix_text else 0,
        }
        out["diagnostic_cells"].append(cell)

    # ---- 5. retained-state identity across baseline cells ------------------------------
    # The four limit=1024 padded cells keep the same 978-token filler prefix, so if their state
    # hashes collide they are the same model input repeated four times, not four measurements.
    by_hash = {}
    for c in out["baseline_cells"]:
        for h in c["state_hashes"]:
            by_hash.setdefault(h, []).append("limit=%d pad=%d" % (c["limit"], c["pad"]))
    out["identical_state_groups"] = {h: v for h, v in by_hash.items() if len(v) > 1}

    # 5b. proof that the limit=1024 padded inputs contain no request: the retained 978 state
    # tokens are byte-identical to the tokenization of a bare, request-free 978-token filler
    # prefix. Same tokens in => the model literally never saw a request in those cells.
    # Control: the same filler document with the request left off entirely. If the retained
    # 978 tokens of the padded cell equal this, the request is provably not in the model input.
    for c in out["baseline_cells"]:
        if c["limit"] != 1024 or c["pad"] == 0:
            continue
        b = (fid * (c["pad"] // max(1, len(fid)) + 1))[:c["pad"]]
        wrapped_only = json.dumps({"text": tok.decode(b)}, ensure_ascii=False)
        ids_only = tok(wrapped_only, add_special_tokens=False)["input_ids"][:room[1024]]
        h_only = hashlib.sha256(json.dumps(ids_only).encode()).hexdigest()[:16]
        c["filler_only_state_hash"] = h_only
        c["request_absent_from_model_input"] = h_only in c["state_hashes"]
    out["limit1024_padded_state_is_bare_filler"] = {
        "limit=%d pad=%d" % (c["limit"], c["pad"]): c.get("request_absent_from_model_input")
        for c in out["baseline_cells"] if c["limit"] == 1024 and c["pad"] > 0}

    d = os.path.join(LAB, "experiments", "V-001")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "measure.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    # ---- console summary --------------------------------------------------------------
    print("\nFILLER base tokens: %d  roundtrip-identical: %s"
          % (out["filler_base"]["tokens"], out["filler_base"]["decoded_roundtrip_identical"]))
    print("request token counts: %s" % out["request_token_counts"])
    print("majority class: %s = %s (%.4f), n=%d" % (out["majority"]["majority_label"],
          out["majority"]["counts"], out["majority"]["majority_accuracy"], out["majority"]["n"]))
    print("needle substring-of-filler: %s ; max shared word ngram: %s"
          % (any(l["needle_is_substring_of_filler"] for l in out["leakage"]),
             max(l["max_shared_word_ngram"] for l in out["leakage"])))
    print("head_len=%d  state room: %s" % (out["head_len_tokens"], out["state_room_by_limit"]))
    print("filler: single sentence of %d tokens repeated; labels in filler text: %s"
          % (out["filler_redundancy"]["base_sentence_tokens"],
             out["filler_redundancy"]["labels_present_in_filler_text"] or "none"))
    print("limit=1024 padded inputs are the identical filler document minus the request: %s"
          % out["limit1024_padded_state_is_bare_filler"])
    print("cells whose retained state is byte-identical to another cell:")
    for h, cells in out["identical_state_groups"].items():
        print("  %s  <- %s" % (h, cells))

    print("\n--- BASELINE cells (needle at end after decode+re-encode) ---")
    print("%-6s %-6s %-18s %-18s %-18s %-9s %-9s %-9s %s" % ("limit", "pad", "doc_tok(min/med/max)",
          "needle@(min/med/max)", "needle_frac(min/med/max)", "d_start", "prefix", "needle", "needle_dropped_at_limit"))
    for c in out["baseline_cells"]:
        room = out["state_room_by_limit"][str(c["limit"])]
        print("%-6d %-6d %-18s %-18s %-18s %-9s %-9s %-9s %s" % (
            c["limit"], c["pad"],
            "%s/%s/%s" % (c["doc_tokens_off"]["min"], c["doc_tokens_off"]["median"], c["doc_tokens_off"]["max"]),
            "%s/%s/%s" % (c["needle_start_tok"]["min"], c["needle_start_tok"]["median"], c["needle_start_tok"]["max"]),
            "%s/%s/%s" % (c["needle_start_frac"]["min"], c["needle_start_frac"]["median"], c["needle_start_frac"]["max"]),
            c["start_minus_label"]["max"],
            c["all_prefix_intact"], c["all_needle_intact"],
            sum(1 for r in c["items"] if r["needle_start_tok"] > room)))

    print("\n--- DIAGNOSTIC cells (place(): needle at `frac` of a `pad`-token filler doc) ---")
    print("%-10s %-5s %-5s %-6s %-18s %-18s %-18s %-9s %-7s %-7s" % ("probe", "pad", "frac", "cut",
          "doc_tok(min/med/max)", "needle@(min/med/max)", "frac_real(min/med/max)", "d_start", "prefix", "needle"))
    for c in out["diagnostic_cells"]:
        print("%-10s %-5d %-5s %-6d %-18s %-18s %-18s %-9s %-7s %-7s" % (
            c["probe"], c["pad_label"], c["frac_label"], c["cut_tokens"],
            "%s/%s/%s" % (c["doc_tokens_off"]["min"], c["doc_tokens_off"]["median"], c["doc_tokens_off"]["max"]),
            "%s/%s/%s" % (c["needle_start_tok"]["min"], c["needle_start_tok"]["median"], c["needle_start_tok"]["max"]),
            "%s/%s/%s" % (c["needle_start_frac"]["min"], c["needle_start_frac"]["median"], c["needle_start_frac"]["max"]),
            c["start_minus_label"]["max"],
            c["all_prefix_intact"], c["all_needle_intact"]))
    print("\nitems whose needle start is off the label (delta != 0):")
    for c in out["baseline_cells"] + out["diagnostic_cells"]:
        if c["items_off_label"]:
            tag = "%s limit=%s pad=%s frac=%s" % (c.get("probe", "baseline"), c.get("limit"),
                                                  c.get("pad_label", c.get("pad")), c.get("frac_label", "end"))
            deltas = sorted({r["delta"] for r in c["items_off_label"]})
            print("  %-46s labels(=%s) start(min/med/max)=%s/%s/%s  delta=%s  n_off=%d/%d"
                  % (tag, c["cut_tokens"] if "cut_tokens" in c else c.get("pad"),
                     c["needle_start_tok"]["min"], c["needle_start_tok"]["median"],
                     c["needle_start_tok"]["max"], deltas, len(c["items_off_label"]), len(c["items"])))
    with open(os.path.join(d, "needle_positions.tsv"), "w") as fh:
        fh.write("script\tcell\tpad\tfrac\tcut\ti\tgold\tdoc_tokens\tneedle_start\tneedle_end\tneedle_tokens\trealized_frac\tdelta_vs_label\n")
        for c in out["baseline_cells"]:
            for r in c["items"]:
                fh.write("baseline\tlimit=%d\t%d\tend\t%d\t%d\t%s\t%d\t%s\t%s\t%s\t%s\t%s\n"
                         % (c["limit"], c["pad"], c["pad"], r["i"], r["gold"], r["doc_tokens_off"],
                            r["needle_start_tok"], r["needle_end_tok"], r["needle_tokens"],
                            r["needle_start_frac"], r["start_minus_label"]))
        for c in out["diagnostic_cells"]:
            for r in c["items"]:
                fh.write("diagnostic\t%s\t%d\t%s\t%d\t%d\t%s\t%d\t%s\t%s\t%s\t%s\t%s\n"
                         % (c["probe"], c["pad_label"], c["frac_label"], c["cut_tokens"], r["i"], r["gold"],
                            r["doc_tokens_off"], r["needle_start_tok"], r["needle_end_tok"],
                            r["needle_tokens"], r["needle_start_frac"], r["start_minus_label"]))
    print("\nwrote experiments/V-001/measure.json and needle_positions.tsv")


if __name__ == "__main__":
    main()

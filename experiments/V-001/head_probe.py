"""V-001 ground truth for the sequence head and the post-truncation state, from the implementation.

Loads the checkpoint on CPU (no CUDA, no GPU lock) and calls Agent._encode_state directly, so the
head length and the retained state are measured rather than re-derived.

    env/venv/bin/python experiments/V-001/head_probe.py
"""
import ast
import json
import os
import sys

os.environ.setdefault("USE_TF", "0")

LAB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(LAB, "fork"))

import laya  # noqa: E402

DIAG = os.path.join(LAB, "experiments", "orch-diagnostic", "run.py")
BASE = os.path.join(LAB, "experiments", "orch-baseline", "run.py")


def consts(path, names=("REQUESTS", "FILLER", "QUESTIONS")):
    ns = {}
    for node in ast.parse(open(path).read()).body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in names:
                    ns[t.id] = ast.literal_eval(node.value)
    return ns


def main():
    base = consts(BASE)
    REQUESTS, FILLER = base["REQUESTS"], base["FILLER"]
    QUESTIONS = base["QUESTIONS"]

    agent = laya.load(os.path.join(LAB, "models", "multilingual"), device="cpu")
    tok = agent.tok
    internal = {qid: agent._to_internal(QUESTIONS[qid]) for qid in ["department"]}
    q = internal["department"]
    print("internal question:", json.dumps(q, ensure_ascii=False)[:200])

    out = {"question_internal": q}

    # --- head: the part before the state, straight out of build_sequence -------------------
    ids0 = agent._encode_state(" " + REQUESTS[0][0], ["department"], internal, max_len=8192)[0]
    state_ids = tok(" " + REQUESTS[0][0], add_special_tokens=False)["input_ids"]
    head = len(ids0["ids"]) - len(state_ids) - 1
    out["head_tokens"] = head
    out["markers"] = ids0["markers"]
    out["head_pieces"] = {
        "head_ids": len(tok("%s question: %s" % (q["t"], q["ins"]), add_special_tokens=False)["input_ids"]),
        "options": [len(tok(" " + o, add_special_tokens=False)["input_ids"]) for o in
                    [str(k) if v is None or v == "" else "%s: %s" % (k, v) for k, v in q["crit"].items()]],
        "n_options": len(q["crit"]),
    }
    print("head_tokens=%d markers=%s pieces=%s" % (head, out["markers"], out["head_pieces"]))
    print("head text as decoded:", repr(tok.decode(ids0["ids"][:head])))

    # --- what the model sees at each limit, per cell ---------------------------------------
    fid = tok(FILLER, add_special_tokens=False)["input_ids"]
    out["cells"] = []
    for limit in (1024, 8192):
        for pad in (0, 1000, 2000, 4000, 7000):
            body = (fid * (pad // max(1, len(fid)) + 1))[:pad]
            rows, state_seen, needle_seen = [], [], []
            for req, gold in REQUESTS:
                text = tok.decode(body) + " " + req
                ids = agent._encode_state(text, ["department"], internal, max_len=limit)[0]["ids"]
                st = tok(text, add_special_tokens=False)["input_ids"]
                n_state = len(ids) - head - 1
                rows.append(len(ids))
                state_seen.append(n_state)
                # needle starts where the request starts in the state
                needle_seen.append(len(st) - n_state < 20)
            out["cells"].append({"limit": limit, "pad": pad,
                                 "model_input_tokens": sorted(set(rows)),
                                 "state_tokens_seen": sorted(set(state_seen)),
                                 "state_room": max(state_seen),
                                 "request_partially_visible_items": sum(needle_seen)})
            print("limit=%-5d pad=%-5d model_input=%s state_seen=%s room=%d req_visible=%d/20"
                  % (limit, pad, sorted(set(rows)), sorted(set(state_seen)), max(state_seen), sum(needle_seen)))

    d = os.path.join(LAB, "experiments", "V-001")
    with open(os.path.join(d, "head_probe.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote experiments/V-001/head_probe.json")


if __name__ == "__main__":
    main()

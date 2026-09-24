"""Probe D — where does the information die: the encoder, or the decision head?

This is the experiment that decides whether an architectural change to the HEAD can work at all.

Setup: the failing regime (document of 4000 filler tokens, needle at the END), balanced labels.
Run the frozen encoder once per item and extract four candidate feature sets:

  marker      mean-pool of H at the option [MASK] marker positions
              -> exactly what the shipped `scorer` reads. The head's only view of the document.
  cls         H at position 0 (the act-head's pooled vector)
  state_mean  mean-pool of H over all state positions
  state_max   max-pool of H over all state positions

Then fit a closed-form linear probe (ridge on one-hot targets) to each, with 5-fold CV, and
compare against the model's own accuracy on the same items.

Reading of the result, fixed BEFORE running:

  state_mean/state_max >> marker
      -> the information IS present in the encoder's output but is not reaching the markers.
         Aggregation is the bottleneck, and a cross-attention head is the right fix. (H5 stands.)
  marker ~= state_* ~= chance
      -> the information is not linearly recoverable from the encoder output at this length.
         A better head cannot fix it; the encoder or the training is the problem. (H5 falls.)
  marker >> chance but the full model is at chance
      -> the head is discarding information it receives. A different, simpler head may suffice.

    env/venv/bin/python experiments/orch-probe-d/run.py
"""
import json
import os
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

LAB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(LAB, "fork"))

import numpy as np  # noqa: E402
import laya  # noqa: E402
import torch  # noqa: E402

QUESTION = {"type": "choice", "instructions": "Which department should handle this request?",
            "criteria": {"billing": "invoices, payments, refunds",
                         "technical": "bugs, outages, system errors",
                         "sales": "pricing, new contracts, plan upgrades",
                         "other": "everything else"}}
QUESTIONS = {"department": QUESTION}
LABELS = ["billing", "technical", "sales", "other"]

# Authored for this probe: 12 surface forms per class, all English, no template reused
# between classes, and no surface form appearing in the filler.
NEEDLES = {
    "billing": [
        "I was charged twice for invoice {n}, please refund the duplicate payment.",
        "My card was billed {n} dollars more than the amount on my last statement.",
        "Please refund invoice {n}, the payment went through twice this morning.",
        "There is an unexpected charge of {n} dollars on my account this month.",
        "I need a refund for the duplicate transaction on invoice {n}.",
        "My last statement shows a payment of {n} dollars I never authorised.",
        "Can you reverse the double charge on invoice {n}?",
        "The renewal payment of {n} dollars was taken from the wrong card.",
        "I am disputing a charge of {n} dollars that appeared on my bill.",
        "Please credit back the overpayment of {n} dollars on my account.",
        "Invoice {n} was paid twice, I would like the second payment returned.",
        "Why was I billed {n} dollars when my plan costs less than that?",
    ],
    "technical": [
        "The dashboard throws an error every time I open report {n}.",
        "Our API returns 500 responses since this morning and the service is down.",
        "The application crashes whenever I try to export file {n}.",
        "Sync stopped working after the last update and nothing uploads now.",
        "The login page hangs and then times out for every user.",
        "We are seeing repeated outages on the {n} endpoint today.",
        "The mobile app closes itself as soon as I open the settings screen.",
        "Database connections are failing and requests are timing out.",
        "The search feature returns an internal error for every query.",
        "Uploads larger than {n} megabytes fail with a server error.",
        "The nightly job did not run and the data is stale again.",
        "Webhooks stopped being delivered after the maintenance window.",
    ],
    "sales": [
        "What would an enterprise contract for {n} seats cost per year?",
        "Can you send pricing for upgrading our team to the business tier?",
        "We would like a quote for {n} licences billed annually.",
        "Is there a volume discount if we purchase {n} seats this quarter?",
        "Please share the price of the premium plan for a team of {n}.",
        "What are the contract terms for an annual plan with {n} users?",
        "I want to compare the cost of the pro and enterprise plans.",
        "Could you prepare a proposal for {n} additional seats?",
        "What discount applies if we commit to {n} seats for two years?",
        "Please quote the enterprise tier for our department of {n} people.",
        "How much more would {n} extra seats add to our current contract?",
        "We are evaluating vendors and need pricing for {n} users.",
    ],
    "other": [
        "Please update the shipping address on my account to the new office.",
        "How do I change the email address associated with my profile?",
        "I would like to close my account and delete all stored data.",
        "Can you confirm which timezone my scheduled reports use?",
        "Where can I download the accessibility documentation?",
        "I need a copy of the data processing agreement for our records.",
        "How do I add a colleague to our workspace as a viewer?",
        "Please tell me the office opening hours for the support desk.",
        "Is there a way to export my settings to another workspace?",
        "I want to change the display language of the interface.",
        "Could you explain how the retention policy applies to archived items?",
        "Please confirm whether the service is available in my region.",
    ],
}
FILLER = ("Thanks for the update on the quarterly planning meeting. We reviewed the roadmap slides, "
          "discussed hiring for the design team, agreed on the offsite venue, and noted that the "
          "parking garage will be closed next week. The finance group circulated the revised "
          "forecast and asked each team lead to comment before Friday. Facilities confirmed the "
          "new badge process and the desk booking system goes live next month. ")


def build_items(n_per_class, pad):
    items = []
    for ci, lab in enumerate(LABELS):
        for j in range(n_per_class):
            txt = NEEDLES[lab][j % len(NEEDLES[lab])].format(n=1000 + ci * 100 + j)
            items.append({"text": txt, "label": lab})
    return items


def ridge_probe(X, y, n_classes=4, lam=1.0, folds=5, seed=0):
    """Closed-form linear probe: ridge on one-hot targets, k-fold CV accuracy."""
    rng = np.random.default_rng(seed)
    n = len(X)
    Y = np.zeros((n, n_classes), dtype=np.float64)
    for i, c in enumerate(y):
        Y[i, c] = 1.0
    idx = rng.permutation(n)
    correct = 0
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        Xtr, Ytr = X[tr], Y[tr]
        mu = Xtr.mean(0, keepdims=True)
        sd = Xtr.std(0, keepdims=True) + 1e-6
        A = (Xtr - mu) / sd
        B = (X[te] - mu) / sd
        d = A.shape[1]
        W = np.linalg.solve(A.T @ A + lam * np.eye(d), A.T @ Ytr)
        pred = (B @ W).argmax(1)
        correct += int((pred == y[te]).sum())
    return correct / n


def main():
    agent = laya.load(os.path.join(LAB, "models", "multilingual"), device="cuda")
    tok = agent.tok
    agent.dtype = torch.float32

    # Capture what the head sees.
    cap = {}

    def pre_hook(_mod, args, kwargs):
        # agent.py:618 calls self.model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
        # positionally, so read from args rather than kwargs.
        if len(args) >= 4:
            cap["marker_pos"], cap["marker_mask"] = args[2], args[3]
        else:
            cap["marker_pos"] = kwargs.get("marker_pos")
            cap["marker_mask"] = kwargs.get("marker_mask")

    def enc_hook(_mod, _inp, out):
        cap["h"] = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

    agent.model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    agent.model.encoder.register_forward_hook(enc_hook)

    pad = 4000
    filler_ids = tok(FILLER, add_special_tokens=False)["input_ids"]
    body = (filler_ids * (pad // max(1, len(filler_ids)) + 1))[:pad]
    filler_text = tok.decode(body)

    results = {}
    for pos_frac, tag in ((1.0, "needle_at_end"), (0.0, "needle_at_start")):
        items = build_items(n_per_class=50, pad=pad)
        feats = {"marker": [], "cls": [], "state_mean": [], "state_max": []}
        ys, head_pred, ntok = [], [], []
        for it in items:
            state = {"text": filler_text + " " + it["text"] if pos_frac == 1.0
                     else it["text"] + " " + filler_text}
            r = agent.predict(state, QUESTIONS, max_len=8192)
            h = cap["h"][0]                       # [L, d]
            mp = cap["marker_pos"][0].clamp(min=0)
            mm = cap["marker_mask"][0].bool()
            mpos = mp[mm]
            L = h.shape[0]
            ntok.append(int(L))
            feats["marker"].append(h[mpos].mean(0).float().cpu().numpy())
            feats["cls"].append(h[0].float().cpu().numpy())
            # state positions = everything after the marker block up to the final [SEP]
            start = int(mpos.max().item()) + 1
            st = h[start:L - 1] if L - 1 > start else h[start:]
            feats["state_mean"].append(st.mean(0).float().cpu().numpy())
            feats["state_max"].append(st.max(0).values.float().cpu().numpy())
            ys.append(LABELS.index(it["label"]))
            head_pred.append(r["answers"]["department"]["choice"])
            cap.clear()

        y = np.array(ys)
        head_acc = float(np.mean([p == LABELS[c] for p, c in zip(head_pred, y)]))
        row = {"n": len(y), "probe_tokens_median": int(np.median(ntok)),
               "head_accuracy": round(head_acc, 4), "majority": 0.25, "random": 0.25}
        for k, v in feats.items():
            X = np.stack(v).astype(np.float64)
            row["probe_" + k] = round(ridge_probe(X, y), 4)
        results[tag] = row
        print("\n[%s]  n=%d  median tokens=%d" % (tag, row["n"], row["probe_tokens_median"]))
        print("   shipped head accuracy : %.3f" % row["head_accuracy"])
        for k in ("marker", "cls", "state_mean", "state_max"):
            print("   linear probe %-11s: %.3f" % (k, row["probe_" + k]))

    d = os.path.join(LAB, "experiments", "orch-probe-d")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "results.json"), "w") as f:
        json.dump({"pad": pad, "model": "multilingual", "dtype": "fp32",
                   "probe": "ridge one-hot, 5-fold CV, lam=1.0", "results": results}, f, indent=1)
    print("\nwrote experiments/orch-probe-d/results.json")


if __name__ == "__main__":
    main()

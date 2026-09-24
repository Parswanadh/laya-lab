"""P1 diagnostic: is long-context degradation precision, distance, or capacity?

Two independent probes, both cheap:

  A. dtype sweep   — CUDA defaults to bf16 autocast (rl_agent_config amp_dtype="bf16").
                     bf16 carries 8 mantissa bits vs fp32's 23. If accuracy at long
                     sequence is precision-limited, fp32 recovers it for free.
  B. position sweep — at FIXED document length, move the needle through the document.
                     Separates "the document is long" from "the evidence is far from the
                     option markers". Upstream's curve conflates the two.

    env/venv/bin/python experiments/orch-diagnostic/run.py
"""
import json
import os
import statistics
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

LAB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(LAB, "fork"))

import laya  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

QUESTIONS = {"department": {"type": "choice", "instructions": "Which department should handle this request?",
                            "criteria": {"billing": "invoices, payments, refunds",
                                         "technical": "bugs, outages, system errors",
                                         "sales": "pricing, new contracts, plan upgrades",
                                         "other": "everything else"}}}

REQUESTS = [
    ("I was charged twice for invoice 4411, please refund the duplicate payment.", "billing"),
    ("The dashboard crashes with an error every time I open the reports page.", "technical"),
    ("What would an enterprise contract for 200 seats cost per year?", "sales"),
    ("Please refund my subscription payment from last month, it was billed by mistake.", "billing"),
    ("Our API returns 500 errors since this morning and the service is down.", "technical"),
    ("Can you send me pricing for upgrading our plan to the business tier?", "sales"),
    ("Me cobraron dos veces la factura de marzo, devuélvanme el cargo duplicado.", "billing"),
    ("La aplicación se cierra cada vez que abro la configuración.", "technical"),
    ("Quisiera una cotización para un contrato anual de 50 licencias.", "sales"),
    ("Fui cobrado duas vezes na fatura de março, quero o reembolso da cobrança duplicada.", "billing"),
    ("O sistema cai toda vez que tento gerar o relatório mensal.", "technical"),
    ("J'ai été facturé deux fois ce mois-ci, merci de rembourser le doublon.", "billing"),
    ("L'application plante dès que j'ouvre la page des paramètres.", "technical"),
    ("Ich wurde zweimal belastet, bitte erstatten Sie die doppelte Zahlung.", "billing"),
    ("Die Anwendung stürzt jedes Mal ab, wenn ich die Einstellungen öffne.", "technical"),
    ("Was kostet ein Jahresvertrag für 100 Nutzer?", "sales"),
    ("मुझसे मार्च में दो बार शुल्क लिया गया, कृपया डुप्लिकेट राशि वापस करें।", "billing"),
    ("ऐप हर बार सेटिंग्स खोलते ही बंद हो जाता है।", "technical"),
    ("請求書が二重に請求されました。重複分を返金してください。", "billing"),
    ("تم خصم المبلغ مرتين من بطاقتي، أرجو استرداد المبلغ المكرر.", "billing"),
]
FILLER = ("Thanks for the update on the quarterly planning meeting. We reviewed the roadmap slides, discussed "
          "hiring for the design team, agreed on the offsite venue, and noted that the parking garage will be "
          "closed next week. ")


def filler_ids(tok, n):
    base = tok(FILLER, add_special_tokens=False)["input_ids"]
    return (base * (n // max(1, len(base)) + 1))[:n]


def place(tok, pad, frac, request):
    """Document of `pad` filler tokens with `request` inserted at fraction `frac`."""
    f = filler_ids(tok, pad)
    cut = int(len(f) * frac)
    ids = f[:cut] + tok(request, add_special_tokens=False)["input_ids"] + f[cut:]
    return tok.decode(ids)


def score(agent, tok, limit, dtype, cases):
    if dtype is not None:
        agent.dtype = dtype
    per_item, lat = [], []
    for text, gold in cases:
        t0 = time.time()
        r = agent.predict({"text": text}, QUESTIONS, max_len=limit)
        lat.append(time.time() - t0)
        ans = r["answers"]["department"]
        per_item.append({"gold": gold, "pred": ans["choice"],
                         "p_gold": round(float(ans["probabilities"].get(gold, 0.0)), 5),
                         "pred_p": round(float(ans["probabilities"][ans["choice"]]), 5)})
    acc = sum(1 for i in per_item if i["pred"] == i["gold"]) / max(1, len(per_item))
    return {"accuracy": round(acc, 4), "n": len(per_item),
            "median_latency_s": round(statistics.median(lat), 4) if lat else None,
            "per_item": per_item}


def main():
    agent = laya.load(os.path.join(LAB, "models", "multilingual"), device="cuda")
    tok = agent.tok
    out = {"device": "cuda", "gpu": torch.cuda.get_device_name(0), "probes": {}}

    print("=== A. dtype sweep  (pad=4000, limit=8192, needle at end) ===", flush=True)
    doc = place(tok, 4000, 1.0, REQUESTS[0][0])
    print("probe tokens: %d" % len(tok(doc, add_special_tokens=False)["input_ids"]), flush=True)
    cases = [(place(tok, 4000, 1.0, req), gold) for req, gold in REQUESTS]
    for name, dt in (("bf16 (shipped default)", torch.bfloat16), ("fp16", torch.float16),
                     ("fp32", torch.float32)):
        r = score(agent, tok, 8192, dt, cases)
        r.pop("per_item")
        out["probes"].setdefault("A_dtype", {})[name] = r
        print("  %-24s acc=%.3f  median=%.3fs" % (name, r["accuracy"], r["median_latency_s"]), flush=True)

    print("\n=== B. needle-position sweep  (pad=4000, limit=8192, fp32) ===", flush=True)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        cases = [(place(tok, 4000, frac, req), gold) for req, gold in REQUESTS]
        r = score(agent, tok, 8192, torch.float32, cases)
        out["probes"].setdefault("B_position", {})[str(frac)] = r
        print("  needle@%.2f  acc=%.3f  p_gold_mean=%.3f" %
              (frac, r["accuracy"],
               statistics.mean(i["p_gold"] for i in r["per_item"])), flush=True)

    print("\n=== C. length sweep at FIXED needle fraction 1.0 (fp32) ===", flush=True)
    for pad in (0, 1000, 2000, 4000, 7000):
        cases = [(place(tok, pad, 1.0, req), gold) for req, gold in REQUESTS]
        r = score(agent, tok, 8192, torch.float32, cases)
        out["probes"].setdefault("C_length", {})[str(pad)] = r
        print("  pad=%-5d acc=%.3f  median=%.3fs" % (pad, r["accuracy"], r["median_latency_s"]), flush=True)

    # Majority-class reference: with these 20 items the majority label is 'billing' (9/20).
    labels = [g for _, g in REQUESTS]
    maj = max(labels.count(l) for l in set(labels)) / len(labels)
    out["majority_class_accuracy"] = round(maj, 4)
    out["label_counts"] = {l: labels.count(l) for l in sorted(set(labels))}
    print("\nmajority-class baseline: %.3f  counts=%s" % (maj, out["label_counts"]))

    d = os.path.join(LAB, "experiments", "orch-diagnostic")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote experiments/orch-diagnostic/results.json")


if __name__ == "__main__":
    main()

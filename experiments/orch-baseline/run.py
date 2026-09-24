"""Orchestrator's independent P0 baseline confirmation.

Not a deliverable — a ground-truth check against which the engineer's harness (issue #1)
and the upstream artifact are both compared. Mirrors upstream's
research/scripts/bench_long_context.py methodology at reduced n.

    env/venv/bin/python experiments/orch-baseline/run.py
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

QUESTIONS = {"department": {"type": "choice", "instructions": "Which department should handle this request?",
                            "criteria": {"billing": "invoices, payments, refunds",
                                         "technical": "bugs, outages, system errors",
                                         "sales": "pricing, new contracts, plan upgrades",
                                         "other": "everything else"}}}

# Same 20 requests as upstream's bench_long_context.py, so the comparison is like-for-like.
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


def main():
    agent = laya.load(os.path.join(LAB, "models", "multilingual"), device="cuda")
    tok = agent.tok
    rows = []
    for limit in (1024, 8192):
        for pad in (0, 1000, 2000, 4000, 7000):
            # Build the padded document: filler first, request last (the case head-truncation loses).
            filler_ids = tok(FILLER, add_special_tokens=False)["input_ids"]
            body = (filler_ids * (pad // max(1, len(filler_ids)) + 1))[:pad]
            text = tok.decode(body) + " " + REQUESTS[0][0]
            probe = tok(text, add_special_tokens=False)["input_ids"]
            ntok = len(probe)
            correct, lat = 0, []
            for req, gold in REQUESTS:
                state = {"text": tok.decode(body) + " " + req}
                t0 = time.time()
                r = agent.predict(state, QUESTIONS, max_len=limit)
                lat.append(time.time() - t0)
                ans = r["answers"]["department"]
                if ans["choice"] == gold:
                    correct += 1
            acc = correct / len(REQUESTS)
            row = {"limit": limit, "pad": pad, "n": len(REQUESTS), "correct": correct,
                   "accuracy": round(acc, 3), "median_latency_s": round(statistics.median(lat), 4),
                   "probe_tokens": ntok}
            rows.append(row)
            print("limit=%-5d pad=%-5d acc=%.3f (%2d/%d)  median=%.3fs  probe_tok=%d"
                  % (limit, pad, acc, correct, len(REQUESTS), row["median_latency_s"], ntok), flush=True)
    os.makedirs(os.path.join(LAB, "experiments", "orch-baseline"), exist_ok=True)
    with open(os.path.join(LAB, "experiments", "orch-baseline", "results.json"), "w") as f:
        json.dump({"device": "cuda", "gpu": torch.cuda.get_device_name(0),
                   "n_per_cell": len(REQUESTS), "rows": rows}, f, indent=1)
    print("wrote experiments/orch-baseline/results.json")


if __name__ == "__main__":
    main()

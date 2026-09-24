import os, time, json, sys
os.environ.setdefault("USE_TF", "0")
t0 = time.time()
sys.path.insert(0, "/home/parshu/projects/contri/laya-lab/fork")
import torch
import transformers
import laya
print("import s", round(time.time()-t0, 2), "torch", torch.__version__, "tf", transformers.__version__, "laya", laya.__version__, flush=True)
t0 = time.time()
agent = laya.load("/home/parshu/projects/contri/laya-lab/models/multilingual", device="cuda")
print("load s", round(time.time()-t0, 2), "device", agent.device, "dtype", agent.dtype, "amp", agent.amp_enabled, flush=True)
print("cfg max_len", agent.cfg.get("max_len"), "head_max_len", agent.cfg.get("head_max_len"), flush=True)
print("tok", type(agent.tok).__name__, "vocab", getattr(agent.tok, "vocab_size", None), "mask", repr(agent.tok.mask_token), flush=True)
Q = {"department": {"type": "choice", "instructions": "Which department should handle this request?",
     "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, system errors",
                  "sales": "pricing, new contracts, plan upgrades", "other": "everything else"}}}
s = "I was charged twice for invoice 4411, please refund the duplicate payment."
t0 = time.time(); r = agent.predict(s, Q, max_len=1024, head_max_len=256); print("fwd1 s", round(time.time()-t0,3), flush=True)
t0 = time.time(); r = agent.predict(s, Q, max_len=1024, head_max_len=256); print("fwd2 s", round(time.time()-t0,3), flush=True)
print(json.dumps(r, ensure_ascii=False), flush=True)
n = agent.tok("filler " * 4000, add_special_tokens=False)["input_ids"]
long_state = "Thanks for the update on the quarterly planning meeting. " * 300
t0 = time.time(); r2 = agent.predict(long_state, Q, max_len=1024, head_max_len=256); print("pad1024 s", round(time.time()-t0,3), "tokens", r2["usage"]["input_tokens"], r2["answers"]["department"]["choice"], flush=True)
t0 = time.time(); r3 = agent.predict(long_state, Q, max_len=8192, head_max_len=256); print("pad8192 s", round(time.time()-t0,3), "tokens", r3["usage"]["input_tokens"], r3["answers"]["department"]["choice"], flush=True)
print("gpu", torch.cuda.memory_allocated()/1e6, "MB allocated", torch.cuda.memory_reserved()/1e6, "MB reserved", flush=True)

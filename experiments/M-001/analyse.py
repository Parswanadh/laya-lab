#!/usr/bin/env python3
"""M-001 analysis: reachability, mixing capacity, cost model, power.

Pure-python (stdlib only); every number in findings/M-001.md is printed by this file.

Run:
    python3 experiments/M-001/analyse.py
"""
import json, math, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.abspath(os.path.join(HERE, "..", ".."))
ART = os.path.join(LAB, "fork", "research", "results", "long_context_multilingual.json")

# ---------------------------------------------------------------- verified cfg
LAYER_TYPES = ["full_attention", "sliding_attention", "sliding_attention"] * 7 + ["full_attention"]
assert len(LAYER_TYPES) == 22
GLOBAL_IDX = [i for i, t in enumerate(LAYER_TYPES) if t == "full_attention"]
SLIDING_IDX = [i for i, t in enumerate(LAYER_TYPES) if t == "sliding_attention"]
W = 64            # sliding half-window = local_attention // 2
D = 768
F = 1152          # intermediate_size
L_FULL = len(GLOBAL_IDX)
L_LOCAL = len(SLIDING_IDX)
HEAD_LAYERS = 2   # rl_agent_config.json: head_layers = 2

def hr(t): print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)

# ================================================================= 1 REACHABILITY
hr("1. REACHABILITY (exact ancestral-set simulation of the programmed mask)")

def reach(marker, n, layer_types=LAYER_TYPES, w=W):
    """EXACT backward ancestral set.  S_0 = {marker}.  Layer t: the query at any position
    in S_{t-1} reads its mask's key set, so
        S_t = union over q in S_{t-1} of keys(q),   keys(q) = [0,n) if full else [q-w,q+w].
    Starting from x_m only, this is the information-flow graph: an edge q -> m exists iff
    q is in some keys() at some layer, i.e. iff q is reachable in one hop, and then two
    hops through the composed sets, etc."""
    pos = {marker}
    sizes, widths, members = [], [], []
    for t, lt in enumerate(layer_types):
        if lt == "full_attention":
            pos = set(range(0, n))
        else:
            pos = {q for p in pos for q in range(max(0, p - w), min(n - 1, p + w) + 1)}
        sizes.append(len(pos))
        widths.append(max(abs(p - marker) for p in pos))
        members.append(set(pos))
    return widths, sizes, members

def reach_horizon(layer_types=LAYER_TYPES, w=W):
    """Per-hop HORIZON model requested by the task: how far past the marker can information
    be pulled, one mask at a time.  A global layer doubles the previous horizon (its query
    can pick any key, and every key already carries what the previous layers pooled); a
    sliding layer adds w.  H_0 = 2w: the fully-connected layer 0 reads every key, and each
    key's own value already pooled its +-w window from the embedding layer."""
    H = 2 * w
    out = [H]
    for lt in layer_types[1:]:
        H = 2 * H if lt == "full_attention" else H + w
        out.append(H)
    return out

# closed form for the programmed pattern (global layers at index 3k, gaps of 2 sliding)
def R_closed(k, w=W):
    """Per-hop horizon at global-layer index k (i.e. layer 3k).
    R_0 = 2w (the marker's window, doubled by the global layer); for k>=1,
    R_k = (8*2^k - 1)w.  Recurrence R_k = 2*R_{k-1} + 6w with R_0 = 2w."""
    return 2 * w if k == 0 else (8 * 2**k - 1) * w

SIM_N = 8192
MARKER = 300
BOUND = SIM_N - 1 - MARKER          # furthest token from the marker inside the sequence
w_sim, s_sim, m_sim = reach(MARKER, SIM_N)
h_sim = reach_horizon()
print(f"marker at {MARKER}, n = {SIM_N} (max possible distance {BOUND})")
print(f"{'layer':>5}  {'type':<18} {'|S_t| ancestral':>15} {'half-width':>11} "
      f"{'per-hop horizon':>16} {'closed form R_k':>16} {'check':<8}")
for t, lt in enumerate(LAYER_TYPES):
    k = t // 3
    if lt == "full_attention":
        rc = R_closed(k)
        chk = "OK" if min(rc, BOUND) == min(h_sim[t], BOUND) else "MISMATCH"
        print(f"{t:>5}  {lt:<18} {s_sim[t]:>15} {w_sim[t]:>11} {h_sim[t]:>16} "
              f"{rc:>16} {chk:<8}")
    else:
        print(f"{t:>5}  {lt:<18} {s_sim[t]:>15} {w_sim[t]:>11} {h_sim[t]:>16}")

print("\nRESULT A (exact ancestral set): because LAYER 0 IS GLOBAL, S_0 = [0, n) after a")
print("single layer. Every token in the sequence, however far, is an ancestor of the")
print("marker at depth 1. The reachability graph has diameter 1.")
print("\nRESULT B (per-hop horizon requested by the task): if the layer-0 short circuit is")
print("excluded and only per-hop composition counts, the horizon at the global layers is")
print(f"{'k':>2} {'layer':>6} {'R_k (tokens)':>13} {'clipped to n=8192':>18} {'frac of doc':>13}")
for k in range(8):
    rc = R_closed(k)
    print(f"{k:>2} {3*k:>6} {rc:>13d} {min(rc, BOUND):>18d} {min(rc, BOUND)/BOUND:>12.2%}")

def min_layer_index(d, w=W):
    for k in range(8):
        if d <= R_closed(k, w): return 3 * k
    return None
print("\nminimum layer index L*(d) under the per-hop horizon:")
print(f"{'d =':>8} {'L*(d)':>6} {'closed form':>13} {'global hops used k':>19} "
      f"{'global hops REMAINING (8-k)':>28} {'entry-dilution 1/(k+1)':>24}")
for d in [1, 64, 65, 128, 129, 449, 960, 961, 1984, 1985, 4032, 4033, 7000, 8128, 8129, 8191]:
    L = min_layer_index(d)
    cf = max(0, 3 * math.ceil(math.log2((d / W + 1) / 8)))
    k = 0 if L is None else L // 3
    print(f"{d:>8} {('>=22' if L is None else str(L)):>6} {cf:>13} {k:>19} "
          f"{L_FULL - k:>28} {1.0/(k+1):>24.3f}")
print("closed form: L*(d) = 0 for d <= 2w;  L*(d) = 3*ceil(log2((d/w + 1)/8)) for d > 2w")
print("(R_k = (8*2^k - 1)w >= d  <=>  2^k >= (d/w + 1)/8  <=>  k >= log2((d/w+1)/8))")

# ============================================================ 2 MIXING CAPACITY
hr("2. MIXING CAPACITY (cross-document edges, head block = 0..H-1, state = H..n-1)")

def cross_edges(n, H):
    """Exact count of (query in head, key in state) attention edges, summed over layers,
    for the programmed layer pattern. Bidirectional sliding mask: |i-j| <= W."""
    tot = {}
    for t, lt in enumerate(LAYER_TYPES):
        c = 0
        if lt == "full_attention":
            c = H * (n - H)
        else:
            for i in range(H):
                lo, hi = max(H, i - W), min(n - 1, i + W)
                c += max(0, hi - lo + 1)
        tot[t] = c
    return tot

for (n, H) in [(1024, 256), (8192, 384)]:
    tot = cross_edges(n, H)
    s = sum(tot.values())
    print(f"\nn = {n}, head block H = {H}  (state block = {n-H} tokens)")
    print(f"  total cross-document (marker-region -> state) edges : {s:,}")
    print(f"  from the 8 global layers  : {sum(tot[i] for i in GLOBAL_IDX):,} "
          f"({sum(tot[i] for i in GLOBAL_IDX)/s:.1%})")
    print(f"  from the 14 sliding layers: {sum(tot[i] for i in SLIDING_IDX):,} "
          f"({sum(tot[i] for i in SLIDING_IDX)/s:.1%})")
    print(f"  per-layer fraction of the state reachable from ONE marker (last marker at H-1):")
    for t in range(22):
        if LAYER_TYPES[t] == "full_attention":
            r = 1.0
        else:
            lo, hi = max(H, (H - 1) - W), min(n - 1, (H - 1) + W)
            r = max(0, hi - lo + 1) / (n - H)
        print(f"    layer {t:>2} {LAYER_TYPES[t]:<18} {r:>8.4%} of state "
              f"({max(0, min(n - 1, H - 1 + W) - max(H, H - 1 - W) + 1):>5} tokens)")

print("\n-- global-layer budget: hops needed to reach distance d vs hops left to MIX with --")
print(f"{'d (tokens from marker)':>24} {'k = global hops used':>21} {'8 - k left to mix':>18}"
      f" {'tail frac reached':>18}")
for d in [64, 321, 960, 1984, 4032, 7000, 8128, 8191]:
    L = min_layer_index(d)
    k = 0 if L is None else L // 3
    print(f"{d:>24} {k:>21} {L_FULL - k:>18} {min(R_closed(k), BOUND)/BOUND:>17.2%}")

print("\n-- information available to the marker as a function of document position --")
print("(under the exact model, every position is reachable at layer 0; what differs is how")
print(" much of the network remains to process it. Under the per-hop horizon model, a token")
print(" at distance d is first read at layer L*(d) and then has (21 - L*(d)) layers left.)")
print(f"{'d':>8} {'L*(d)':>7} {'layers remaining after first read':>35} "
      f"{'token count at this distance is':>32}")
for d in [64, 320, 960, 1984, 4032, 7000, 8191]:
    L = min_layer_index(d)
    L = 22 if L is None else L
    lo = max(1, d - 320)
    print(f"{d:>8} {L:>7} {22 - L:>35} {'~%d tokens in [%d, %d]' % (320, lo, d):>32}")

# head block width from build_sequence, computed exactly (published tokenizer)
hr("2b. HEAD BLOCK WIDTH (computed with laya.build_sequence, fork source, CPU only)")

# ================================================================ 3 COST MODEL
hr("3. COST MODEL")

def flops(n, head_layers=HEAD_LAYERS, with_head=True):
    """MODELLED inference FLOPs (multiply-accumulate = 2 flops), forward pass only,
    batch 1, no attention-mask materialisation, no softmax/norm/rope/embedding terms."""
    proj = 17 * n * D * D                     # qkv = 3*d^2, out = d^2, wi = d*F, wo = F*d
    mlp = 4 * n * D * F                       # (2*n*d*F) for each of Wi, Wo... 2 n d F each
    q_enc = 2 * L_FULL * n * n * D + 2 * L_LOCAL * n * (2 * W + 1) * D
    q_head = head_layers * 2 * n * n * D if with_head else 0
    return proj + mlp + q_enc + q_head, proj, q_enc, q_head, mlp

n0 = 1024
tot0 = flops(n0)[0]
print(f"d={D}  F={F}  22 layers = {L_FULL} full + {L_LOCAL} sliding (w={W})  head_layers={HEAD_LAYERS}")
print(f"\n{'n':>7} {'total FLOPs':>16} {'ratio vs n=1024':>16} {'encoder q share':>16} {'head q share':>14}")
for n in [61, 512, 1024, 2048, 3006, 3972, 4938, 5946, 6912, 8192]:
    T, proj, qe, qh, mlp = flops(n)
    print(f"{n:>7} {T:>16.4e} {T/tot0:>16.3f} {qe/T:>16.1%} {qh/T:>14.1%}")

# closed form.  For n and n0 = 1024 both below saturation, encoder quadratic terms
# contribute 2*L_FULL*n*d and the linear coefficient is
#   b = 2*L_FULL*d (full-layer attn) + 2*L_LOCAL*(2w+1)*d (sliding attn) + 17*d^2 + 4*d*F
b_c = 2 * L_FULL * D + 2 * L_LOCAL * (2 * W + 1) * D + 17 * D * D + 4 * D * F
c_head = 2 * HEAD_LAYERS * D
T8, T0 = flops(8192)[0], flops(1024)[0]
print(f"\nMODELLED  FLOPs(8192)/FLOPs(1024) = {T8/T0:.3f}x")
print(f"MODELLED  FLOPs(5946)/FLOPs(1024) = {flops(5946)[0]/T0:.3f}x")
print(f"MODELLED  FLOPs(7000)/FLOPs(1024) = {flops(7000)[0]/T0:.3f}x")
print(f"closed form: ratio = (b*n8 + c_head*n8^2) / (b*n0 + c_head*n0^2),")
print(f"  b = 2*{L_FULL}*{D} + 2*{L_LOCAL}*{2*W+1}*{D} + 17*{D}^2 + 4*{D}*{F} = {b_c:,}")
print(f"  c_head = 2*head_layers*d = {c_head:,}  (the DECISION HEAD's own 2 self-attention")
print(f"  layers are quadratic in n too - the head is forward()ed over the whole sequence)")
num = b_c * 8192 + c_head * 8192**2
den = b_c * 1024 + c_head * 1024**2
print(f"  = ({b_c}*8192 + {c_head}*8192^2) / ({b_c}*1024 + {c_head}*1024^2) = {num/den:.3f}")

print(f"\nBOUND: modelled ratio = {T8/T0:.3f}x. In the family lat-proportional-to-FLOPs with")
print(f"  a fixed linear coefficient b = {b_c:,} and a variable quadratic coefficient c >= 0,")
print(f"  the ratio (b*8192 + c*8192^2)/(b*1024 + c*1024^2) increases monotonically in c and")
print(f"  tends to (8192/1024)^2 = 64.0x from below. So the SUPREMUM is exactly 64x and the")
print(f"  headline 219x is UNREACHABLE for any number of global layers, hidden size, or head")
print(f"  width - 219 > 64. The only way to see 219x is if the measured cost is not FLOPs-")
print(f"  proportional, which the section-4 comparison tests.")
for ctest in [c_head, 10 * c_head, 100 * c_head, 1e6]:
    r = (b_c * 8192 + ctest * 8192**2) / (b_c * 1024 + ctest * 1024**2)
    print(f"    c = {ctest:>12,.0f}  -> ratio {r:>7.3f}x")

# ============================================================= 4 MEASURED DATA
hr("4. MEASURED UPSTREAM DATA + NORMALISED COMPARISON")
data = json.load(open(ART))
rows = data["rows"]
cases = data["cases"]
print(f"artifact : {os.path.relpath(ART, LAB)}")
print(f"device={data['device']} platform={data['platform']} torch={data['torch']} "
      f"transformers={data['transformers']} laya={data['laya']}")
print(f"checkpoint={data['checkpoint']}\n")
print(f"{'pad':>6} {'limit':>6} {'n':>4} {'acc':>5} {'median lat s':>13} {'median tok':>11}")
for r in rows:
    print(f"{r['pad_tokens']:>6} {r['limit']:>6} {r['n']:>4} {r['accuracy']:>5.2f} "
          f"{r['median_latency_s']:>13.3f} {r['median_input_tokens']:>11d}")

print("\n-- the headline number the task quotes --")
a = [r for r in rows if r["limit"] == 8192 and r["pad_tokens"] == 6000][0]
b = [r for r in rows if r["limit"] == 8192 and r["pad_tokens"] == 0][0]
print(f"  limit=8192, pad=0    : {b['median_input_tokens']} tok, {b['median_latency_s']} s")
print(f"  limit=8192, pad=6000 : {a['median_input_tokens']} tok, {a['median_latency_s']} s")
raw = a['median_latency_s'] / b['median_latency_s']
grow = a['median_input_tokens'] / b['median_input_tokens']
pred_norm = flops(a['median_input_tokens'])[0] / flops(b['median_input_tokens'])[0]
print(f"  MEASURED raw ratio   = {raw:.2f}x  for a {grow:.2f}x input growth")
print(f"  MODELLED ratio for the same two input lengths = {pred_norm:.3f}x  "
      f"-> measured/modelled = {raw/pred_norm:.3f} (model OVER-predicts by {pred_norm/raw:.2f}x)")
print("  The FLOP model has NO constant term, so it cannot describe the 61-token endpoint:")
print(f"    modelled latency growth n=61 -> n=5946 (pure FLOPs) = {grow:.1f}x,")
print(f"    measured growth at the same two points           = {raw:.1f}x,")
print(f"    so the model OVER-predicts by {pred_norm/raw:.2f}x (anything < 1 means measured < modelled).")
print(f"    n=61 -> n=1032: modelled {flops(1032)[0]/flops(61)[0]:.1f}x vs measured "
      f"{0.211/0.016:.1f}x - the model OVER-predicts the small-n end by "
      f"{(flops(1032)[0]/flops(61)[0])/(0.211/0.016):.2f}x.")
print(f"    Conversely at n=5946 -> 6912 the model UNDER-predicts by "
      f"{(4.503/3.505)/(flops(6912)[0]/flops(5946)[0]):.2f}x. No single FLOPs-only model")
print("    fits both ends: it needs a constant term at small n and a super-quadratic term at large n.")

print("\n-- matched-input pairs: SAME input length, different max_len --")
print("(this is the only comparison that isolates the max_len knob from the input-growth")
print(" confound; the pad=1000 pair has 1024 vs 1032 tokens, i.e. within 0.8%)")
print(f"{'n(1024)':>8} {'lat(1024)':>10} {'n(8192)':>8} {'lat(8192)':>10} "
      f"{'meas ratio':>11} {'modelled':>9} {'meas/mod':>9} {'per-token norm':>15}")
pairs = []
for pad in [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000]:
    r1 = [r for r in rows if r["pad_tokens"] == pad and r["limit"] == 1024][0]
    r2 = [r for r in rows if r["pad_tokens"] == pad and r["limit"] == 8192][0]
    mr = r2["median_latency_s"] / r1["median_latency_s"]
    pr = flops(r2["median_input_tokens"])[0] / flops(r1["median_input_tokens"])[0]
    # per-token normalised: scale the 8192-limit latency back to the 1024-limit input size
    nn = (r2["median_latency_s"] * r1["median_input_tokens"] / r2["median_input_tokens"]) / r1["median_latency_s"]
    pairs.append((r1, r2, mr, pr, nn))
    print(f"{r1['median_input_tokens']:>8} {r1['median_latency_s']:>10.3f} "
          f"{r2['median_input_tokens']:>8} {r2['median_latency_s']:>10.3f} "
          f"{mr:>11.3f} {pr:>9.3f} {mr/pr:>9.3f} {nn:>15.3f}")
print("  'meas ratio' is contaminated when n(1024) != n(8192): at pad=4000 the 1024-limit")
print("  run is truncated to 1024 tokens and the 8192-limit run sees 3972, so the 7.9x is")
print("  mostly input growth. 'per-token norm' removes it and is the honest comparison.")
xs = [p[1]["median_input_tokens"] for p in pairs[2:]]
ys = [p[4] for p in pairs[2:]]
n_pts = len(xs)
mx, my = sum(math.log(x) for x in xs) / n_pts, sum(math.log(y) for y in ys) / n_pts
slope = sum((math.log(x) - mx) * (math.log(y) - my) for x, y in zip(xs, ys)) / \
        sum((math.log(x) - mx) ** 2 for x in xs)
print(f"  log-log slope of per-token-normalised measured ratio vs n = {slope:.3f} +- "
      f"over {n_pts} points (a pure n^1 FLOP model would give 0.0 here)")

print("\n-- per-token cost (s/token), limit=8192, from the median latencies --")
print(f"{'n':>7} {'s/token':>10} {'x vs n=61':>11} {'n x vs 61':>11}")
base = None
for r in rows:
    if r["limit"] != 8192: continue
    pt = r["median_latency_s"] / r["median_input_tokens"]
    if base is None: base = pt
    print(f"{r['median_input_tokens']:>7} {pt:>10.5f} {pt/base:>11.2f} "
          f"{r['median_input_tokens']/61:>11.2f}")

print("\n-- scaling from n=1024 to each larger n, limit=8192 (measured) --")
r1024 = [r for r in rows if r["limit"] == 8192 and r["pad_tokens"] == 1000][0]
print(f"{'n':>8} {'measured ratio':>15} {'implied exponent':>17} {'modelled ratio':>15} "
      f"{'meas/modelled':>14}")
for r in rows:
    if r["limit"] == 8192 and r["median_input_tokens"] > 1100:
        n = r["median_input_tokens"]
        ratio = r["median_latency_s"] / r1024["median_latency_s"]
        expo = math.log(ratio) / math.log(n / 1024)
        mod = flops(n)[0] / flops(1024)[0]
        print(f"{n:>8} {ratio:>15.3f} {expo:>17.3f} {mod:>15.3f} {ratio/mod:>14.3f}")
print("  the measured exponent settles near 1.6; the FLOP model's local exponent is higher")
print("  because the quadratic term only takes over above ~2000 tokens.")

# --- empirical fit of lat(n) = F + a*n^p  (own model, validated against these 8 points)
pts = [(r["median_input_tokens"], r["median_latency_s"]) for r in rows if r["limit"] == 8192]
def sse(F, a, p):
    return sum((F + a * n**p - lat)**2 for n, lat in pts)
best = None
for F in [x / 100 for x in range(0, 4001)]:
    for p in [x / 100 for x in range(100, 301)]:
        num = den = 0.0
        for n, lat in pts:
            x = n**p
            num += x * (lat - F); den += x * x
        a = num / den
        e = sse(F, a, p)
        if best is None or e < best[0]:
            best = (e, F, a, p)
e, F, a, p = best
print(f"\n-- fitted lat(n) = F + a*n^p on the 8 limit=8192 medians (own model, [MODELLED]) --")
print(f"   F = {F:.3f} s   a = {a:.3e}   p = {p:.3f}   RMSE = {math.sqrt(e/len(pts)):.4f} s")
for n, lat in pts:
    print(f"   n={n:>5} measured {lat:>6.3f}  fitted {F + a*n**p:>6.3f}  "
          f"resid {lat-(F+a*n**p):>+7.3f} s ({(lat-(F+a*n**p))/lat:>+6.1%})")
print("   F is the fitted constant: dispatch, python, tokenisation, the fixed 3-token")
print("   prompt scaffold and the decision head's own cost floor. The pure FLOP model has")
print(f"   no such term, which is why it cannot fit n=61. F alone is {F/0.016:.2f}x the")
print(f"   smallest measured latency ({F:.3f} s vs 0.016 s).")

# fit with p forced to 2 (FLOP-quadratic) for comparison
num = den = 0.0
for n, lat in pts:
    x = n*n
    num += x*(lat-F); den += x*x
a2 = num/den
e2 = sse(F, a2, 2.0)
print(f"   p fixed at 2: a = {a2:.3e}, RMSE = {math.sqrt(e2/len(pts)):.4f} s")

# ================================================================ 5 POWER
hr("5. POWER / SAMPLE SIZE")
def z_for(conf=0.95):
    # inverse normal via bisection on erf
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if math.erf(mid / math.sqrt(2)) < conf: lo = mid
        else: hi = mid
    return (lo + hi) / 2
z = z_for()
print(f"z_(1-alpha/2) for 95% = {z:.5f}")
print(f"\nWald half-width at p=0.5:  h = z*sqrt(0.25/n)")
print(f"{'n':>6} {'half-width at p=0.5':>21} {'Wilson [lo,hi] at x=n/2':>26}")
for n in [20, 100, 200, 300, 400, 600, 1000]:
    h = z * math.sqrt(0.25 / n)
    # Wilson interval at phat = 0.5
    ph = 0.5
    den = 1 + z*z/n
    c = (ph + z*z/(2*n)) / den
    hw_w = z*math.sqrt(ph*(1-ph)/n + z*z/(4*n*n)) / den
    print(f"{n:>6} {h:>21.4f} {('[' + format(c-hw_w, '.4f') + ', ' + format(c+hw_w, '.4f') + ']'):>26}")
print("\nexact 95% Clopper-Pearson interval at x = n/2 (binomial tails, computed in log space):")
def binom_log_pmf(k, n, p):
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))
def binom_cdf(k, n, p):
    return sum(math.exp(binom_log_pmf(i, n, p)) for i in range(0, k + 1))
def cp_interval(n, x, alpha=0.05):
    """Exact Clopper-Pearson two-sided interval for x successes in n trials.
    P(X >= x | p) increases as p falls, so to find the LARGEST p with
    P(X >= x | p) >= alpha/2 we bisect on that predicate.  Symmetrically
    P(X <= x | p) increases with p."""
    lo, hi = 0.0, 1.0
    if x > 0:
        a, b = 1.0, 0.0                   # a: predicate true, b: predicate false; move a down
        for _ in range(200):
            m = (a + b) / 2
            if 1 - binom_cdf(x - 1, n, m) >= alpha / 2: a = m
            else: b = m
        lo = a
    if x < n:
        a, b = 0.0, 1.0
        for _ in range(200):
            m = (a + b) / 2
            if binom_cdf(x, n, m) >= alpha / 2: a = m
            else: b = m
        hi = a
    return lo, hi
# self-check against a known value: 7/20, Clopper-Pearson 95% = [0.1539, 0.5922]
_c = cp_interval(20, 7)
assert abs(_c[0] - 0.1539) < 5e-4 and abs(_c[1] - 0.5922) < 5e-4, _c
print(f"  [self-check] cp_interval(20, 7) = [{_c[0]:.4f}, {_c[1]:.4f}] "
      f"(published 0.1539, 0.5922) OK")
def exact_at_half(n, alpha=0.05):
    return cp_interval(n, n // 2, alpha)
for n in [20, 200, 300, 400, 600]:
    lo, hi = exact_at_half(n)
    print(f"   n={n:>4}, x={n//2:>4}, phat=0.5: exact 95% [{lo:.4f}, {hi:.4f}]  "
          f"half-width {(hi-lo)/2:.4f}")
print("   this is the WORST CASE for a single proportion (p=0.5 maximises p(1-p)); at")
print("   phat=0.35 or 0.9 the interval is narrower.")

print("\nSanity check of the power arithmetic: 'n>=200 gives +-0.066 at 95% for p~0.5'")
print("(plan.md section 6) is the SINGLE-PROPORTION Wald half-width. It is NOT the")
print("resolution of an ARM-vs-ARM comparison; for that the paired (McNemar) design applies.")

print("\nMcNemar power.  Two formulations are reported because they disagree, and the")
print("disagreement is itself the finding:")
print("  (a) EXACT conditional sign test: condition on the observed number of discordant")
print("      pairs m and ask whether X ~ Bin(m, 1/2).  This is what 'exact McNemar' means,")
print("      and it is CONSERVATIVE - for every m below the point where 2*P(X<=k) <= 0.05")
print("      has a solution the conditional rejection probability is 0, so power stays at")
print("      alpha however large delta is.  With delta=0.05 you need m ~ 85 discordant")
print("      pairs for the test to bite at all (see the self-checks below).")
print("  (b) UNCONDITIONAL normal approximation on the discordant counts: the standard")
print("      design-stage formula, which is what a sample-size argument must use.")
def binom_pmf_list(m, p):
    """All m+1 binomial probabilities for p <= 0.5, built by the ratio recurrence:
    pmf(0) = (1-p)^m in log space, then pmf(k+1) = pmf(k) * (m-k)/(k+1) * p/(1-p)."""
    if p <= 0.0:
        out = [0.0] * (m + 1); out[0] = 1.0
        return out
    log_first = m * math.log1p(-p)
    ratio = p / (1.0 - p)
    out = [0.0] * (m + 1)
    cur = log_first
    out[0] = math.exp(cur)
    for k in range(0, m):
        cur += math.log(m - k) - math.log(k + 1) + math.log(ratio)
        out[k + 1] = math.exp(cur) if cur > -700 else 0.0
    return out
_SIGN_CACHE = {}
def sign_reject_prob(m, alpha=0.05):
    """(a) P(reject | m discordant pairs): exact two-sided sign test, critical region
    {k : 2*P(X <= k) <= alpha} union its mirror.  Cached by m."""
    if m in _SIGN_CACHE:
        return _SIGN_CACHE[m]
    if m == 0:
        _SIGN_CACHE[m] = 0.0
        return 0.0
    pmf = binom_pmf_list(m, 0.5)
    cum = 0.0
    kcrit = -1
    for k in range(0, m // 2 + 1):
        cum += pmf[k]
        if 2.0 * cum <= alpha:
            kcrit = k
        else:
            break
    val = 0.0 if kcrit < 0 else 2.0 * sum(pmf[0:kcrit + 1])
    _SIGN_CACHE[m] = val
    return val
def Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
def mcnemar_power_conditional(n, delta):
    """(a) exact-conditional power.  Under H1 the discordant count is m ~ Bin(n, delta)
    (the boundary case where the baseline never wins a discordant item); conditional on m
    the exact sign test's rejection probability is sign_reject_prob(m)."""
    pmf_n = binom_pmf_list(n, min(max(delta, 0.0), 1.0))
    return sum(pmf_n[m] * sign_reject_prob(m) for m in range(1, n + 1))
def zq(target):
    lo, hi = 0.0, 12.0
    for _ in range(300):
        mid = (lo + hi) / 2
        if Phi(mid) < target: lo = mid
        else: hi = mid
    return (lo + hi) / 2
def mcnemar_power_normal(n, delta, q, alpha=0.05):
    """(b') unconditional normal-approximation power for the PAIRED design, from first
    principles.  X ~ Bin(n, p12) counts items the new arm wins, Y ~ Bin(n, p21) counts
    items the baseline wins, X and Y independent, p12 = (q+delta)/2, p21 = (q-delta)/2.
    Under H0 (p12 = p21 = q/2) the McNemar statistic (X-Y)/sqrt(X+Y) has SD
    0.5*sqrt(n*q); under H1, E[X-Y] = n*delta and Var(X-Y) = n*q - n*delta^2.
    q - the ANTI-correlation between the two arms - is the parameter that controls power;
    delta alone does not."""
    if delta <= 0 or q <= abs(delta):
        return 0.0
    zcrit = zq(1 - alpha / 2)
    lam = n * delta
    sd = math.sqrt(max(n * q - n * delta * delta, 1e-12))
    crit = zcrit * 0.5 * math.sqrt(n * q)
    return (1 - Phi((crit - lam) / sd) + Phi((-crit - lam) / sd))
def rejection_threshold_exact(n, q, alpha=0.05):
    """Smallest |X-Y| at which the exact McNemar test rejects, under H0 p12=p21=q/2.
    The distribution of X-Y is the self-convolution of the pmf with its reverse; the
    rejection region is its UPPER tail, accumulated from the largest difference down."""
    p = q / 2.0
    pmf = binom_pmf_list(n, p)
    conv = [0.0] * (2 * n + 1)              # conv[d + n] = P(X - Y = d)
    for k, pk in enumerate(pmf):
        if pk == 0.0:
            continue
        for l, pl in enumerate(pmf):
            conv[k - l + n] += pk * pl
    tot = 0.0
    for d in range(0, n + 1):               # upper tail, one side only
        pr = conv[d + n]
        if tot + pr > alpha / 2.0:
            return d
        tot += pr
    return n + 1
# self-check: at q=0.1, n=200 the threshold should sit near the normal value
for _n, _q in [(200, 0.10), (200, 0.20), (400, 0.10)]:
    _d = rejection_threshold_exact(_n, _q)
    _nn = math.sqrt(2 * _n * (_q / 2) * (1 - _q / 2))
    print(f"  [self-check] n={_n} q={_q:.2f}: exact |X-Y| threshold = {_d} "
          f"(normal approx z*SD = {1.959964*_nn:.1f}, SD = {_nn:.2f})")
def mcnemar_power_exact_uncond(n, delta, q):
    """(b) EXACT unconditional power: sum over (X, Y) of the exact McNemar rejection
    region {|X-Y| >= d*}.  O(n) per call via prefix sums of the binomial pmfs.
    This is the design-grade number; the conditional test in (a) is not."""
    if delta <= 0 or q <= abs(delta):
        return 0.0
    p12, p21 = (q + delta) / 2.0, (q - delta) / 2.0
    dstar = rejection_threshold_exact(n, q)
    fx = binom_pmf_list(n, p12)
    fy = binom_pmf_list(n, p21)
    # prefix / suffix sums of fx
    pref = [0.0] * (n + 1)
    acc = 0.0
    for i in range(0, n + 1):
        acc += fx[i]
        pref[i] = acc
    suf = [0.0] * (n + 2)
    acc = 0.0
    for i in range(n, -1, -1):
        acc += fx[i]
        suf[i] = acc
    pw = 0.0
    for y in range(0, n + 1):
        p = fy[y]
        if p == 0.0:
            continue
        lo = y - dstar                     # X <= y - d*
        if lo >= 0:
            pw += p * pref[lo]
        hi = y + dstar                     # X >= y + d*
        if hi <= n:
            pw += p * suf[hi]
    return min(pw, 1.0)
for _m in [100, 200, 600]:
    _pmf = binom_pmf_list(_m, 0.5)
    _c = 0.0
    _k = -1
    for k in range(0, _m // 2 + 1):
        _c += _pmf[k]
        if 2.0 * _c <= 0.05: _k = k
        else: break
    print(f"  [self-check] m={_m}: exact sign test rejects for X <= {_k} or X >= {_m-_k}; "
          f"P(reject|H0) = {2.0*sum(_pmf[0:_k+1]):.4f}")
print("\n(a) EXACT-conditional McNemar power (conservative; conditions on m = X+Y):")
print(f"{'n':>6} {'delta=0.03':>11} {'delta=0.05':>11} {'delta=0.10':>11}")
for n in [100, 200, 400, 800, 2000]:
    print(f"{n:>6} " + " ".join(f"{mcnemar_power_conditional(n, d):>11.3f}"
                                 for d in (0.03, 0.05, 0.10)))
print("\n(b) EXACT unconditional McNemar power (design-grade), by discordance rate q:")
for q in [0.10, 0.20]:
    print(f"  q = {q:.2f}:")
    print(f"{'n':>6} {'delta=0.03':>11} {'delta=0.05':>11} {'delta=0.10':>11}")
    for n in [100, 200, 400, 800, 2000]:
        row = []
        for d in (0.03, 0.05, 0.10):
            row.append(f"{mcnemar_power_exact_uncond(n, d, q):>11.3f}" if d < q else f"{'--':>11}")
        print(f"{n:>6} " + " ".join(row))
print("\nnormal approximation (b') vs exact unconditional (b), same parameters:")
print(f"{'n':>6} {'q':>6} {'delta':>6} {'normal approx':>14} {'exact uncond':>13}")
for n, q, d in [(200, 0.10, 0.05), (400, 0.10, 0.05), (800, 0.10, 0.05),
                (200, 0.20, 0.10), (400, 0.20, 0.10), (800, 0.20, 0.10)]:
    print(f"{n:>6} {q:>6.2f} {d:>6.2f} {mcnemar_power_normal(n, d, q):>14.3f} "
          f"{mcnemar_power_exact_uncond(n, d, q):>13.3f}")
print("\npower at the protocol's candidate sample sizes (EXACT unconditional):")
print(f"{'q':>5} {'delta':>6} " + " ".join(f"{'n=' + str(n):>9}" for n in [100, 200, 300, 400, 600, 800]))
for q in [0.10, 0.20, 0.30]:
    for delta in [0.03, 0.05, 0.10]:
        if delta >= q:
            continue
        row = " ".join(f"{mcnemar_power_exact_uncond(n, delta, q):>9.3f}"
                       for n in [100, 200, 300, 400, 600, 800])
        print(f"{q:>5.2f} {delta:>6.2f} " + row)
print("\nsmallest n (multiple of 20) with EXACT unconditional power >= 0.80:")
for q in [0.10, 0.20, 0.30]:
    for delta in [0.03, 0.05, 0.10]:
        if delta >= q:
            continue
        n = 20
        while n < 4000 and mcnemar_power_exact_uncond(n, delta, q) < 0.80:
            n += 20
        print(f"   q={q:.2f}, delta={delta:.2f} -> n = {n:>4} "
              f"(power {mcnemar_power_exact_uncond(n, delta, q):.3f})")

print("\nUPSTREAM'S OWN n=20 CELL, tested against the chance rate (4 options -> 0.25):")
for x, lab in [(7, "pad>=2000, limit=1024 (acc 0.35)"),
               (17, "pad=2000, limit=8192 (acc 0.85)")]:
    pv = sum(math.comb(20, i) * 0.25**i * 0.75**(20 - i) for i in range(x, 21))
    print(f"   {lab}: exact one-sided binomial p vs 0.25 = {pv:.5f}")
print("   the 0.35 cell is NOT distinguishable from the 0.25 chance rate at n=20, so")
print("   upstream's claim that 0.35 IS the majority prior is itself underpowered.")
for x, n, lab in [(7, 20, "upstream 0.35 cell"), (17, 20, "upstream 0.85 cell"),
                  (19, 20, "upstream 0.95 cell")]:
    lo, hi = cp_interval(n, x)
    print(f"   {lab}: exact Clopper-Pearson 95% = [{lo:.3f}, {hi:.3f}] "
          f"(width {hi-lo:.3f}, +-{(hi-lo)/2:.3f})")

print("\nfamily-wise error: 25 cells x 2 arms, independent two-proportion tests at alpha=0.05")
for m in [10, 25, 50]:
    print(f"   m={m:>3} comparisons: P(>=1 false positive | all null) = {1-(1-0.05)**m:.3f}; "
          f"Bonferroni alpha = {0.05/m:.5f}")
print("\nBonferroni-corrected alpha needs a larger raw p; per-cell n to keep the SAME MDE")
print("under alpha=0.05/m scales as n * (z_(1-a/2m)/z_(1-a/2))^2 (zq defined above):")
for m in [1, 10, 25, 50]:
    zm = zq(1 - 0.05/(2*m))
    print(f"   m={m:>3}: z={zm:.4f}, inflation factor vs m=1 = {(zm/z)**2:.3f}")

# binomial resolution floor
print("\nresolution floor: with n items and a 4-option task, accuracy is a multiple of 1/n.")
for n in [20, 200, 400, 600]:
    print(f"   n={n:>4}: one item = {1/n:.4f} accuracy; floor resolution with a "
          f"majority-class baseline of 0.35 is {(0.35*n):.0f}/{n}")

# ================================================================ 6 LAYOUT PROBE
hr("6. HEAD-BLOCK WIDTH PROBE (fork source, CPU tokenizer only)")
print("Builds the real sequence with the shipped tokenizer and the real upstream question,")
print("so the marker positions and the state start are COMPUTED, not assumed.")
try:
    import os as _os
    _os.environ.setdefault("USE_TF", "0")
    import sys
    sys.path.insert(0, _os.path.join(LAB, "fork"))
    from laya.common import build_sequence
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(_os.path.join(LAB, "models", "multilingual", "tokenizer"))
    qs = {
        "upstream department (4 long options)": {
            "t": "department",
            "ins": data["questions"]["department"]["instructions"],
            "criteria": data["questions"]["department"]["criteria"],
        },
        "short 2-option yes/no": {
            "t": "sentiment",
            "ins": "Does the statement hold?",
            "criteria": {"yes": "the statement holds", "no": "the statement does not hold"},
        },
    }
    doc = "The quick brown fox jumps over the lazy dog. " * 900
    for label, q in qs.items():
        print(f"\n  question: {label}")
        for max_len, head_max_len in [(1024, 256), (8192, 256), (8192, 192), (512, 192)]:
            ids, markers = build_sequence(tok, doc, q, max_len=max_len, head_max_len=head_max_len)
            n_state = len(ids) - (markers[-1] + 1 + len(tok(" " + list(q["criteria"].values())[0], add_special_tokens=False)["input_ids"]) + 1) if markers else None
            print(f"    max_len={max_len:>5} head_max_len={head_max_len:>3}: len(ids)={len(ids):>4} "
                  f"markers={markers} -> last marker m*={max(markers)}, "
                  f"state block starts at {markers[-1]+1}+len(last option), "
                  f"HEAD WIDTH for the marker = {max(markers)+1} tokens")
    print("\n  The marker's position m* is the parameter that matters for reachability, and it")
    print("  is set by head_max_len and the option rendering; the state block sits AFTER it.")
except Exception as exc:  # noqa: BLE001
    print(f"  PROBE FAILED ({type(exc).__name__}: {exc}) - marker positions stay MODELLED")

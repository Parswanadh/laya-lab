import sys, re, os, urllib.request, html, gzip, io
UA={"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36","Accept-Encoding":"gzip"}
def get(url):
    req=urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        d=r.read()
        if r.headers.get("Content-Encoding")=="gzip":
            d=gzip.decompress(d)
        return d.decode("utf-8","replace")
def strip(t):
    t=re.sub(r'(?is)<(script|style|nav|footer|header)[^>]*>.*?</\1>',' ',t)
    t=re.sub(r'(?is)<br\s*/?>|</p>|</div>|</li>|</tr>|</h[1-6]>','\n',t)
    t=re.sub(r'(?is)</t[dh]>',' | ',t)
    t=re.sub(r'(?s)<[^>]+>',' ',t)
    t=html.unescape(t)
    t=re.sub(r'[ \t\xa0]+',' ',t)
    t=re.sub(r'\n\s*\n\s*\n+','\n\n',t)
    return t.strip()
def arxiv_id(u):
    m=re.search(r'(\d{4}\.\d{4,5})',u); return m.group(1) if m else None
for u in sys.argv[1:]:
    aid=arxiv_id(u) or re.sub(r'\W+','_',u)[:60]
    out=f".scratch/papers/{aid}.txt"
    if os.path.exists(out) and os.path.getsize(out)>3000:
        print("CACHED", out); continue
    txt=""
    for cand in ([f"https://arxiv.org/html/{aid}v3",f"https://arxiv.org/html/{aid}v2",f"https://arxiv.org/html/{aid}v1",f"https://arxiv.org/html/{aid}"] if aid else []) + [u]:
        try:
            raw=get(cand); s=strip(raw)
            if len(s)>3000: txt=f"SOURCE_URL: {cand}\n\n"+s; break
        except Exception as e:
            pass
    if not txt and aid:
        try:
            txt=f"SOURCE_URL: https://arxiv.org/abs/{aid}\n\n"+strip(get(f"https://arxiv.org/abs/{aid}"))
        except Exception as e: txt=""
    if txt:
        open(out,"w").write(txt); print(f"OK {out} {len(txt)} chars")
    else:
        print(f"FAIL {u}")

import sys, urllib.parse, urllib.request, re, html
def search(q, n=12):
    url = "https://lite.duckduckgo.com/lite/?q=" + urllib.parse.quote(q)
    req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"})
    with urllib.request.urlopen(req, timeout=30) as r:
        t = r.read().decode("utf-8","replace")
    # result links
    links = re.findall(r'<a[^>]+class="result-link"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', t, re.S)
    snips  = re.findall(r'class="result-snippet">(.*?)</td>', t, re.S)
    out=[]
    for i,(u,title) in enumerate(links[:n]):
        u = html.unescape(u)
        m = re.search(r'uddg=([^&]+)', u)
        if m: u = urllib.parse.unquote(m.group(1))
        title = html.unescape(re.sub(r'<[^>]+>','',title)).strip()
        s = html.unescape(re.sub(r'<[^>]+>','',snips[i])).strip() if i < len(snips) else ""
        out.append((u,title,s))
    return out
for q in sys.argv[1:]:
    print("="*100); print("QUERY:", q)
    try:
        for u,t,s in search(q):
            print(f"- {t}\n  {u}\n  {s[:300]}")
    except Exception as e:
        print("ERR", e)

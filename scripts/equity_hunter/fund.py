#!/usr/bin/env python3
"""SEC XBRL frames -> quarterly revenue per ticker, point-in-time (assumed public at period end + 75 days, conservative)."""
import json, urllib.request, urllib.error, time, pathlib, gzip
from datetime import datetime, timezone

try:
    from scripts.authorization_extensions import sec_user_agent
except ModuleNotFoundError as error:
    if error.name not in ('scripts', 'scripts.authorization_extensions'):
        raise
    from authorization_extensions import sec_user_agent
OUT = pathlib.Path.home() / "data/equity-hunter"
AUTHORIZATION = pathlib.Path.home() / "authorizations/news-radar-sec-fed-telegram-v1.json"
def get(u, *, authorization=AUTHORIZATION, clock=lambda: datetime.now(timezone.utc),
        send=urllib.request.urlopen, pause=time.sleep, evidence=OUT / "fund-authorization.json"):
    for _ in range(4):
        pause(0.3)
        # Outside the transport retry handler: an invalid/expired record stops immediately.
        ua, metadata = sec_user_agent(authorization, clock())
        if evidence is not None:
            temporary = evidence.with_suffix('.tmp')
            temporary.write_text(json.dumps(metadata))
            temporary.replace(evidence)
        try:
            r = send(urllib.request.Request(u, headers={"User-Agent": ua, "Accept-Encoding": "gzip"}), timeout=90)
            b = r.read(); return json.loads(gzip.decompress(b) if r.headers.get("Content-Encoding") == "gzip" else b)
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
            pause(10)
        except Exception: pause(10)
    return None
def main():
    tick = get("https://www.sec.gov/files/company_tickers.json")
    cik2t = {}
    for v in tick.values(): cik2t.setdefault(int(v["cik_str"]), v["ticker"].upper())
    CONCEPTS = ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"]
    rev = {}; n = 0
    for y in range(2019, 2027):
        for qn in range(1, 5):
            per = f"CY{y}Q{qn}"
            if (y, qn) > (2026, 2): continue
            for c in CONCEPTS:
                d = get(f"https://data.sec.gov/api/xbrl/frames/us-gaap/{c}/USD/{per}.json"); n += 1
                if not d: continue
                for it in d.get("data", []):
                    t = cik2t.get(int(it["cik"]))
                    if t: rev.setdefault(t, {}).setdefault(c, {})[per] = [it["end"].replace("-", ""), it["val"]]
            print(per, "companies", len(rev), "requests", n, flush=True)
    (OUT / "revenue_frames.json").write_text(json.dumps(rev)); print("DONE", len(rev), "tickers", n, "requests")


if __name__ == "__main__":
    main()

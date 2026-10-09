"""
Russell 2000 constituent list for method1's universe.

Unlike the S&P 500 (a clean Wikipedia table), there's no free bulk
ticker-list source for the Russell 2000 -- FTSE Russell doesn't publish
one, and iShares' own IWM holdings download is behind bot-protection
that blocks non-browser requests. The workaround: IWM (iShares Russell
2000 ETF) is a registered fund and FILES its exact holdings with the SEC
quarterly via Form N-PORT-P, same EDGAR system used elsewhere in this
project (see method3/edgar_data.py). That filing gives CUSIPs, not
ticker symbols, so a second step maps CUSIP -> ticker via OpenFIGI's
free public mapping API (Bloomberg's open security-identifier service;
no key required, just rate-limited).

Two free-but-imperfect data sources chained together, cached for 90 days
(the Russell indices only reconstitute once/year, in June, so this is
far more tolerant than the S&P 500 list's 7-day cache) -- some tickers
may be missed if OpenFIGI has no mapping for a given CUSIP, same
"absent, not wrong" tolerance already applied to missing yfinance/SEC
data elsewhere in this project.
"""

import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

HEADERS = {"User-Agent": "method1-personal-research-project research-script@example.com",
           "Accept-Encoding": "gzip, deflate"}

ISHARES_TRUST_CIK = 1100663
IWM_SERIES_ID = "S000004344"

CACHE_FILE = Path(__file__).with_name("russell2000_constituents.json")
CACHE_MAX_AGE_DAYS = 90

OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"
OPENFIGI_BATCH_SIZE = 10
OPENFIGI_DELAY_SECONDS = 2.5  # free tier: 25 requests/minute


def _utcnow():
    return datetime.now(timezone.utc)


def _find_iwm_nport_url(max_scan: int = 400) -> str | None:
    """Scans iShares Trust's recent N-PORT-P filings (newest first) for
    the one whose seriesId matches IWM, stopping at the first match.
    Direct full-text search for "iShares Russell 2000 ETF" is unreliable
    -- other iShares funds that simply HOLD IWM as a position (e.g. the
    LifePath target-date funds) also contain that exact text, so matching
    on the fund's own seriesId (from SEC's company_tickers_mf.json,
    hardcoded above since it almost never changes) is the only reliable
    discriminator.
    """
    resp = requests.get(f"https://data.sec.gov/submissions/CIK{ISHARES_TRUST_CIK:010d}.json",
                         headers=HEADERS, timeout=20)
    resp.raise_for_status()
    recent = resp.json()["filings"]["recent"]
    accessions = [a for f, a in zip(recent["form"], recent["accessionNumber"]) if f == "NPORT-P"]

    for acc in accessions[:max_scan]:
        acc_nodash = acc.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{ISHARES_TRUST_CIK}/{acc_nodash}/primary_doc.xml"
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if IWM_SERIES_ID in r.text:
                return url
        except Exception:
            continue
        time.sleep(0.1)
    return None


def _parse_cusips(nport_xml: str) -> list[str]:
    return re.findall(r"<cusip>([^<]*)</cusip>", nport_xml)


def _map_cusips_to_tickers(cusips: list[str]) -> list[str]:
    tickers = []
    for i in range(0, len(cusips), OPENFIGI_BATCH_SIZE):
        batch = cusips[i:i + OPENFIGI_BATCH_SIZE]
        jobs = [{"idType": "ID_CUSIP", "idValue": c, "exchCode": "US"} for c in batch]
        try:
            resp = requests.post(OPENFIGI_URL, json=jobs, timeout=20)
            resp.raise_for_status()
            results = resp.json()
        except Exception:
            time.sleep(OPENFIGI_DELAY_SECONDS)
            continue
        for r in results:
            data = r.get("data") or []
            if data and data[0].get("ticker"):
                tickers.append(data[0]["ticker"])
        if (i // OPENFIGI_BATCH_SIZE + 1) % 20 == 0:
            print(f"  ...mapped {i + len(batch)}/{len(cusips)} CUSIPs, {len(tickers)} tickers so far")
        time.sleep(OPENFIGI_DELAY_SECONDS)
    return tickers


def get_russell2000_tickers(force_refresh: bool = False) -> list[str]:
    if not force_refresh and CACHE_FILE.exists():
        try:
            cached = json.loads(CACHE_FILE.read_text())
            cached_at = datetime.fromisoformat(cached["cached_at"])
            if _utcnow() - cached_at < timedelta(days=CACHE_MAX_AGE_DAYS):
                return cached["tickers"]
        except Exception:
            pass

    url = _find_iwm_nport_url()
    if url is None:
        if CACHE_FILE.exists():
            print("  Could not find a fresh IWM N-PORT filing; using stale cache.")
            return json.loads(CACHE_FILE.read_text())["tickers"]
        print("  Could not find IWM N-PORT filing and no cache exists; returning empty list.")
        return []

    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    cusips = _parse_cusips(resp.text)
    print(f"  Found {len(cusips)} holdings in IWM's N-PORT filing; mapping CUSIPs to tickers via OpenFIGI...")
    tickers = sorted(set(_map_cusips_to_tickers(cusips)))
    print(f"  Resolved {len(tickers)}/{len(cusips)} CUSIPs to US tickers")

    CACHE_FILE.write_text(json.dumps({"cached_at": _utcnow().isoformat(), "tickers": tickers,
                                       "source_url": url}, indent=2))
    return tickers


if __name__ == "__main__":
    tickers = get_russell2000_tickers(force_refresh=True)
    print(f"\n{len(tickers)} Russell 2000 tickers cached.")

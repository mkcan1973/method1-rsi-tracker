"""
S&P 500 constituent list, ported from the get_universe() pattern in
C:\\mark\\scripts\\csp\\old\\puts_tracker.py: scrape Wikipedia (cached for a
week), fall back to a static list if that fails.

This is the individual-stock complement to universe.py's ETF/index list --
added specifically to fill the gaps between correlated sector-ETF signals,
since idiosyncratic single-stock moves are less correlated with each other
than SPY/XLF/XLY all getting oversold on the same market-wide selloff.
"""

import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP500_CACHE_FILE = Path(__file__).with_name("sp500_constituents.json")
SP500_CACHE_MAX_AGE_DAYS = 7

SP500_FALLBACK_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "MA", "DIS", "BAC", "XOM", "PFE", "KO",
    "PEP", "CSCO", "ABBV", "CVX", "WMT", "MRK", "TMO", "COST", "MCD", "NKE",
    "ADBE", "CRM", "LIN", "ABT", "ORCL", "ACN", "DHR", "TXN", "NEE", "PM",
    "RTX", "HON", "UNP", "LOW", "QCOM", "IBM", "INTU", "CAT", "AMGN", "SPGI",
]


def _utcnow():
    return datetime.now(timezone.utc)


def get_sp500_tickers(force_refresh: bool = False) -> list[str]:
    if not force_refresh and SP500_CACHE_FILE.exists():
        try:
            cached = json.loads(SP500_CACHE_FILE.read_text())
            cached_at = datetime.fromisoformat(cached["cached_at"])
            if _utcnow() - cached_at < timedelta(days=SP500_CACHE_MAX_AGE_DAYS):
                return cached["tickers"]
        except Exception:
            pass

    tickers = None
    try:
        import requests
        # Wikipedia 403s unidentified clients - pd.read_html's own request
        # doesn't set a User-Agent, so fetch with requests first.
        resp = requests.get(SP500_WIKI_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        symbols = tables[0]["Symbol"].tolist()
        tickers = [str(s).strip().replace(".", "-") for s in symbols if str(s).strip()]
    except Exception as e:
        print(f"  Could not fetch S&P 500 list from Wikipedia ({e}); using static fallback list.")

    if not tickers:
        tickers = SP500_FALLBACK_TICKERS
        print(f"  Using static fallback universe of {len(tickers)} tickers (may be stale).")
    else:
        SP500_CACHE_FILE.write_text(json.dumps(
            {"cached_at": _utcnow().isoformat(), "tickers": tickers}, indent=2))

    return tickers

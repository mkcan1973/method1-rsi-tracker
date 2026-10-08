"""
Trailing EPS cache, used to approximate "P/E at entry" = entry_price /
current trailing EPS.

This is necessarily an approximation for anything but very recent trades:
EPS updates every quarter when a company reports earnings, so a position
entered a year ago was priced against DIFFERENT earnings than today's
trailing EPS reflects. There's no free historical EPS/P/E time series to
compute this exactly -- yfinance only exposes the CURRENT trailing EPS.
This is the honest approximation, not a precise historical figure, and
is most reliable for trades entered in roughly the last two quarters.

Cached with a 7-day TTL (same convention as stock_universe.py's S&P 500
list) since EPS only changes on quarterly earnings reports -- no need to
refetch 500+ symbols daily for data that moves four times a year.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yfinance as yf

CACHE_FILE = Path(__file__).with_name("trailing_eps_cache.json")
CACHE_MAX_AGE_DAYS = 7


def _utcnow():
    return datetime.now(timezone.utc)


def get_trailing_eps(symbols: list[str], force_refresh: bool = False) -> dict:
    """Return {symbol: trailing_eps}. Symbols with no reported EPS (e.g.
    yfinance has no data for them) are simply absent from the dict --
    callers should treat a missing symbol the same as an unprofitable one
    (no meaningful P/E), not silently default it to zero.
    """
    if not force_refresh and CACHE_FILE.exists():
        try:
            cached = json.loads(CACHE_FILE.read_text())
            cached_at = datetime.fromisoformat(cached["cached_at"])
            if _utcnow() - cached_at < timedelta(days=CACHE_MAX_AGE_DAYS):
                return cached["eps"]
        except Exception:
            pass

    eps = {}
    for symbol in symbols:
        try:
            info = yf.Ticker(symbol).info
            val = info.get("trailingEps")
            if val is not None:
                eps[symbol] = val
        except Exception:
            continue

    CACHE_FILE.write_text(json.dumps({"cached_at": _utcnow().isoformat(), "eps": eps}, indent=2))
    return eps

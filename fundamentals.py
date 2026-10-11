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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yfinance as yf

CACHE_FILE = Path(__file__).with_name("trailing_eps_cache.json")
CACHE_MAX_AGE_DAYS = 7
# yfinance's .info has no built-in request timeout in the versions this
# project has hit -- a single hung connection (no response, no error,
# just silence) blocks forever. Found the hard way: a dashboard rebuild
# sat for 10+ hours on what should have been a ~30-second incremental
# fetch, with zero CPU usage most of that time, stuck on presumably one
# symbol. Run each fetch in its own thread and give up on it past this
# timeout rather than trust the network to always behave.
PER_SYMBOL_TIMEOUT_SECONDS = 15
# Save progress every N symbols, not just once at the very end -- so a
# hang, crash, or Ctrl-C (or the next timeout-related kill) doesn't throw
# away everything fetched so far in a long run.
SAVE_EVERY = 50


def _utcnow():
    return datetime.now(timezone.utc)


def get_trailing_eps(symbols: list[str], force_refresh: bool = False) -> dict:
    """Return {symbol: trailing_eps}. Symbols with no reported EPS (e.g.
    yfinance has no data for them) are simply absent from the dict --
    callers should treat a missing symbol the same as an unprofitable one
    (no meaningful P/E), not silently default it to zero.

    Incremental like the price-history fetchers: the cache tracks which
    symbols were ever ATTEMPTED (separately from which ones actually had
    an EPS value), so a symbol missing from the EPS dict because
    yfinance has nothing for it is never re-fetched every call, but a
    symbol that's simply NEW to this request (e.g. Russell 2000 added
    to the universe after the cache was last built) still gets fetched
    even if the cache file itself is only a day old. Before this, a
    "freshness" check that only looked at the cache's timestamp -- not
    whether it actually covered the requested symbols -- silently
    returned empty results for any symbol added to the universe after
    the cache was built, for up to the full 7-day TTL.
    """
    cached_eps, attempted = {}, set()
    if not force_refresh and CACHE_FILE.exists():
        try:
            cached = json.loads(CACHE_FILE.read_text())
            cached_at = datetime.fromisoformat(cached["cached_at"])
            if _utcnow() - cached_at < timedelta(days=CACHE_MAX_AGE_DAYS):
                cached_eps = cached["eps"]
                # Back-compat with caches written before "attempted" existed:
                # treat every symbol that HAD a value as attempted, but with
                # no way to know which no-EPS symbols were already tried --
                # those get harmlessly re-attempted once.
                attempted = set(cached.get("attempted", cached["eps"].keys()))
        except Exception:
            pass

    missing = [s for s in symbols if force_refresh or s not in attempted]

    def _save():
        CACHE_FILE.write_text(json.dumps(
            {"cached_at": _utcnow().isoformat(), "eps": cached_eps, "attempted": sorted(attempted)}, indent=2))

    def _fetch_one(symbol):
        return yf.Ticker(symbol).info.get("trailingEps")

    if missing:
        print(f"  Fetching trailing EPS for {len(missing)} symbols (timeout {PER_SYMBOL_TIMEOUT_SECONDS}s each)...")
    # A pool bigger than 1 matters here: .result(timeout=...) only makes
    # the CALLER stop waiting -- it can't actually kill a thread that's
    # truly stuck in a blocking socket read, so that thread (and its one
    # worker slot) is gone for good. With max_workers=1, one hung symbol
    # would permanently starve every symbol after it of a worker to run
    # on, turning "skip this one" into "hang forever" all over again just
    # spread across many individually-timed-out calls. Multiple workers
    # mean a stuck call only costs ONE slot, not the whole pool.
    with ThreadPoolExecutor(max_workers=10) as executor:
        for i, symbol in enumerate(missing):
            try:
                val = executor.submit(_fetch_one, symbol).result(timeout=PER_SYMBOL_TIMEOUT_SECONDS)
                if val is not None:
                    cached_eps[symbol] = val
            except FutureTimeoutError:
                print(f"  WARNING: {symbol} timed out after {PER_SYMBOL_TIMEOUT_SECONDS}s -- skipping.")
            except Exception:
                pass
            attempted.add(symbol)
            if (i + 1) % SAVE_EVERY == 0:
                _save()
                print(f"  ...{i + 1}/{len(missing)} symbols attempted, {len(cached_eps)} with EPS so far")

    if missing:
        _save()
    return cached_eps

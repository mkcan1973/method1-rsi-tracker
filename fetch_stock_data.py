"""
Pull daily OHLCV history for the S&P 500 + Russell 2000 individual-stock
universe from Yahoo Finance into the same method1.sqlite3 bars table
fetch_data.py uses for ETFs. Batches downloads (yfinance supports many
tickers per call) since doing thousands of one-at-a-time calls would be
painfully slow; still incremental and idempotent like fetch_data.py --
a brand-new ticker (no prior rows) automatically gets a full period="max"
history pull, an already-tracked one only fetches what's missing since
its last stored date.

auto_adjust=True is mandatory here, more so than for the ETF fetch:
individual stocks split far more often, and an unadjusted split shows up
as a fake multi-day price "crash" that corrupts returns/RSI/backtests for
that symbol around the split date.
"""

import sys
from collections import defaultdict

import pandas as pd
import yfinance as yf

import data_db
from stock_universe import get_sp500_tickers
from russell2000 import get_russell2000_tickers

SOURCE = "yfinance"
BATCH_SIZE = 50  # keep each yf.download call small enough to be reliable


RECENT_SPLIT_LOOKBACK_DAYS = 5
# A full period="max" pull for a brand-new ticker (every Russell 2000
# symbol, on first fetch) can contain an unadjusted split ANYWHERE in its
# history, not just in the last few days -- the trailing-window check
# below only catches one still propagating during an incremental update.
# Found via a 2273-symbol backtest producing a nonsense 126% CAGR: 35
# small/micro-cap symbols had single-day "moves" up to 16,079% (AMPY) or
# infinite (DEC, a div-by-near-zero print) buried in their full history.
# A real stock doesn't 160x in a day without a trading halt; this
# threshold is set far above even an extreme genuine small-cap move (a
# short squeeze or binary biotech readout rarely clears 500% close-to-
# close) specifically so it only catches near-certain data corruption,
# not real (if dramatic) volatility.
EXTREME_MOVE_THRESHOLD = 5.0  # 500% in one day, anywhere in history


def _drop_unadjusted_split_rows(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    """See fetch_data.py's version of this function for the original
    trailing-window-only check and its rationale (an old large move is
    far more likely a real event -- 2008 crisis, a Yahoo-never-adjusted
    ancient split -- than a propagation artifact, so don't blanket-delete
    on magnitude alone). This adds a second, much higher-bar check across
    the FULL history for genuinely extreme moves, since those are a
    reliable sign of corrupted data regardless of how old they are.

    An EXTREME_MOVE_THRESHOLD hit drops the ticker's ENTIRE history, not
    just the one transition row: if yfinance's split-adjustment is wrong
    at that point, every price on one side of it is mis-scaled, not just
    the transition day itself, and there's no reliable way to tell which
    side (before/after) is the correct one from here. 35/2273 symbols hit
    this on the Russell 2000's first full fetch -- excluding them
    entirely costs little universe coverage against the alternative of
    silently backtesting on mis-scaled prices.
    """
    if len(df) < 2:
        return df
    day_ret = df["Close"].pct_change()
    # abs() of +-inf (a divide-by-near-zero print) is still inf, which
    # compares True against any finite threshold -- no separate inf case
    # needed, the magnitude check below already catches it.
    extreme = (day_ret.abs() > EXTREME_MOVE_THRESHOLD).fillna(False)
    if extreme.any():
        bad_dates = [ts.date() for ts in df.index[extreme]]
        threshold_pct = EXTREME_MOVE_THRESHOLD * 100
        print(f"  WARNING: {ticker} shows an implausible >{threshold_pct:.0f}% single-day move "
              f"on {bad_dates} (likely corrupted/unadjusted split data) "
              f"-- dropping this ticker's entire history.")
        return df.iloc[0:0]

    cutoff = df.index[-1] - pd.Timedelta(days=RECENT_SPLIT_LOOKBACK_DAYS)
    bad = ((day_ret.abs() > 0.5) & (df.index > cutoff)).fillna(False)
    if bad.any():
        for ts in df.index[bad]:
            print(f"  WARNING: {ticker} {ts.date()} shows a >50% single-day move "
                  f"in the last {RECENT_SPLIT_LOOKBACK_DAYS} days (likely an unadjusted "
                  f"split still propagating) -- dropping this row.")
        df = df[~bad]
    return df


def _rows_for_upsert(ticker: str, df: pd.DataFrame):
    df = _drop_unadjusted_split_rows(ticker, df)
    for ts, r in df.dropna().iterrows():
        yield (
            ticker, ts.strftime("%Y-%m-%d"),
            float(r["Open"]), float(r["High"]), float(r["Low"]),
            float(r["Close"]), float(r["Volume"]), SOURCE,
        )


def _download_batch(tickers: list[str], start: str | None) -> dict:
    kwargs = {"start": start} if start is not None else {"period": "max"}
    data = yf.download(tickers, auto_adjust=True, progress=False, group_by="ticker", **kwargs)
    if data.empty:
        return {}
    if len(tickers) == 1:
        # Single-ticker batches are usually flat-columned, but not always
        # (observed: an incremental start= fetch kept the ticker level
        # while a full-history period="max" fetch didn't) -- handle both
        # rather than assume one, which crashed on a KeyError.
        if isinstance(data.columns, pd.MultiIndex):
            return {tickers[0]: data[tickers[0]]} if tickers[0] in data.columns.get_level_values(0) else {}
        return {tickers[0]: data}
    return {t: data[t] for t in tickers if t in data.columns.get_level_values(0)}


def main():
    conn = data_db.connect()
    tickers = list(dict.fromkeys(get_sp500_tickers() + get_russell2000_tickers()))

    # Group by identical start date so symbols caught up to the same day
    # can be fetched together in one batch call.
    groups = defaultdict(list)
    for ticker in tickers:
        last = data_db.last_date(conn, ticker)
        start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if last is not None else None
        groups[start].append(ticker)

    total_new_rows = 0
    for start, group_tickers in groups.items():
        label = start or "full history"
        for i in range(0, len(group_tickers), BATCH_SIZE):
            batch = group_tickers[i:i + BATCH_SIZE]
            print(f"Fetching {len(batch)} tickers from {label} "
                  f"({i + 1}-{i + len(batch)} of {len(group_tickers)} in this group)...")
            try:
                results = _download_batch(batch, start)
            except Exception as e:
                print(f"  ERROR on batch: {e}", file=sys.stderr)
                continue

            for ticker in batch:
                df = results.get(ticker)
                if df is None or df.empty:
                    continue
                rows = list(_rows_for_upsert(ticker, df))
                if rows:
                    data_db.upsert_bars(conn, rows)
                    total_new_rows += len(rows)

    print(f"\nTotal rows upserted: {total_new_rows}")
    conn.close()


if __name__ == "__main__":
    main()

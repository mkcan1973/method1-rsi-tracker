"""
Pull daily OHLCV history for the S&P 500 individual-stock universe from
Yahoo Finance into the same method1.sqlite3 bars table fetch_data.py uses
for ETFs. Batches downloads (yfinance supports many tickers per call) since
doing 500 one-at-a-time calls would be painfully slow; still incremental
and idempotent like fetch_data.py.

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

SOURCE = "yfinance"
BATCH_SIZE = 50  # keep each yf.download call small enough to be reliable


RECENT_SPLIT_LOOKBACK_DAYS = 5


def _drop_unadjusted_split_rows(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    """See fetch_data.py's version of this function -- same fix, same
    reason, same restriction to only the trailing few days: an old large
    move is far more likely a real event (2008 crisis, a Yahoo-never-
    adjusted ancient split) than a propagation artifact, and blanket-
    deleting on magnitude alone destroys real data.
    """
    if len(df) < 2:
        return df
    cutoff = df.index[-1] - pd.Timedelta(days=RECENT_SPLIT_LOOKBACK_DAYS)
    day_ret = df["Close"].pct_change().abs()
    bad = (day_ret > 0.5) & (df.index > cutoff)
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
    tickers = get_sp500_tickers()

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

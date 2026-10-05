"""
Pull daily OHLCV history for the method1 universe from Yahoo Finance and
upsert it into method1.sqlite3. Safe to rerun: only fetches dates missing
since each symbol's last stored bar (or full history on first run).

auto_adjust=True is required, not optional: with it off, yfinance returns
raw unadjusted prices, and any stock split shows up as a fake multi-day
price "crash" (e.g. a 5:1 split looks like an 80% one-day loss) that
corrupts every return/RSI/backtest calculation touching that date. ETFs
rarely split but individual stocks (fetch_stock_data.py) split often
enough that this matters a lot there.
"""

import sys

import pandas as pd
import yfinance as yf

import data_db
from universe import UNIVERSE

SOURCE = "yfinance"


def fetch_symbol(ticker: str, start: str | None) -> pd.DataFrame:
    kwargs = {"start": start} if start is not None else {"period": "max"}
    df = yf.download(
        ticker,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
        **kwargs,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.lower)
    return _drop_unadjusted_split_rows(ticker, df[["open", "high", "low", "close", "volume"]])


RECENT_SPLIT_LOOKBACK_DAYS = 5


def _drop_unadjusted_split_rows(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    """A split that happened in the last few days sometimes shows up in
    Yahoo's feed before the back-adjustment of older history has propagated
    -- producing a fake 50%+ single-day "crash"/"spike" that would corrupt
    returns/RSI until it self-corrects. Only check the TRAILING few days,
    never older history: a >50% move further back is far more likely a real
    event (2008 financial-crisis bank stocks, a VIX spike, an old split
    Yahoo never adjusted) than a propagation artifact, and deleting those
    would destroy real data, not fix bad data.
    """
    if len(df) < 2:
        return df
    cutoff = df.index[-1] - pd.Timedelta(days=RECENT_SPLIT_LOOKBACK_DAYS)
    day_ret = df["close"].pct_change().abs()
    bad = (day_ret > 0.5) & (df.index > cutoff)
    if bad.any():
        for ts in df.index[bad]:
            print(f"  WARNING: {ticker} {ts.date()} shows a >50% single-day move "
                  f"in the last {RECENT_SPLIT_LOOKBACK_DAYS} days (likely an unadjusted "
                  f"split still propagating) -- dropping this row.")
        df = df[~bad]
    return df


def rows_for_upsert(ticker: str, df: pd.DataFrame):
    for ts, r in df.iterrows():
        yield (
            ticker,
            ts.strftime("%Y-%m-%d"),
            float(r["open"]),
            float(r["high"]),
            float(r["low"]),
            float(r["close"]),
            float(r["volume"]),
            SOURCE,
        )


def main():
    conn = data_db.connect()
    for ticker, spec in UNIVERSE.items():
        last = data_db.last_date(conn, ticker)
        start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if last is not None else None

        print(f"{ticker:8s} ({spec.name}): fetching from {start or 'full history'}...", end=" ")
        try:
            df = fetch_symbol(ticker, start)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            continue

        if df.empty:
            print("no new bars")
            continue

        data_db.upsert_bars(conn, rows_for_upsert(ticker, df))
        print(f"{len(df)} bars ({df.index.min().date()} to {df.index.max().date()})")

    conn.close()


if __name__ == "__main__":
    main()

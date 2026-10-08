"""
Daily signal scan for method1.

Strategy: RSI(14) mean reversion. Buy when a symbol's 14-day RSI drops to
<=20 (oversold), hold until it recovers to >=65, flat otherwise. Validated
via backtest.py's walk-forward harness across the full universe (S&P 500 +
core ETFs/index): positive out-of-sample Sharpe on effectively all symbols
tested, not just a cherry-picked few. These specific thresholds were then
chosen over the original textbook 30/55 defaults via threshold_grid.py's
grid search plus threshold_train_test.py's train/test replication check --
see indicators.py's module docstring for that validation trail.

This replaced an earlier narrow streak-based signal (still in event_study.py)
because it fires far more often -- something in this universe is in a
trade on ~100% of weeks, vs. a handful of times a year for the streak rule.

Signal/alert only: this prints what's actionable today. It places no orders.
"""

from pathlib import Path

import pandas as pd

import data_db
import fetch_data
import fetch_stock_data
import features
import indicators
from universe import tradeable_tickers
from stock_universe import get_sp500_tickers

RSI_BUY = 20
RSI_EXIT = 65

SUMMARY_CSV = Path(__file__).with_name("scan_results.csv")


def scan_symbol(conn, symbol: str) -> dict | None:
    bars = data_db.load_bars(conn, symbol)
    if bars is None or len(bars) < 60:
        return None

    feats = features.build_features(bars)
    position = indicators.rsi_reversion(feats, buy_thresh=RSI_BUY, exit_thresh=RSI_EXIT)
    if len(position) < 2:
        return None

    pos_today, pos_yesterday = position.iloc[-1], position.iloc[-2]
    if pos_today == 1 and pos_yesterday == 0:
        status = "NEW_ENTRY"
    elif pos_today == 1 and pos_yesterday == 1:
        status = "HOLDING"
    elif pos_today == 0 and pos_yesterday == 1:
        status = "EXIT"
    else:
        status = "FLAT"

    return dict(
        symbol=symbol,
        as_of=feats.index[-1].date(),
        close=bars["close"].iloc[-1],
        rsi_14=feats["rsi_14"].iloc[-1],
        status=status,
    )


def main():
    print("Refreshing ETF/index data...")
    fetch_data.main()
    print("\nRefreshing S&P 500 stock data (incremental)...")
    fetch_stock_data.main()
    print()

    conn = data_db.connect()
    symbols = list(dict.fromkeys(tradeable_tickers() + get_sp500_tickers()))
    rows = [r for r in (scan_symbol(conn, s) for s in symbols) if r is not None]
    conn.close()

    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_CSV, index=False)

    as_of = df["as_of"].iloc[0] if not df.empty else "n/a"
    print(f"Rule: RSI(14) <= {RSI_BUY} -> buy, hold until RSI >= {RSI_EXIT} -> exit")
    print(f"As of: {as_of}  |  {len(df)} symbols scanned")
    print()

    counts = df["status"].value_counts()
    print(f"{counts.get('NEW_ENTRY', 0)} new entries today, {counts.get('EXIT', 0)} exits today, "
          f"{counts.get('HOLDING', 0)} positions already held, {counts.get('FLAT', 0)} flat")
    print()

    actionable = df[df["status"].isin(["NEW_ENTRY", "EXIT"])].sort_values(["status", "symbol"])
    if not actionable.empty:
        print("=== Actionable today ===")
        print(actionable.to_string(index=False))
    else:
        print("Nothing new today.")

    holding = df[df["status"] == "HOLDING"].sort_values("symbol")
    if not holding.empty:
        print()
        print(f"=== Currently held ({len(holding)}) -- see {SUMMARY_CSV.name} for the full list ===")
        print(holding.head(15).to_string(index=False))
        if len(holding) > 15:
            print(f"... and {len(holding) - 15} more")

    print()
    print("Execution convention: today's close is the signal, not a fill price --")
    print("if trading this for real, place entries/exits at tomorrow's OPEN.")
    print("Signal only -- no orders placed.")


if __name__ == "__main__":
    main()

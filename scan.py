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

Discord notifications: entries/exits are also posted to Discord (best
effort, see notify.py), filtered to P/E @ entry and EPS thresholds read
from notify_thresholds.json (see notify.get_notify_thresholds) -- not
because the underlying RSI signal uses P/E or EPS (it doesn't; every
symbol trades on the same rule), but because cagr_by_year_pe.py /
cagr_by_year_eps.py's local backtests showed a P/E<30, EPS>0 subset of
trades outperforming in 2023-2025. The filter narrows what you get
PINGED about, not what the dashboard tracks or what the strategy trades
-- the full, unfiltered signal is still in scan_results.csv and on the
dashboard with its own live P/E/EPS filter fields. To change the
threshold without touching code, edit notify_thresholds.json directly
(e.g. via GitHub's web/mobile file editor) and commit -- the next scan
run picks it up.
"""

from pathlib import Path

import pandas as pd

import data_db
import fetch_data
import fetch_stock_data
import features
import fundamentals
import indicators
import notify
from universe import tradeable_tickers
from stock_universe import get_sp500_tickers
from russell2000 import get_russell2000_tickers

RSI_BUY = 20
RSI_EXIT = 65

SUMMARY_CSV = Path(__file__).with_name("scan_results.csv")


def scan_symbol(conn, symbol: str, eps_by_symbol: dict) -> dict | None:
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

    close = bars["close"].iloc[-1]
    eps = eps_by_symbol.get(symbol)
    pe = round(close / eps, 1) if eps and eps > 0 else None

    return dict(
        symbol=symbol,
        as_of=feats.index[-1].date(),
        close=close,
        rsi_14=feats["rsi_14"].iloc[-1],
        status=status,
        pe=pe,
        eps=round(eps, 2) if eps is not None else None,
    )


def main():
    print("Refreshing ETF/index data...")
    fetch_data.main()
    print("\nRefreshing S&P 500 stock data (incremental)...")
    fetch_stock_data.main()
    print()

    symbols = list(dict.fromkeys(tradeable_tickers() + get_sp500_tickers() + get_russell2000_tickers()))
    print("Fetching trailing EPS (for the Discord P/E+EPS notification filter)...")
    eps_by_symbol = fundamentals.get_trailing_eps(symbols)

    conn = data_db.connect()
    rows = [r for r in (scan_symbol(conn, s, eps_by_symbol) for s in symbols) if r is not None]
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

    thresholds = notify.get_notify_thresholds()
    pe_max, eps_min = thresholds["pe_max"], thresholds["eps_min"]
    notify_rows = actionable[
        (actionable["pe"].notna()) & (actionable["pe"] < pe_max)
        & (actionable["eps"].notna()) & (actionable["eps"] > eps_min)
    ]
    if not notify_rows.empty:
        lines = [
            f"{'+' if r.status == 'NEW_ENTRY' else '-'} "
            f"{'BUY ' if r.status == 'NEW_ENTRY' else 'SELL'}  {r.symbol:6s} ${r.close:<10.2f} "
            f"(RSI={r.rsi_14:5.1f}  P/E={r.pe:5.1f}  EPS={r.eps:6.2f})"
            for r in notify_rows.itertuples()
        ]
        message = (
            f"**method1 RSI signal -- {as_of}** (P/E < {pe_max:g}, EPS > {eps_min:g})\n"
            "```diff\n" + "\n".join(lines) + "\n```\n"
            "<https://mkcan1973.github.io/method1-rsi-tracker/>"
        )
        notify.notify_discord(message)
        print(f"\nSent Discord notification: {len(notify_rows)} signal(s) passed the P/E+EPS filter "
              f"(P/E < {pe_max:g}, EPS > {eps_min:g}).")

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

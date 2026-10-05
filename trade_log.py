"""
Reconstructs the RSI-reversion trade history (entries/exits, open vs
closed, per-trade ROI) over a trailing window, for the web summary.

Uses the exact same lagged position logic validated in backtest.py and
used live in scan.py -- this is a reporting view of that same signal, not
a separate calculation that could drift out of sync with it.
"""

import pandas as pd

import data_db
import features
import indicators
from universe import tradeable_tickers
from stock_universe import get_sp500_tickers


def build_trade_log(months: int = 12) -> pd.DataFrame:
    conn = data_db.connect()
    symbols = list(dict.fromkeys(tradeable_tickers() + get_sp500_tickers()))
    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(months=months)

    trades = []
    for symbol in symbols:
        bars = data_db.load_bars(conn, symbol)
        # 120 trading days comfortably covers feature warmup (the longest
        # rolling window is the 50-day MA) -- this isn't backtest.py's
        # walk-forward harness, which needs much deeper history for
        # multiple yearly folds; this just needs enough for RSI/MACD to be
        # valid before the 12-month trade-log window begins.
        if bars is None or len(bars) < 120:
            continue
        feats = features.build_features(bars)
        position = indicators.rsi_reversion(feats).shift(1).fillna(0)
        close = bars["close"]

        entry_date = None
        for i in range(1, len(position)):
            prev, curr = position.iloc[i - 1], position.iloc[i]
            date = position.index[i]
            if prev == 0 and curr == 1:
                entry_date = date
            elif prev == 1 and curr == 0 and entry_date is not None:
                if date >= cutoff:
                    entry_price, exit_price = close.loc[entry_date], close.loc[date]
                    trades.append(dict(
                        symbol=symbol, entry_date=entry_date, exit_date=date,
                        entry_price=entry_price, exit_price=exit_price,
                        roi=exit_price / entry_price - 1, status="closed",
                        hold_days=(date - entry_date).days,
                    ))
                entry_date = None

        # A still-open trade at the end of history is always relevant,
        # regardless of when it entered.
        if entry_date is not None and position.iloc[-1] == 1:
            entry_price, current_price = close.loc[entry_date], close.iloc[-1]
            trades.append(dict(
                symbol=symbol, entry_date=entry_date, exit_date=None,
                entry_price=entry_price, exit_price=current_price,
                roi=current_price / entry_price - 1, status="open",
                hold_days=(position.index[-1] - entry_date).days,
            ))

    conn.close()
    df = pd.DataFrame(trades)
    if not df.empty:
        df = df.sort_values(["status", "entry_date"], ascending=[True, False])
    return df


def summarize(df: pd.DataFrame) -> dict:
    closed = df[df["status"] == "closed"]
    open_ = df[df["status"] == "open"]
    return dict(
        n_closed=len(closed),
        n_open=len(open_),
        win_rate=(closed["roi"] > 0).mean() if len(closed) else None,
        avg_roi_closed=closed["roi"].mean() if len(closed) else None,
        avg_roi_open=open_["roi"].mean() if len(open_) else None,
        avg_hold_days_closed=closed["hold_days"].mean() if len(closed) else None,
        total_normalized_pnl=closed["roi"].sum() * 1000 if len(closed) else 0.0,  # $1000/trade, for comparability
    )


if __name__ == "__main__":
    log = build_trade_log(months=12)
    log.to_csv("trade_log_12mo.csv", index=False)
    print(f"{len(log)} trades in the last 12 months")
    print(summarize(log))

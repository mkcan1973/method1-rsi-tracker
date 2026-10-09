"""
Reconstructs the RSI-reversion trade history (entries/exits, open vs
closed, per-trade ROI) over a trailing window, for the web summary.

Uses the exact same signal logic validated in backtest.py and used live
in scan.py -- this is a reporting view of that same signal, not a
separate calculation that could drift out of sync with it.

Entry/exit price convention: the day a signal fires (RSI crosses the
threshold at that day's close) is a detection, not a fill -- you can't
act on a close until the next session opens. Both entry and exit price
are therefore the NEXT trading day's open after the signal-crossing
close, applied symmetrically (an earlier version used the signal day's
own close as a same-day exit fill, which was inconsistent: realistic
about entry timing but not about exit timing, since both signals are
only knowable after that day's close).
"""

import pandas as pd

import data_db
import features
import indicators
from universe import tradeable_tickers
from stock_universe import get_sp500_tickers
from russell2000 import get_russell2000_tickers


# Fixed anchor rather than "N months ago from today": the web dashboard's
# year-toggle buttons need full-year data available at all times for
# each year shown, not just a trailing window that would eventually slide
# past them. Pulled forward from 2022-10-01 to 2025-01-01 (losing the
# 2023/2024 buttons) specifically because adding Russell 2000 made the
# per-day data footprint ~4.5x bigger -- the old anchor's DB no longer
# fits GitHub's 100MB hard per-file push limit (was 226MB pruned to
# 1588 days). This only applies to the deployed tracker's copy of this
# constant; the local research copy keeps full history and has no such
# constraint. prune.py's KEEP_DAYS derives from this constant
# automatically -- revisit both the anchor and the size math before
# adding a 2027 button.
DEFAULT_START_DATE = "2025-06-01"


def build_trade_log(start_date: str = DEFAULT_START_DATE) -> pd.DataFrame:
    return build_trade_log_with_mtm(start_date=start_date)[0]


def build_trade_log_with_mtm(start_date: str = DEFAULT_START_DATE) -> tuple[pd.DataFrame, dict]:
    """Same trade reconstruction as build_trade_log(), plus (in the same
    pass over the universe, to avoid doubling the ~2-3min runtime) each
    currently-open position's daily mark-to-market P&L series from its own
    entry date through today -- needed to plot unrealized P&L on a real
    calendar axis rather than just today's single snapshot value.

    Returns (trades_df, open_mtm_series) where open_mtm_series maps
    symbol -> pd.Series of (price/entry_price - 1)*1000, indexed by date,
    for every currently-open position.
    """
    conn = data_db.connect()
    symbols = list(dict.fromkeys(tradeable_tickers() + get_sp500_tickers() + get_russell2000_tickers()))
    cutoff = pd.Timestamp(start_date)

    trades = []
    open_mtm_series = {}
    for symbol in symbols:
        bars = data_db.load_bars(conn, symbol)
        # 120 trading days comfortably covers feature warmup (the longest
        # rolling window is the 50-day MA) -- this isn't backtest.py's
        # walk-forward harness, which needs much deeper history for
        # multiple yearly folds; this just needs enough for RSI/MACD to be
        # valid before the trade-log window begins.
        if bars is None or len(bars) < 120:
            continue
        feats = features.build_features(bars)
        # Unshifted: position flips the day its *close* crosses the
        # threshold -- that close is the signal, not a tradeable price.
        # indicators.rsi_reversion() itself has no lookahead (RSI at day t
        # uses only bars <= t), so this is safe to read directly.
        position = indicators.rsi_reversion(feats).fillna(0)
        close, open_ = bars["close"], bars["open"]

        entry_signal_date = None
        for i in range(1, len(position)):
            prev, curr = position.iloc[i - 1], position.iloc[i]
            date = position.index[i]
            if prev == 0 and curr == 1:
                entry_signal_date = date
            elif prev == 1 and curr == 0 and entry_signal_date is not None:
                # Earliest a human could act on each signal is the next
                # trading day's open -- the close that triggered it is
                # already known history by the time the market reopens.
                entry_idx = position.index.get_loc(entry_signal_date)
                exit_idx = position.index.get_loc(date)
                if entry_idx + 1 >= len(position) or exit_idx + 1 >= len(position):
                    entry_signal_date = None
                    continue
                entry_fill_date = position.index[entry_idx + 1]
                exit_fill_date = position.index[exit_idx + 1]
                if exit_fill_date >= cutoff:
                    entry_price, exit_price = open_.loc[entry_fill_date], open_.loc[exit_fill_date]
                    trades.append(dict(
                        symbol=symbol, entry_date=entry_fill_date, exit_date=exit_fill_date,
                        entry_price=entry_price, exit_price=exit_price,
                        roi=exit_price / entry_price - 1, status="closed",
                        hold_days=(exit_fill_date - entry_fill_date).days,
                    ))
                entry_signal_date = None

        # A still-open trade at the end of history is always relevant,
        # regardless of when it entered. No exit signal yet, so "current
        # price" stays today's close (the best available mark, not a fill).
        if entry_signal_date is not None and position.iloc[-1] == 1:
            entry_idx = position.index.get_loc(entry_signal_date)
            if entry_idx + 1 < len(position):
                entry_fill_date = position.index[entry_idx + 1]
                entry_price, current_price = open_.loc[entry_fill_date], close.iloc[-1]
                trades.append(dict(
                    symbol=symbol, entry_date=entry_fill_date, exit_date=None,
                    entry_price=entry_price, exit_price=current_price,
                    roi=current_price / entry_price - 1, status="open",
                    hold_days=(position.index[-1] - entry_fill_date).days,
                ))
                open_mtm_series[symbol] = (close.loc[entry_fill_date:] / entry_price - 1) * 1000

    conn.close()
    df = pd.DataFrame(trades)
    if not df.empty:
        df = df.sort_values(["status", "entry_date"], ascending=[True, False])
    return df, open_mtm_series


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
    log = build_trade_log()
    log.to_csv("trade_log_12mo.csv", index=False)
    print(f"{len(log)} trades since {DEFAULT_START_DATE}")
    print(summarize(log))

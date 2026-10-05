"""
Traditional technical-indicator strategies (RSI, MACD, and a combination),
plugged into the same walk-forward harness as baselines.py and model.py.

Parameters are standard textbook defaults (RSI 14/30/55, MACD 12/26/9), not
fitted to this data -- picking whichever threshold looks best in hindsight
is exactly the data-dredging trap this project has been built to avoid.
"""

import numpy as np
import pandas as pd


def rsi_reversion(full_feats: pd.DataFrame, buy_thresh: float = 30, exit_thresh: float = 55, **_) -> pd.Series:
    """Long-only: buy when RSI drops to oversold, hold until it recovers
    past exit_thresh, flat otherwise. Classic buy-the-dip mean reversion.
    """
    rsi = full_feats["rsi_14"]
    position = np.zeros(len(rsi))
    in_pos = False
    for i, r in enumerate(rsi.to_numpy()):
        if np.isnan(r):
            continue
        if not in_pos and r <= buy_thresh:
            in_pos = True
        elif in_pos and r >= exit_thresh:
            in_pos = False
        position[i] = 1.0 if in_pos else 0.0
    return pd.Series(position, index=rsi.index)


def rsi_reversion_with_stop(full_bars: pd.DataFrame, full_feats: pd.DataFrame,
                             buy_thresh: float = 30, exit_thresh: float = 55,
                             stop_loss_pct: float = 0.15, **_) -> pd.Series:
    """rsi_reversion, plus a hard stop-loss: exit if price falls stop_loss_pct
    below the entry price, regardless of what RSI is doing. Without this, a
    stock that's oversold for a real fundamental reason (fraud, bankruptcy,
    a short-seller report) rather than ordinary noise can sit in the book
    for months with RSI pinned near zero, never triggering the RSI exit --
    see indicators.py's module docstring history / the method1 ROI
    discussion for why this matters at portfolio scale, not just per trade.
    """
    rsi = full_feats["rsi_14"]
    close = full_bars["close"].reindex(rsi.index)
    position = np.zeros(len(rsi))
    in_pos = False
    entry_price = None
    rsi_vals, close_vals = rsi.to_numpy(), close.to_numpy()
    for i in range(len(rsi)):
        r, c = rsi_vals[i], close_vals[i]
        if np.isnan(r):
            continue
        if not in_pos and r <= buy_thresh:
            in_pos, entry_price = True, c
        elif in_pos and (r >= exit_thresh or c <= entry_price * (1 - stop_loss_pct)):
            in_pos, entry_price = False, None
        position[i] = 1.0 if in_pos else 0.0
    return pd.Series(position, index=rsi.index)


def rsi_overbought_short(full_feats: pd.DataFrame, sell_thresh: float = 70, exit_thresh: float = 45, **_) -> pd.Series:
    """Short-only mirror of rsi_reversion: short when RSI rises to
    overbought, hold the short until it cools back down past exit_thresh,
    flat otherwise. Same mean-reversion logic, opposite direction -- note
    shorting fights the market's long-run upward drift in a way buying
    dips doesn't, so this needs its own out-of-sample check, not an
    assumption that symmetry holds.
    """
    rsi = full_feats["rsi_14"]
    position = np.zeros(len(rsi))
    in_pos = False
    for i, r in enumerate(rsi.to_numpy()):
        if np.isnan(r):
            continue
        if not in_pos and r >= sell_thresh:
            in_pos = True
        elif in_pos and r <= exit_thresh:
            in_pos = False
        position[i] = -1.0 if in_pos else 0.0
    return pd.Series(position, index=rsi.index)


def macd_crossover(full_feats: pd.DataFrame, **_) -> pd.Series:
    """Long/short trend-following: long while MACD line is above its
    signal line, short while below. Always in a position (no flat state).
    """
    macd = full_feats["macd"]
    signal = full_feats["macd_signal"]
    position = pd.Series(np.where(macd > signal, 1.0, -1.0), index=macd.index)
    position[macd.isna() | signal.isna()] = 0.0
    return position


def rsi_macd_combo(full_feats: pd.DataFrame, buy_thresh: float = 35, exit_thresh: float = 55, **_) -> pd.Series:
    """Long-only: enter only when RSI is oversold AND MACD histogram has
    turned positive (momentum confirmation), exit when RSI recovers past
    exit_thresh OR momentum rolls back over -- requiring both indicators to
    agree is the whole point of testing a "combination" rather than either
    alone.
    """
    rsi = full_feats["rsi_14"]
    hist = full_feats["macd_hist"]
    position = np.zeros(len(rsi))
    in_pos = False
    rsi_vals, hist_vals = rsi.to_numpy(), hist.to_numpy()
    for i in range(len(rsi)):
        r, h = rsi_vals[i], hist_vals[i]
        if np.isnan(r) or np.isnan(h):
            continue
        if not in_pos and r <= buy_thresh and h > 0:
            in_pos = True
        elif in_pos and (r >= exit_thresh or h < 0):
            in_pos = False
        position[i] = 1.0 if in_pos else 0.0
    return pd.Series(position, index=rsi.index)

"""
Feature engineering for method1.

Every column here is computed using only data up to and including the row's
own date (close-to-close) -- no lookahead. Forward-looking labels live in
labels.py, kept deliberately separate so it's obvious nothing in this file
can leak the future into a feature.
"""

import numpy as np
import pandas as pd

RETURN_WINDOWS = (1, 5, 10, 20)
VOL_WINDOWS = (10, 20)
MA_WINDOWS = (10, 20, 50)
RSI_WINDOW = 14
VOLUME_WINDOW = 20


def _streak_features(close: pd.Series) -> pd.DataFrame:
    """Consecutive up/down day count and streak magnitude, as of each date.

    Reproduces "SPX closed down n days in a row, averaging m points/day" as
    plain numeric columns (streak_len, streak_len signed by direction,
    cumulative move, average move) so a model can find its own (n, m)
    thresholds instead of them being hand-coded.
    """
    day_chg = close.diff()
    day_pct = close.pct_change()
    sign = np.sign(day_chg).fillna(0)

    # streak_len: consecutive days with the same sign as today (>0 up, <0 down).
    streak_len = np.zeros(len(close))
    for i in range(1, len(close)):
        if sign.iloc[i] != 0 and sign.iloc[i] == sign.iloc[i - 1]:
            streak_len[i] = streak_len[i - 1] + 1
        else:
            streak_len[i] = 1 if sign.iloc[i] != 0 else 0

    signed_streak_len = streak_len * sign.to_numpy()

    # Cumulative/average move over the current streak, in both points (as in
    # "SPX down m points/day") and percent (comparable across symbols at
    # different price levels, e.g. SPX vs SPY vs a sector ETF).
    cum_move = np.zeros(len(close))
    avg_move = np.zeros(len(close))
    cum_move_pct = np.zeros(len(close))
    avg_move_pct = np.zeros(len(close))
    chg_vals = day_chg.to_numpy()
    pct_vals = day_pct.to_numpy()
    for i in range(len(close)):
        n = int(streak_len[i])
        if n == 0:
            continue
        window = chg_vals[i - n + 1 : i + 1]
        window_pct = pct_vals[i - n + 1 : i + 1]
        cum_move[i] = np.nansum(window)
        avg_move[i] = cum_move[i] / n
        cum_move_pct[i] = np.nansum(window_pct)
        avg_move_pct[i] = cum_move_pct[i] / n

    return pd.DataFrame(
        {
            "streak_len": signed_streak_len,
            "streak_cum_move": cum_move,
            "streak_avg_move": avg_move,
            "streak_cum_move_pct": cum_move_pct,
            "streak_avg_move_pct": avg_move_pct,
        },
        index=close.index,
    )


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": macd_line - signal_line,
    })


def _atr(bars: pd.DataFrame, window: int) -> pd.Series:
    high, low, close = bars["high"], bars["low"], bars["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(window).mean()


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    """bars: DataFrame[open,high,low,close,volume] indexed by date for one symbol.

    Returns a DataFrame of features indexed the same way. Rows near the start
    of history will contain NaNs until rolling windows fill in -- callers
    should dropna() after joining with labels, not before.
    """
    close = bars["close"]
    feats = pd.DataFrame(index=bars.index)

    for w in RETURN_WINDOWS:
        feats[f"ret_{w}d"] = close.pct_change(w)

    feats = feats.join(_streak_features(close))

    for w in VOL_WINDOWS:
        feats[f"vol_{w}d"] = close.pct_change().rolling(w).std()

    feats["atr_14"] = _atr(bars, 14) / close

    for w in MA_WINDOWS:
        ma = close.rolling(w).mean()
        feats[f"dist_ma_{w}"] = close / ma - 1
        feats[f"ma_{w}_slope"] = ma.pct_change(5)

    feats[f"rsi_{RSI_WINDOW}"] = _rsi(close, RSI_WINDOW)
    feats = feats.join(_macd(close))

    roll_mean = close.rolling(20).mean()
    roll_std = close.rolling(20).std()
    feats["zscore_20"] = (close - roll_mean) / roll_std.replace(0, np.nan)

    vol = bars["volume"]
    feats["rel_volume_20"] = vol / vol.rolling(VOLUME_WINDOW).mean()

    return feats


def join_macro_context(feats: pd.DataFrame, vix: pd.Series, rate_spread: pd.Series) -> pd.DataFrame:
    """Add cross-asset macro columns (VIX level/change, TLT-vs-SHY spread as a
    yield-curve proxy), aligned on date. vix/rate_spread must be computed from
    data no later than each row's own date -- callers pass already-aligned,
    as-of-date series.
    """
    out = feats.copy()
    out["vix_level"] = vix.reindex(out.index)
    out["vix_chg_5d"] = vix.pct_change(5).reindex(out.index)
    out["rate_spread"] = rate_spread.reindex(out.index)
    return out

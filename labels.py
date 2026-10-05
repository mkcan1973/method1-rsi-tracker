"""
Forward-return labels for method1. Kept separate from features.py so it's
obvious nothing in that file can see the future -- only this one does, and
only to build the training target.
"""

import pandas as pd

DEFAULT_HORIZON = 10  # trading days (~2 weeks)


def forward_return(close: pd.Series, horizon: int = DEFAULT_HORIZON) -> pd.Series:
    """Forward horizon-day return as of each date: (close[t+h] / close[t]) - 1.

    The last `horizon` rows will be NaN (no future data yet) -- that's
    expected and must stay NaN rather than be filled, since there is no
    real label there.
    """
    return close.shift(-horizon) / close - 1

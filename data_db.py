"""
SQLite store of daily OHLCV bars, one row per (symbol, date).

Mirrors the connect()/upsert()/load_*() pattern from seasonal/seasonal_db.py.
Every row is tagged with its data source so backtest history and anything
logged later (e.g. live scan predictions in a separate table) are never
ambiguous about provenance.
"""

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).with_name("method1.sqlite3")


def connect(path=DB_PATH):
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS bars (
               symbol TEXT NOT NULL,
               date TEXT NOT NULL,
               open REAL NOT NULL,
               high REAL NOT NULL,
               low REAL NOT NULL,
               close REAL NOT NULL,
               volume REAL NOT NULL,
               source TEXT NOT NULL,
               PRIMARY KEY (symbol, date)
           )"""
    )
    return conn


def upsert_bars(conn, rows):
    """rows: iterable of (symbol, 'YYYY-MM-DD', open, high, low, close, volume, source)."""
    conn.executemany(
        "INSERT OR REPLACE INTO bars VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
    )
    conn.commit()


def load_bars(conn, symbol=None):
    """Return {symbol: pd.DataFrame[open,high,low,close,volume] indexed by date}.

    If symbol is given, return just that one DataFrame (or None if absent).
    """
    query = "SELECT symbol, date, open, high, low, close, volume FROM bars"
    params = ()
    if symbol is not None:
        query += " WHERE symbol = ?"
        params = (symbol,)
    df = pd.read_sql_query(query, conn, params=params, parse_dates=["date"])

    if symbol is not None:
        if df.empty:
            return None
        return df.set_index("date").sort_index()[["open", "high", "low", "close", "volume"]]

    return {
        sym: g.set_index("date").sort_index()[["open", "high", "low", "close", "volume"]]
        for sym, g in df.groupby("symbol")
    }


def last_date(conn, symbol):
    """Return the latest stored date (pd.Timestamp) for symbol, or None if absent."""
    row = conn.execute(
        "SELECT MAX(date) FROM bars WHERE symbol = ?", (symbol,)
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return pd.Timestamp(row[0])


def prune_old_bars(conn, keep_days: int):
    """Delete bars older than `keep_days` before the latest stored date.

    For a deployment (e.g. a git-hosted mirror) that only needs a trailing
    window -- RSI(14)/MACD need at most ~90 days of warmup, and the 12-month
    trade log needs 12 months -- rather than the full multi-year history a
    local research copy keeps. keep_days should cover trade-log window +
    warmup with room to spare (450 days = 12mo + ~90d warmup + buffer).
    """
    cutoff = conn.execute("SELECT date(MAX(date), ?) FROM bars", (f"-{keep_days} days",)).fetchone()[0]
    conn.execute("DELETE FROM bars WHERE date < ?", (cutoff,))
    conn.commit()
    conn.execute("VACUUM")

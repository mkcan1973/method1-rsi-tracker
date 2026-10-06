"""
Trims the database to a trailing window before each commit, so this
git-hosted mirror's method1.sqlite3 stays small and roughly flat in size
over time, unlike the full-history copy kept for local research.

KEEP_DAYS is computed from trade_log.DEFAULT_START_DATE (the dashboard's
2025 / YTD / Live chart needs data back to that fixed calendar anchor)
rather than hardcoded, so it can't silently go stale as "today" advances
and the gap to that fixed anchor grows -- a hardcoded day-count would
eventually prune away data the chart still expects to find. The warmup
buffer covers RSI/MACD/the 50-day MA before the anchor date.
"""

import pandas as pd

import data_db
from trade_log import DEFAULT_START_DATE

WARMUP_BUFFER_DAYS = 120
KEEP_DAYS = (pd.Timestamp.today().normalize() - pd.Timestamp(DEFAULT_START_DATE)).days + WARMUP_BUFFER_DAYS

if __name__ == "__main__":
    conn = data_db.connect()
    before = conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
    data_db.prune_old_bars(conn, keep_days=KEEP_DAYS)
    after = conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
    conn.close()
    print(f"Pruned bars table: {before} -> {after} rows (kept trailing {KEEP_DAYS} days, "
          f"anchored to {DEFAULT_START_DATE} + {WARMUP_BUFFER_DAYS}d warmup)")

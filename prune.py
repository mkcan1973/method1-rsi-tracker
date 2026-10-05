"""
Trims the database to a trailing window before each commit, so this
git-hosted mirror's method1.sqlite3 stays small and roughly flat in size
over time, unlike the full-history copy kept for local research.

600 calendar days (~430 trading days) = the 12-month trade-log window
(~252 trading days) + a comfortable ~175 trading days of RSI/MACD/MA
warmup before it, which is all the live scan and web summary ever need.
"""

import data_db

KEEP_DAYS = 600

if __name__ == "__main__":
    conn = data_db.connect()
    before = conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
    data_db.prune_old_bars(conn, keep_days=KEEP_DAYS)
    after = conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
    conn.close()
    print(f"Pruned bars table: {before} -> {after} rows (kept trailing {KEEP_DAYS} days)")

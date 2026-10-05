# method1 RSI Tracker

Tracks an RSI(14) mean-reversion signal across the S&P 500 and core ETFs/index:
buy when a symbol's 14-day RSI drops to <=30 (oversold), hold until it recovers
to >=55, flat otherwise. Validated via walk-forward backtesting (positive
out-of-sample Sharpe on effectively the whole universe tested) -- see the
companion research project for the backtest harness and validation history.

## Key features

The tracker runs automatically on weekdays via GitHub Actions, once daily
after market close. It generates:

- A web dashboard (`index.html`, served via GitHub Pages) showing open and
  closed positions over the trailing 12 months, with ROI and an equity curve
- `scan_results.csv` -- today's RSI scan across the full universe
- `trade_log_12mo.csv` -- the full open/closed trade log behind the dashboard
- `method1.sqlite3` -- daily OHLCV price history, pruned to a trailing ~600-day
  window (enough for RSI/MACD warmup + the 12-month trade log) so this repo's
  database stays small; a separate local copy keeps full multi-year history
  for research/backtesting

## Usage

Signal only -- this places no orders. Run `python scan.py` for today's scan,
or `python web_summary.py` to rebuild the dashboard. Both handle their own
data refresh from Yahoo Finance.

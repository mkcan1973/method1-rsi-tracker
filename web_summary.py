"""
Builds the method1 trade-log web dashboard: open vs closed RSI-reversion
positions over the trailing 12 months, with ROI and an equity curve.

Reuses trade_log.build_trade_log() (which itself reuses the validated,
correctly-lagged indicators.rsi_reversion position logic) -- this is a
reporting view of that same signal, nothing recalculated independently.

Writes a full standalone HTML file (open directly in a browser, or host
via GitHub Pages later) and a trimmed fragment (title+style+body only, no
doctype/html/head wrapper) for publishing as a Claude Artifact.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trade_log import build_trade_log, summarize

FULL_HTML_PATH = Path(__file__).with_name("web_summary.html")


def _equity_curve(df: pd.DataFrame) -> list[dict]:
    closed = df[df["status"] == "closed"].sort_values("exit_date")
    cum = (closed["roi"] * 1000).cumsum()
    return [
        {"date": d.strftime("%Y-%m-%d"), "cum_pnl": round(c, 2)}
        for d, c in zip(closed["exit_date"], cum)
    ]


def _rows_for_js(df: pd.DataFrame, status: str) -> list[dict]:
    sub = df[df["status"] == status].copy()
    out = []
    for r in sub.itertuples():
        out.append({
            "symbol": r.symbol,
            "entry_date": r.entry_date.strftime("%Y-%m-%d"),
            "exit_date": r.exit_date.strftime("%Y-%m-%d") if pd.notna(r.exit_date) else None,
            "entry_price": round(r.entry_price, 2),
            "exit_price": round(r.exit_price, 2),
            "roi": round(r.roi * 100, 2),
            "hold_days": int(r.hold_days),
        })
    return out


STYLE = """
<title>RSI Trade Log</title>
<style>
  :root {
    color-scheme: light;
    --bg-page: #f9f9f7; --surface: #fcfcfb;
    --ink-1: #0b0b0b; --ink-2: #52514e; --ink-muted: #898781;
    --grid: #e1e0d9; --baseline: #c3c2b7;
    --accent: #2a78d6; --accent-soft: #cde2fb;
    --good: #0ca30c; --critical: #d03b3b;
    --border: rgba(11,11,11,0.10);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --bg-page: #0d0d0d; --surface: #1a1a19;
      --ink-1: #ffffff; --ink-2: #c3c2b7; --ink-muted: #898781;
      --grid: #2c2c2a; --baseline: #383835;
      --accent: #3987e5; --accent-soft: #15325a;
      --border: rgba(255,255,255,0.10);
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --bg-page: #0d0d0d; --surface: #1a1a19;
    --ink-1: #ffffff; --ink-2: #c3c2b7; --ink-muted: #898781;
    --grid: #2c2c2a; --baseline: #383835;
    --accent: #3987e5; --accent-soft: #15325a;
    --border: rgba(255,255,255,0.10);
  }

  body { background: var(--bg-page); color: var(--ink-1);
         font-family: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif; }
  .wrap { max-width: 1120px; margin-inline: auto; padding-inline: 16px; padding-block: 28px 48px; }
  .num { font-family: "IBM Plex Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; }

  h1 { font-size: 1.5rem; font-weight: 600; margin: 0; letter-spacing: -0.01em; }
  .subtitle { color: var(--ink-2); font-size: 0.92rem; margin-top: 4px; }
  .asof { color: var(--ink-muted); font-size: 0.8rem; margin-top: 2px; }

  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
           gap: 10px; margin-top: 22px; }
  .tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
          padding: 14px 16px; }
  .tile .label { font-size: 0.72rem; letter-spacing: 0.04em; text-transform: uppercase;
                 color: var(--ink-muted); }
  .tile .value { font-size: 1.5rem; font-weight: 600; margin-top: 4px; }
  .tile .value.good { color: var(--good); }
  .tile .value.critical { color: var(--critical); }
  .tile .sub { font-size: 0.78rem; color: var(--ink-2); margin-top: 2px; }

  h2 { font-size: 1.05rem; font-weight: 600; margin: 32px 0 10px; }
  .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
           padding: 16px; }
  .chart-note { color: var(--ink-muted); font-size: 0.78rem; margin-top: 6px; }

  .tablebar { display: flex; gap: 10px; align-items: center; margin: 10px 0; flex-wrap: wrap; }
  .tablebar input { background: var(--bg-page); border: 1px solid var(--border); border-radius: 7px;
                     color: var(--ink-1); padding: 7px 10px; font-size: 0.85rem; width: 160px; }
  .tablebar .count { color: var(--ink-muted); font-size: 0.8rem; margin-left: auto; }

  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  thead th { text-align: left; font-size: 0.72rem; letter-spacing: 0.03em; text-transform: uppercase;
             color: var(--ink-muted); border-bottom: 1px solid var(--baseline); padding: 7px 10px;
             cursor: pointer; white-space: nowrap; user-select: none; }
  thead th:hover { color: var(--ink-1); }
  thead th.num, tbody td.num { text-align: right; }
  tbody td { padding: 7px 10px; border-bottom: 1px solid var(--grid); }
  tbody tr:hover { background: var(--accent-soft); }
  .roi.good { color: var(--good); font-weight: 600; }
  .roi.critical { color: var(--critical); font-weight: 600; }
  .tablewrap { overflow-x: auto; }

  footer { color: var(--ink-muted); font-size: 0.78rem; margin-top: 36px; border-top: 1px solid var(--border);
           padding-top: 14px; }

  #tooltip { position: absolute; pointer-events: none; background: var(--ink-1); color: var(--bg-page);
             font-size: 0.78rem; padding: 6px 9px; border-radius: 6px; opacity: 0; transition: opacity 0.1s;
             font-family: "IBM Plex Mono", monospace; white-space: nowrap; z-index: 10; }
</style>
"""

HEAD_LINK = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">'


def _svg_equity_chart(curve: list[dict]) -> str:
    if len(curve) < 2:
        return '<p class="chart-note">Not enough closed trades yet to chart an equity curve.</p>'

    W, H, PAD_L, PAD_R, PAD_T, PAD_B = 1040, 260, 56, 16, 16, 28
    vals = [p["cum_pnl"] for p in curve]
    lo, hi = min(0, min(vals)), max(vals)
    span = (hi - lo) or 1
    n = len(curve)

    def x(i):
        return PAD_L + (W - PAD_L - PAD_R) * i / (n - 1)

    def y(v):
        return PAD_T + (H - PAD_T - PAD_B) * (1 - (v - lo) / span)

    pts = [(x(i), y(p["cum_pnl"])) for i, p in enumerate(curve)]
    path = "M " + " L ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
    zero_y = y(0)
    area = path + f" L {pts[-1][0]:.1f},{zero_y:.1f} L {pts[0][0]:.1f},{zero_y:.1f} Z"

    gridlines = ""
    for frac in (0, 0.25, 0.5, 0.75, 1):
        gy = PAD_T + (H - PAD_T - PAD_B) * frac
        val = hi - span * frac
        gridlines += (f'<line x1="{PAD_L}" y1="{gy:.1f}" x2="{W-PAD_R}" y2="{gy:.1f}" '
                      f'stroke="var(--grid)" stroke-width="1"/>'
                      f'<text x="{PAD_L-8}" y="{gy+3:.1f}" text-anchor="end" font-size="10" '
                      f'fill="var(--ink-muted)" class="num">${val:,.0f}</text>')

    dots = "".join(
        f'<circle class="eqdot" data-i="{i}" cx="{px:.1f}" cy="{py:.1f}" r="7" fill="transparent"/>'
        for i, (px, py) in enumerate(pts)
    )

    dates_json = json.dumps([p["date"] for p in curve])
    vals_json = json.dumps([p["cum_pnl"] for p in curve])

    return f"""
    <div style="position:relative;">
      <svg viewBox="0 0 {W} {H}" style="width:100%; height:auto; display:block;">
        {gridlines}
        <line x1="{PAD_L}" y1="{zero_y:.1f}" x2="{W-PAD_R}" y2="{zero_y:.1f}" stroke="var(--baseline)" stroke-width="1"/>
        <path d="{area}" fill="var(--accent)" opacity="0.12"/>
        <path d="{path}" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linecap="round"/>
        <circle cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="4" fill="var(--accent)"/>
        <g id="eqHover">{dots}</g>
        <line id="crosshair" x1="0" y1="{PAD_T}" x2="0" y2="{H-PAD_B}" stroke="var(--ink-muted)"
              stroke-width="1" stroke-dasharray="3,3" style="opacity:0;"/>
      </svg>
      <div id="tooltip"></div>
    </div>
    <script>
      (function() {{
        var dates = {dates_json};
        var vals = {vals_json};
        var svg = document.currentScript.previousElementSibling.querySelector('svg')
                  || document.getElementById('eqHover').closest('svg');
        var tooltip = document.getElementById('tooltip');
        var crosshair = document.getElementById('crosshair');
        var container = document.getElementById('eqHover').closest('div[style]');
        document.querySelectorAll('.eqdot').forEach(function(dot) {{
          dot.addEventListener('mouseenter', function() {{
            var i = parseInt(dot.getAttribute('data-i'), 10);
            var rect = dot.closest('svg').getBoundingClientRect();
            var cx = dot.cx.baseVal.value / {W} * rect.width;
            var cy = dot.cy.baseVal.value / {H} * rect.height;
            tooltip.textContent = dates[i] + '  $' + vals[i].toLocaleString('en-US', {{maximumFractionDigits:0}});
            tooltip.style.left = (cx + 10) + 'px';
            tooltip.style.top = (cy - 28) + 'px';
            tooltip.style.opacity = 1;
            crosshair.setAttribute('x1', dot.cx.baseVal.value);
            crosshair.setAttribute('x2', dot.cx.baseVal.value);
            crosshair.style.opacity = 1;
          }});
          dot.addEventListener('mouseleave', function() {{
            tooltip.style.opacity = 0;
            crosshair.style.opacity = 0;
          }});
        }});
      }})();
    </script>
    """


def _table_section(title: str, rows: list[dict], status: str) -> str:
    table_id = f"tbl-{status}"
    search_id = f"search-{status}"
    count_id = f"count-{status}"

    if status == "open":
        cols = [("symbol", "Symbol"), ("entry_date", "Entry"), ("entry_price", "Entry $"),
                ("exit_price", "Now $"), ("roi", "Unrlzd ROI"), ("hold_days", "Days held")]
    else:
        cols = [("symbol", "Symbol"), ("entry_date", "Entry"), ("exit_date", "Exit"),
                ("entry_price", "Entry $"), ("exit_price", "Exit $"), ("roi", "ROI"),
                ("hold_days", "Hold days")]

    header_cells = "".join(
        f'<th class="{"num" if k in ("entry_price","exit_price","roi","hold_days") else ""}" '
        f'data-key="{k}" data-tbl="{table_id}">{label}</th>'
        for k, label in cols
    )

    return f"""
    <h2>{title} <span class="num" style="color:var(--ink-muted); font-size:0.85rem;">({len(rows)})</span></h2>
    <div class="panel">
      <div class="tablebar">
        <input type="text" id="{search_id}" placeholder="Filter symbol..." oninput="filterTable('{table_id}','{search_id}','{count_id}')">
        <span class="count" id="{count_id}"></span>
      </div>
      <div class="tablewrap">
        <table id="{table_id}">
          <thead><tr>{header_cells}</tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
    """


BODY_SCRIPT_TEMPLATE = """
<script>
  var DATA = {data_json};

  function fmtRoi(v) {
    var cls = v >= 0 ? 'good' : 'critical';
    return '<span class="roi ' + cls + '">' + (v >= 0 ? '+' : '') + v.toFixed(2) + '%</span>';
  }

  function renderTable(tblId, rows, status) {
    var tbody = document.querySelector('#' + tblId + ' tbody');
    tbody.innerHTML = rows.map(function(r) {
      if (status === 'open') {
        return '<tr>' +
          '<td>' + r.symbol + '</td>' +
          '<td class="num">' + r.entry_date + '</td>' +
          '<td class="num">' + r.entry_price.toFixed(2) + '</td>' +
          '<td class="num">' + r.exit_price.toFixed(2) + '</td>' +
          '<td class="num">' + fmtRoi(r.roi) + '</td>' +
          '<td class="num">' + r.hold_days + '</td>' +
          '</tr>';
      }
      return '<tr>' +
        '<td>' + r.symbol + '</td>' +
        '<td class="num">' + r.entry_date + '</td>' +
        '<td class="num">' + r.exit_date + '</td>' +
        '<td class="num">' + r.entry_price.toFixed(2) + '</td>' +
        '<td class="num">' + r.exit_price.toFixed(2) + '</td>' +
        '<td class="num">' + fmtRoi(r.roi) + '</td>' +
        '<td class="num">' + r.hold_days + '</td>' +
        '</tr>';
    }).join('');
  }

  var sortState = {};
  function sortRows(status, key) {
    var rows = DATA[status].slice();
    var asc = sortState[status + ':' + key] !== true;
    sortState = {};
    sortState[status + ':' + key] = asc;
    rows.sort(function(a, b) {
      var av = a[key], bv = b[key];
      if (av === null) av = -Infinity;
      if (bv === null) bv = -Infinity;
      if (av < bv) return asc ? -1 : 1;
      if (av > bv) return asc ? 1 : -1;
      return 0;
    });
    DATA[status + '_sorted'] = rows;
    applyFilter(status);
  }

  function applyFilter(status) {
    var tblId = 'tbl-' + status, searchId = 'search-' + status, countId = 'count-' + status;
    var q = (document.getElementById(searchId).value || '').trim().toUpperCase();
    var base = DATA[status + '_sorted'] || DATA[status];
    var rows = q ? base.filter(function(r) { return r.symbol.indexOf(q) !== -1; }) : base;
    renderTable(tblId, rows, status);
    document.getElementById(countId).textContent = rows.length + ' of ' + base.length;
  }

  function filterTable(tblId, searchId, countId) {
    var status = tblId.replace('tbl-', '');
    applyFilter(status);
  }

  ['open', 'closed'].forEach(function(status) {
    DATA[status + '_sorted'] = DATA[status];
    applyFilter(status);
    document.querySelectorAll('#tbl-' + status + ' thead th').forEach(function(th) {
      th.addEventListener('click', function() { sortRows(status, th.getAttribute('data-key')); });
    });
  });
</script>
"""


def build_body(stats: dict, open_rows: list[dict], closed_rows: list[dict],
               equity_curve: list[dict], today: str) -> str:
    win_rate = stats["win_rate"] or 0
    avg_roi_closed = (stats["avg_roi_closed"] or 0) * 100
    avg_roi_open = (stats["avg_roi_open"] or 0) * 100
    total_pnl = stats["total_normalized_pnl"]

    data_json = json.dumps({"open": open_rows, "closed": closed_rows})

    return f"""
<div class="wrap">
  <h1>RSI Trade Log</h1>
  <div class="subtitle">RSI(14) mean reversion &middot; S&amp;P 500 + core ETFs/index &middot; trailing 12 months</div>
  <div class="asof">As of {today} &middot; signal only, no orders placed &middot; normalized to $1,000 risked per trade</div>

  <div class="tiles">
    <div class="tile"><div class="label">Open positions</div><div class="value">{len(open_rows)}</div></div>
    <div class="tile"><div class="label">Closed trades (12mo)</div><div class="value">{len(closed_rows)}</div></div>
    <div class="tile"><div class="label">Win rate (closed)</div><div class="value">{win_rate*100:.1f}%</div></div>
    <div class="tile"><div class="label">Avg ROI / closed trade</div>
      <div class="value {'good' if avg_roi_closed>=0 else 'critical'}">{avg_roi_closed:+.2f}%</div></div>
    <div class="tile"><div class="label">Avg unrealized ROI (open)</div>
      <div class="value {'good' if avg_roi_open>=0 else 'critical'}">{avg_roi_open:+.2f}%</div></div>
    <div class="tile"><div class="label">Total realized P&amp;L</div>
      <div class="value {'good' if total_pnl>=0 else 'critical'} num">${total_pnl:,.0f}</div>
      <div class="sub">across {len(closed_rows)} trades, $1,000 risk each</div></div>
  </div>

  <h2>Cumulative realized P&amp;L</h2>
  <div class="panel">
    {_svg_equity_chart(equity_curve)}
    <div class="chart-note">Running total of normalized ($1,000/trade) realized P&amp;L across closed trades, by exit date.</div>
  </div>

  {_table_section("Open positions", open_rows, "open")}
  {_table_section("Closed positions (last 12 months)", closed_rows, "closed")}

  <footer>
    Strategy: buy when RSI(14) &le; 30 (oversold), hold until RSI &ge; 55, flat otherwise &mdash;
    validated via walk-forward backtesting across the S&amp;P 500 + core ETFs/index (see method1/backtest.py).
    This page is a reporting view of that signal, not investment advice.
  </footer>
</div>
{BODY_SCRIPT_TEMPLATE.replace("{data_json}", data_json)}
"""


def main():
    log = build_trade_log(months=12)
    stats = summarize(log)
    equity_curve = _equity_curve(log)
    open_rows = _rows_for_js(log, "open")
    closed_rows = _rows_for_js(log, "closed")
    today = pd.Timestamp.today().strftime("%Y-%m-%d")

    body = build_body(stats, open_rows, closed_rows, equity_curve, today)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{HEAD_LINK}
{STYLE}
</head>
<body>
{body}
</body>
</html>"""

    FULL_HTML_PATH.write_text(full_html, encoding="utf-8")
    print(f"Wrote {FULL_HTML_PATH}")
    print(f"open={len(open_rows)} closed={len(closed_rows)} win_rate={stats['win_rate']:.1%}")

    fragment = f"{HEAD_LINK}\n{STYLE}\n{body}"
    return fragment


if __name__ == "__main__":
    main()

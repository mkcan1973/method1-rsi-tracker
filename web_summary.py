"""
Builds the method1 trade-log web dashboard: open vs closed RSI-reversion
positions over the trailing 12 months, with ROI and a realized/unrealized/
total mark-to-market P&L chart on one shared calendar axis.

Reuses trade_log.build_trade_log_with_mtm() (which itself reuses the
validated, correctly-lagged indicators.rsi_reversion position logic) --
this is a reporting view of that same signal, nothing recalculated
independently.

Writes a full standalone HTML file (open directly in a browser, or host
via GitHub Pages later) and a trimmed fragment (title+style+body only, no
doctype/html/head wrapper) for publishing as a Claude Artifact.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import data_db
from fundamentals import get_trailing_eps
from trade_log import DEFAULT_START_DATE, build_trade_log_with_mtm, summarize

FULL_HTML_PATH = Path(__file__).with_name("web_summary.html")

# Reference trading calendar: SPY's own bars give the real NYSE trading
# days, so realized/unrealized get reindexed onto every actual trading day
# since DEFAULT_START_DATE -- not just the days spanned by currently-open
# positions, which would silently truncate the chart to however far back
# the single oldest-still-open position happens to reach.
_CALENDAR_REF_SYMBOL = "SPY"


def _n_active_curve(df: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.Series:
    """Count of positions actively held on each calendar date -- BOTH open
    and already-closed trades, unlike open_mtm_series (which only covers
    positions still open today). Needed as the capital-base denominator for
    an honest CAGR: "how much was actually tied up at once," not just
    today's open count. A trade is active on [entry_date, exit_date) --
    matching indicators.py's own convention that the exit day itself is no
    longer held -- or [entry_date, today] if still open.
    """
    # Clamp entry_date to the calendar's own start date. trade_log.py keeps
    # a closed trade if its EXIT is >= start_date, even when its entry was
    # earlier -- so a trade that began before the window but exited inside
    # it would otherwise contribute a -1 (its exit) with no matching +1
    # (its entry falls before `calendar` and reindex() silently drops it),
    # permanently corrupting the cumulative count negative from then on.
    entry_dates = df["entry_date"].clip(lower=calendar[0])
    entry_delta = entry_dates.value_counts()
    exit_delta = df["exit_date"].dropna().value_counts()
    delta = pd.Series(0.0, index=calendar)
    delta = delta.add(entry_delta.reindex(calendar, fill_value=0), fill_value=0)
    delta = delta.add(-exit_delta.reindex(calendar, fill_value=0), fill_value=0)
    return delta.cumsum()


def _mtm_curve(df: pd.DataFrame, open_mtm_series: dict, start_date: str = DEFAULT_START_DATE) -> list[dict]:
    """Realized, unrealized, and total P&L on ONE shared calendar axis
    spanning every trading day since start_date:
      realized(d)   = cumulative $ from trades already closed by d (a step
                      function -- flat between exits, jumps on each one)
      unrealized(d) = sum of every currently-open position's mark-to-market
                      P&L AT d (0 before that position's own entry date)
      total(d)      = realized(d) + unrealized(d)
      n_active(d)   = how many positions (open + closed) were held on d --
                      the capital-base denominator the page uses for CAGR
    """
    conn = data_db.connect()
    ref_bars = data_db.load_bars(conn, _CALENDAR_REF_SYMBOL)
    conn.close()
    calendar = ref_bars.loc[start_date:].index

    if open_mtm_series:
        mtm_df = pd.DataFrame(open_mtm_series).reindex(calendar)
        unrealized = mtm_df.sum(axis=1, skipna=True)
    else:
        unrealized = pd.Series(0.0, index=calendar)

    closed = df[df["status"] == "closed"]
    if len(closed):
        # Multiple trades can exit on the same date, which would otherwise
        # leave duplicate index labels reindex() can't ffill against --
        # group same-day exits together first.
        daily_realized = (closed["roi"] * 1000).groupby(closed["exit_date"]).sum().sort_index()
        closed_cum = daily_realized.cumsum()
        realized = closed_cum.reindex(calendar, method="ffill").fillna(0)
    else:
        realized = pd.Series(0.0, index=calendar)

    total = realized + unrealized
    n_active = _n_active_curve(df, calendar)
    return [
        {"date": d.strftime("%Y-%m-%d"), "realized": round(r, 2), "unrealized": round(u, 2),
         "total": round(t, 2), "n_active": round(n, 1)}
        for d, r, u, t, n in zip(calendar, realized, unrealized, total, n_active)
    ]


def _rows_for_js(df: pd.DataFrame, status: str, eps_by_symbol: dict) -> list[dict]:
    sub = df[df["status"] == status].copy()
    out = []
    for r in sub.itertuples():
        eps = eps_by_symbol.get(r.symbol)
        pe_at_entry = round(r.entry_price / eps, 1) if eps and eps > 0 else None
        out.append({
            "symbol": r.symbol,
            "entry_date": r.entry_date.strftime("%Y-%m-%d"),
            "exit_date": r.exit_date.strftime("%Y-%m-%d") if pd.notna(r.exit_date) else None,
            "entry_price": round(r.entry_price, 2),
            "exit_price": round(r.exit_price, 2),
            "roi": round(r.roi * 100, 2),
            "hold_days": int(r.hold_days),
            "pe_at_entry": pe_at_entry,
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
    --series-total: #2a78d6; --series-realized: #eb6834; --series-unrealized: #1baf7a;
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
      --series-total: #3987e5; --series-realized: #d95926; --series-unrealized: #199e70;
      --border: rgba(255,255,255,0.10);
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --bg-page: #0d0d0d; --surface: #1a1a19;
    --ink-1: #ffffff; --ink-2: #c3c2b7; --ink-muted: #898781;
    --grid: #2c2c2a; --baseline: #383835;
    --accent: #3987e5; --accent-soft: #15325a;
    --series-total: #3987e5; --series-realized: #d95926; --series-unrealized: #199e70;
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
  .legend { display: flex; gap: 16px; margin-bottom: 10px; font-size: 0.82rem; color: var(--ink-2); flex-wrap: wrap; }
  .legend-item { display: inline-flex; align-items: center; gap: 6px; }
  .legend .swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
  .togglebar { display: flex; gap: 6px; margin-bottom: 14px; flex-wrap: wrap; align-items: center; }
  .toggle-btn { background: var(--bg-page); border: 1px solid var(--border); border-radius: 7px;
                color: var(--ink-2); padding: 6px 12px; font-size: 0.82rem; font-family: inherit;
                cursor: pointer; }
  .toggle-btn:hover { color: var(--ink-1); }
  .toggle-btn.active { background: var(--accent); border-color: var(--accent); color: #fff; }
  #liveDateInput { background: var(--bg-page); border: 1px solid var(--border); border-radius: 7px;
                    color: var(--ink-1); padding: 6px 10px; font-size: 0.82rem; font-family: inherit; }

  .tablebar { display: flex; gap: 10px; align-items: center; margin: 10px 0; flex-wrap: wrap; }
  .tablebar input[type="text"] { background: var(--bg-page); border: 1px solid var(--border); border-radius: 7px;
                     color: var(--ink-1); padding: 7px 10px; font-size: 0.85rem; width: 160px; }
  .tablebar .count { color: var(--ink-muted); font-size: 0.8rem; margin-left: auto; }
  .pe-filter-label { display: flex; align-items: center; gap: 6px; font-size: 0.78rem; color: var(--ink-muted); }
  .pe-filter-label input[type="number"] { background: var(--bg-page); border: 1px solid var(--border);
                     border-radius: 7px; color: var(--ink-1); padding: 7px 8px; font-size: 0.85rem; width: 64px; }

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

  .tooltip { position: absolute; pointer-events: none; background: var(--ink-1); color: var(--bg-page);
             font-size: 0.78rem; padding: 6px 9px; border-radius: 6px; opacity: 0; transition: opacity 0.1s;
             font-family: "IBM Plex Mono", monospace; white-space: nowrap; z-index: 10; }
</style>
"""

HEAD_LINK = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">'


def _year_buttons_html(first_date: str, last_date: str) -> str:
    """One button per COMPLETE calendar year the data covers (data starts
    partway through DEFAULT_START_DATE's year, and the current year is
    handled by the YTD button instead).
    """
    current_year = last_date[:4]
    start_year, end_year = int(first_date[:4]), int(last_date[:4])
    years_present = [
        str(y) for y in range(start_year, end_year + 1)
        if str(y) != current_year and f"{y}-01-01" >= first_date
    ]
    return "".join(f'<button class="toggle-btn" data-range="{y}">{y}</button>' for y in years_present)


def _svg_mtm_chart(curve: list[dict], open_mtm_series: dict) -> str:
    """One chart, three lines, one shared calendar axis: realized (step
    function, flat between exits), unrealized (sum of currently-open
    positions' mark-to-market P&L as of each date), and total (their sum).

    Rendering happens in JS, not here: the 2025 / 2026 YTD / Live toggle
    needs to re-slice the full embedded series and rescale the axes to
    whatever window is selected, which a statically-rendered SVG path
    can't do. Python's job is just to hand the complete daily series over.

    Also embeds each currently-open position's own daily mark-to-market
    series (OPEN_MTM) alongside the pre-aggregated curve, so the P/E
    filter (page-level, build_body) can recompute realized/unrealized/
    total from only the positions that pass it, instead of being stuck
    with Python's unfiltered aggregate -- window.mtmSetData() lets the
    page's filter handler push a recomputed curve in before re-rendering.

    The toggle bar itself lives at the page level (build_body), not here,
    since it now drives the stat tiles and tables too, not just this
    chart -- this function only exposes window.mtmRender()/mtmSetData()
    for the page's shared toggle/filter handler to call.
    """
    if len(curve) < 2:
        return '<p class="chart-note">Not enough data yet to chart.</p>'

    data_json = json.dumps(curve)
    open_mtm_json = json.dumps({
        symbol: [[d.strftime("%Y-%m-%d"), round(v, 2)] for d, v in series.items()]
        for symbol, series in open_mtm_series.items()
    })

    return f"""
    <div class="legend">
      <span class="legend-item"><span class="swatch" style="background:var(--series-total)"></span>Total (realized + unrealized)</span>
      <span class="legend-item"><span class="swatch" style="background:var(--series-realized)"></span>Realized</span>
      <span class="legend-item"><span class="swatch" style="background:var(--series-unrealized)"></span>Unrealized</span>
    </div>
    <div style="position:relative;">
      <svg viewBox="0 0 1040 300" style="width:100%; height:auto; display:block;" id="mtmSvg">
        <g id="mtmGrid"></g>
        <line id="mtmZero" x1="56" y1="0" x2="1024" y2="0" stroke="var(--baseline)" stroke-width="1"/>
        <path id="mtmPathRealized" fill="none" stroke="var(--series-realized)" stroke-width="2" stroke-linecap="round"/>
        <path id="mtmPathUnrealized" fill="none" stroke="var(--series-unrealized)" stroke-width="2" stroke-linecap="round"/>
        <path id="mtmPathTotal" fill="none" stroke="var(--series-total)" stroke-width="2.5" stroke-linecap="round"/>
        <line id="crosshair-mtm" x1="0" y1="16" x2="0" y2="272" stroke="var(--ink-muted)"
              stroke-width="1" stroke-dasharray="3,3" style="opacity:0;"/>
        <rect id="hoverRect-mtm" x="56" y="16" width="968" height="256" fill="transparent"/>
      </svg>
      <div id="tooltip-mtm" class="tooltip"></div>
    </div>
    <script>
      (function() {{
        var ALL = {data_json};
        var OPEN_MTM = {open_mtm_json};  // {{symbol: [[date, value], ...]}}, entry date to today
        var CALENDAR_DATES = ALL.map(function(c) {{ return c.date; }});  // stable even if ALL is replaced
        var W = 1040, H = 300, PAD_L = 56, PAD_R = 16, PAD_T = 16, PAD_B = 28;
        var svg = document.getElementById('mtmSvg');
        var hoverRect = document.getElementById('hoverRect-mtm');
        var tooltip = document.getElementById('tooltip-mtm');
        var crosshair = document.getElementById('crosshair-mtm');
        var current = ALL;  // the currently-displayed (filtered) slice

        function fmt(v) {{
          return (v >= 0 ? '+$' : '-$') + Math.abs(v).toLocaleString('en-US', {{maximumFractionDigits: 0}});
        }}

        function render(startStr, endStr) {{
          var sliced = ALL.filter(function(c) {{
            return c.date >= startStr && (!endStr || c.date <= endStr);
          }});
          if (sliced.length < 2) {{ sliced = ALL.slice(-2); }}

          // Rebase so every line starts at zero within the selected
          // window, rather than carrying forward wherever the running
          // total happened to stand before this window began.
          var base = sliced[0];
          current = sliced.map(function(c) {{
            return {{
              date: c.date,
              realized: c.realized - base.realized,
              unrealized: c.unrealized - base.unrealized,
              total: c.total - base.total,
            }};
          }});

          var n = current.length;
          var allVals = [];
          current.forEach(function(c) {{ allVals.push(c.realized, c.unrealized, c.total); }});
          var lo = Math.min(0, Math.min.apply(null, allVals));
          var hi = Math.max.apply(null, allVals);
          var span = (hi - lo) || 1;

          function x(i) {{ return PAD_L + (W - PAD_L - PAD_R) * i / (n - 1); }}
          function y(v) {{ return PAD_T + (H - PAD_T - PAD_B) * (1 - (v - lo) / span); }}

          function pathFor(key) {{
            var pts = current.map(function(c, i) {{ return x(i).toFixed(1) + ',' + y(c[key]).toFixed(1); }});
            return 'M ' + pts.join(' L ');
          }}

          document.getElementById('mtmPathRealized').setAttribute('d', pathFor('realized'));
          document.getElementById('mtmPathUnrealized').setAttribute('d', pathFor('unrealized'));
          document.getElementById('mtmPathTotal').setAttribute('d', pathFor('total'));

          var zeroY = y(0);
          document.getElementById('mtmZero').setAttribute('y1', zeroY);
          document.getElementById('mtmZero').setAttribute('y2', zeroY);

          var gridHtml = '';
          [0, 0.25, 0.5, 0.75, 1].forEach(function(frac) {{
            var gy = PAD_T + (H - PAD_T - PAD_B) * frac;
            var val = hi - span * frac;
            gridHtml += '<line x1="' + PAD_L + '" y1="' + gy.toFixed(1) + '" x2="' + (W - PAD_R) + '" y2="' + gy.toFixed(1) +
              '" stroke="var(--grid)" stroke-width="1"/>' +
              '<text x="' + (PAD_L - 8) + '" y="' + (gy + 3).toFixed(1) + '" text-anchor="end" font-size="10" ' +
              'fill="var(--ink-muted)" class="num">$' + val.toLocaleString('en-US', {{maximumFractionDigits: 0}}) + '</text>';
          }});
          document.getElementById('mtmGrid').innerHTML = gridHtml;
        }}

        hoverRect.addEventListener('mousemove', function(evt) {{
          var n = current.length;
          var svgRect = svg.getBoundingClientRect();
          var relX = (evt.clientX - svgRect.left) / svgRect.width * W;
          var i = Math.round((relX - PAD_L) / (W - PAD_L - PAD_R) * (n - 1));
          i = Math.max(0, Math.min(n - 1, i));
          var px = PAD_L + (W - PAD_L - PAD_R) * i / (n - 1);
          var cxScreen = px / W * svgRect.width;
          var cyScreen = evt.clientY - svgRect.top;
          var c = current[i];
          tooltip.innerHTML = c.date + '<br>Total: ' + fmt(c.total) +
                               '<br>Realized: ' + fmt(c.realized) + '<br>Unrealized: ' + fmt(c.unrealized);
          tooltip.style.left = (cxScreen + 12) + 'px';
          tooltip.style.top = (cyScreen - 52) + 'px';
          tooltip.style.opacity = 1;
          crosshair.setAttribute('x1', px);
          crosshair.setAttribute('x2', px);
          crosshair.style.opacity = 1;
        }});
        hoverRect.addEventListener('mouseleave', function() {{
          tooltip.style.opacity = 0;
          crosshair.style.opacity = 0;
        }});

        // The page-level toggle bar (build_body) drives this, not this
        // chart script itself -- it calls window.mtmRender(start, end)
        // whenever the selected window changes, so the chart stays in
        // sync with the stat tiles and tables under one shared control.
        // mtmSetData lets the P/E filter push in a recomputed (filtered)
        // curve before calling mtmRender, using mtmOpenMtm/mtmCalendarDates
        // to do that recomputation itself.
        window.mtmRender = render;
        window.mtmAll = ALL;
        window.mtmOriginalAll = ALL;  // stable reference to restore when the P/E filter clears
        window.mtmSetData = function(newAll) {{ ALL = newAll; }};
        window.mtmOpenMtm = OPEN_MTM;
        window.mtmCalendarDates = CALENDAR_DATES;
      }})();
    </script>
    """


def _table_section(title: str, rows: list[dict], status: str, initial_count: int) -> str:
    table_id = f"tbl-{status}"
    search_id = f"search-{status}"
    count_id = f"count-{status}"
    header_count_id = f"header-count-{status}"

    if status == "open":
        cols = [("symbol", "Symbol"), ("entry_date", "Entry"), ("entry_price", "Entry $"),
                ("exit_price", "Now $"), ("roi", "Unrlzd ROI"), ("hold_days", "Days held"),
                ("pe_at_entry", "P/E @ entry")]
    else:
        cols = [("symbol", "Symbol"), ("entry_date", "Entry"), ("exit_date", "Exit"),
                ("entry_price", "Entry $"), ("exit_price", "Exit $"), ("roi", "ROI"),
                ("hold_days", "Hold days"), ("pe_at_entry", "P/E @ entry")]

    header_cells = "".join(
        f'<th class="{"num" if k in ("entry_price","exit_price","roi","hold_days","pe_at_entry") else ""}" '
        f'data-key="{k}" data-tbl="{table_id}">{label}</th>'
        for k, label in cols
    )

    return f"""
    <h2>{title} <span class="num" id="{header_count_id}" style="color:var(--ink-muted); font-size:0.85rem;">({initial_count})</span></h2>
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
  var TODAY = "{today}";
  // The window-filtered (by the shared toggle) subset each table/tile
  // actually reads from -- separate from DATA itself, which always holds
  // the complete, unfiltered dataset so any window can be re-selected.
  var WINDOWED = { open: DATA.open, closed: DATA.closed };

  function fmtRoi(v) {
    var cls = v >= 0 ? 'good' : 'critical';
    return '<span class="roi ' + cls + '">' + (v >= 0 ? '+' : '') + v.toFixed(2) + '%</span>';
  }
  function fmtMoney(v) {
    return (v >= 0 ? '+$' : '-$') + Math.abs(v).toLocaleString('en-US', {maximumFractionDigits: 0});
  }
  function setTileValue(id, text, cls) {
    var el = document.getElementById(id);
    if (!el) return;
    el.textContent = text;
    if (cls) el.className = 'value ' + cls;
  }

  function fmtPe(v) {
    return (v === null || v === undefined) ? '&mdash;' : v.toFixed(1);
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
          '<td class="num">' + fmtPe(r.pe_at_entry) + '</td>' +
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
        '<td class="num">' + fmtPe(r.pe_at_entry) + '</td>' +
        '</tr>';
    }).join('');
  }

  var sortState = {};
  function sortRows(status, key) {
    var rows = WINDOWED[status].slice();
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
    WINDOWED[status] = rows;
    applyFilter(status);
  }

  function applyFilter(status) {
    var tblId = 'tbl-' + status, searchId = 'search-' + status, countId = 'count-' + status;
    var q = (document.getElementById(searchId).value || '').trim().toUpperCase();
    var base = WINDOWED[status];
    var rows = q ? base.filter(function(r) { return r.symbol.indexOf(q) !== -1; }) : base;
    renderTable(tblId, rows, status);
    document.getElementById(countId).textContent = rows.length + ' of ' + base.length;
  }

  function filterTable(tblId, searchId, countId) {
    var status = tblId.replace('tbl-', '');
    applyFilter(status);
  }

  // Tracks the currently-selected date window so the global P/E filter
  // (which lives in the toggle bar, not per-table) can re-apply it without
  // needing its own copy of which toggle button is active.
  var currentWindowStart = null, currentWindowEnd = null;

  function applyDateWindow(startStr, endStr) {
    currentWindowStart = startStr;
    currentWindowEnd = endStr;

    var peMinEl = document.getElementById('pe-min-global'), peMaxEl = document.getElementById('pe-max-global');
    var peMin = peMinEl && peMinEl.value !== '' ? parseFloat(peMinEl.value) : null;
    var peMax = peMaxEl && peMaxEl.value !== '' ? parseFloat(peMaxEl.value) : null;
    function peOk(r) {
      if (peMin !== null && (r.pe_at_entry === null || r.pe_at_entry < peMin)) return false;
      if (peMax !== null && (r.pe_at_entry === null || r.pe_at_entry > peMax)) return false;
      return true;
    }

    WINDOWED.open = DATA.open.filter(function(r) {
      return r.entry_date >= startStr && (!endStr || r.entry_date <= endStr) && peOk(r);
    });
    WINDOWED.closed = DATA.closed.filter(function(r) {
      return r.exit_date >= startStr && (!endStr || r.exit_date <= endStr) && peOk(r);
    });
    sortState = {};

    document.getElementById('header-count-open').textContent = '(' + WINDOWED.open.length + ')';
    document.getElementById('header-count-closed').textContent = '(' + WINDOWED.closed.length + ')';
    applyFilter('open');
    applyFilter('closed');

    var closedRois = WINDOWED.closed.map(function(r) { return r.roi; });
    var openRois = WINDOWED.open.map(function(r) { return r.roi; });
    var winRate = closedRois.length ? closedRois.filter(function(r) { return r > 0; }).length / closedRois.length : 0;
    var avgRoiClosed = closedRois.length ? closedRois.reduce(function(a, b) { return a + b; }, 0) / closedRois.length : 0;
    var avgRoiOpen = openRois.length ? openRois.reduce(function(a, b) { return a + b; }, 0) / openRois.length : 0;
    var totalPnl = closedRois.reduce(function(a, b) { return a + b; }, 0) * 10;  // roi is %, normalized to $1000/trade

    setTileValue('tileOpenCount', WINDOWED.open.length);
    setTileValue('tileClosedCount', WINDOWED.closed.length);
    setTileValue('tileWinRate', (winRate * 100).toFixed(1) + '%');
    setTileValue('tileAvgRoiClosed', (avgRoiClosed >= 0 ? '+' : '') + avgRoiClosed.toFixed(2) + '%', avgRoiClosed >= 0 ? 'good' : 'critical');
    setTileValue('tileAvgRoiOpen', (avgRoiOpen >= 0 ? '+' : '') + avgRoiOpen.toFixed(2) + '%', avgRoiOpen >= 0 ? 'good' : 'critical');
    setTileValue('tileTotalPnl', fmtMoney(totalPnl), totalPnl >= 0 ? 'good num' : 'critical num');
    document.getElementById('tileTotalPnlSub').textContent = 'across ' + WINDOWED.closed.length + ' trades, $1,000 risk each';

    // Chart: when no P/E filter is active, restore Python's original
    // (unfiltered) curve exactly rather than recomputing it -- cheaper and
    // avoids any drift between the two code paths. When a filter IS
    // active, rebuild realized/unrealized/total from only the trades that
    // pass it, over the FULL calendar (render() does its own date-window
    // slicing and rebasing on top of whatever curve it's given).
    if (window.mtmSetData && window.mtmRender) {
      if (peMin === null && peMax === null) {
        if (window.mtmOriginalAll) window.mtmSetData(window.mtmOriginalAll);
      } else {
        window.mtmSetData(recomputeChartCurve(peOk));
      }
      window.mtmRender(startStr, endStr);
    }

    // CAGR, computed the honest way (not by extrapolating trade frequency --
    // see method1's own earlier false-start on that): compound the window's
    // ACTUAL $ change (realized from WINDOWED.closed + each WINDOWED.open
    // position's own unrealized ROI) over its REAL elapsed calendar days,
    // against a stated capital base of $1,000 x the average number of
    // positions concurrently held -- both computed directly from the same
    // date+P/E-filtered WINDOWED trades the tiles/tables use, via total
    // time each trade overlaps the window (equivalent to averaging a daily
    // active-count series, without needing to build one).
    var cagrEl = document.getElementById('tileCagr'), subEl = document.getElementById('tileCagrSub');
    if (cagrEl) {
      var windowEndDate = endStr || TODAY;
      var windowStartMs = new Date(startStr).getTime(), windowEndMs = new Date(windowEndDate).getTime();
      var windowDays = (windowEndMs - windowStartMs) / 86400000;

      function overlapDays(entryStr, exitStrOrNull) {
        var entryMs = new Date(entryStr).getTime();
        var exitMs = exitStrOrNull ? new Date(exitStrOrNull).getTime() : windowEndMs;
        var startMs = Math.max(entryMs, windowStartMs);
        var endMs = Math.min(exitMs, windowEndMs);
        var days = (endMs - startMs) / 86400000;
        return days > 0 ? days : 0;
      }

      var totalOverlapDays = 0;
      WINDOWED.open.concat(WINDOWED.closed).forEach(function(r) {
        totalOverlapDays += overlapDays(r.entry_date, r.exit_date);
      });
      var avgNActive = windowDays > 0 ? totalOverlapDays / windowDays : 0;
      var capitalBase = avgNActive * 1000;
      var finalTotal = totalPnl + openRois.reduce(function(a, b) { return a + b; }, 0) * 10;

      var cagr = null;
      if (capitalBase > 0 && windowDays > 0) {
        var totalReturnFrac = finalTotal / capitalBase;
        if (totalReturnFrac > -1) { cagr = Math.pow(1 + totalReturnFrac, 365 / windowDays) - 1; }
      }
      if (cagr !== null) {
        setTileValue('tileCagr', (cagr >= 0 ? '+' : '') + (cagr * 100).toFixed(1) + '%', cagr >= 0 ? 'good' : 'critical');
      } else {
        setTileValue('tileCagr', 'n/a');
      }
      if (subEl) {
        subEl.textContent = 'capital base: $' + Math.round(capitalBase).toLocaleString() +
          ' (avg ' + avgNActive.toFixed(1) + ' positions × $1,000), ' + Math.round(windowDays) + ' days';
      }
    }
  }

  // Rebuilds the full-calendar realized/unrealized/total curve from only
  // the trades that pass peOk -- the chart's equivalent of WINDOWED, just
  // not date-restricted (render() does that part itself).
  function recomputeChartCurve(peOk) {
    var calendarDates = window.mtmCalendarDates || [];
    var n = calendarDates.length;
    var openFiltered = DATA.open.filter(peOk);
    var closedFiltered = DATA.closed.filter(peOk);

    var byExitDate = {};
    closedFiltered.forEach(function(r) {
      byExitDate[r.exit_date] = (byExitDate[r.exit_date] || 0) + r.roi * 10;
    });
    var realized = new Array(n), running = 0;
    for (var i = 0; i < n; i++) {
      var d = calendarDates[i];
      if (byExitDate.hasOwnProperty(d)) { running += byExitDate[d]; }
      realized[i] = running;
    }

    var unrealized = new Array(n).fill(0);
    var openMtm = window.mtmOpenMtm || {};
    openFiltered.forEach(function(r) {
      var series = openMtm[r.symbol];
      if (!series) return;
      var si = 0, lastVal = 0, started = false;
      for (var i = 0; i < n; i++) {
        var d = calendarDates[i];
        while (si < series.length && series[si][0] <= d) { lastVal = series[si][1]; si++; started = true; }
        if (started) unrealized[i] += lastVal;
      }
    });

    var curve = new Array(n);
    for (var i = 0; i < n; i++) {
      curve[i] = { date: calendarDates[i], realized: realized[i], unrealized: unrealized[i], total: realized[i] + unrealized[i] };
    }
    return curve;
  }

  var liveInput = document.getElementById('liveDateInput');
  function setActiveToggle(btn) {
    document.querySelectorAll('.toggle-btn').forEach(function(b) { b.classList.remove('active'); });
    if (btn) btn.classList.add('active');
  }
  document.querySelectorAll('.toggle-btn').forEach(function(btn) {
    btn.addEventListener('click', function() {
      var range = btn.getAttribute('data-range');
      if (range === 'live') {
        liveInput.hidden = false;
        liveInput.focus();
        setActiveToggle(btn);
        return;
      }
      liveInput.hidden = true;
      setActiveToggle(btn);
      if (range === 'ytd') {
        applyDateWindow(TODAY.slice(0, 4) + '-01-01', null);
      } else {
        applyDateWindow(range + '-01-01', range + '-12-31');
      }
    });
  });
  liveInput.addEventListener('change', function() {
    if (liveInput.value) { applyDateWindow(liveInput.value, null); }
  });

  // The global P/E filter re-applies the CURRENT date window on every
  // edit, rather than needing its own copy of which toggle is active.
  ['pe-min-global', 'pe-max-global'].forEach(function(id) {
    document.getElementById(id).addEventListener('input', function() {
      if (currentWindowStart !== null) { applyDateWindow(currentWindowStart, currentWindowEnd); }
    });
  });

  document.querySelectorAll('table thead th').forEach(function(th) {
    th.addEventListener('click', function() {
      sortRows(th.getAttribute('data-tbl').replace('tbl-', ''), th.getAttribute('data-key'));
    });
  });

  // Default view: current year to date.
  setActiveToggle(document.querySelector('.toggle-btn[data-range="ytd"]'));
  applyDateWindow(TODAY.slice(0, 4) + '-01-01', null);
</script>
"""


def build_body(stats: dict, open_rows: list[dict], closed_rows: list[dict],
               open_count_ytd: int, closed_count_ytd: int,
               mtm_curve: list[dict], open_mtm_series: dict, today: str) -> str:
    win_rate = stats["win_rate"] or 0
    avg_roi_closed = (stats["avg_roi_closed"] or 0) * 100
    avg_roi_open = (stats["avg_roi_open"] or 0) * 100
    total_pnl = stats["total_normalized_pnl"]

    data_json = json.dumps({"open": open_rows, "closed": closed_rows})
    year_buttons = _year_buttons_html(mtm_curve[0]["date"], mtm_curve[-1]["date"]) if len(mtm_curve) >= 2 else ""
    live_min = mtm_curve[0]["date"] if mtm_curve else today
    live_max = mtm_curve[-1]["date"] if mtm_curve else today

    return f"""
<div class="wrap">
  <h1>RSI Trade Log</h1>
  <div class="subtitle">RSI(14) mean reversion &middot; S&amp;P 500 + core ETFs/index</div>
  <div class="asof">As of {today} &middot; signal only, no orders placed &middot; normalized to $1,000 risked per trade</div>

  <div class="togglebar">
    {year_buttons}
    <button class="toggle-btn" data-range="ytd">{today[:4]} YTD</button>
    <button class="toggle-btn" data-range="live">Live from...</button>
    <input type="date" id="liveDateInput" min="{live_min}" max="{live_max}" hidden>
    <label class="pe-filter-label">P/E @ entry
      <input type="number" id="pe-min-global" placeholder="min">
      <span>&ndash;</span>
      <input type="number" id="pe-max-global" placeholder="max">
    </label>
  </div>
  <div class="chart-note" style="margin-top:-6px; margin-bottom: 14px;">P/E filter applies to the stat tiles, chart, and tables below.</div>

  <div class="tiles">
    <div class="tile"><div class="label">Open positions</div><div class="value" id="tileOpenCount">{open_count_ytd}</div></div>
    <div class="tile"><div class="label">Closed trades</div><div class="value" id="tileClosedCount">{closed_count_ytd}</div></div>
    <div class="tile"><div class="label">Win rate (closed)</div><div class="value" id="tileWinRate">{win_rate*100:.1f}%</div></div>
    <div class="tile"><div class="label">Avg ROI / closed trade</div>
      <div class="value {'good' if avg_roi_closed>=0 else 'critical'}" id="tileAvgRoiClosed">{avg_roi_closed:+.2f}%</div></div>
    <div class="tile"><div class="label">Avg unrealized ROI (open)</div>
      <div class="value {'good' if avg_roi_open>=0 else 'critical'}" id="tileAvgRoiOpen">{avg_roi_open:+.2f}%</div></div>
    <div class="tile"><div class="label">Total realized P&amp;L</div>
      <div class="value {'good' if total_pnl>=0 else 'critical'} num" id="tileTotalPnl">${total_pnl:,.0f}</div>
      <div class="sub" id="tileTotalPnlSub">across {closed_count_ytd} trades, $1,000 risk each</div></div>
    <div class="tile"><div class="label">Annualized (CAGR)</div>
      <div class="value" id="tileCagr">--</div>
      <div class="sub" id="tileCagrSub"></div></div>
  </div>

  <h2>Realized, unrealized &amp; total P&amp;L</h2>
  <div class="panel">
    {_svg_mtm_chart(mtm_curve, open_mtm_series)}
    <div class="chart-note">All three lines normalized to $1,000/position, on the same calendar axis. Realized is a step function
      (flat between exits, jumps on each one); unrealized is the mark-to-market value of positions still open at each date
      (0 before a position's own entry date); total is their sum &mdash; today's rightmost point is your actual current P&amp;L.</div>
  </div>

  {_table_section("Open positions", open_rows, "open", open_count_ytd)}
  {_table_section("Closed positions", closed_rows, "closed", closed_count_ytd)}

  <footer>
    Strategy: buy when RSI(14) &le; 20 (oversold), hold until RSI &ge; 65, flat otherwise &mdash;
    validated via walk-forward backtesting across the S&amp;P 500 + core ETFs/index, then chosen over the
    original 30/55 textbook defaults via a grid search that replicated out-of-sample (see method1/threshold_grid.py
    and threshold_train_test.py). This page is a reporting view of that signal, not investment advice.
    Entry/exit prices use the next trading day's OPEN after the signal-crossing close, since a close that
    triggers RSI crossing a threshold isn't itself a tradeable price.
    <br><br>
    P/E @ entry is an approximation (entry price &divide; CURRENT trailing EPS, not the EPS that was actually
    current on the entry date) -- there's no free historical EPS series to compute it exactly. It's reasonably
    accurate for trades entered in the last ~2 quarters and increasingly approximate for older ones, since EPS
    changes every earnings report. Trades in unprofitable companies (negative/missing EPS) show as &mdash; and
    are excluded by the P/E filter, not treated as having a P/E of zero.
  </footer>
</div>
{BODY_SCRIPT_TEMPLATE.replace("{data_json}", data_json).replace("{today}", today)}
"""


def main():
    # The tiles, chart, and tables are now all driven by the SAME toggle
    # (2025 / YTD / Live) -- so all three need the full extended range
    # embedded in JS, not just a trailing-12-month slice. A separate
    # trade_log_12mo.csv snapshot is still written for anyone consuming
    # the data outside the page.
    log, open_mtm_series = build_trade_log_with_mtm()
    twelve_months_ago = pd.Timestamp.today().normalize() - pd.DateOffset(months=12)
    log_12mo = log[(log["status"] == "open") | (log["exit_date"] >= twelve_months_ago)]
    log_12mo.to_csv(Path(__file__).with_name("trade_log_12mo.csv"), index=False)

    today_ts = pd.Timestamp.today().normalize()
    today = today_ts.strftime("%Y-%m-%d")
    ytd_start = pd.Timestamp(year=today_ts.year, month=1, day=1)
    log_ytd = log[(log["status"] == "open") & (log["entry_date"] >= ytd_start)
                  | (log["status"] == "closed") & (log["exit_date"] >= ytd_start)]

    stats = summarize(log_ytd)  # initial server-rendered paint matches JS's own YTD default
    mtm_curve = _mtm_curve(log, open_mtm_series)
    eps_by_symbol = get_trailing_eps(sorted(log["symbol"].unique()))
    open_rows = _rows_for_js(log, "open", eps_by_symbol)
    closed_rows = _rows_for_js(log, "closed", eps_by_symbol)
    open_count_ytd = int((log_ytd["status"] == "open").sum())
    closed_count_ytd = int((log_ytd["status"] == "closed").sum())

    body = build_body(stats, open_rows, closed_rows, open_count_ytd, closed_count_ytd, mtm_curve, open_mtm_series, today)

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

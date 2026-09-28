"""Self-contained HTML report.

No CDN, no external stylesheet, no build step -- everything is inlined so the file can be
copied off the box, emailed, or opened from a share and still render. Charts are generated as
plain SVG in Python rather than by a charting library, for the same reason.

Deliberate presentation choices:
  * The answer (best config + whether it met the constraints) comes first, above the data.
  * Offline rows sit in their own section behind an explicit banner, because they do not
    share a measurement boundary across backends and putting them in the same sortable table
    as online rows would invite exactly the comparison that is invalid.
  * Missing metrics render blank, never 0.
"""
from __future__ import annotations

import html
import json
from typing import Any

from .common import (LOWER_IS_BETTER, ReportContext, cell, fmt, host_lines, rows_for,
                     server_commands, shell_command, used_columns)

CSS = """
pre{white-space:pre-wrap;word-break:break-all;background:var(--chip);border:1px solid var(--line);
padding:.55rem .75rem;border-radius:8px;font:12px/1.5 ui-monospace,Menlo,monospace}
:root{--bg:#fbfbfa;--fg:#1a1a19;--muted:#6b6b68;--line:#e3e3e0;--card:#fff;
--accent:#2f6f4e;--warn:#8a5a00;--bad:#a33;--good:#2f6f4e;--chip:#f0f0ed;--bar:#5b8c72;}
@media (prefers-color-scheme:dark){:root{--bg:#16171a;--fg:#e8e8e6;--muted:#9a9a96;
--line:#2c2e33;--card:#1d1f23;--accent:#7fc0a0;--warn:#d8a33a;--bad:#e07a7a;--good:#7fc0a0;
--chip:#25272c;--bar:#5b8c72;}}
*{box-sizing:border-box}
body{margin:0;padding:2rem 1.25rem 5rem;background:var(--bg);color:var(--fg);
font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.wrap{max-width:1180px;margin:0 auto}
h1{font-size:1.7rem;margin:0 0 .3rem;letter-spacing:-.01em}
h2{font-size:1.2rem;margin:2.5rem 0 .75rem;padding-bottom:.35rem;border-bottom:1px solid var(--line)}
h3{font-size:1rem;margin:1.5rem 0 .5rem}
p{margin:.5rem 0}
.sub{color:var(--muted);font-size:.9rem;margin-bottom:1.5rem}
.meta{display:flex;flex-wrap:wrap;gap:.4rem;margin:.75rem 0 0}
.chip{background:var(--chip);border:1px solid var(--line);border-radius:999px;
padding:.15rem .6rem;font-size:.8rem;color:var(--muted);white-space:nowrap}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:1.1rem 1.25rem;margin:1rem 0}
.hero{border-left:4px solid var(--accent)}
.hero .val{font-size:1.5rem;font-weight:600;letter-spacing:-.01em}
.hero .cfg{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.9rem;
color:var(--muted);margin-top:.3rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.75rem;margin-top:.9rem}
.kv{background:var(--chip);border-radius:8px;padding:.55rem .7rem}
.kv .k{font-size:.72rem;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
.kv .v{font-size:1rem;font-weight:600;font-family:ui-monospace,Menlo,monospace}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;border:1px solid var(--line);
border-radius:8px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:.83rem}
th,td{padding:.4rem .6rem;text-align:left;white-space:nowrap;border-bottom:1px solid var(--line)}
th{position:sticky;top:0;background:var(--card);font-weight:600;font-size:.75rem;
text-transform:uppercase;letter-spacing:.03em;color:var(--muted);cursor:pointer;user-select:none}
th:hover{color:var(--fg)}
th.sorted::after{content:" \\2193";opacity:.6}
th.sorted.asc::after{content:" \\2191"}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;
font-family:ui-monospace,Menlo,monospace}
tbody tr:hover{background:var(--chip)}
.pass{color:var(--good);font-weight:600}
.fail{color:var(--bad);font-weight:600}
.banner{border-left:4px solid var(--warn);background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:.85rem 1.1rem;margin:1rem 0;font-size:.9rem}
.banner b{color:var(--warn)}
.note{color:var(--muted);font-size:.88rem;margin:.5rem 0}
ul.warn{margin:.5rem 0;padding-left:1.2rem;font-size:.88rem;color:var(--muted)}
ul.warn li{margin:.25rem 0}
.chart{margin:1rem 0}
.chart svg{max-width:100%;height:auto;display:block}
.filter{margin:.75rem 0}
.filter input{width:100%;max-width:340px;padding:.45rem .7rem;border-radius:8px;
border:1px solid var(--line);background:var(--card);color:var(--fg);font-size:.88rem}
.legend{display:flex;flex-wrap:wrap;gap:.75rem;font-size:.78rem;color:var(--muted);margin:.4rem 0}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:.3rem}
code{font-family:ui-monospace,Menlo,monospace;background:var(--chip);padding:.1rem .3rem;
border-radius:4px;font-size:.85em}
footer{margin-top:3rem;color:var(--muted);font-size:.8rem;border-top:1px solid var(--line);
padding-top:1rem}
"""

JS = """
function sortTable(th){
  const table=th.closest('table'), tb=table.tBodies[0];
  const idx=[...th.parentNode.children].indexOf(th);
  const asc=!(th.classList.contains('sorted')&&!th.classList.contains('asc'));
  [...th.parentNode.children].forEach(h=>h.classList.remove('sorted','asc'));
  th.classList.add('sorted'); if(asc) th.classList.add('asc');
  const num=th.classList.contains('num');
  const rows=[...tb.rows];
  rows.sort((a,b)=>{
    let x=a.cells[idx].textContent.trim(), y=b.cells[idx].textContent.trim();
    if(x===''&&y==='')return 0; if(x==='')return 1; if(y==='')return -1;
    if(num){x=parseFloat(x.replace(/,/g,''));y=parseFloat(y.replace(/,/g,''));
      return asc?x-y:y-x;}
    return asc?x.localeCompare(y):y.localeCompare(x);
  });
  rows.forEach(r=>tb.appendChild(r));
}
function filterTable(input, tableId){
  const q=input.value.toLowerCase(), tb=document.getElementById(tableId).tBodies[0];
  let shown=0;
  [...tb.rows].forEach(r=>{
    const hit=r.textContent.toLowerCase().includes(q);
    r.style.display=hit?'':'none'; if(hit)shown++;
  });
  const c=document.getElementById(tableId+'-count'); if(c)c.textContent=shown+' rows';
}
document.addEventListener('DOMContentLoaded',()=>{
  document.querySelectorAll('table.sortable th').forEach(th=>{
    th.addEventListener('click',()=>sortTable(th));
  });
});
"""

PALETTE = ["#5b8c72", "#4a7fa5", "#a5794a", "#8a5f9e", "#a35a5a", "#4f9e9e",
           "#9e8a4f", "#6b6f9e", "#7a9e4f", "#9e4f7a"]


def _e(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def _bar_chart(title: str, series: list[tuple[str, float]], *, unit: str = "",
               lower_better: bool = False) -> str:
    """Horizontal bar chart as inline SVG. Bars are sorted best-first."""
    if not series:
        return ""
    series = sorted(series, key=lambda kv: kv[1], reverse=not lower_better)
    top = max(v for _, v in series) or 1.0
    row_h, pad_l, pad_t = 24, 260, 28
    width, height = 940, pad_t + row_h * len(series) + 12
    bar_w = width - pad_l - 90

    parts = [
        f'<div class="chart"><svg viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{_e(title)}" xmlns="http://www.w3.org/2000/svg">',
        f'<text x="0" y="16" font-size="13" font-weight="600" fill="currentColor">{_e(title)}'
        f'{f" ({_e(unit)})" if unit else ""}</text>',
    ]
    for i, (label, value) in enumerate(series):
        y = pad_t + i * row_h
        w = max(1.0, (value / top) * bar_w)
        colour = PALETTE[i % len(PALETTE)]
        parts.append(
            f'<text x="{pad_l - 8}" y="{y + 13}" font-size="11" text-anchor="end" '
            f'fill="currentColor" opacity=".75" font-family="ui-monospace,Menlo,monospace">'
            f'{_e(label[:44])}</text>'
            f'<rect x="{pad_l}" y="{y + 3}" width="{w:.1f}" height="{row_h - 8}" rx="3" '
            f'fill="{colour}" opacity=".85"/>'
            f'<text x="{pad_l + w + 6:.1f}" y="{y + 13}" font-size="11" fill="currentColor" '
            f'font-family="ui-monospace,Menlo,monospace">{value:,.2f}</text>'
        )
    parts.append("</svg></div>")
    return "".join(parts)


def _table_html(headers: list[str], rows: list[list[str]], *, table_id: str,
                numeric_cols: set[int], sortable: bool = True,
                value_classes: dict[str, str] | None = None) -> str:
    """`value_classes` maps an exact cell string to an extra CSS class (pass/FAIL colouring).

    It is applied here rather than by string-replacing the finished table, which is what an
    earlier version did: that produced a second `class=` attribute on a cell that already had
    one, and browsers keep the first, so the colours silently never rendered.
    """
    value_classes = value_classes or {}
    cls = "sortable" if sortable else ""
    head = "".join(
        f'<th class="{"num" if i in numeric_cols else ""}">{_e(h)}</th>'
        for i, h in enumerate(headers)
    )

    def td(i: int, v: str) -> str:
        classes = " ".join(c for c in ("num" if i in numeric_cols else "",
                                       value_classes.get(v, "")) if c)
        return f'<td class="{classes}">{_e(v)}</td>'

    body = "".join(
        "<tr>" + "".join(td(i, v) for i, v in enumerate(row)) + "</tr>"
        for row in rows
    )
    return (f'<div class="scroll"><table id="{table_id}" class="{cls}">'
            f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>")


def _numeric_columns(columns, results) -> set[int]:
    out = set()
    for i, (_, source, key) in enumerate(columns):
        values = [cell(r, source, key) for r in results]
        values = [v for v in values if v is not None and v != ""]
        if values and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            out.add(i)
    return out


def render_html(ctx: ReportContext) -> str:
    m = ctx.manifest
    rank = ctx.ranking
    topo = ctx.plan.get("topology", {})
    env = m.get("env", {})
    p: list[str] = []
    a = p.append

    a(f"<title>{_e(m.get('name', 'llmbench sweep'))} — llmbench report</title>")
    a(f"<style>{CSS}</style>")
    a('<div class="wrap">')

    # ---- header ----
    a(f"<h1>{_e(m.get('name', 'sweep'))}</h1>")
    a(f'<div class="sub">llmbench sweep report · run <code>{_e(m.get("run_id"))}</code></div>')
    a('<div class="meta">')
    for label, value in (
        ("host", env.get("hostname")),
        ("cpu", topo.get("model_name")),
        (f"{topo.get('n_physical')} cores", f"{topo.get('n_ccds')} CCDs × {topo.get('cores_per_ccd')}"),
        ("mode", m.get("mode")),
        ("started", m.get("started_at_utc", "")[:19].replace("T", " ")),
        ("elapsed", f"{m.get('elapsed_s')}s"),
        ("rows", len(ctx.results)),
    ):
        if value:
            a(f'<span class="chip">{_e(label)}: {_e(value)}</span>')
    for status, count in sorted(ctx.status_counts().items()):
        a(f'<span class="chip">{_e(status)}: {count}</span>')
    a("</div>")

    # ---- the answer ----
    a("<h2>Result</h2>")
    a(f'<p class="note">Objective: <code>{_e(rank.objective)}</code></p>')
    a('<div class="card hero">')
    a(f'<div class="val">{_e(rank.headline())}</div>')
    if rank.best is not None:
        a(f'<div class="cfg">{_e(rank.best.config_label)}</div>')
        a('<div class="grid">')
        for k in ("instances", "cores_per_instance", "threads_per_instance", "n_parallel",
                  "n_ctx", "batch", "ubatch", "lb", "concurrency"):
            v = rank.best.axes.get(k)
            if v is not None:
                a(f'<div class="kv"><div class="k">{_e(k.replace("_", " "))}</div>'
                  f'<div class="v">{_e(v)}</div></div>')
        a("</div>")
    a("</div>")

    if rank.best is not None and rank.best.verdicts:
        a("<h3>Constraints for the winning configuration</h3>")
        a(_table_html(
            ["constraint", "measured", "verdict"],
            [[v.described, fmt(v.value), "pass" if v.satisfied else "FAIL"]
             for v in rank.best.verdicts],
            table_id="constraints", numeric_cols={1}, sortable=False,
            value_classes={"pass": "pass", "FAIL": "fail"},
        ))

    for note in rank.notes:
        a(f'<div class="banner"><b>Note</b> — {_e(note)}</div>')

    # ---- charts ----
    if rank.best_per_test:
        a("<h2>Best configuration per workload</h2>")
        a('<p class="note">A single global winner hides real trade-offs; these are the '
          "per-workload answers.</p>")
        a(_table_html(
            ["test", "best config", rank.metric, "feasible"],
            [[t, c.config_label, fmt(c.value), "yes" if c.feasible else "no"]
             for t, c in sorted(rank.best_per_test.items())],
            table_id="perTest", numeric_cols={2},
        ))

    online_ok = [r for r in ctx.online if r.status == "ok" and r.src == rank.src]
    by_test: dict[str, list[tuple[str, float]]] = {}
    for r in online_ok:
        v = r.metrics.get(rank.metric)
        if isinstance(v, (int, float)):
            from ..objective import config_label as _cfg
            label = f"{_cfg(r.axes)} c{r.axes.get('concurrency')}"
            by_test.setdefault(r.test, []).append((label, float(v)))
    if by_test:
        a("<h2>Measured throughput by workload</h2>")
        a(f'<p class="note">Metric: <code>{_e(rank.metric)}</code>, '
          f'measurement path <code>src={_e(rank.src)}</code>. '
          f'{"Lower is better." if rank.metric in LOWER_IS_BETTER else "Higher is better."}</p>')
        for test in sorted(by_test):
            a(_bar_chart(test, by_test[test], unit=rank.metric,
                         lower_better=rank.metric in LOWER_IS_BETTER))

    # ---- ranked configs ----
    if rank.config_scores:
        a("<h2>Configurations ranked across the workload mix</h2>")
        a('<p class="note"><code>score</code> normalises each workload to that workload\'s '
          "winner and then averages, so a high-throughput workload cannot outvote a "
          "low-throughput one by scale alone. <code>1.000</code> means it won every workload."
          "</p>")
        a(_table_html(
            ["#", "config", "backend", "score", "tests", "feasible", "all feasible"],
            [[str(i + 1), s.config_label, s.backend, f"{s.normalised_score:.3f}",
              str(s.n_tests), str(s.n_feasible), "yes" if s.fully_feasible else "no"]
             for i, s in enumerate(rank.config_scores)],
            table_id="configs", numeric_cols={0, 3, 4, 5},
        ))

    if rank.pareto:
        a("<h2>Pareto front</h2>")
        a('<p class="note">Configurations that are not beaten on every axis at once.</p>')
        extra = [v.metric for v in rank.pareto[0].verdicts]
        a(_table_html(
            ["config", "test", rank.metric] + extra,
            [[c.config_label, c.test, fmt(c.value)] + [fmt(v.value) for v in c.verdicts]
             for c in rank.pareto],
            table_id="pareto", numeric_cols=set(range(2, 3 + len(extra))),
        ))

    # ---- full online table ----
    online = [r for r in ctx.online if r.status == "ok"]
    if online:
        a("<h2>Online results</h2>")
        a('<p class="note"><code>src=client</code> is wall-clock and comparable across '
          "backends. <code>src=server</code> is backend-reported — a diagnostic, not a "
          "cross-backend result.</p>")
        cols = used_columns(online)
        a('<div class="filter"><input type="text" placeholder="filter rows…" '
          'oninput="filterTable(this,\'onlineTable\')"> '
          f'<span class="note" id="onlineTable-count">{len(online)} rows</span></div>')
        a(_table_html([h for h, _, _ in cols], rows_for(online, cols),
                      table_id="onlineTable", numeric_cols=_numeric_columns(cols, online)))

    # ---- offline ----
    offline = [r for r in ctx.offline if r.status == "ok"]
    if offline:
        a("<h2>Offline results (native tools)</h2>")
        a('<div class="banner"><b>Not comparable across backends.</b> These rows come from '
          "each backend's own benchmark tool, and those tools measure different things: "
          "<code>llama-bench</code> times <code>llama_decode()</code> in-process with no "
          "scheduler and no HTTP, while <code>vllm bench latency</code> runs the full vLLM "
          "engine including its scheduler. Compare each backend's offline number to its own "
          "online number — that gap is its HTTP-plus-scheduler tax — never to the other "
          "backend's offline number.</div>")
        cols = used_columns(offline)
        a(_table_html([h for h, _, _ in cols], rows_for(offline, cols),
                      table_id="offlineTable", numeric_cols=_numeric_columns(cols, offline)))

    # ---- deployments / placement ----
    deployments = ctx.plan.get("deployments", [])
    if deployments:
        a("<h2>Core placement</h2>")
        rows = []
        for entry in deployments:
            d = entry.get("deployment", entry)
            for inst in d.get("instances", []):
                rows.append([
                    d.get("id", ""), d.get("axes", {}).get("backend", ""),
                    str(inst.get("index")), str(inst.get("port")), inst.get("cpus", ""),
                    str(inst.get("n_physical_cores")), str(inst.get("n_threads")),
                    ",".join(str(x) for x in inst.get("ccds", [])),
                    ",".join(str(x) for x in inst.get("membind", [])),
                    "yes" if inst.get("cross_socket") else "no",
                ])
        a(_table_html(
            ["deployment", "backend", "inst", "port", "cpus", "cores", "threads",
             "ccds", "membind", "cross-socket"],
            rows, table_id="placement", numeric_cols={2, 3, 5, 6},
        ))

    # ---- failures ----
    failures = ctx.failures
    if failures:
        a("<h2>Failed and skipped trials</h2>")
        a(_table_html(
            ["trial", "backend", "test", "status", "error"],
            [[r.trial_id, r.backend, r.test, r.status,
              (r.error or "").replace("\n", " ")[:200]] for r in failures],
            table_id="failures", numeric_cols=set(),
        ))

    warnings = ctx.all_warnings()
    if warnings:
        a("<h2>Warnings</h2><ul class='warn'>")
        for w in warnings:
            a(f"<li>{_e(w)}</li>")
        a("</ul>")

    # ---- provenance ----
    host = host_lines(m)
    if host:
        a("<h2>Host and software</h2>")
        a(_table_html(["", "value"], [[k, str(v)] for k, v in host],
                      table_id="host", numeric_cols=set(), sortable=False))

    commands = server_commands(ctx.results)
    if commands:
        a("<h2>Server commands</h2>")
        a('<p class="note">Exactly as launched. The variables in front are the ones the sweep '
          "set on top of its own environment.</p>")
        for dep_id, (backend, cmds) in commands.items():
            a(f"<h3>{_e(dep_id)} — {_e(backend)}</h3>")
            a("<pre>" + _e("\n".join(shell_command(c) for c in cmds)) + "</pre>")

    # ---- artifacts ----
    a("<h2>Artifacts</h2><ul class='warn'>")
    for label, path in (m.get("artifacts") or {}).items():
        a(f"<li><code>{_e(path)}</code> — {_e(label.replace('_', ' '))}</li>")
    a("</ul>")

    a("<footer>Generated by llmbench. Every number here is recomputed from the raw "
      "per-request records in <code>records/</code> and <code>offline.jsonl</code>; nothing "
      "is aggregated at collection time.</footer>")
    a("</div>")
    a(f"<script>{JS}</script>")
    return "\n".join(p)


__all__ = ["render_html"]

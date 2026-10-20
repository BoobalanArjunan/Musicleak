"""Turning a Report into text, JSON and CSV."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .scan import Report


def _t(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    return f"{m}:{s:02d}"


def _semis(n: int) -> str:
    return "same key" if n == 0 else f"{n:+d} semitones"


def report_to_text(rep: Report) -> str:
    c = rep.calibration
    lines = [f"musicleak: {len(rep.results)} query file(s) against {rep.settings.get('references', '?')} reference song(s): {rep.verdict()}"]
    if c.calibrated:
        lines.append(f"unrelated-song level measured on {c.n} windows from the {c.source}: lone-window level {c.single:.2f}, "
                     f"confirmed-match level {c.chain:.2f}")
    else:
        lines.append(f"levels NOT calibrated (fixed {c.single:.2f} / {c.chain:.2f}); see notes")
    lines.append("")
    for r in rep.results:
        v = r.verdict()
        if r.error:
            lines.append(f"{v:4s} {r.name}: {r.error}")
            continue
        head = f"{v:4s} {r.name} ({_t(r.duration_s)})"
        if v == "FAIL":
            head += f": {r.matched_s():.0f} s matched ({100 * r.matched_s() / r.duration_s:.0f}% of the song)"
        lines.append(head)
        for m in r.matches:
            lines.append(f"       {m.kind:9s} {_t(m.q_start)}-{_t(m.q_end)} matches {m.ref} {_t(m.r_start)}-{_t(m.r_end)}, "
                         f"{_semis(m.transposition)}, tempo x{m.tempo:.2f}, score {m.score:.2f}")
    if rep.notes:
        lines += ["", "Notes:"] + [f"  - {n}" for n in rep.notes]
    lines += ["", "FAIL = a stretch of about two windows (20 s by default) matches one reference song consistently. "
              "WARN = a lone window scored above every unrelated window we measured: look at it, it is also what chance produces now and then. "
              "Chroma finds the same notes, not the same recording: a cover or a re-recording also matches."]
    return "\n".join(lines)


def report_to_dict(rep: Report) -> dict:
    return {
        "verdict": rep.verdict(),
        "calibration": {"single": rep.calibration.single, "chain": rep.calibration.chain, "windows": rep.calibration.n,
                        "source": rep.calibration.source},
        "settings": rep.settings, "notes": rep.notes,
        "files": [{"name": r.name, "verdict": r.verdict(), "duration_s": r.duration_s, "matched_s": r.matched_s(), "error": r.error,
                   "matches": [{"kind": m.kind, "reference": m.ref, "query_start": m.q_start, "query_end": m.q_end,
                                "reference_start": m.r_start, "reference_end": m.r_end, "transposition_semitones": m.transposition,
                                "tempo_ratio": m.tempo, "score": m.score} for m in r.matches]} for r in rep.results],
    }


def write_reports(rep: Report, out: Path) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.txt").write_text(report_to_text(rep), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report_to_dict(rep), indent=2), encoding="utf-8")
    with open(out / "matches.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query", "kind", "query_start_s", "query_end_s", "reference", "reference_start_s", "reference_end_s",
                    "transposition_semitones", "tempo_ratio", "score"])
        for r in rep.results:
            for m in r.matches:
                w.writerow([r.name, m.kind, f"{m.q_start:.1f}", f"{m.q_end:.1f}", m.ref, f"{m.r_start:.1f}", f"{m.r_end:.1f}",
                            m.transposition, f"{m.tempo:.2f}", f"{m.score:.3f}"])


def report_to_html(rep: Report) -> str:
    """Render an interactive glassmorphic visual HTML report for MusicLeak."""
    data = report_to_dict(rep)
    verdict = rep.verdict()
    cal = rep.calibration

    total_files = len(rep.results)
    fail_count = sum(1 for r in rep.results if r.verdict() == "FAIL")
    warn_count = sum(1 for r in rep.results if r.verdict() == "WARN")
    pass_count = sum(1 for r in rep.results if r.verdict() == "PASS")
    error_count = sum(1 for r in rep.results if r.error)

    total_dur_s = sum(r.duration_s for r in rep.results if not r.error)
    total_matched_s = sum(r.matched_s() for r in rep.results if not r.error)
    leakage_pct = (100.0 * total_matched_s / total_dur_s) if total_dur_s > 0 else 0.0

    verdict_colors = {
        "FAIL": ("#ef4444", "rgba(239, 68, 68, 0.15)", "rgba(239, 68, 68, 0.4)", "LEAKAGE DETECTED"),
        "WARN": ("#f59e0b", "rgba(245, 158, 11, 0.15)", "rgba(245, 158, 11, 0.4)", "SUSPICIOUS SEGMENTS"),
        "PASS": ("#10b981", "rgba(16, 185, 129, 0.15)", "rgba(16, 185, 129, 0.4)", "CLEAN - NO LEAKAGE"),
    }
    v_color, v_bg, v_border, v_label = verdict_colors.get(verdict, ("#6b7280", "rgba(107, 114, 128, 0.15)", "rgba(107, 114, 128, 0.4)", verdict))

    data_json = json.dumps(data)

    cards_html = []
    for r in rep.results:
        v = r.verdict()
        tag_class = "tag-" + v.lower()
        dur_str = _t(r.duration_s) if r.duration_s else "0:00"
        matched_str = f"{r.matched_s():.1f}s" if r.duration_s else "0s"
        matched_pct = f"{100.0 * r.matched_s() / r.duration_s:.0f}%" if r.duration_s and r.duration_s > 0 else "0%"
        match_color = "#ef4444" if r.matched_s() > 0 else "var(--text)"

        card = f'''
    <div class="result-card" data-verdict="{v}" data-name="{r.name.lower()}">
      <div class="result-header">
        <div class="result-title">
          <span class="tag {tag_class}">{v}</span>
          <span>{r.name}</span>
          <span style="color: var(--text-muted); font-size: 13px; font-weight: normal;">({dur_str})</span>
        </div>
        <div style="font-size: 13px; color: var(--text-muted);">
          Matched: <strong style="color: {match_color}">{matched_str}</strong> ({matched_pct})
        </div>
      </div>
'''
        if r.error:
            card += f'<div style="color: #ef4444; font-size: 13px;">Error: {r.error}</div></div>'
            cards_html.append(card)
            continue

        if r.matches and r.duration_s > 0:
            card += '<div class="timeline-bar-bg">'
            for m in r.matches:
                left_pct = max(0.0, min(100.0, 100.0 * m.q_start / r.duration_s))
                width_pct = max(1.0, min(100.0 - left_pct, 100.0 * (m.q_end - m.q_start) / r.duration_s))
                seg_bg = "#ef4444" if m.kind == "confirmed" else "#f59e0b"
                card += f'<div class="timeline-seg" style="left: {left_pct:.1f}%; width: {width_pct:.1f}%; background: {seg_bg};"></div>'
            card += '</div>'

        if r.matches:
            card += '''
      <table class="match-table">
        <thead>
          <tr>
            <th>Kind</th>
            <th>Query Window</th>
            <th>Reference Song</th>
            <th>Ref Window</th>
            <th>Transposition</th>
            <th>Tempo</th>
            <th>Confidence Score</th>
          </tr>
        </thead>
        <tbody>
'''
            for m in r.matches:
                kind_style = "color: #ef4444;" if m.kind == "confirmed" else "color: #f59e0b;"
                q_range = f"{_t(m.q_start)} - {_t(m.q_end)}"
                r_range = f"{_t(m.r_start)} - {_t(m.r_end)}"
                shift_str = _semis(m.transposition)
                tempo_str = f"x{m.tempo:.2f}"
                score_pct = int(min(1.0, max(0.0, m.score)) * 100)
                card += f'''
          <tr>
            <td style="{kind_style} font-weight: 600;">{m.kind}</td>
            <td><code>{q_range}</code></td>
            <td><strong>{m.ref}</strong></td>
            <td><code>{r_range}</code></td>
            <td><span class="badge">{shift_str}</span></td>
            <td><span class="badge">{tempo_str}</span></td>
            <td>
              <div class="score-bar-bg"><div class="score-bar-fill" style="width: {score_pct}%;"></div></div>
              <span>{m.score:.3f}</span>
            </td>
          </tr>
'''
            card += '''
        </tbody>
      </table>
'''
        else:
            card += '<div style="font-size: 13px; color: var(--text-muted); margin-top: 6px;">No matching reference segments detected. Clean audio.</div>'

        card += '</div>'
        cards_html.append(card)

    rendered_cards = "\n".join(cards_html)
    cal_type = "Empirical" if cal.calibrated else "Fixed"

    # HTML template with pure string replacement to avoid javascript escaping issues
    template = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MusicLeak Audit Report - __VERDICT__</title>
<style>
  :root {
    --bg: #0b0f17;
    --card-bg: rgba(22, 27, 34, 0.75);
    --card-border: rgba(255, 255, 255, 0.08);
    --text: #f0f6fc;
    --text-muted: #8b949e;
    --accent: #6366f1;
    --accent-glow: rgba(99, 102, 241, 0.25);
    --pass: #10b981;
    --warn: #f59e0b;
    --fail: #ef4444;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, var(--bg) 70%);
    color: var(--text);
    padding: 32px 24px;
    min-height: 100vh;
    line-height: 1.5;
  }
  .container { max-width: 1200px; margin: 0 auto; }
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 20px;
    margin-bottom: 32px;
    padding-bottom: 24px;
    border-bottom: 1px solid var(--card-border);
  }
  .brand { display: flex; align-items: center; gap: 14px; }
  .brand-logo {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    box-shadow: 0 4px 16px var(--accent-glow);
  }
  .brand h1 { font-size: 24px; font-weight: 700; letter-spacing: -0.5px; }
  .brand p { font-size: 13px; color: var(--text-muted); }
  
  .verdict-banner {
    padding: 10px 24px;
    border-radius: 9999px;
    background: __V_BG__;
    border: 1px solid __V_BORDER__;
    color: __V_COLOR__;
    font-weight: 700;
    font-size: 15px;
    letter-spacing: 0.5px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    box-shadow: 0 0 24px __V_BG__;
  }
  .verdict-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: __V_COLOR__;
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.3); opacity: 0.7; }
  }

  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
    margin-bottom: 32px;
  }
  .card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    backdrop-filter: blur(16px);
    border-radius: 14px;
    padding: 20px;
    transition: transform 0.2s, border-color 0.2s;
  }
  .card:hover {
    transform: translateY(-2px);
    border-color: rgba(255, 255, 255, 0.16);
  }
  .card-label { font-size: 12px; text-transform: uppercase; letter-spacing: 0.8px; color: var(--text-muted); margin-bottom: 6px; }
  .card-val { font-size: 28px; font-weight: 700; }
  .card-sub { font-size: 13px; color: var(--text-muted); margin-top: 4px; }

  .controls {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 24px;
  }
  .search-input {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid var(--card-border);
    border-radius: 10px;
    padding: 10px 16px;
    color: var(--text);
    font-size: 14px;
    width: 320px;
    outline: none;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  .search-input:focus {
    border-color: var(--accent);
    box-shadow: 0 0 12px var(--accent-glow);
  }
  .pills { display: flex; gap: 8px; flex-wrap: wrap; }
  .pill {
    padding: 6px 14px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--card-border);
    color: var(--text-muted);
    transition: all 0.2s;
  }
  .pill:hover, .pill.active {
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
  }
  .btn-group { display: flex; gap: 8px; }
  .btn {
    padding: 8px 16px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid var(--card-border);
    color: var(--text);
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: background 0.2s;
  }
  .btn:hover { background: rgba(255, 255, 255, 0.15); }

  .results-list { display: flex; flex-direction: column; gap: 16px; }
  .result-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 20px;
    backdrop-filter: blur(12px);
  }
  .result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 14px;
  }
  .result-title { font-size: 16px; font-weight: 600; display: flex; align-items: center; gap: 10px; }
  .tag {
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }
  .tag-fail { background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }
  .tag-warn { background: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
  .tag-pass { background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
  .tag-err  { background: rgba(107, 114, 128, 0.2); color: #9ca3af; border: 1px solid rgba(107, 114, 128, 0.3); }

  .timeline-bar-bg {
    width: 100%;
    height: 10px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 5px;
    position: relative;
    overflow: hidden;
    margin: 12px 0 16px 0;
  }
  .timeline-seg {
    position: absolute;
    top: 0;
    bottom: 0;
    background: linear-gradient(90deg, #ef4444, #f59e0b);
    border-radius: 4px;
  }

  .match-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-top: 10px;
  }
  .match-table th {
    text-align: left;
    padding: 8px 12px;
    color: var(--text-muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 1px solid var(--card-border);
  }
  .match-table td {
    padding: 10px 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.03);
  }
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    background: rgba(255, 255, 255, 0.06);
  }
  .score-bar-bg {
    width: 80px;
    height: 6px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
    display: inline-block;
    vertical-align: middle;
    margin-right: 6px;
    overflow: hidden;
  }
  .score-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #6366f1, #ef4444);
    border-radius: 3px;
  }

  footer {
    margin-top: 48px;
    padding-top: 24px;
    border-top: 1px solid var(--card-border);
    display: flex;
    justify-content: space-between;
    color: var(--text-muted);
    font-size: 12px;
    flex-wrap: wrap;
    gap: 12px;
  }
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="brand">
      <div class="brand-logo">&#127925;</div>
      <div>
        <h1>MusicLeak Audit Report</h1>
        <p>ComposerMind Labs &middot; AI Model Memorization & Training Leakage Auditor</p>
      </div>
    </div>
    <div class="verdict-banner">
      <span class="verdict-dot"></span>
      <span>__V_LABEL__</span>
    </div>
  </header>

  <div class="grid">
    <div class="card">
      <div class="card-label">Overall Verdict</div>
      <div class="card-val" style="color: __V_COLOR__">__VERDICT__</div>
      <div class="card-sub">__V_LABEL__</div>
    </div>
    <div class="card">
      <div class="card-label">Query Songs Audited</div>
      <div class="card-val">__TOTAL_FILES__</div>
      <div class="card-sub"><span style="color:#ef4444">__FAIL_COUNT__ Leaked</span> &middot; <span style="color:#f59e0b">__WARN_COUNT__ Warned</span> &middot; <span style="color:#10b981">__PASS_COUNT__ Clean</span></div>
    </div>
    <div class="card">
      <div class="card-label">Contaminated Duration</div>
      <div class="card-val">__MATCHED_S__s</div>
      <div class="card-sub">__LEAKAGE_PCT__% of total audio (__TOTAL_DUR_S__s)</div>
    </div>
    <div class="card">
      <div class="card-label">Calibration Baseline</div>
      <div class="card-val">__CAL_TYPE__</div>
      <div class="card-sub">Null floor: __CAL_SINGLE__ &middot; Confirmed: __CAL_CHAIN__ (__CAL_N__ windows)</div>
    </div>
  </div>

  <div class="controls">
    <div class="pills">
      <div class="pill active" onclick="filterStatus('all', this)">All (__TOTAL_FILES__)</div>
      <div class="pill" onclick="filterStatus('FAIL', this)">Leaked (__FAIL_COUNT__)</div>
      <div class="pill" onclick="filterStatus('WARN', this)">Suspicious (__WARN_COUNT__)</div>
      <div class="pill" onclick="filterStatus('PASS', this)">Clean (__PASS_COUNT__)</div>
    </div>
    <input type="text" id="search" class="search-input" placeholder="Search song or reference track..." oninput="applySearch()">
    <div class="btn-group">
      <button class="btn" onclick="downloadJSON()">&#11015; Export JSON</button>
      <button class="btn" onclick="downloadCSV()">&#11015; Export CSV</button>
    </div>
  </div>

  <div class="results-list" id="results-container">
__RENDERED_CARDS__
  </div>

  <footer>
    <div>MusicLeak v0.1.0a1 &middot; Signal Processing Audio Leakage Engine &middot; MIT Licence</div>
    <div>Empirical Null Distribution Windows: __CAL_N__ &middot; Window Size: __WINDOW_S__s</div>
  </footer>
</div>

<script>
const reportData = __DATA_JSON__;
let currentFilter = 'all';

function filterStatus(status, el) {
  currentFilter = status;
  document.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
  applySearch();
}

function applySearch() {
  const query = document.getElementById('search').value.toLowerCase();
  const cards = document.querySelectorAll('.result-card');
  cards.forEach(card => {
    const name = card.getAttribute('data-name');
    const verdict = card.getAttribute('data-verdict');
    const matchesFilter = currentFilter === 'all' || verdict === currentFilter;
    const matchesSearch = !query || name.includes(query) || card.innerText.toLowerCase().includes(query);
    card.style.display = (matchesFilter && matchesSearch) ? 'block' : 'none';
  });
}

function downloadJSON() {
  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(reportData, null, 2));
  const a = document.createElement('a');
  a.setAttribute('href', dataStr);
  a.setAttribute('download', 'musicleak_report.json');
  a.click();
}

function downloadCSV() {
  let csv = 'query,kind,query_start_s,query_end_s,reference,reference_start_s,reference_end_s,transposition_semitones,tempo_ratio,score\n';
  reportData.files.forEach(f => {
    f.matches.forEach(m => {
      csv += f.name + ',' + m.kind + ',' + m.query_start.toFixed(1) + ',' + m.query_end.toFixed(1) + ',' + m.reference + ',' + m.reference_start.toFixed(1) + ',' + m.reference_end.toFixed(1) + ',' + m.transposition_semitones + ',' + m.tempo_ratio.toFixed(2) + ',' + m.score.toFixed(3) + '\n';
    });
  });
  const dataStr = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv);
  const a = document.createElement('a');
  a.setAttribute('href', dataStr);
  a.setAttribute('download', 'musicleak_matches.csv');
  a.click();
}
</script>
</body>
</html>"""

    html = (template
        .replace("__VERDICT__", verdict)
        .replace("__V_COLOR__", v_color)
        .replace("__V_BG__", v_bg)
        .replace("__V_BORDER__", v_border)
        .replace("__V_LABEL__", v_label)
        .replace("__TOTAL_FILES__", str(total_files))
        .replace("__FAIL_COUNT__", str(fail_count))
        .replace("__WARN_COUNT__", str(warn_count))
        .replace("__PASS_COUNT__", str(pass_count))
        .replace("__TOTAL_DUR_S__", f"{total_dur_s:.1f}")
        .replace("__MATCHED_S__", f"{total_matched_s:.1f}")
        .replace("__LEAKAGE_PCT__", f"{leakage_pct:.1f}")
        .replace("__CAL_TYPE__", cal_type)
        .replace("__CAL_SINGLE__", f"{cal.single:.2f}")
        .replace("__CAL_CHAIN__", f"{cal.chain:.2f}")
        .replace("__CAL_N__", str(cal.n))
        .replace("__WINDOW_S__", str(rep.settings.get("window_s", 10.0)))
        .replace("__RENDERED_CARDS__", rendered_cards)
        .replace("__DATA_JSON__", data_json)
    )
    return html


def write_html_report(rep: Report, html_path: Path) -> None:
    """Write interactive HTML report to designated path."""
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(report_to_html(rep), encoding="utf-8")

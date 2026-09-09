"""Static HTML dashboard for one decision cycle -- prompt Block L (optional).

Deliberately dependency-free (no streamlit/plotly): it writes a self-contained
HTML file with inline SVG, so it opens from a clean checkout with no server and
no network.

    python -m src.dashboard --at "2026-05-02 07:20:00"
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _svg_series(dates, values, limit=None, band=None, width=900, height=220,
                label="") -> str:
    v = np.asarray(values, float)
    m = np.isfinite(v)
    if m.sum() < 2:
        return "<p>no data</p>"
    x = np.arange(len(v))[m]
    y = v[m]
    lo = float(np.nanmin(y)); hi = float(np.nanmax(y))
    if limit is not None:
        lo, hi = min(lo, limit), max(hi, limit)
    if band is not None:
        hi = max(hi, float(np.nanmax(band[m])))
    pad = 0.08 * max(hi - lo, 1e-9)
    lo, hi = lo - pad, hi + pad
    def sx(i): return 40 + (i - x[0]) / max(x[-1] - x[0], 1) * (width - 60)
    def sy(val): return height - 30 - (val - lo) / max(hi - lo, 1e-9) * (height - 50)
    pts = " ".join(f"{sx(i):.1f},{sy(val):.1f}" for i, val in zip(x, y))
    parts = [f'<svg width="{width}" height="{height}" role="img" '
             f'aria-label="{html.escape(label)}">']
    parts.append(f'<rect width="{width}" height="{height}" fill="var(--panel)"/>')
    if band is not None:
        b = np.asarray(band, float)[m]
        up = " ".join(f"{sx(i):.1f},{sy(val):.1f}" for i, val in zip(x, y + b))
        dn = " ".join(f"{sx(i):.1f},{sy(val):.1f}"
                      for i, val in reversed(list(zip(x, y - b))))
        parts.append(f'<polygon points="{up} {dn}" fill="var(--band)" '
                     'opacity="0.35"/>')
    if limit is not None:
        parts.append(f'<line x1="40" y1="{sy(limit):.1f}" x2="{width - 20}" '
                     f'y2="{sy(limit):.1f}" stroke="var(--danger)" '
                     'stroke-width="2" stroke-dasharray="6 4"/>')
        parts.append(f'<text x="{width - 24}" y="{sy(limit) - 6:.1f}" '
                     'text-anchor="end" fill="var(--danger)" font-size="12">'
                     f'limit {limit:g}</text>')
    parts.append(f'<polyline points="{pts}" fill="none" stroke="var(--accent)" '
                 'stroke-width="2"/>')
    parts.append(f'<text x="40" y="16" fill="var(--muted)" font-size="12">'
                 f'{html.escape(label)}</text>')
    parts.append(f'<text x="40" y="{height - 8}" fill="var(--muted)" '
                 f'font-size="11">{dates.iloc[int(x[0])]}</text>')
    parts.append(f'<text x="{width - 20}" y="{height - 8}" text-anchor="end" '
                 f'fill="var(--muted)" font-size="11">'
                 f'{dates.iloc[int(x[-1])]}</text>')
    parts.append("</svg>")
    return "".join(parts)


def render(system, ts: str, window_h: float = 168.0) -> str:
    d = system.decide(ts, persist=False)
    f = system.ds.frame
    r = system.replay_frame
    t = pd.Timestamp(ts)
    lo = t - pd.Timedelta(hours=window_h)
    m = (f["date"] >= lo) & (f["date"] <= t)
    sub_d = f.loc[m, "date"].reset_index(drop=True)
    yhat = r.loc[m, "y_hat"].to_numpy(float)
    sigma = float(d.confidence_block["sigma_pred_mgkg"])
    band = np.full(yhat.shape, sigma * 1.6448536)
    lims = f.loc[m, "LIMS:sulfur"].to_numpy(float)

    def block(title, body):
        return (f'<section><h2>{html.escape(title)}</h2>{body}</section>')

    def table(rows):
        out = ["<table>"]
        for k, v in rows:
            out.append(f"<tr><th>{html.escape(str(k))}</th>"
                       f"<td>{html.escape(str(v))}</td></tr>")
        out.append("</table>")
        return "".join(out)

    badge = {"recommend": "ok", "hold": "ok", "mitigate": "warn",
             "refuse": "danger"}.get(d.action, "muted")
    changes = ("<p class=muted>no change proposed</p>" if not
               d.action_block.get("changes") else "<table><tr><th>tag</th>"
               "<th>parameter</th><th>now</th><th>recommended</th>"
               "<th>unit</th></tr>" + "".join(
                   f"<tr><td>{html.escape(c['tag'])}</td>"
                   f"<td>{html.escape(c['label'])}</td><td>{c['current']}</td>"
                   f"<td><b>{c['recommended']}</b></td>"
                   f"<td>{html.escape(c['unit'])}</td></tr>"
                   for c in d.action_block["changes"]) + "</table>")
    checks = "".join(
        f"<li class=\"{'ok' if c.get('ok') else 'danger'}\">"
        f"{html.escape(str(c.get('detail', '')))}</li>"
        for c in d.constraint_block.get("checks", []))
    gate = "".join(f"<li class=danger>{html.escape(g)}</li>"
                   for g in d.constraint_block.get("gate_eq66", []))

    return f"""<title>Diesel quality decision {html.escape(ts)}</title>
<style>
:root {{ --bg:#fbfaf8; --panel:#ffffff; --ink:#1a1a1a; --muted:#6b6b6b;
        --accent:#2f6f8f; --band:#7fb2c8; --ok:#2e7d4f; --warn:#a86a12;
        --danger:#b03030; --line:#e3e0da; }}
:root:not([data-theme="light"]) {{ }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#16181a; --panel:#1e2124; --ink:#eceae6; --muted:#9a9a9a;
  --accent:#79b8d6; --band:#3d6d84; --ok:#5fb383; --warn:#d6a24a;
  --danger:#e08585; --line:#2e3236; }} }}
:root[data-theme="dark"] {{ --bg:#16181a; --panel:#1e2124; --ink:#eceae6;
  --muted:#9a9a9a; --accent:#79b8d6; --band:#3d6d84; --ok:#5fb383;
  --warn:#d6a24a; --danger:#e08585; --line:#2e3236; }}
body {{ background:var(--bg); color:var(--ink); margin:0;
  font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
main {{ max-width:1000px; margin:0 auto; padding:24px 20px 60px; }}
h1 {{ font-size:20px; margin:0 0 4px; }}
h2 {{ font-size:14px; text-transform:uppercase; letter-spacing:.06em;
  color:var(--muted); margin:26px 0 8px; font-weight:600; }}
section {{ background:var(--panel); border:1px solid var(--line);
  border-radius:8px; padding:14px 16px; margin-bottom:14px; }}
table {{ border-collapse:collapse; width:100%; }}
th,td {{ text-align:left; padding:5px 8px; border-bottom:1px solid var(--line);
  vertical-align:top; }}
th {{ color:var(--muted); font-weight:600; width:34%; }}
.badge {{ display:inline-block; padding:3px 10px; border-radius:99px;
  font-weight:700; font-size:12px; letter-spacing:.04em; }}
.badge.ok {{ background:var(--ok); color:#fff; }}
.badge.warn {{ background:var(--warn); color:#fff; }}
.badge.danger {{ background:var(--danger); color:#fff; }}
ul {{ margin:6px 0; padding-left:18px; }}
li.ok {{ color:var(--ok); }} li.danger {{ color:var(--danger); }}
.muted {{ color:var(--muted); }}
.wrap {{ overflow-x:auto; }}
p.expl {{ margin:6px 0 0; }}
</style>
<main>
<h1>Product diesel sulfur &mdash; decision cycle</h1>
<p class=muted>{html.escape(ts)} &nbsp;
   <span class="badge {badge}">{html.escape(d.action.upper())}</span></p>
<p><b>{html.escape(d.headline)}</b></p>
{block("Forecast with the 95% one-sided band",
       '<div class=wrap>' + _svg_series(sub_d, yhat, limit=10.0, band=band,
       label="predicted product sulfur, mg/kg (eq 55) with the eq 63 back-off")
       + '</div>')}
{block("State and data freshness",
       table(list(d.state_block["key_parameters"].items())
             + [(f"age {k}", v) for k, v in
                d.state_block["freshness_h"].items()]))}
{block("Risk", table(list(d.risk_block.items())[:7]))}
{block("Proposed action", changes)}
{block("Expected effect", table(list(d.effect_block.items())) if
       d.effect_block else "<p class=muted>none</p>")}
{block("Constraint checks (eq 63-66)",
       ("<ul>" + gate + "</ul>" if gate else "") +
       ("<ul>" + checks + "</ul>" if checks else ""))}
{block("Confidence",
       table([("level", d.confidence_block["confidence"]),
              ("sigma_pred, mg/kg", d.confidence_block["sigma_pred_mgkg"]),
              ("back-off, mg/kg", d.confidence_block["backoff_mgkg"])])
       + "<ul>" + "".join(f"<li class=muted>{html.escape(w)}</li>"
                          for w in d.confidence_block["warnings"]) + "</ul>")}
{block("Explanation", f'<p class=expl>{html.escape(d.explanation)}</p>')}
</main>"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.dashboard")
    ap.add_argument("--at", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    from src.system import build_system
    S = build_system()
    html_text = render(S, args.at)
    out = Path(args.out) if args.out else (
        S.cfg.artifacts_dir / f"dashboard_{args.at.replace(':', '').replace(' ', 'T')}.html")
    out.write_text(html_text, encoding="utf-8")
    print(f"dashboard written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

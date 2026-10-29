"""Formatting helpers and page template for build_report.py (standard library + pandas)."""
import html
import math

import pandas as pd

CSS = """
@page { size: A4; margin: 17mm 16mm 18mm 16mm;
        @bottom-center { content: counter(page) " / " counter(pages); font: 8.5pt 'Segoe UI', Arial, sans-serif; color: #6b6a66; } }
:root { --ink:#0b0b0b; --ink2:#3d3c39; --muted:#6b6a66; --line:#d6d5ce; --head:#eef3f9; --accent:#184f95; }
* { box-sizing: border-box; }
body { font: 10pt/1.42 'Segoe UI', Calibri, Arial, sans-serif; color: var(--ink); margin: 0; text-align: justify; hyphens: auto; }
h1 { font-size: 19pt; line-height: 1.2; margin: 0 0 4pt; text-align: left; }
h2 { font-size: 13pt; margin: 17pt 0 5pt; color: var(--accent); text-align: left; break-after: avoid; }
h3 { font-size: 10.5pt; margin: 11pt 0 3pt; text-align: left; break-after: avoid; }
p { margin: 0 0 6pt; orphans: 3; widows: 3; }
.meta { color: var(--ink2); margin-bottom: 8pt; text-align: left; font-size: 9.5pt; }
.abstract { border-left: 3px solid var(--accent); padding: 2pt 0 2pt 9pt; margin: 8pt 0 6pt; background: #f7f9fc; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 9pt; font-size: 8.3pt; font-variant-numeric: tabular-nums; break-inside: avoid; }
th { background: var(--head); text-align: left; padding: 3pt 5pt; border-bottom: 1.2px solid var(--accent); }
td { padding: 2.2pt 5pt; border-bottom: 0.6px solid var(--line); vertical-align: top; text-align: left; hyphens: manual; }
td.num, th.num { text-align: right; }
td.num { white-space: nowrap; }
caption { caption-side: top; text-align: left; font-weight: 600; font-size: 8.8pt; margin-bottom: 3pt; color: var(--ink2); }
figure { margin: 7pt 0 9pt; break-inside: avoid; text-align: center; }
figure img { max-width: 100%; }
figcaption { font-size: 8.8pt; color: var(--ink2); text-align: left; margin-top: 3pt; }
code { font: 8.6pt Consolas, 'Cascadia Mono', monospace; background: #f1f0ec; padding: 0 2pt; border-radius: 2pt; }
pre { font: 8pt/1.35 Consolas, 'Cascadia Mono', monospace; background: #f6f5f1; padding: 6pt 8pt; border: 0.6px solid var(--line);
      border-radius: 3pt; overflow: hidden; white-space: pre-wrap; text-align: left; break-inside: avoid; margin: 4pt 0 8pt; }
ul, ol { margin: 0 0 6pt; padding-left: 16pt; } li { margin-bottom: 2.5pt; }
.small { font-size: 8.6pt; color: var(--ink2); }
.refs li { font-size: 8.8pt; text-align: left; }
.pass { color: #006300; font-weight: 600; } .warn { color: #8a5a00; font-weight: 600; }
"""


def esc(x):
    return html.escape(str(x))


def fmt_time(s):
    """Seconds with 3 significant digits in a sensible unit."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return "–"
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("µs", 1e-6), ("ns", 1e-9)):
        if s >= scale or unit == "ns":
            v = s / scale
            return f"{v:.3g} {unit}" if v < 100 else f"{v:.0f} {unit}"
    return str(s)


def fmt_time_iqr(med, q1, q3):
    return f"{fmt_time(med)} [{fmt_time(q1)}–{fmt_time(q3)}]"


def fmt_num(v, digits=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v:.{digits}g}"


def table(headers, rows, caption=None, num_cols=(), col_widths=None):
    """headers: list of str; rows: list of lists of str (already formatted, may contain safe HTML)."""
    out = ["<table>"]
    if caption:
        out.append(f"<caption>{caption}</caption>")
    out.append("<tr>" + "".join(
        f"<th class='{'num' if i in num_cols else ''}'>{h}</th>" for i, h in enumerate(headers)) + "</tr>")
    for r in rows:
        out.append("<tr>" + "".join(
            f"<td class='{'num' if i in num_cols else ''}'>{c}</td>" for i, c in enumerate(r)) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def figure(path, caption, width="100%"):
    return f"<figure><img src='{esc(path)}' style='width:{width}'><figcaption>{caption}</figcaption></figure>"


def page(title, body):
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{esc(title)}</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")


def df_rows(df, cols):
    return [[row[c] for c in cols] for _, row in df.iterrows()]

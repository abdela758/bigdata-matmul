"""Assemble the Assignment 1 report (HTML -> PDF) from the tables and figures written by analyze.py.

    python scripts/build_report.py [--repo-url URL] [--no-pdf]

The PDF is printed with headless Microsoft Edge (found automatically on Windows); the HTML is always written to report/.
"""
import argparse
import datetime
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

import report_static as S
import report_text as TXT
from report_lib import esc, figure, fmt_num, fmt_time, fmt_time_iqr, page, table

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
REPORT = ROOT / "report"
TABLES = RESULTS / "tables"
LANGS = ["python", "java", "c"]
LABEL = {"python": "Python", "java": "Java", "c": "C"}
AUTHOR = "Abdelaziz Choukri Mohamed"
PDF_NAME = "Assignment01_Choukri_Report.pdf"


def T(name, required=True):
    p = TABLES / f"{name}.csv"
    if not p.exists():
        if required:
            raise FileNotFoundError(p)
        return None
    return pd.read_csv(p, keep_default_na=False, na_values=[""])


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


# ---------------------------------------------------------------- tables
def correctness_table():
    c = pd.read_csv(RESULTS / "correctness.csv", keep_default_na=False)
    desc = {
        "identity_left": ("I × X = X", "bit-exact"), "identity_right": ("X × I = X", "bit-exact"),
        "zero": ("0 × X = 0", "bit-exact"), "known_2x2": ("hand-computed 2 × 2 product", "bit-exact"),
        "known_3x3": ("hand-computed 3 × 3 product", "bit-exact"),
        "integer_valued": ("random integers 0..9 vs NumPy int64 product", "bit-exact"),
        "random_01": ("random reals in [0, 1)", "error ≤ γₙ Σ|aᵢₖbₖⱼ|"),
        "random_pm1": ("random reals in [−1, 1)", "error ≤ γₙ Σ|aᵢₖbₖⱼ|"),
        "shared_inputs": ("the shared benchmark inputs", "error ≤ γₙ Σ|aᵢₖbₖⱼ|"),
        "cross_language_bit_identical": ("C, Java and Python results identical", "bit-identical"),
    }
    rows = []
    for test, (what, crit) in desc.items():
        d = c[c.test == test]
        if d.empty:
            continue
        ratios = []
        for s in d.detail:
            m = re.search(r"exact-sample ([0-9.]+), NumPy-all ([0-9.]+)", s)
            if m:
                ratios.append((float(m.group(1)), float(m.group(2))))
        worst = (f"{max(r[0] for r in ratios):.3f} / {max(r[1] for r in ratios):.3f}" if ratios else "–")
        ns = ", ".join(str(n) for n in sorted(d.n.astype(str).unique(), key=lambda x: int(x)))
        rows.append([what, ns, crit, f"<span class='pass'>{int(d.passed.sum())}/{len(d)}</span>", worst])
    total, ok = len(c), int(c.passed.sum())
    rows.append(["<b>Total</b>", "", "", f"<span class='pass'><b>{ok}/{total}</b></span>", ""])
    return table(["Check", "Sizes n", "Criterion", "Passed", "Worst error ÷ bound (exact / NumPy)"], rows,
                 caption="Table 3. Correctness checks (each row counts the three implementations × test cases). "
                         "An error ÷ bound of 1 would be the theoretical worst case.", num_cols=(3, 4))


def runtime_table(ts):
    p = ts.set_index(["n", "language"])
    rows = []
    for n in sorted(ts.n.unique()):
        row = [str(n)]
        for lang in LANGS:
            if (n, lang) in p.index:
                r = p.loc[(n, lang)]
                row.append(fmt_time_iqr(r.median_s, r.q1_s, r.q3_s))
            else:
                row.append("not run / timeout")
        rows.append(row)
    return table(["n", "Python", "Java", "C"], rows,
                 caption="Table 4. Time of one multiplication: median [Q1–Q3] over all measured batches of all launches (counts in Appendix A).",
                 num_cols=(0, 1, 2, 3))


def normalised_table(ts):
    p = ts.set_index(["n", "language"])
    rows = []
    for n in sorted(ts.n.unique()):
        rows.append([str(n)] + [(f"{p.loc[(n, l)].ns_per_multiply_add:.2f}" if (n, l) in p.index else "–") for l in LANGS])
    return table(["n", "Python", "Java", "C"], rows,
                 caption="Table 5. Median time per inner iteration, <i>t</i>/<i>n</i>³ (ns). Constant = exactly cubic.", num_cols=(0, 1, 2, 3))


def fits_tables(fits, loc):
    rows = []
    for _, r in fits.iterrows():
        rows.append([LABEL[r.language], r.window, str(int(r.points)), f"{r.exponent:.2f}",
                     f"{r.ci95_low:.2f}–{r.ci95_high:.2f}", f"{r.r_squared:.4f}"])
    t1 = table(["Language", "Sizes", "Points", "Exponent", "95 % CI", "R²"], rows,
               caption="Table 6. Log-log slope of median time against <i>n</i> over the sizes all three languages completed (ordinary least squares on the six medians; the CI reflects only the scatter of those points).",
               num_cols=(2, 3, 4, 5))
    ns = sorted(loc.n_to.unique())
    heads = ["Local exponent between consecutive sizes"] + [f"→ {n}" for n in ns]
    # languages have different size lists: build a size-aligned table
    rows = []
    for lang in LANGS:
        d = loc[loc.language == lang].set_index("n_to")
        rows.append([LABEL[lang]] + [(f"{d.loc[n].local_exponent:.2f}" if n in d.index else "–") for n in ns])
    t2 = table(heads, rows, caption="Table 7. Local exponent log(<i>t</i>₂/<i>t</i>₁)/log(<i>n</i>₂/<i>n</i>₁) from each size to the next listed size (3 = cubic).",
               num_cols=tuple(range(1, len(ns) + 1)))
    return t1 + t2


def ratio_table(rt):
    rows = []
    for r in rt.itertuples():
        rows.append([str(r.n), f"{r.python_over_c:.1f}×" if pd.notna(r.python_over_c) else "–",
                     f"{r.java_over_c:.2f}×" if pd.notna(r.java_over_c) else "–",
                     f"{r.python_over_java:.1f}×" if pd.notna(r.python_over_java) else "–"])
    return table(["n", "Python ÷ C", "Java ÷ C", "Python ÷ Java"], rows,
                 caption="Table 8. Ratios of median times (where both runs completed).", num_cols=(0, 1, 2, 3))


def memory_table(mem):
    m = mem[mem.variant == "main"].set_index(["n", "language"])
    rows = []
    for n in sorted(mem[mem.variant == "main"].n.unique()):
        row = [str(n)]
        for lang in LANGS:
            if (n, lang) in m.index:
                r = m.loc[(n, lang)]
                rng = f" ({r.peak_ws_min_mib:.1f}–{r.peak_ws_max_mib:.1f})" if r.launches > 1 else ""
                row.append(f"{r.peak_ws_median_mib:.1f}{rng}")
            else:
                row.append("–")
        row.append(f"{3 * 8 * n * n / 2**20:.2f}")
        rows.append(row)
    return table(["n", "Python", "Java", "C", "24<i>n</i>² bytes"], rows,
                 caption="Table 9. Peak working set of the process (MiB): median over launches (min–max when more than one launch). Last column: array storage of the three matrices at 8 bytes per element.",
                 num_cols=(0, 1, 2, 3, 4))


def bytes_per_element_table(mem):
    m = mem[(mem.variant == "main") & (mem.n >= 300)]
    rows = []
    for lang in LANGS:
        d = m[m.language == lang].sort_values("n")
        if d.empty:
            continue
        rows.append([LABEL[lang], f"{d.baseline_mib.iloc[0]:.1f}",
                     f"{d.net_bytes_per_element.median():.1f}",
                     f"{d.net_bytes_per_element.min():.1f}–{d.net_bytes_per_element.max():.1f}"])
    return table(["Language", "Baseline at n = 10 (MiB)", "Median net bytes per matrix element, n ≥ 300", "Range"], rows,
                 caption="Table 10. Memory beyond the baseline per matrix element: (peak − baseline) ÷ (3<i>n</i>²). Array storage alone would be 8.",
                 num_cols=(1, 2, 3))


def limits_tables(lim, fails, limit_s):
    rows = [[LABEL[r.language], str(int(r.largest_feasible_n)), str(int(r.largest_attempted_n)), f"{limit_s:g} s"]
            for r in lim.itertuples()]
    t1 = table(["Language", "Largest size meeting all limits", "Largest size attempted", "Limit per multiplication"], rows,
               caption="Table 11. Feasibility limits of the main comparison.", num_cols=(1, 2, 3))
    if fails.empty:
        return t1 + "<p>No launch failed.</p>"
    rows = [[LABEL[r.language], str(int(r.n)), esc(r.status), f"{r.wall_seconds:.0f} s", str(int(r.completed_batches_before_failure))]
            for r in fails[fails.variant == "main"].itertuples()]
    t2 = table(["Language", "n", "Outcome", "Wall time until stopped", "Batches completed before"], rows,
               caption="Table 12. Launches that did not complete (results, not omissions): killed because a single multiplication exceeded the limit.",
               num_cols=(1, 3, 4))
    return t1 + t2


def appendix_table(ts):
    rows = []
    for lang in LANGS:
        for r in ts[ts.language == lang].sort_values("n").itertuples():
            rows.append([LABEL[lang], str(r.n), str(r.launches), str(r.samples), str(r.inner_batch), fmt_time(r.median_s),
                         fmt_time(r.q1_s), fmt_time(r.q3_s), f"{r.rel_iqr_percent:.1f}", fmt_time(r.min_s), fmt_time(r.max_s),
                         f"{r.between_launch_spread_percent:.1f}" if r.launches > 1 else "–"])
    return table(["Lang.", "n", "Launches", "Samples", "Mults per batch", "Median", "Q1", "Q3", "IQR/med (%)", "Min", "Max",
                  "Launch spread (%)"], rows,
                 caption="Table A1. Every tested configuration of the main comparison: time per multiplication (all measured batches pooled).",
                 num_cols=tuple(range(1, 12)))


# ---------------------------------------------------------------- pdf
def find_edge():
    for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"):
        if Path(p).exists():
            return p
    return shutil.which("msedge") or shutil.which("google-chrome") or shutil.which("chromium")


def print_pdf(html_path, pdf_path):
    edge = find_edge()
    if not edge:
        print("No Edge/Chrome found: open report/report.html and print it to PDF manually.")
        return False
    with tempfile.TemporaryDirectory() as profile:
        cmd = [edge, "--headless=new", "--disable-gpu", "--no-sandbox", f"--user-data-dir={profile}",
               "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}", html_path.as_uri()]
        subprocess.run(cmd, capture_output=True, timeout=180)
    return pdf_path.exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-url", default="(GitHub repository URL to be inserted before submission)")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()

    cfg = read_json(ROOT / "config" / "experiment.json")
    env = read_json(RESULTS / "environment.json")
    build = read_json(ROOT / "build" / "build-info.json")
    ctx = dict(cfg=cfg, env=env, build=build, ts=T("time_summary"), mem=T("memory_summary"), fits=T("fits"),
               loc=T("local_exponents"), rt=T("ratios"), lim=T("limits"), fails=T("failed_launches"),
               warm=T("warmup"), chk=T("checksums"), bg=T("background_cpu"), launches=pd.read_csv(RESULTS / "launches.csv", keep_default_na=False),
               limit_s=cfg["limits"]["seconds_per_multiplication"])
    ts, mem = ctx["ts"], ctx["mem"]

    fig = lambda name, cap, w="100%": figure(f"../results/figures/{name}.png", cap, w)  # noqa: E731
    parts = [
        S.header(AUTHOR, args.repo_url, datetime.date.today().isoformat()),
        TXT.abstract(ctx),
        S.introduction(), S.background(), S.methodology(ctx),
        "<h2>4. Results</h2>",
        "<h3>4.1 Correctness</h3>", TXT.correctness_text(ctx), correctness_table(),
        "<h3>4.2 Runtime</h3>", TXT.runtime_text(ctx),
        runtime_table(ts),
        fig("fig1_runtime", "Figure 1. Median time of one multiplication against matrix size (log-log). Vertical bars are the IQR; × marks a launch stopped at the per-multiplication limit. The dashed line only shows the slope of <i>n</i>³; its position is arbitrary."),
        normalised_table(ts),
        fig("fig2_normalised", "Figure 2. Median time per inner iteration, <i>t</i>/<i>n</i>³. Dotted lines mark where one matrix B stops fitting in the L2 and L3 caches (hypotheses from the cache sizes, not measurements)."),
        TXT.scaling_text(ctx), fits_tables(ctx["fits"], ctx["loc"]), ratio_table(ctx["rt"]),
        "<h3>4.3 Memory</h3>", TXT.memory_text(ctx), memory_table(mem),
        fig("fig3_memory", "Figure 3. Peak working set of the process against matrix size (log-log)."),
        bytes_per_element_table(mem),
        "<h3>4.4 Limits, timeouts and failures</h3>", TXT.limits_text(ctx), limits_tables(ctx["lim"], ctx["fails"], ctx["limit_s"]),
        "<h3>4.5 Warm-up and variability</h3>", TXT.warmup_text(ctx),
        fig("fig5_warmup", "Figure 4. Batch times of the first launch of selected configurations, relative to that launch’s measured median (log scale): open circles = calibration batches, grey = warm-up batches, filled = measured batches."),
        "<h2>5. Discussion</h2>", TXT.discussion(ctx),
        "<h2>6. Limitations</h2>", TXT.limitations(ctx),
        "<h2>7. Conclusions</h2>", TXT.conclusions(ctx),
        S.disclosure(), S.references(),
        "<h2>Appendix A. Full results of the main comparison</h2>", appendix_table(ts),
        "<h2>Appendix B. Reproducibility materials</h2>", TXT.reproducibility(ctx, args.repo_url),
    ]
    html = page("Basic Matrix Multiplication in Python, Java and C", "\n".join(parts))
    REPORT.mkdir(exist_ok=True)
    html_path = REPORT / "report.html"
    html_path.write_text(html, encoding="utf-8")
    print("wrote", html_path)
    if not args.no_pdf:
        pdf = REPORT / PDF_NAME
        print("wrote" if print_pdf(html_path, pdf) else "FAILED to write", pdf)


if __name__ == "__main__":
    sys.exit(main())

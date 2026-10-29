"""Assemble the Assignment 2 report (HTML -> PDF with headless Edge) from results/tables and results/figures.

    python scripts/build_report.py --repo-url URL      (run from PowerShell on Windows)
"""
import argparse
import datetime
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import report_static as S
import report_text as TXT
from report_lib import figure, page

ROOT = Path(__file__).resolve().parent.parent
RESULTS, REPORT = ROOT / "results", ROOT / "report"
AUTHOR = "Abdelaziz Choukri Mohamed"
PDF_NAME = "Assignment02_Choukri_Mohamed_Report.pdf"


def print_pdf(html_path, pdf_path):
    edge = next((p for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                             r"C:\Program Files\Microsoft\Edge\Application\msedge.exe") if Path(p).exists()), None) \
        or shutil.which("msedge") or shutil.which("google-chrome") or shutil.which("chromium")
    if not edge:
        print("No Edge/Chrome found: print report/report.html to PDF manually.")
        return False
    pdf_path.unlink(missing_ok=True)  # an old PDF can never be mistaken for a fresh one
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([edge, "--headless=new", "--disable-gpu", "--no-sandbox", f"--user-data-dir={profile}",
                        "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}", html_path.as_uri()], capture_output=True, timeout=180)
    return pdf_path.exists() and pdf_path.stat().st_size > 10_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-url", default="https://github.com/abdela758/bigdata-matmul")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    cfg = json.loads((ROOT / "config" / "experiment.json").read_text(encoding="utf-8-sig"))
    env = json.loads((RESULTS / "environment.json").read_text(encoding="utf-8"))
    c = TXT.Ctx()
    fig = lambda name, cap: figure(f"../results/figures/{name}.png", cap)  # noqa: E731
    parts = [
        S.header(AUTHOR, args.repo_url, datetime.date.today().isoformat()),
        TXT.abstract(c), S.introduction(), S.background(), S.methodology({"cfg": cfg, "env": env}),
        "<h2>4. Results</h2>",
        "<h3>4.1 Correctness</h3>", TXT.correctness_text(c),
        "<h3>4.2 Part A: dense optimisations</h3>", TXT.dense_text(c), TXT.dense_table(c),
        fig("fig1_dense_time", "Figure 1. Part A: median time of one multiplication (log-log); bars are the IQR, × marks the baseline’s timeouts at the 120 s limit."),
        fig("fig2_dense_per_iteration", "Figure 2. Part A: median time per multiply-add, <i>t</i>/<i>n</i>³. Dotted lines mark where one matrix B outgrows the L2 and L3 caches (computed from the cache sizes)."),
        TXT.speedup_table(c),
        "<h3>4.3 Part A: tile size</h3>", TXT.blocksize_text(c), TXT.blocksize_table(c),
        "<h3>4.4 Part B: sparse versus dense</h3>", TXT.sparse_text(c), TXT.sparse_table(c),
        fig("fig3_sparse_time", "Figure 3. Part B: median times against the density of the inputs, for the same logical matrices (log scales). The dense time is flat because it always performs <i>n</i>³ operations."),
        TXT.fill_table(c),
        "<h2>5. Discussion</h2>", TXT.discussion(c),
        "<h2>6. Limitations</h2>", TXT.limitations(),
        "<h2>7. Conclusions</h2>", TXT.conclusions(c),
        S.disclosure(), S.references(), TXT.materials(args.repo_url),
        "<h2>Appendix A. Part A, all configurations</h2>", TXT.appendix_table(c),
    ]
    REPORT.mkdir(exist_ok=True)
    html_path = REPORT / "report.html"
    html_path.write_text(page("Optimised and Sparse Matrix Multiplication in C", "\n".join(parts)), encoding="utf-8")
    print("wrote", html_path)
    if not args.no_pdf:
        pdf = REPORT / PDF_NAME
        print("wrote" if print_pdf(html_path, pdf) else "FAILED to write", pdf)


if __name__ == "__main__":
    sys.exit(main())

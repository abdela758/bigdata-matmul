"""Analysis for Assignment 2: summary tables (results/tables) and figures (results/figures) from the raw data.

Statistics as in Assignment 1: median and interquartile range of all measured batches of all successful launches;
calibration and warm-up batches are excluded (kept in the raw file). Needs pandas, numpy, matplotlib.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS, TABLES, FIGURES = ROOT / "results", ROOT / "results" / "tables", ROOT / "results" / "figures"

METHODS = ["ijk", "ikj", "blocked"]
LABEL = {"ijk": "ijk (baseline)", "ikj": "ikj (loop reordering)", "blocked": "blocked, S = 64",
         "csr": "CSR sparse multiplication", "dense_blocked": "dense blocked (S = 64)", "convert": "dense → CSR conversion of A"}
# validated categorical palette (slots 1-3), markers repeat the identity
COLOR = {"ijk": "#2a78d6", "ikj": "#eb6834", "blocked": "#1baf7a", "csr": "#2a78d6", "dense_blocked": "#1baf7a", "convert": "#eb6834"}
MARKER = {"ijk": "o", "ikj": "s", "blocked": "^", "csr": "o", "dense_blocked": "^", "convert": "s"}
INK2, MUTED, GRID, AXIS, SURFACE = "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
L2_BYTES, L3_BYTES = 1.25 * 2**20, 8 * 2**20


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9.5, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 9,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "savefig.dpi": 200,
    })


def load():
    s = pd.read_csv(RESULTS / "raw_samples.csv")
    l = pd.read_csv(RESULTS / "launches.csv")
    s["spm"] = s.seconds_per_multiplication.astype(float)
    ok = l[l.status == "ok"].launch_id
    m = s[(s.phase == "measure") & s.launch_id.isin(ok)]
    return s, l, m


def summary(m, keys):
    rows = []
    for k, g in m.groupby(keys):
        v = g.spm.to_numpy()
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        per_launch = g.groupby("launch_id").spm.median()
        rows.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,))), "launches": g.launch_id.nunique(),
                     "samples": len(v), "median_s": med, "q1_s": q1, "q3_s": q3, "rel_iqr_percent": 100 * (q3 - q1) / med,
                     "min_s": v.min(), "max_s": v.max(), "inner_batch": int(g.inner.median()),
                     "launch_ratio": per_launch.max() / per_launch.min()})
    return pd.DataFrame(rows)


def nonzero_products(n, d):
    """Number of nonzero products a_ik * b_kj in C = A x B: sum over k of nnz(column k of A) * nnz(row k of B)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from common import sparse_inputs
    pa, pb = sparse_inputs(n, d, "csr")
    def read(path):
        with open(path, "rb") as f:
            n_, nnz = np.fromfile(f, "<i8", 2)
            rowptr = np.fromfile(f, "<i8", n_ + 1)
            col = np.fromfile(f, "<i4", nnz)
        return rowptr, col
    _, col_a = read(pa)
    rowptr_b, _ = read(pb)
    return int((np.bincount(col_a, minlength=n) * np.diff(rowptr_b)).sum())


def write(df, name):
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{name}.csv", index=False, float_format="%.6g")


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def log_x(ax, ticks):
    ax.set_xscale("log"); ax.set_xticks(ticks); ax.set_xticklabels([f"{t:g}" for t in ticks]); ax.minorticks_off()


def main():
    style()
    cfg = json.loads((ROOT / "config" / "experiment.json").read_text(encoding="utf-8-sig"))
    limit = cfg["limits"]["seconds_per_multiplication"]
    s, l, m = load()

    # ---------------- Part A: dense methods
    md = m[m.experiment == "dense"]
    ts = summary(md, ["method", "n"])
    ts["ns_per_iteration"] = ts.median_s / ts.n.astype(float) ** 3 * 1e9
    write(ts, "dense_time")
    p = ts.pivot(index="n", columns="method", values="median_s")
    sp = pd.DataFrame({"n": p.index, "ijk_s": p.get("ijk"), "ikj_s": p.get("ikj"), "blocked_s": p.get("blocked")})
    sp["speedup_ikj"] = sp.ijk_s / sp.ikj_s
    sp["speedup_blocked"] = sp.ijk_s / sp.blocked_s
    sp["blocked_vs_ikj"] = sp.ikj_s / sp.blocked_s
    write(sp.reset_index(drop=True), "dense_speedup")
    ld = l[l.experiment == "dense"]
    lim = []
    for meth in METHODS:
        d = ld[ld.method == meth]
        okn = d[d.status == "ok"].n
        lim.append({"method": meth, "largest_feasible_n": int(okn.max()), "largest_attempted_n": int(d.n.max()),
                    "failed_sizes": " ".join(str(int(x)) for x in sorted(d[d.status != "ok"].n.unique()))})
    write(pd.DataFrame(lim), "dense_limits")
    write(l[l.status != "ok"][["launch_id", "experiment", "method", "n", "density", "status", "wall_seconds", "measured_samples"]], "failed_launches")
    mem_d = ld[ld.status == "ok"].groupby(["method", "n"]).peak_working_set_bytes.median().div(2**20).rename("peak_ws_mib").reset_index()
    write(mem_d, "dense_memory")

    # ---------------- block-size sensitivity
    mb = m[m.experiment == "blocksize"]
    if len(mb):
        tb = summary(mb, ["method", "n"])
        tb["S"] = tb.method.str.replace("blocked", "").astype(int)
        tb["ns_per_iteration"] = tb.median_s / tb.n.astype(float) ** 3 * 1e9
        write(tb.sort_values(["n", "S"]), "blocksize")

    # ---------------- Part B: sparse
    ms = m[m.experiment == "sparse"]
    if len(ms):
        tsp = summary(ms, ["method", "n", "density"])
        write(tsp, "sparse_time")
        ls = l[(l.experiment == "sparse") & (l.status == "ok")]
        nn = ls[ls.method == "csr"].groupby(["n", "density"])[["nnz_a", "nnz_b", "nnz_c"]].first().reset_index()
        nn["out_density"] = nn.nnz_c / nn.n.astype(float) ** 2
        mem = ls.groupby(["method", "n", "density"]).peak_working_set_bytes.median().div(2**20).unstack(0)
        piv = tsp.pivot_table(index=["n", "density"], columns="method", values="median_s")
        comp = nn.set_index(["n", "density"]).join(piv).join(mem, rsuffix="_mib")
        comp = comp.rename(columns={"csr": "csr_s", "dense_blocked": "dense_s", "convert": "convert_a_s",
                                    "csr_mib": "csr_peak_mib", "dense_blocked_mib": "dense_peak_mib", "convert_mib": "convert_peak_mib"})
        comp["speedup_csr_vs_dense"] = comp.dense_s / comp.csr_s
        comp["nonzero_products"] = [nonzero_products(n, d) for n, d in comp.index]
        comp["csr_ns_per_product"] = comp.csr_s / comp.nonzero_products * 1e9
        comp["dense_ns_per_iteration"] = comp.dense_s / np.array([n for n, _ in comp.index], dtype=float) ** 3 * 1e9
        comp["csr_plus_convert_both_s"] = comp.csr_s + 2 * comp.convert_a_s
        comp["convert_over_csr"] = comp.convert_a_s / comp.csr_s
        comp = comp.reset_index()
        write(comp, "sparse_comparison")
        fig_sparse(tsp, comp)

    fig_dense(ts, l, limit)
    fig_dense_norm(ts)
    print(f"tables -> {TABLES}\nfigures -> {FIGURES}")


def fig_dense(ts, l, limit):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for meth in METHODS:
        d = ts[ts.method == meth].sort_values("n")
        ax.errorbar(d.n, d.median_s, yerr=[d.median_s - d.q1_s, d.q3_s - d.median_s], fmt="none", ecolor=COLOR[meth], elinewidth=1, capsize=2)
        ax.plot(d.n, d.median_s, color=COLOR[meth], marker=MARKER[meth], markersize=6, linewidth=1.8,
                markeredgecolor=SURFACE, label=LABEL[meth])
    fails = l[(l.experiment == "dense") & (l.status != "ok")]
    first = True
    for r in fails.itertuples():
        if ((ts.method == r.method) & (ts.n == r.n)).any():
            continue
        ax.plot(r.n, limit, marker="X", color=COLOR[r.method], markersize=9, markeredgecolor="#0b0b0b", linestyle="none",
                label="timeout (> limit)" if first else None)
        first = False
    ax.axhline(limit, color=MUTED, linestyle=(0, (2, 3)), linewidth=1)
    ax.text(110, limit * 1.25, f"limit: {limit:g} s per multiplication", color=MUTED, fontsize=8.5)
    ax.set_yscale("log"); log_x(ax, [100, 200, 500, 1000, 2000, 3000])
    ax.set_xlabel("Matrix size n (n × n)"); ax.set_ylabel("Median time per multiplication (s)")
    ax.legend(loc="lower right"); ax.set_title("Part A: time of one dense multiplication")
    save(fig, "fig1_dense_time")


def fig_dense_norm(ts):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for meth in METHODS:
        d = ts[ts.method == meth].sort_values("n")
        ax.plot(d.n, d.ns_per_iteration, color=COLOR[meth], marker=MARKER[meth], markersize=6, linewidth=1.8,
                markeredgecolor=SURFACE, label=LABEL[meth])
    for size in (L2_BYTES, L3_BYTES):
        ax.axvline(np.sqrt(size / 8), color=MUTED, linestyle=(0, (2, 3)), linewidth=1)
    ax.set_yscale("log"); log_x(ax, [100, 200, 500, 1000, 2000, 3000])
    ymax = ts.ns_per_iteration.max() * 1.6
    ax.set_ylim(ts.ns_per_iteration.min() * 0.6, ymax)
    for size, text in ((L2_BYTES, "B > L2"), (L3_BYTES, "B > L3")):
        ax.text(np.sqrt(size / 8) * 0.96, ymax / 1.25, text, color=MUTED, fontsize=8.5, ha="right")
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    ax.yaxis.set_major_locator(FixedLocator([0.5, 0.7, 1, 1.5, 2, 3, 4, 5]))
    ax.yaxis.set_major_formatter(ScalarFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Matrix size n (n × n)"); ax.set_ylabel("Median time per inner iteration, t / n³ (ns)")
    ax.legend(loc="upper left"); ax.set_title("Part A: cost per multiply-add")
    save(fig, "fig2_dense_per_iteration")


def fig_sparse(tsp, comp):
    sizes = sorted(tsp.n.unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(8.8, 3.9), sharey=True)
    for ax, n in zip(np.atleast_1d(axes), sizes):
        for meth in ("dense_blocked", "csr", "convert"):
            d = tsp[(tsp.method == meth) & (tsp.n == n)].sort_values("density")
            ax.plot(d.density * 100, d.median_s, color=COLOR[meth], marker=MARKER[meth], markersize=6, linewidth=1.8,
                    markeredgecolor=SURFACE, label=LABEL[meth])
        ax.set_yscale("log"); log_x(ax, [0.1, 1, 5, 10, 25, 50])
        ax.set_xlabel("Density of A and B (% nonzeros)"); ax.set_title(f"n = {n}", fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("Median time (s)")
    np.atleast_1d(axes)[0].legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Part B: sparse (CSR) versus dense multiplication of the same matrices", x=0.01, ha="left", fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig3_sparse_time")


if __name__ == "__main__":
    sys.exit(main())

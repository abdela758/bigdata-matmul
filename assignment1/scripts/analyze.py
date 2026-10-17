"""Analysis for Assignment 1: summary tables and figures from the raw measurements.

Reads  results/raw_samples.csv and results/launches.csv
Writes results/tables/*.csv and results/figures/*.png

Statistics: central value = median, spread = interquartile range (quartiles by
linear interpolation, NumPy's default). Every measured batch of every
successful launch is pooled; warm-up and calibration batches are excluded from
the statistics but kept in the raw file and shown in the warm-up figure.

Needs pandas, numpy, scipy, matplotlib (see requirements.txt).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

LANGS = ["python", "java", "c"]
LABEL = {"python": "Python", "java": "Java", "c": "C"}
# fixed categorical order (validated palette, slots 1-3, all-pairs CVD-safe); markers repeat the identity
COLOR = {"python": "#2a78d6", "java": "#eb6834", "c": "#1baf7a"}
MARKER = {"python": "o", "java": "s", "c": "^"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
L2_BYTES, L3_BYTES = 1.25 * 2**20, 8 * 2**20  # per-core L2 and shared L3 of the i5-1135G7


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9.5,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "figure.dpi": 100, "savefig.dpi": 200,
    })


# ------------------------------------------------------------------ loading
def load(tag=""):
    samples = pd.read_csv(RESULTS / f"raw_samples{tag}.csv", keep_default_na=False)
    launches = pd.read_csv(RESULTS / f"launches{tag}.csv", keep_default_na=False)
    for df in (samples, launches):
        df["variant"] = df["variant"].replace("", "main")
    return samples, launches


def measured_ok(samples, launches):
    ok = launches[launches.status == "ok"][["launch_id", "peak_working_set_bytes", "peak_private_bytes"]]
    m = samples[samples.phase == "measure"].merge(ok, on="launch_id")
    m["seconds_per_multiplication"] = m.seconds_per_multiplication.astype(float)
    return m


# ------------------------------------------------------------------ tables
def time_summary(m, keys):
    rows = []
    for k, g in m.groupby(keys):
        v = g.seconds_per_multiplication.to_numpy()
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        per_launch = g.groupby("launch_id").seconds_per_multiplication.median()
        n = int(g.n.iloc[0])
        key = k if isinstance(k, tuple) else (k,)
        rows.append({**dict(zip(keys, key)), "launches": g.launch_id.nunique(), "samples": len(v),
                     "median_s": med, "q1_s": q1, "q3_s": q3, "iqr_s": q3 - q1,
                     "rel_iqr_percent": 100 * (q3 - q1) / med, "min_s": v.min(), "max_s": v.max(),
                     "launch_median_min_s": per_launch.min(), "launch_median_max_s": per_launch.max(),
                     "between_launch_spread_percent": 100 * (per_launch.max() / per_launch.min() - 1),
                     "inner_batch": int(g.inner.astype(int).median()),
                     "ns_per_multiply_add": med / n**3 * 1e9})
    return pd.DataFrame(rows)


def memory_summary(launches_ok, baseline_n=10):
    rows = []
    for (lang, variant, n), g in launches_ok.groupby(["language", "variant", "n"]):
        ws = g.peak_working_set_bytes.astype(float)
        priv = pd.to_numeric(g.peak_private_bytes, errors="coerce")
        rows.append({"language": lang, "variant": variant, "n": n, "launches": len(g),
                     "peak_ws_median_mib": ws.median() / 2**20, "peak_ws_min_mib": ws.min() / 2**20,
                     "peak_ws_max_mib": ws.max() / 2**20, "peak_private_median_mib": priv.median() / 2**20,
                     "array_storage_mib": 3 * 8 * n**2 / 2**20})
    mem = pd.DataFrame(rows)
    base = mem[mem.n == baseline_n].set_index(["language", "variant"]).peak_ws_median_mib
    mem["baseline_mib"] = [base.get((r.language, r.variant), np.nan) for r in mem.itertuples()]
    mem["net_mib"] = mem.peak_ws_median_mib - mem.baseline_mib
    mem["net_bytes_per_element"] = mem.net_mib * 2**20 / (3 * mem.n**2)  # 3 matrices of n*n elements
    mem["net_over_array_storage"] = mem.net_mib / mem.array_storage_mib
    return mem


def fit_exponent(df, lo, hi):
    d = df[(df.n >= lo) & (df.n <= hi)]
    if len(d) < 3:
        return None
    x, y = np.log(d.n.to_numpy(float)), np.log(d.median_s.to_numpy(float))
    res = stats.linregress(x, y)
    t = stats.t.ppf(0.975, len(d) - 2)
    return {"points": len(d), "exponent": res.slope, "ci95_low": res.slope - t * res.stderr,
            "ci95_high": res.slope + t * res.stderr, "r_squared": res.rvalue**2}


def fits_table(ts):
    windows = [("n=100-1000", 100, 1000)]  # the sizes completed by all three languages (6 points each)
    rows = []
    for lang in LANGS:
        d = ts[ts.language == lang]
        for name, lo, hi in windows:
            f = fit_exponent(d, lo, hi)
            if f:
                rows.append({"language": lang, "window": name, **f})
    return pd.DataFrame(rows)


def local_exponents(ts):
    rows = []
    for lang in LANGS:
        d = ts[ts.language == lang].sort_values("n")
        for a, b in zip(d.itertuples(), d.iloc[1:].itertuples()):
            rows.append({"language": lang, "n_from": a.n, "n_to": b.n,
                         "local_exponent": np.log(b.median_s / a.median_s) / np.log(b.n / a.n)})
    return pd.DataFrame(rows)


def ratios_table(ts):
    p = ts.pivot(index="n", columns="language", values="median_s")
    out = pd.DataFrame({"n": p.index, "python_s": p.get("python"), "java_s": p.get("java"), "c_s": p.get("c")})
    out["python_over_c"] = out.python_s / out.c_s
    out["java_over_c"] = out.java_s / out.c_s
    out["python_over_java"] = out.python_s / out.java_s
    return out.reset_index(drop=True)


def limits_table(launches, samples, limit_s):
    rows = []
    for lang in LANGS:
        d = launches[launches.language == lang]
        okn = sorted(d[d.status == "ok"].n.unique())
        rows.append({"language": lang, "largest_feasible_n": max(okn) if okn else None,
                     "largest_attempted_n": int(d.n.max()), "limit_seconds_per_multiplication": limit_s})
    summary = pd.DataFrame(rows)
    fails = launches[launches.status != "ok"][["launch_id", "language", "variant", "n", "status", "exit_code",
                                                 "wall_seconds", "peak_working_set_bytes"]].copy()
    done = samples.groupby("launch_id").size()
    fails["completed_batches_before_failure"] = fails.launch_id.map(done).fillna(0).astype(int)
    return summary, fails


def warmup_table(samples, launches):
    """First (calibration) batch versus the median measured batch, per launch."""
    ok = launches[launches.status == "ok"].launch_id
    s = samples[samples.launch_id.isin(ok)].copy()
    s["spm"] = s.seconds_per_multiplication.astype(float)
    rows = []
    for lid, g in s.groupby("launch_id"):
        first = g[g.phase == "calibration"].sort_values("index", key=lambda c: c.astype(int)).iloc[0]
        med = g[g.phase == "measure"].spm.median()
        rows.append({"launch_id": lid, "language": first.language, "variant": first.variant, "n": int(first.n),
                     "first_batch_s_per_mult": first.spm, "measured_median_s_per_mult": med,
                     "first_over_measured": first.spm / med,
                     "calibration_batches": int((g.phase == "calibration").sum())})
    return pd.DataFrame(rows)


def checksum_table(samples, launches):
    ok = launches[launches.status == "ok"].launch_id
    s = samples[samples.launch_id.isin(ok)]
    g = s.groupby(["variant", "language", "n"]).checksum.agg(lambda c: sorted(set(c)))
    rows = [{"variant": v, "language": l, "n": n, "distinct_checksums_in_launch": len(c), "checksum": c[0]}
            for (v, l, n), c in g.items()]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def direct_label(ax, x, y, text, color_key, dx=1.06):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center", ha="left",
                color=INK2, fontsize=9, fontweight="bold", annotation_clip=False)


def series(ax, d, lang, ycol, yerr=None, label=True, **kw):
    d = d.sort_values("n")
    if yerr is not None:
        ax.errorbar(d.n, d[ycol], yerr=yerr(d), fmt="none", ecolor=COLOR[lang], elinewidth=1, capsize=2, alpha=.8)
    ax.plot(d.n, d[ycol], color=COLOR[lang], marker=MARKER[lang], markersize=6, linewidth=1.8,
            markeredgecolor=SURFACE, markeredgewidth=0.8, label=LABEL[lang], **kw)
    if label:
        direct_label(ax, d.n.iloc[-1], d[ycol].iloc[-1], LABEL[lang], lang)


def log_axes(ax, xlabel="Matrix size n (n × n)", ylabel=""):
    ax.set_xscale("log"); ax.set_yscale("log")
    ticks = [10, 20, 50, 100, 200, 500, 1000, 2000, 3000]
    ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks]); ax.minorticks_off()
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def fig_runtime(ts, fails, limit_s):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for lang in LANGS:
        series(ax, ts[ts.language == lang], lang, "median_s", label=False,
               yerr=lambda d: [d.median_s - d.q1_s, d.q3_s - d.median_s])
    ax.axhline(limit_s, color=MUTED, linestyle=(0, (2, 3)), linewidth=1)
    ax.text(45, limit_s * 1.25, f"limit: {limit_s:g} s per multiplication", color=MUTED, fontsize=8.5, va="bottom")
    first = True
    for lang in LANGS:  # timeouts drawn at the limit, never at an invented time
        for n in sorted(fails[(fails.language == lang) & fails.status.str.startswith("timeout")].n.unique()):
            if ((ts.language == lang) & (ts.n == n)).any():
                continue  # another launch of this size completed: not a failed size
            ax.plot(n, limit_s, marker="X", color=COLOR[lang], markersize=9, markeredgecolor=INK, markeredgewidth=0.8,
                    linestyle="none", label="timeout (> limit)" if first else None)
            first = False
    # unanchored slope-3 guide (position is arbitrary; only the slope carries information)
    gx = np.array([300.0, 1000.0]); ax.plot(gx, 2e-4 * (gx / 300) ** 3, color=MUTED, linestyle="--", linewidth=1)
    ax.text(560, 6e-4, "slope 3 (∝ n³)", color=MUTED, fontsize=8.5)
    log_axes(ax, ylabel="Median time per multiplication (s)")
    ax.set_ylim(3e-7, 600)
    ax.legend(loc="upper left")
    ax.set_title("Runtime of one n × n multiplication")
    save(fig, "fig1_runtime")


def fig_normalised(ts):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for lang in LANGS:
        series(ax, ts[ts.language == lang], lang, "ns_per_multiply_add")
    for size, text in ((L2_BYTES, "B > L2"), (L3_BYTES, "B > L3")):
        n_cross = np.sqrt(size / 8)
        ax.axvline(n_cross, color=MUTED, linestyle=(0, (2, 3)), linewidth=1)
        ax.text(n_cross * 0.97, 250, text, color=MUTED, fontsize=8.5, ha="right", va="center")
    log_axes(ax, ylabel="Median time per inner iteration, t / n³ (ns)")
    ax.set_ylim(0.4, 400)
    ax.legend(loc="upper right")
    ax.set_title("Cost per inner iteration (flat = exactly cubic)")
    save(fig, "fig2_normalised")


def fig_memory(mem):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for lang in LANGS:
        d = mem[(mem.language == lang) & (mem.variant == "main")]
        series(ax, d, lang, "peak_ws_median_mib", label=False)
    nn = np.array([10.0, 3000.0])
    ax.plot(nn, 3 * 8 * nn**2 / 2**20, color=MUTED, linestyle="--", linewidth=1)
    ax.text(40, 3 * 8 * 40**2 / 2**20 * 0.28, "3 matrices × 8 B per element (24 n² bytes)", color=MUTED, fontsize=8.5,
            rotation=34, rotation_mode="anchor")
    log_axes(ax, ylabel="Peak working set of the process (MiB)")
    ax.legend(loc="lower right")
    ax.set_title("Peak memory of the process")
    save(fig, "fig3_memory")


def fig_warmup(samples, launches, sizes=(20, 100)):
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    ok = launches[launches.status == "ok"]
    first = ok[(ok.variant == "main") & (ok.sweep == 1)].set_index(["language", "n"]).launch_id
    fig, axes = plt.subplots(len(sizes), 3, figsize=(8.6, 3.0 * len(sizes)), sharey="row")
    for r, n in enumerate(sizes):
        for c, lang in enumerate(LANGS):
            ax = axes[r, c]
            g = samples[samples.launch_id == first[(lang, n)]].copy()
            g["spm"] = g.seconds_per_multiplication.astype(float)
            g["i"] = np.arange(len(g))
            g["rel"] = g.spm / g[g.phase == "measure"].spm.median()
            for phase, face, edge in (("calibration", SURFACE, COLOR[lang]), ("warmup", "#c3c2b7", "#c3c2b7"),
                                      ("measure", COLOR[lang], COLOR[lang])):
                d = g[g.phase == phase]
                ax.plot(d.i, d.rel, linestyle="none", marker="o", markersize=5, markerfacecolor=face,
                        markeredgecolor=edge, markeredgewidth=1.2, label=phase)
            ax.axhline(1, color=AXIS, linewidth=1)
            ax.set_yscale("log"); ax.set_title(f"{LABEL[lang]}, n={n}", fontsize=9.5)
            ax.set_xlabel("batch number in launch")
            ax.yaxis.set_major_locator(FixedLocator([0.5, 0.7, 1, 1.5, 2, 3, 5, 10, 20, 30]))
            ax.yaxis.set_major_formatter(ScalarFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
            if c == 0:
                ax.set_ylabel("batch time / measured median")
            if r == 0 and c == 0:
                ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("First launch of each configuration: batch times relative to the measured median", x=0.01, ha="left",
                 fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig5_warmup")


# ------------------------------------------------------------------ main
def write(df, name):
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{name}.csv", index=False, float_format="%.6g")


def main():
    style()
    cfg = json.loads((ROOT / "config" / "experiment.json").read_text(encoding="utf-8-sig"))
    limit_s = cfg["limits"]["seconds_per_multiplication"]

    samples, launches = load()
    main_launches = launches[launches.variant == "main"]
    m = measured_ok(samples, launches)
    mm = m[m.variant == "main"]
    ts = time_summary(mm, ["language", "n"])
    write(ts, "time_summary")
    mem = memory_summary(launches[launches.status == "ok"])
    write(mem, "memory_summary")
    write(fits_table(ts), "fits")
    write(local_exponents(ts), "local_exponents")
    rt = ratios_table(ts)
    write(rt, "ratios")
    lim, fails = limits_table(main_launches, samples, limit_s)
    write(lim, "limits")
    write(fails, "failed_launches")
    write(warmup_table(samples, launches), "warmup")
    write(checksum_table(samples, launches), "checksums")
    write(launches.groupby(["language", "n"]).background_cpu_percent.apply(pd.to_numeric, errors="coerce")
          .groupby(["language", "n"]).median().reset_index(), "background_cpu")

    fig_runtime(ts, fails, limit_s)
    fig_normalised(ts)
    fig_memory(mem)
    fig_warmup(samples, launches)

    print(f"tables -> {TABLES}\nfigures -> {FIGURES}")


if __name__ == "__main__":
    sys.exit(main())

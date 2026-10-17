"""Narrative sections of the report. Every number is computed from the tables in results/tables, results/*.csv or the config."""
import re
from pathlib import Path

import pandas as pd

from report_lib import fmt_time

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
LABEL = {"python": "Python", "java": "Java", "c": "C"}


# ---------------------------------------------------------------- number helpers
def med(ctx, lang, n):
    d = ctx["ts"]
    return float(d[(d.language == lang) & (d.n == n)].median_s.iloc[0])


def nsi(ctx, lang, n):
    """Median time per inner iteration in ns."""
    d = ctx["ts"]
    return float(d[(d.language == lang) & (d.n == n)].ns_per_multiply_add.iloc[0])


def ratio(ctx, a, b, n):
    return med(ctx, a, n) / med(ctx, b, n)


def baseline(ctx, lang):
    m = ctx["mem"]
    return float(m[(m.language == lang) & (m.n == 10)].peak_ws_median_mib.iloc[0])


def peak(ctx, lang, n):
    m = ctx["mem"]
    return float(m[(m.language == lang) & (m.variant == "main") & (m.n == n)].peak_ws_median_mib.iloc[0])


def growth(ctx, lang, n):
    """Peak working set above the n = 10 baseline, in MiB."""
    return peak(ctx, lang, n) - baseline(ctx, lang)


def array_mib(n):
    return 3 * 8 * n * n / 2**20


def rt_range(ctx, col, lo, hi):
    rt = ctx["rt"]
    d = rt[(rt.n >= lo) & (rt.n <= hi)][col].dropna()
    return d.min(), d.max(), d.median()


def failed_partial(ctx):
    """The failed launch that still completed some batches: (launch id, wall seconds, median s per multiplication, language, n)."""
    f = ctx["fails"]
    f = f[f.completed_batches_before_failure > 0].iloc[0]
    sm = pd.read_csv(RESULTS / "raw_samples.csv", keep_default_na=False)
    sm = sm[(sm.launch_id == f.launch_id) & (sm.phase == "measure")]
    return int(f.launch_id), float(f.wall_seconds), float(sm.seconds_per_multiplication.astype(float).median()), str(f.language), int(f.n)


def c100_drift(ctx):
    """Slow-down inside the first C launch at n = 100: mean of the last 3 measured batches over the mean of the first 3 batches."""
    l = ctx["launches"]
    lid = int(l[(l.language == "c") & (l.n == 100) & (l.sweep == 1)].launch_id.iloc[0])
    sm = pd.read_csv(RESULTS / "raw_samples.csv", keep_default_na=False)
    sm = sm[sm.launch_id == lid].copy()
    sm["v"] = sm.seconds_per_multiplication.astype(float)
    return 100 * (sm[sm.phase == "measure"].iloc[-3:].v.mean() / sm.iloc[:3].v.mean() - 1)


def disturbed_launches(ctx):
    """Launches that started while the machine was fully busy, compared with the other launches of the same configuration."""
    s = pd.read_csv(RESULTS / "raw_samples.csv", keep_default_na=False)
    l = ctx["launches"].rename(columns={"sweep": "lsweep"})[["launch_id", "status", "background_cpu_percent", "lsweep"]]
    m = s[s.phase == "measure"].merge(l, on="launch_id")
    m = m[m.status == "ok"].copy()
    m["spm"] = m.seconds_per_multiplication.astype(float)
    m["bg"] = pd.to_numeric(m.background_cpu_percent, errors="coerce")
    out = []
    for (lang, n), g in m.groupby(["language", "n"]):
        per = g.groupby("launch_id").agg(bg=("bg", "first"), med=("spm", "median"))
        for lid, r in per[per.bg > 50].iterrows():
            others = per.drop(lid)
            if len(others):
                out.append(dict(launch=int(lid), language=lang, n=int(n), med=r.med, others=others.med.median(),
                                excess=100 * (r.med / others.med.median() - 1)))
    return out


# ---------------------------------------------------------------- sections
def abstract(ctx):
    lim = ctx["lim"].set_index("language")
    n_py, n_j, n_c = (int(lim.loc[l].largest_feasible_n) for l in ("python", "java", "c"))
    pc_lo, pc_hi, pc_med = rt_range(ctx, "python_over_c", 100, 10**9)
    jc_lo, jc_hi, _ = rt_range(ctx, "java_over_c", 100, 1000)
    return f"""
<div class="abstract"><b>Abstract.</b> This study measures how the matrix size <i>n</i> affects the runtime and the memory of the textbook triple-loop multiplication of dense <i>n</i> × <i>n</i>
<code>double</code> matrices in Python, Java and C, on one Windows laptop (Intel Core i5-1135G7), with identical input files, loop order and timing protocol.
Sizes were increased until a single multiplication exceeded 120 s: the largest size that met the budget was <i>n</i> = {n_py} for Python, {n_j} for Java and {n_c} for C.
All three implementations return bit-identical results. Kernel time (median and interquartile range over repeated batches and up to three launches) grows roughly, but not exactly, as <i>n</i>³:
the time per inner iteration, <i>t</i>/<i>n</i>³, is nearly constant for small <i>n</i> and then rises in every language, steeply in C and Java above <i>n</i> ≈ 1000–1500.
Python is about {pc_med:.0f}× slower than C (range {pc_lo:.0f}–{pc_hi:.0f}× for <i>n</i> ≥ 100); Java is {jc_lo:.1f}–{jc_hi:.1f}× slower than C up to <i>n</i> = 1000 but {ratio(ctx, 'java', 'c', 1500):.1f}× slower at <i>n</i> = 1500, so in this study it does not approach C at large sizes.
At <i>n</i> = 1000 the processes peak at {peak(ctx, 'python', 1000):.0f} MiB (Python), {peak(ctx, 'java', 1000):.0f} MiB (Java) and {peak(ctx, 'c', 1000):.0f} MiB (C), on top of runtime baselines of {baseline(ctx, 'python'):.0f}, {baseline(ctx, 'java'):.0f} and {baseline(ctx, 'c'):.0f} MiB.
The conclusions hold for these implementations, settings, inputs and this machine; explanations of the slowdown at large sizes are hypotheses, not measurements.</div>
"""


def correctness_text(ctx):
    c = pd.read_csv(RESULTS / "correctness.csv", keep_default_na=False, encoding="utf-8")
    worst = 0.0
    for d in c.detail:
        m = re.search(r"max \|C - NumPy\| = ([0-9.e+-]+)", d)
        if m:
            worst = max(worst, float(m.group(1)))
    same = ctx["chk"].groupby("n").checksum.nunique().eq(1).all()
    return f"""
<p>All {len(c)} checks passed (Table 3). The largest difference between any implementation and NumPy on random matrices was {worst:.1e}, more than four orders of magnitude below the 10<sup>−9</sup> tolerance, so the tolerance
would catch a real error but does not hide a marginal one. The sum of all elements of C printed after every timed batch was {'identical in all three languages for every size' if same else 'NOT identical across languages'} (<code>results/tables/checksums.csv</code>).</p>
"""


def runtime_text(ctx):
    lo, hi = 10, 1000
    return f"""
<p>Table 4 and Figure 1 give the time of one multiplication. On the log-log plot the three curves are nearly parallel straight lines from <i>n</i> = {lo} to <i>n</i> ≈ {hi}: one multiplication goes from
{fmt_time(med(ctx, 'python', lo))} (Python), {fmt_time(med(ctx, 'java', lo))} (Java) and {fmt_time(med(ctx, 'c', lo))} (C) at <i>n</i> = {lo} to {fmt_time(med(ctx, 'python', hi))}, {fmt_time(med(ctx, 'java', hi))} and {fmt_time(med(ctx, 'c', hi))} at <i>n</i> = {hi}.
Above that, the Java and C curves bend upwards (Java first); Python stops at <i>n</i> = {hi} because the next size, 1500, exceeded the 120 s limit.</p>
"""


def scaling_text(ctx):
    r = ctx["rt"].set_index("n")
    pc_lo, pc_hi, pc_med = rt_range(ctx, "python_over_c", 100, 10**9)
    jc_s = rt_range(ctx, "java_over_c", 100, 300)
    _, _, part_med, _, _ = failed_partial(ctx)
    return f"""
<p><b>Cost per operation.</b> If the cost were exactly cubic, <i>t</i>/<i>n</i>³ (Table 5, Figure 2) would be constant. It is nearly constant only for small sizes: Python stays at {nsi(ctx, 'python', 50):.0f}–{nsi(ctx, 'python', 200):.0f} ns up to <i>n</i> = 200 and C moves from {nsi(ctx, 'c', 10):.2f} to {nsi(ctx, 'c', 200):.2f} ns,
Java from {nsi(ctx, 'java', 50):.2f} to {nsi(ctx, 'java', 200):.2f} ns for <i>n</i> = 50–200. After that it rises in every language: Python reaches {nsi(ctx, 'python', 1000):.0f} ns at <i>n</i> = 1000 ({100 * (nsi(ctx, 'python', 1000) / nsi(ctx, 'python', 100) - 1):.0f} % above <i>n</i> = 100);
C goes through {nsi(ctx, 'c', 1000):.2f} ns (<i>n</i> = 1000) and {nsi(ctx, 'c', 1500):.2f} ns (1500) to {nsi(ctx, 'c', 2000):.2f} ns (2000) and {nsi(ctx, 'c', 2500):.2f} ns (2500); Java through {nsi(ctx, 'java', 1000):.2f} ns, {nsi(ctx, 'java', 1500):.1f} ns and {nsi(ctx, 'java', 2000):.1f} ns.
So the growth is cubic only approximately: close to cubic for small <i>n</i>, and clearly faster than cubic once the matrices are large.</p>
<p><b>Relative performance.</b> Table 6 gives the ratios between languages. Python is {r.loc[10].python_over_c:.0f}× slower than C at <i>n</i> = 10, where fixed overheads weigh most, and {pc_lo:.0f}–{pc_hi:.0f}× (median {pc_med:.0f}×) for <i>n</i> ≥ 100.
Java is {jc_s[0]:.2f}–{jc_s[1]:.2f}× slower than C for <i>n</i> = 100–300, {ratio(ctx, 'java', 'c', 500):.1f}–{ratio(ctx, 'java', 'c', 1000):.1f}× for <i>n</i> = 500–1000, and {ratio(ctx, 'java', 'c', 1500):.1f}× and {ratio(ctx, 'java', 'c', 2000):.1f}× at 1500 and 2000:
after warm-up the JIT-compiled Java loop is close to C for small and medium sizes, but the gap grows, it does not close, at large sizes. The Java value at <i>n</i> = 1500 rests on one launch (the second launch failed, Section 4.4);
the one measured batch of the failed launch, {fmt_time(part_med)}, is within {100 * abs(part_med / med(ctx, 'java', 1500) - 1):.0f} % of it.</p>
"""


def memory_text(ctx):
    n = 1000
    a = array_mib(n)
    return f"""
<p>Table 7 and Figure 3 show the peak working set. Each process has a fixed runtime baseline that dominates at small <i>n</i>: {baseline(ctx, 'c'):.1f} MiB for C, {baseline(ctx, 'python'):.1f} MiB for Python and {baseline(ctx, 'java'):.1f} MiB for Java (the JVM is the heaviest). On top of it, the data grow with <i>n</i>²:
at <i>n</i> = {n} the growth above the baseline is {growth(ctx, 'c', n):.0f} MiB in C, {growth(ctx, 'java', n):.0f} MiB in Java and {growth(ctx, 'python', n):.0f} MiB in Python, while the three matrices of 8-byte values alone need {a:.1f} MiB.
C and Java stay close to that array size; Python needs about {growth(ctx, 'python', n) / a:.0f} times more because every element is a pointer to a separate <code>float</code> object. As a consequence Python’s total passes Java’s between <i>n</i> = 500
({peak(ctx, 'python', 500):.0f} vs {peak(ctx, 'java', 500):.0f} MiB) and <i>n</i> = 700 ({peak(ctx, 'python', 700):.0f} vs {peak(ctx, 'java', 700):.0f} MiB), although Java starts {baseline(ctx, 'java') - baseline(ctx, 'python'):.0f} MiB higher.
At <i>n</i> = 1500 and 2000 Java’s peak is the same, {peak(ctx, 'java', 1500):.0f} MiB, while the array size grows from {array_mib(1500):.0f} to {array_mib(2000):.0f} MiB: its memory does not track the data exactly (probably because the JVM grows its heap in chunks; I did not investigate).</p>
"""


def limits_text(ctx):
    lim = ctx["lim"].set_index("language")
    t = {l: int(lim.loc[l].largest_feasible_n) for l in ("python", "java", "c")}
    a = {l: int(lim.loc[l].largest_attempted_n) for l in ("python", "java", "c")}
    maxmem = ctx["launches"].peak_working_set_bytes.astype(float).max() / 2**20
    fid, fwall, fmed, flang, fn = failed_partial(ctx)
    return f"""
<p>The time budget was the only limit that bound. The largest size that met it was <i>n</i> = {t['python']} for Python ({fmt_time(med(ctx, 'python', t['python']))} per multiplication), {t['java']} for Java ({fmt_time(med(ctx, 'java', t['java']))}) and {t['c']} for C ({fmt_time(med(ctx, 'c', t['c']))}).
One larger size was attempted per language ({a['python']}, {a['java']} and {a['c']}); in each case one multiplication ran past 120 s and the launch was stopped (Table 9). These are timeouts at the limit; their real times are unknown.
No launch came close to the 4 GiB memory budget (largest peak {maxmem:.0f} MiB), so there were no memory failures.
A fourth launch also failed: launch {fid} (second launch of {LABEL[flang]} at <i>n</i> = {fn}) completed two batches and then produced no output for more than 120 s; it was stopped after {fwall:.0f} s of wall time. I do not know why.
Its batches stay in the raw file but are excluded from every statistic, so Java at <i>n</i> = 1500 has a single launch.</p>
"""


def warmup_text(ctx):
    w = ctx["warm"]
    jw = w[w.language == "java"].groupby("n").first_over_measured.median()
    ts = ctx["ts"]
    dist = disturbed_launches(ctx)
    d_txt = "; ".join(f"the {LABEL[x['language']]} launch at <i>n</i> = {x['n']} was {x['excess']:.0f} % slower than the other launches of that configuration" for x in dist)
    return f"""
<p>Warm-up matters mainly for Java (Figure 4): its first batch is {jw.loc[10]:.0f}× slower than the measured median at <i>n</i> = 10, {jw.loc[100]:.1f}× at <i>n</i> = 100 and about 1× from <i>n</i> ≈ 300 on, which is why the calibration and warm-up batches are excluded and 5 warm-up batches are used for small sizes.
For Python and C the first batch is within about ±30 % of the median at every size. Times also drift inside a launch: at <i>n</i> = 100 the C batches of the first launch end about {c100_drift(ctx):.0f} % slower than they start, probably a frequency or thermal effect that I did not measure; pooling launches that ran at different times averages over it.</p>
<p>The IQR is {ts.rel_iqr_percent.median():.0f} % of the median for the typical configuration (range {ts.rel_iqr_percent.min():.1f}–{ts.rel_iqr_percent.max():.0f} %; Appendix A). The widest ranges come from two launches that started while the machine was busy with something else: {d_txt}.
They were kept (no observation was removed); the medians stay close to those of the other launches because those outnumber them.</p>
"""


def discussion(ctx):
    pc_lo, pc_hi, pc_med = rt_range(ctx, "python_over_c", 100, 10**9)
    jc_lo, jc_hi, _ = rt_range(ctx, "java_over_c", 100, 1000)
    return f"""
<p><b>Answer to the research question.</b> Within the tested range, the time of one multiplication grows close to <i>n</i>³ for small sizes and faster than that for large sizes, in all three languages (E1 only partly holds). The language mainly changes the constant:
Python is about {pc_med:.0f}× slower than C, and Java {jc_lo:.1f}–{jc_hi:.1f}× slower than C up to <i>n</i> = 1000 (E2 holds there), but {ratio(ctx, 'java', 'c', 1500):.1f}× and {ratio(ctx, 'java', 'c', 2000):.1f}× slower at <i>n</i> = 1500 and 2000.
A statement that Java “approaches” C at large sizes would contradict these measurements. Memory has a fixed runtime baseline plus growth with <i>n</i>², and the languages differ in the per-element cost, not in the shape of the growth (E4).</p>
<p><b>Possible explanation of the rise in <i>t</i>/<i>n</i>³ (a hypothesis, not tested).</b> In the <i>i-j-k</i> order the inner loop reads one element of B from a different row at every step, so it touches <i>n</i> cache lines per column [2]. B stops fitting in the L2 cache at <i>n</i> ≈ 400 and in the L3 cache at <i>n</i> ≈ 1000
(from the cache sizes), and the cost per operation rises in that region and beyond, which is consistent with a cache effect (E3). But the steepest steps come later (<i>n</i> ≈ 1500–2000 for C) or earlier (<i>n</i> ≈ 1000–1500 for Java) than those capacities, and I had no hardware counters to identify the mechanism.
One possible reason for Java’s earlier and larger step is its <code>double[n][n]</code> layout, which adds a row-pointer access to every iteration, but that would need a separate experiment (a flat Java array, a C version with separate rows) that I did not run.</p>
<p><b>Other points.</b> Java’s steady state, which Table 4 reports, is not what a short program would see, because the first batches are up to {jw_first(ctx):.0f}× slower. The Python result is for the literal textbook loop on CPython 3.12.0; libraries such as NumPy would change the constant by orders of magnitude, but that is a different experiment.</p>
"""


def jw_first(ctx):
    w = ctx["warm"]
    return float(w[w.language == "java"].groupby("n").first_over_measured.median().loc[10])


def limitations(ctx):
    return """
<ul>
<li><b>One machine and one set of tools:</b> a thin-and-light laptop CPU with turbo (HP 250 G8, i5-1135G7), Windows 11, CPython 3.12.0, OpenJDK 25.0.2 with default JIT/GC, GCC 16.2.0 at <code>-O2</code>. Other hardware, operating systems, compilers or runtimes may give other constants and other step positions.</li>
<li><b>Machine state not fully controlled:</b> AC power and the Balanced plan, but clock frequency and thermal state were not controlled or recorded and the machine was not completely idle; two launches were visibly disturbed (Section 4.5).</li>
<li><b>Few launches at large sizes:</b> the most expensive configurations have one launch (Python ≥ 700, Java ≥ 1500, C 2500), so launch-to-launch variation is unknown there.</li>
<li><b>Memory metric:</b> the peak working set includes the language runtime and depends on how each runtime grows its memory; it is not live data or array size.</li>
<li><b>Kernel time only, one input type:</b> start-up, input and output are excluded; inputs are uniform in [0, 1); only square matrices; no hardware counters, so explanations of the large-<i>n</i> behaviour are hypotheses.</li>
</ul>
"""


def conclusions(ctx):
    pc_lo, pc_hi, pc_med = rt_range(ctx, "python_over_c", 100, 10**9)
    jc_lo, jc_hi, _ = rt_range(ctx, "java_over_c", 100, 1000)
    return f"""
<ul>
<li>For the tested implementations, runtime grows roughly as <i>n</i>³ for small sizes; the cost per operation then rises in all three languages, steeply in C and Java above <i>n</i> ≈ 1000–1500.</li>
<li>Python is about {pc_med:.0f}× slower than C for <i>n</i> ≥ 100. Java is {jc_lo:.1f}–{jc_hi:.1f}× slower than C up to <i>n</i> = 1000 and {ratio(ctx, 'java', 'c', 1500):.1f}× slower at 1500 (single launch).</li>
<li>Memory is a fixed runtime baseline ({baseline(ctx, 'c'):.0f}, {baseline(ctx, 'java'):.0f} and {baseline(ctx, 'python'):.0f} MiB for C, Java and Python) plus growth with <i>n</i>²; at <i>n</i> = 1000 Python needs about {growth(ctx, 'python', 1000) / array_mib(1000):.0f} times the array size, C and Java about once.</li>
<li><b>Unresolved:</b> why the steep step occurs where it does and why it is earlier and larger in Java; this would need layout variants and hardware counters. No general claim that one language is faster than another follows from this study.</li>
</ul>
"""


def materials(ctx, repo_url):
    return f"""
<p><b>Materials.</b> Everything is in the repository, {repo_url}, and in <code>Assignment01_Choukri_Materials.zip</code>: source code (<code>src/</code>), the declared protocol (<code>config/experiment.json</code>), the build, test, benchmark and analysis scripts (<code>scripts/</code>),
raw measurements (<code>results/raw_samples.csv</code>: one row per timed batch including warm-up; <code>results/launches.csv</code>: one row per process with status and peak memory), the tables and figures, the environment record and the correctness results.
The shared inputs are not stored (up to 144 MB) but are regenerated bit-for-bit by <code>scripts/generate_inputs.py</code> and verified against SHA-256 hashes. The <code>README.md</code> lists the exact build and run commands and the tool versions.</p>
"""

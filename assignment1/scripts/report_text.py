"""Narrative sections of the report. Every number is computed from the tables in results/tables, results/*.csv or the config."""
import re
from pathlib import Path

import pandas as pd

from report_lib import fmt_time, table

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


def fit(ctx, lang):
    r = ctx["fits"][ctx["fits"].language == lang].iloc[0]
    return r.exponent, r.ci95_low, r.ci95_high


def local_exp(ctx, lang, n_to):
    d = ctx["loc"]
    return float(d[(d.language == lang) & (d.n_to == n_to)].local_exponent.iloc[0])


def net_bytes(ctx, lang, lo=300, hi=10**9):
    m = ctx["mem"]
    d = m[(m.language == lang) & (m.variant == "main") & (m.n >= lo) & (m.n <= hi)]
    return d.net_bytes_per_element


def baseline(ctx, lang):
    m = ctx["mem"]
    return float(m[(m.language == lang) & (m.n == 10)].peak_ws_median_mib.iloc[0])


def peak(ctx, lang, n):
    m = ctx["mem"]
    return float(m[(m.language == lang) & (m.variant == "main") & (m.n == n)].peak_ws_median_mib.iloc[0])


def disturbed_launches(ctx):
    """Launches that started with > 50 % background CPU, compared with the other launches of the same configuration."""
    s = pd.read_csv(RESULTS / "raw_samples.csv", keep_default_na=False)
    l = ctx["launches"].rename(columns={"sweep": "lsweep"})[["launch_id", "status", "background_cpu_percent", "lsweep"]]
    m = s[s.phase == "measure"].merge(l, on="launch_id")
    m = m[m.status == "ok"].copy()
    m["spm"] = m.seconds_per_multiplication.astype(float)
    m["bg"] = pd.to_numeric(m.background_cpu_percent, errors="coerce")
    out = []
    for (lang, n), g in m.groupby(["language", "n"]):
        per = g.groupby("launch_id").agg(bg=("bg", "first"), med=("spm", "median"))
        bad = per[per.bg > 50]
        for lid, r in bad.iterrows():
            others = per.drop(lid)
            if len(others):
                out.append(dict(launch=int(lid), language=lang, n=int(n), bg=r.bg, med=r.med, others=others.med.median(),
                                excess=100 * (r.med / others.med.median() - 1)))
    return out


def failed_partial(ctx):
    """The failed launch that still completed some batches: (launch id, wall seconds, median s per multiplication of its measured batches)."""
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
    first = sm.iloc[:3].v.mean()
    last = sm[sm.phase == "measure"].iloc[-3:].v.mean()
    return 100 * (last / first - 1)


# ---------------------------------------------------------------- sections
def abstract(ctx):
    n_py, n_j, n_c = (int(ctx["lim"].set_index("language").loc[l].largest_feasible_n) for l in ("python", "java", "c"))
    rt = ctx["rt"]
    big = rt[(rt.n >= 100) & rt.python_over_c.notna()].python_over_c
    jc = rt[(rt.n >= 100) & (rt.n <= 1000)].java_over_c
    pe, pl, ph = fit(ctx, "python"); ce, _, _ = fit(ctx, "c"); je, _, _ = fit(ctx, "java")
    return f"""
<div class="abstract"><b>Abstract.</b> This study measures how the matrix size <i>n</i> affects the runtime and the memory of the textbook triple-loop multiplication of dense <i>n</i> × <i>n</i>
<code>double</code> matrices in Python, Java and C, on one Windows laptop (Intel Core i5-1135G7), with identical input files, loop order and timing protocol.
Sizes were increased until a single multiplication exceeded 120 s: the largest size that met the budget was <i>n</i> = {n_py} for Python, {n_j} for Java and {n_c} for C.
All three implementations return bit-identical results. Kernel time (median and interquartile range over repeated batches and up to three process launches) grows roughly, but not exactly, cubically:
the log-log slope over <i>n</i> = 100–1000 is {pe:.2f} for Python, {je:.2f} for Java and {ce:.2f} for C, and the time per inner iteration rises with <i>n</i> in every language, steeply in C and Java above <i>n</i> ≈ 1500.
Python is about {big.min():.0f}–{big.max():.0f}× slower than C (≈ {big.median():.0f}× typically); Java is {jc.min():.1f}–{jc.max():.1f}× slower than C up to <i>n</i> = 1000 but {ratio(ctx, 'java', 'c', 1500):.1f}× slower at <i>n</i> = 1500.
Peak process memory beyond the runtime baseline is about {net_bytes(ctx, 'c', 1000).median():.1f} bytes per matrix element in C, {net_bytes(ctx, 'java', 500, 1000).median():.1f} in Java (plus a {baseline(ctx, 'java'):.0f} MiB JVM baseline) and {net_bytes(ctx, 'python').median():.1f} in Python.
The conclusions hold for these implementations, settings, inputs and this machine; the causes of the memory-related slowdowns were not measured with hardware counters and are discussed as hypotheses.</div>
"""


def correctness_text(ctx):
    c = pd.read_csv(RESULTS / "correctness.csv", keep_default_na=False)
    worst = 0.0
    for d in c.detail:
        m = re.search(r"exact-sample ([0-9.]+), NumPy-all ([0-9.]+)", d)
        if m:
            worst = max(worst, float(m.group(1)))
    chk = ctx["chk"]
    same = chk.groupby("n").checksum.nunique().eq(1).all()
    return f"""
<p>All {len(c)} checks passed ({int(c.passed.sum())}/{len(c)}; Table 3). For random real matrices the largest error was {worst:.2f} times the theoretical bound, so the bound is a meaningful tolerance and the results are not merely “close”.
The sum of all elements of C that every program prints after every timed batch was {'identical in all three languages for every size' if same else 'NOT identical across languages'} (<code>results/tables/checksums.csv</code>),
and within a launch every batch gave the same checksum, so the timed work was really performed on every batch.</p>
"""


def runtime_text(ctx):
    lo, hi = 10, 1000
    return f"""
<p>Table 4 and Figure 1 give the time of one multiplication. On the log-log plot the three curves are nearly parallel straight lines from <i>n</i> = {lo} to <i>n</i> ≈ {hi}: the cost of one multiplication goes from
{fmt_time(med(ctx, 'python', lo))} (Python), {fmt_time(med(ctx, 'java', lo))} (Java) and {fmt_time(med(ctx, 'c', lo))} (C) at <i>n</i> = {lo} to {fmt_time(med(ctx, 'python', hi))}, {fmt_time(med(ctx, 'java', hi))} and {fmt_time(med(ctx, 'c', hi))} at <i>n</i> = {hi}.
Above that, the Java and C curves bend upwards (Java first); Python stops at <i>n</i> = {hi} because the next size, 1500, exceeded the 120 s limit.</p>
"""


def scaling_text(ctx):
    pe, pl, ph = fit(ctx, "python"); je, jl, jh = fit(ctx, "java"); ce, cl, ch = fit(ctx, "c")
    _, _, part_med, _, _ = failed_partial(ctx)
    return f"""
<p>If the cost were exactly cubic, <i>t</i>/<i>n</i>³ (Figure 2, Table 5) would be flat. It is almost flat only for small <i>n</i>: in Python it stays at {nsi(ctx, 'python', 50):.0f}–{nsi(ctx, 'python', 200):.0f} ns up to <i>n</i> = 200 and then rises to {nsi(ctx, 'python', 1000):.0f} ns at <i>n</i> = 1000
({100 * (nsi(ctx, 'python', 1000) / nsi(ctx, 'python', 100) - 1):.0f} % above <i>n</i> = 100). In C it grows from {nsi(ctx, 'c', 10):.2f} ns at <i>n</i> = 10 through {nsi(ctx, 'c', 500):.2f} ns (500) and {nsi(ctx, 'c', 1000):.2f} ns (1000) and jumps
between <i>n</i> = 1500 ({nsi(ctx, 'c', 1500):.2f} ns) and <i>n</i> = 2000 ({nsi(ctx, 'c', 2000):.2f} ns) to {nsi(ctx, 'c', 2500):.2f} ns at 2500. Java starts at {nsi(ctx, 'java', 50):.2f}–{nsi(ctx, 'java', 10):.2f} ns, reaches {nsi(ctx, 'java', 1000):.2f} ns at 1000, then {nsi(ctx, 'java', 1500):.1f} ns at 1500 and {nsi(ctx, 'java', 2000):.1f} ns at 2000.
The fitted slope over the sizes all three languages completed (<i>n</i> = 100–1000, Table 6) is {pe:.2f} (95 % CI {pl:.2f}–{ph:.2f}) for Python, {je:.2f} ({jl:.2f}–{jh:.2f}) for Java and {ce:.2f} ({cl:.2f}–{ch:.2f}) for C;
{'none of the intervals contains 3, so the growth is cubic only approximately.' if min(pl, jl, cl) > 3 else 'the intervals are close to 3.'} The local exponents (Table 7) show where the extra growth appears: for C the steep step is between <i>n</i> = 1500 and 2000 (local exponent {local_exp(ctx, 'c', 2000):.1f}),
for Java between 1000 and 1500 ({local_exp(ctx, 'java', 1500):.1f}).</p>
<p>Table 8 gives the ratios between languages. Python is {ctx['rt'].set_index('n').loc[10].python_over_c:.0f}× slower than C at <i>n</i> = 10 (fixed loop overhead matters most there) and between
{ctx['rt'][ctx['rt'].n >= 100].python_over_c.min():.0f}× and {ctx['rt'][ctx['rt'].n >= 100].python_over_c.max():.0f}× for <i>n</i> ≥ 100. Java is {ctx['rt'][(ctx['rt'].n >= 100) & (ctx['rt'].n <= 300)].java_over_c.min():.2f}–{ctx['rt'][(ctx['rt'].n >= 100) & (ctx['rt'].n <= 300)].java_over_c.max():.2f}× slower than C for
<i>n</i> = 100–300, {ratio(ctx, 'java', 'c', 500):.1f}–{ratio(ctx, 'java', 'c', 1000):.1f}× for <i>n</i> = 500–1000, and {ratio(ctx, 'java', 'c', 1500):.1f}× and {ratio(ctx, 'java', 'c', 2000):.1f}× at 1500 and 2000.
The Java value at <i>n</i> = 1500 rests on one launch (the second launch failed, Section 4.4); the one measured batch of the failed launch,
{fmt_time(part_med)}, is within {100 * abs(part_med / med(ctx, 'java', 1500) - 1):.0f} % of it.</p>
"""


def memory_text(ctx):
    nb = ctx["mem"].set_index(["language", "n"]).net_bytes_per_element
    py, jv, cc = net_bytes(ctx, "python"), net_bytes(ctx, "java", 200, 1000), net_bytes(ctx, "c", 1000)
    return f"""
<p>Table 9 and Figure 3 show the peak working set. Every process has a runtime baseline that dominates at small <i>n</i>: {baseline(ctx, 'c'):.1f} MiB for C, {baseline(ctx, 'python'):.1f} MiB for Python and {baseline(ctx, 'java'):.1f} MiB for Java.
Above the baseline, C needs {cc.min():.1f}–{cc.max():.1f} bytes per matrix element at <i>n</i> ≥ 1000 (Table 10), i.e. exactly the 8 bytes of array storage per element of A, B and C.
Java needs {jv.min():.1f}–{jv.max():.1f} bytes per element for <i>n</i> = 500–1000 (close to 8, plus its larger baseline; below that the net amount is too small to separate from baseline noise). At <i>n</i> = 1500 and 2000 its peak is the same, {peak(ctx, 'java', 1500):.0f} MiB ({nb.loc[('java', 1500)]:.1f} and {nb.loc[('java', 2000)]:.1f} bytes per element): the JVM grows its heap in steps,
so the working set of the JVM does not track live data exactly; I did not investigate this further. Python needs {py.median():.1f} bytes per element ({py.min():.1f}–{py.max():.1f}), about five times C: each element is an 8-byte pointer to a separate <code>float</code> object of 24 bytes (<code>sys.getsizeof(1.0)</code>),
and the figure is consistent with the allocator rounding that object up to 32 bytes (8 + 32 = 40) [6], although I did not verify the rounding in the interpreter source.
As a result Python overtakes Java in total memory between <i>n</i> = 500 ({peak(ctx, 'python', 500):.0f} vs {peak(ctx, 'java', 500):.0f} MiB) and 700 ({peak(ctx, 'python', 700):.0f} vs {peak(ctx, 'java', 700):.0f} MiB), and at <i>n</i> = 1000 it uses {peak(ctx, 'python', 1000):.0f} MiB against {peak(ctx, 'java', 1000):.0f} (Java) and {peak(ctx, 'c', 1000):.0f} (C).</p>
"""


def limits_text(ctx):
    lim = ctx["lim"].set_index("language")
    t = {l: int(lim.loc[l].largest_feasible_n) for l in ("python", "java", "c")}
    a = {l: int(lim.loc[l].largest_attempted_n) for l in ("python", "java", "c")}
    extra = nsi(ctx, "python", 1000) * 1500**3 * 1e-9
    maxmem = ctx["launches"].peak_working_set_bytes.astype(float).max() / 2**20
    fid, fwall, fmed, flang, fn = failed_partial(ctx)
    return f"""
<p>The time budget was the only limit that bound. The largest size that met it was <i>n</i> = {t['python']} for Python ({fmt_time(med(ctx, 'python', t['python']))} per multiplication), {t['java']} for Java ({fmt_time(med(ctx, 'java', t['java']))}) and {t['c']} for C ({fmt_time(med(ctx, 'c', t['c']))}).
One larger size was attempted per language ({a['python']}, {a['java']} and {a['c']}); in each case a single multiplication ran past 120 s and the launch was stopped (Table 12). These are timeouts at the limit; their true times are unknown. For orientation only,
extrapolating the Python time per iteration at <i>n</i> = 1000 to <i>n</i> = 1500 would give about {extra:.0f} s, which is an extrapolation, not a measurement.
No launch came close to the 4 GiB memory budget (the largest peak working set of any launch was {maxmem:.0f} MiB), so there were no memory failures.
A fourth launch also failed: launch {fid} (second launch of {LABEL[flang]} at <i>n</i> = {fn}) completed two batches ({fmt_time(fmed)} per multiplication for the measured one) and then produced no output for more than 120 s; it was stopped after {fwall:.0f} s of wall time.
I do not know why (a system suspend or heavy foreground activity are possible); its batches stay in the raw file but are excluded from every statistic, which is why Java at <i>n</i> = 1500 has a single launch.</p>
"""


def warmup_text(ctx):
    w = ctx["warm"]
    jw = w[w.language == "java"].groupby("n").first_over_measured.median()
    dist = disturbed_launches(ctx)
    ts = ctx["ts"]
    d_txt = "; ".join(f"launch {x['launch']} (C, <i>n</i> = {x['n']}) was {x['excess']:.0f} % slower than the other launches of that configuration" for x in dist)
    return f"""
<p>Warm-up matters only for Java (Figure 4, <code>results/tables/warmup.csv</code>). Its first batch is {jw.loc[10]:.0f}× slower than the measured median at <i>n</i> = 10, {jw.loc[100]:.1f}× at <i>n</i> = 100 and about 1× from <i>n</i> ≈ 300 on, which is why
5 warm-up batches were used for small sizes and why the first (calibration) batches are excluded from all statistics. For Python and C the first batch is within about ±30 % of the median at every size. Within a launch the times also drift:
at <i>n</i> = 100 the C batches of the first launch become about {c100_drift(ctx):.0f} % slower between the first batches and the last measured ones (Figure 4), which is consistent with the CPU leaving its turbo frequency, but I did not record the clock frequency, so this is an interpretation. Pooling launches that ran at different times is meant to average over this.</p>
<p>The interquartile range of the pooled batches is {ts.rel_iqr_percent.median():.0f} % of the median for the typical configuration (range {ts.rel_iqr_percent.min():.1f}–{ts.rel_iqr_percent.max():.0f} %; Appendix A). The widest ranges come from two launches that started while the machine
was fully busy with something else (background CPU 100 % in the 0.5 s before the launch, <code>background_cpu_percent</code>): {d_txt}. They were kept (no observation was removed), and the medians stay close to those of the undisturbed launches because the other launches outnumber them.
The median background load before launches was {pd.to_numeric(ctx['launches'].background_cpu_percent, errors='coerce').median():.0f} % of all logical processors.</p>
"""


def discussion(ctx):
    rows = [
        ["P1: time ∝ <i>n</i>³, <i>t</i>/<i>n</i>³ flat", "<span class='warn'>partly</span>",
         f"Straight lines on log-log from 10 to 1000, but slopes {fit(ctx, 'python')[0]:.2f} / {fit(ctx, 'java')[0]:.2f} / {fit(ctx, 'c')[0]:.2f} (Python / Java / C), all CIs above 3, and <i>t</i>/<i>n</i>³ grows {nsi(ctx, 'c', 2500) / nsi(ctx, 'c', 10):.0f}× in C between <i>n</i> = 10 and 2500."],
        ["P2: Python ≫ Java ≳ C", "<span class='pass'>confirmed</span>",
         f"Python ≈ {ctx['rt'][ctx['rt'].n >= 100].python_over_c.median():.0f}× C; Java {ctx['rt'][(ctx['rt'].n >= 100) & (ctx['rt'].n <= 1000)].java_over_c.min():.1f}–{ctx['rt'][(ctx['rt'].n >= 100) & (ctx['rt'].n <= 1000)].java_over_c.max():.1f}× C up to <i>n</i> = 1000, but {ratio(ctx, 'java', 'c', 1500):.1f}× at 1500."],
        ["P3: <i>t</i>/<i>n</i>³ rises once B exceeds L2 (<i>n</i> ≈ 400) and L3 (<i>n</i> ≈ 1000)", "<span class='warn'>partly</span>",
         "The rise is real in all three languages, but it is gradual from small <i>n</i> and the steep step is at <i>n</i> ≈ 1500–2000 for C and 1000–1500 for Java, not at the cache capacities I predicted."],
        ["P4: memory ≈ 24<i>n</i>² (C), + larger JVM baseline (Java), much larger per element (Python)", "<span class='pass'>confirmed</span>",
         f"C {net_bytes(ctx, 'c', 1000).median():.1f} B/element; Java {net_bytes(ctx, 'java', 500, 1000).median():.1f} B/element + {baseline(ctx, 'java'):.0f} MiB baseline (heap steps at 1500); Python {net_bytes(ctx, 'python').median():.1f} B/element."],
    ]
    t = table(["Prediction (Section 1)", "Outcome", "Evidence"], rows, caption="Table 13. Predictions against results.")
    return f"""
<p><b>Answer to the research question.</b> Within the tested range, the time of one multiplication grows close to <i>n</i>³ but a little faster, in all three languages, and the language mainly changes the constant:
about {ctx['rt'][ctx['rt'].n >= 100].python_over_c.median():.0f}× between Python and C, and a factor of {ctx['rt'][(ctx['rt'].n >= 100) & (ctx['rt'].n <= 1000)].java_over_c.min():.1f}–{ctx['rt'][(ctx['rt'].n >= 100) & (ctx['rt'].n <= 1000)].java_over_c.max():.1f} between Java and C for <i>n</i> ≤ 1000, after the JIT has warmed up.
Memory beyond a fixed runtime baseline grows with <i>n</i>² and the per-element cost is where the languages differ: about {net_bytes(ctx, 'c', 1000).median():.1f} bytes in C and {net_bytes(ctx, 'java', 500, 1000).median():.1f} in Java, about {net_bytes(ctx, 'python').median():.1f} in Python.</p>
{t}
<p><b>Interpreting the memory-related slowdown (hypotheses, not tested).</b> In the <i>i-j-k</i> order every inner iteration reads one element of B from a different row, so the loop touches <i>n</i> cache lines and <i>n</i> memory pages per column [2].
The gradual rise of <i>t</i>/<i>n</i>³ from small <i>n</i> and the steps at large <i>n</i> fit that picture, but my data do not identify the exact mechanism: the step comes later than the L3 capacity for C, and I had no hardware counters on this machine
(<code>perf</code>, which the course material uses [10], is Linux-only). One possible explanation for the earlier and larger step in Java is its <code>double[n][n]</code> layout, which adds a row-header and row-pointer access to every iteration; the C program, by contrast, uses one contiguous block.
Testing this would need a Java version with a flat array (and a C version with separate rows), which I did not run. The same applies to compiler flags: only <code>-O2</code> was measured, and <code>-O3</code> may reorder loops [5].</p>
<p><b>Other points.</b> The JVM’s start is expensive (first batch {ctx['warm'][(ctx['warm'].language == 'java')].groupby('n').first_over_measured.median().loc[10]:.0f}× slower at <i>n</i> = 10), so Java’s steady state, which is what Table 4 reports, is not what a short program would see.
The Python result is for the literal textbook loop on CPython 3.12.0; libraries such as NumPy or other implementations would change the constant by orders of magnitude, but that is a different experiment.</p>
"""


def limitations(ctx):
    return """
<ul>
<li><b>One machine and one set of tools.</b> A 15 W-class laptop CPU with turbo, Windows 11, CPython 3.12.0, OpenJDK 25.0.2 with default JIT/GC, GCC 16.2.0 at <code>-O2</code>. Other CPUs, operating systems, compilers or runtimes may give different constants and different step positions.</li>
<li><b>Uncontrolled machine state.</b> The laptop was on AC power in the Balanced plan, but clock frequency and thermal state were not controlled or recorded, and the machine was not completely idle (background load is recorded for every launch). Two launches were visibly disturbed (Section 4.5).</li>
<li><b>Few launches at large sizes.</b> The most expensive configurations have one launch (Python ≥ 700, Java ≥ 1500, C 2500), so launch-to-launch variation is unknown there; their IQR describes only the batches inside one process.</li>
<li><b>Memory metric.</b> Peak working set includes the language runtime and depends on how the JVM and allocators grow their memory; it is not live data or array size, and one number per launch says nothing about when the peak occurs.</li>
<li><b>Kernel time only, one input type.</b> Start-up, input and output are excluded; inputs are uniform in [0, 1) only; only square matrices were used.</li>
<li><b>No hardware counters.</b> Explanations of the large-<i>n</i> behaviour are hypotheses.</li>
<li><b>Linux path untested.</b> The driver also contains a Linux memory-measurement branch that was not run.</li>
</ul>
"""


def conclusions(ctx):
    rt = ctx["rt"]
    return f"""
<ul>
<li>For the tested implementations, runtime grows roughly as <i>n</i>³ with a growing per-operation cost (slope {fit(ctx, 'python')[0]:.2f}–{fit(ctx, 'java')[0]:.2f} over <i>n</i> = 100–1000); the per-operation cost rises sharply in C and Java at <i>n</i> ≈ 1500–2000.</li>
<li>Python is about {rt[rt.n >= 100].python_over_c.median():.0f}× slower than C at <i>n</i> ≥ 100; Java is {rt[(rt.n >= 100) & (rt.n <= 1000)].java_over_c.min():.1f}–{rt[(rt.n >= 100) & (rt.n <= 1000)].java_over_c.max():.1f}× slower than C up to <i>n</i> = 1000 and {ratio(ctx, 'java', 'c', 1500):.1f}× slower at 1500.</li>
<li>Memory: ≈ {net_bytes(ctx, 'c', 1000).median():.1f} bytes per element in C, ≈ {net_bytes(ctx, 'java', 500, 1000).median():.1f} in Java and ≈ {net_bytes(ctx, 'python').median():.1f} in Python, on top of baselines of {baseline(ctx, 'c'):.0f}, {baseline(ctx, 'java'):.0f} and {baseline(ctx, 'python'):.0f} MiB (C, Java, Python).</li>
<li><b>Unresolved:</b> why the steep step occurs where it does and why it is larger and earlier in Java; this needs layout variants and hardware counters, which were out of scope. No general claim that one language is faster than another follows from this study.</li>
</ul>
"""


def reproducibility(ctx, repo_url):
    return f"""
<p>Everything is in the repository — {repo_url} — and in <code>Assignment01_Choukri_Materials.zip</code>: source code (<code>src/</code>), the declared protocol (<code>config/experiment.json</code>), the build/test/benchmark/analysis scripts (<code>scripts/</code>),
raw measurements (<code>results/raw_samples.csv</code>, one row per timed batch including warm-up; <code>results/launches.csv</code>, one row per process with status and peak memory), the generated tables and figures, the environment record and the correctness results.
The shared inputs are not stored (up to 144 MB for the largest pair) but are regenerated bit-for-bit by <code>scripts/generate_inputs.py</code> and verified against SHA-256 hashes (<code>--check</code>). The <code>README.md</code> lists the exact build and run commands and the tool versions.
No material from earlier work is reused; the only external material is the course’s benchmarking guidance [10].</p>
"""

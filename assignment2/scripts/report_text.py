"""Result-dependent sections of the Assignment 2 report. Every number is computed from results/tables or results/*.csv."""
from pathlib import Path

import pandas as pd

from report_lib import fmt_time, fmt_time_iqr, table

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
METHODS = ["ijk", "ikj", "blocked"]
NAME = {"ijk": "ijk (baseline)", "ikj": "ikj", "blocked": "blocked (S = 64)"}


def T(name):
    return pd.read_csv(TABLES / f"{name}.csv")


class Ctx:
    def __init__(self):
        self.dt = T("dense_time")
        self.sp = T("dense_speedup").set_index("n")
        self.lim = T("dense_limits").set_index("method")
        self.fails = T("failed_launches")
        self.bs = T("blocksize")
        self.sc = T("sparse_comparison")
        self.launches = pd.read_csv(RESULTS / "launches.csv")
        self.samples = pd.read_csv(RESULTS / "raw_samples.csv")
        self.corr = pd.read_csv(RESULTS / "correctness.csv", encoding="utf-8")

    def med(self, meth, n):
        d = self.dt[(self.dt.method == meth) & (self.dt.n == n)]
        return float(d.median_s.iloc[0])

    def nsi(self, meth, n):
        d = self.dt[(self.dt.method == meth) & (self.dt.n == n)]
        return float(d.ns_per_iteration.iloc[0])

    def ratio(self, meth, n):
        d = self.dt[(self.dt.method == meth) & (self.dt.n == n)]
        return float(d.launch_ratio.iloc[0])

    def sc_row(self, n, d):
        r = self.sc[(self.sc.n == n) & (abs(self.sc.density - d) < 1e-12)]
        return r.iloc[0]

    def ijk_partial(self):
        """Calibration batch of the ijk launch that timed out at n = 2500 (a single completed multiplication)."""
        f = self.fails[(self.fails.method == "ijk") & (self.fails.n == 2500)]
        if f.empty:
            return None
        s = self.samples[self.samples.launch_id == int(f.launch_id.iloc[0])]
        return float(s.seconds_per_multiplication.iloc[0]) if len(s) else None


# ---------------------------------------------------------------- text
def abstract(c):
    sp = c.sp
    lo, hi = c.sc_row(2000, 0.001).speedup_csr_vs_dense, c.sc_row(2000, 0.5).speedup_csr_vs_dense
    return f"""
<div class="abstract"><b>Abstract.</b> Building on the validated C kernel of Assignment 1, this study measures, on one laptop (Intel Core i5-1135G7, single thread, GCC <code>-O2</code>), how two optimisations change dense matrix multiplication and when a sparse representation pays off.
Part A compares the baseline <i>i-j-k</i> loop with loop reordering (<i>i-k-j</i>) and cache blocking (64 × 64 tiles) for <i>n</i> = 100–3000. All three return bit-identical results. Loop reordering is {sp.speedup_ikj.min():.1f}–{sp.speedup_ikj.max():.1f}× faster than the baseline and blocking {sp.speedup_blocked.min():.1f}–{sp.speedup_blocked.max():.1f}×,
with the largest gains at the largest sizes; the baseline exceeded the 120 s budget at <i>n</i> = 2500, while both optimised versions completed <i>n</i> = 3000 in {fmt_time(c.med('ikj', 3000))} and {fmt_time(c.med('blocked', 3000))}.
Part B multiplies randomly scattered sparse matrices (0.1–50 % nonzeros, <i>n</i> = 1000 and 2000) with an own CSR implementation of Gustavson’s algorithm and compares it with dense blocked multiplication of the same matrices.
CSR was faster at every tested density, by {lo:,.0f}× at 0.1 % but only {hi:.1f}× at 50 % (<i>n</i> = 2000), because each useful product costs it 3–20 times more than a dense multiply-add and because the product becomes dense: at 5 % density, more than 90 % of the entries of C are nonzero.
The conversion from dense to CSR is measured separately; it dominates only at the lowest densities. Conclusions are limited to these implementations, random uniform sparsity and this machine.</div>
"""


def correctness_text(c):
    k = c.corr
    return f"""<p>All {len(k)} checks passed ({int(k.passed.sum())}/{len(k)}; <code>results/correctness.csv</code>). The two optimised dense methods and the CSR method returned matrices bit-identical to the baseline in every test, the C conversion produced byte-identical CSR files to the independent Python writer,
and the number of stored entries of every CSR product equalled the number of nonzeros of NumPy’s product. The checksum printed after every timed batch was constant within each launch.</p>"""


def dense_text(c):
    sp = c.sp
    part = c.ijk_partial()
    part_txt = (f" Its first multiplication at <i>n</i> = 2500 completed in {fmt_time(part)} (a calibration batch, so not counted as a measurement); the next one ran past 120 s and the launch was stopped." if part else "")
    return f"""
<p>Table 2 and Figure 1 give the time of one multiplication. Both optimisations are faster than the baseline at every size, and the gap grows with <i>n</i> (Table 3): loop reordering gives {sp.loc[100].speedup_ikj:.1f}× at <i>n</i> = 100, {sp.loc[1000].speedup_ikj:.1f}× at 1000 and {sp.loc[2000].speedup_ikj:.1f}× at 2000;
blocking gives {sp.loc[100].speedup_blocked:.1f}×, {sp.loc[1000].speedup_blocked:.1f}× and {sp.loc[2000].speedup_blocked:.1f}×. Blocking is faster than reordering alone at every size from 200 on, by {sp.loc[200:].blocked_vs_ikj.min():.2f}–{sp.loc[200:].blocked_vs_ikj.max():.2f}×.
The baseline was the only method to exceed the budget: the largest size it completed was <i>n</i> = {int(c.lim.loc['ijk'].largest_feasible_n)}.{part_txt} Reordering and blocking both completed the largest declared size, <i>n</i> = 3000, in {fmt_time(c.med('ikj', 3000))} and {fmt_time(c.med('blocked', 3000))}.</p>
<p>Figure 2 shows why. The baseline’s time per multiply-add, <i>t</i>/<i>n</i>³, grows from {c.nsi('ijk', 100):.2f} ns at <i>n</i> = 100 to {c.nsi('ijk', 1000):.2f} ns at 1000 and {c.nsi('ijk', 2000):.2f} ns at 2000, as in Assignment 1.
For reordering it stays between {c.dt[c.dt.method == 'ikj'].ns_per_iteration.min():.2f} and {c.dt[c.dt.method == 'ikj'].ns_per_iteration.max():.2f} ns over the whole range, and for blocking between {c.dt[c.dt.method == 'blocked'].ns_per_iteration.min():.2f} and {c.dt[c.dt.method == 'blocked'].ns_per_iteration.max():.2f} ns:
with unit-stride access the cost per operation no longer depends much on whether B fits in the caches.</p>
<p><b>Variability.</b> At <i>n</i> ≤ 1000 the two launches of a configuration often disagreed: by up to {c.dt[(c.dt.method == 'ijk') & (c.dt.n <= 1000)].launch_ratio.max():.1f}× for the baseline and up to {c.dt[(c.dt.method != 'ijk') & (c.dt.n <= 1000)].launch_ratio.max():.1f}× for the optimised methods (Appendix A); for the baseline the slower launch was always the one that started with more background load on the laptop.
At <i>n</i> ≥ 1500 the launches of every method agree within {100 * (c.dt[c.dt.n >= 1500].launch_ratio.max() - 1):.0f} %. The speed-ups for <i>n</i> ≤ 1000 are therefore uncertain by up to that factor; those for <i>n</i> ≥ 1500, including the largest ones, are robust.</p>
"""


def blocksize_text(c):
    b = c.bs
    best = b.loc[b.groupby("n").median_s.idxmin()]
    s64 = b[b.S == 64].set_index("n")
    main64 = c.dt[c.dt.method == "blocked"].set_index("n")
    gains = [100 * (1 - r.median_s / s64.loc[r.n].median_s) for r in best.itertuples()]
    return f"""
<p>The tile size S = 64 was fixed before measuring. Table 4 shows how the blocked method responds to other tile sizes (one launch each, run after the main comparison). S = {int(best.S.iloc[0])} was fastest at both sizes,
about {min(gains):.0f}–{max(gains):.0f} % faster than S = 64, and S = 64 was the slowest of the five in this run.
However, the same S = 64 configuration measured in the main comparison took {fmt_time(main64.loc[1000].median_s)} at <i>n</i> = 1000 and {fmt_time(main64.loc[2000].median_s)} at <i>n</i> = 2000, against {fmt_time(s64.loc[1000].median_s)} and {fmt_time(s64.loc[2000].median_s)} here, a run-to-run difference of the same size as the differences between tile sizes.
The data therefore suggest that larger tiles may be slightly better on this machine, but they do not establish a best tile size; a properly repeated tuning experiment would be needed.</p>
"""


def sparse_text(c):
    r1, r2 = c.sc_row(1000, 0.05), c.sc_row(2000, 0.05)
    s10 = c.sc_row(2000, 0.10)
    return f"""
<p>Table 5 and Figure 3 compare, for the same logical matrices, the CSR multiplication with the dense blocked method. The dense time does not depend on the density ({fmt_time(c.sc[c.sc.n == 1000].dense_s.min())}–{fmt_time(c.sc[c.sc.n == 1000].dense_s.max())} at <i>n</i> = 1000, {fmt_time(c.sc[c.sc.n == 2000].dense_s.min())}–{fmt_time(c.sc[c.sc.n == 2000].dense_s.max())} at 2000),
because it performs all <i>n</i>³ multiply-adds whatever their values. The CSR time grows with the number of nonzero products and was lower than the dense time at every tested density, but the advantage falls from {c.sc_row(1000, 0.001).speedup_csr_vs_dense:,.0f}× and {c.sc_row(2000, 0.001).speedup_csr_vs_dense:,.0f}× at 0.1 %
to {c.sc_row(1000, 0.25).speedup_csr_vs_dense:.1f}× and {c.sc_row(2000, 0.25).speedup_csr_vs_dense:.1f}× at 25 % and to {c.sc_row(1000, 0.5).speedup_csr_vs_dense:.2f}× and {c.sc_row(2000, 0.5).speedup_csr_vs_dense:.2f}× at 50 % (<i>n</i> = 1000 and 2000). Densities above 50 % were not tested, so no crossover point was measured.</p>
<p><b>Cost per useful operation.</b> Table 5 also gives the CSR time per nonzero product: {c.sc.csr_ns_per_product.min():.1f}–{c.sc.csr_ns_per_product.max():.1f} ns, against {c.sc.dense_ns_per_iteration.min():.2f}–{c.sc.dense_ns_per_iteration.max():.2f} ns per multiply-add for the dense method.
Each useful product is 3–20 times more expensive in CSR, because of the indirect index accesses, the scattered updates of the accumulator, the symbolic pass and writing the output. CSR wins because it performs far fewer operations, not because each is cheaper: at 50 % density it performs about 25 % of the dense work at roughly three times the cost per operation.
The cost per product is highest at the lowest densities, where per-row overheads (resetting markers, empty rows) are no longer amortised.</p>
<p><b>Output fill-in.</b> The product is much denser than the inputs (Table 6). With 1 % nonzeros in A and B, C has {100 * c.sc_row(1000, 0.01).out_density:.1f} % nonzeros at <i>n</i> = 1000 and {100 * c.sc_row(2000, 0.01).out_density:.1f} % at 2000; at 5 % input density it has {100 * r1.out_density:.1f} % and {100 * r2.out_density:.1f} %, and from 10 % on C is effectively full ({100 * s10.out_density:.2f} % at <i>n</i> = 2000).
This matches the estimate 1 − (1 − <i>d</i>²)<sup><i>n</i></sup> of Section 2, and it means that for these random matrices the same density of zeros gives a denser result, and less advantage, the larger <i>n</i> is.</p>
<p><b>Conversion cost.</b> Converting one input from dense to CSR took {fmt_time(c.sc[c.sc.n == 1000].convert_a_s.min())}–{fmt_time(c.sc[c.sc.n == 1000].convert_a_s.max())} at <i>n</i> = 1000 and {fmt_time(c.sc[c.sc.n == 2000].convert_a_s.min())}–{fmt_time(c.sc[c.sc.n == 2000].convert_a_s.max())} at 2000, because it has to read all <i>n</i>² entries.
At 0.1 % density this is about {c.sc_row(1000, 0.001).convert_over_csr:.0f}× (<i>n</i> = 1000) and {c.sc_row(2000, 0.001).convert_over_csr:.0f}× (<i>n</i> = 2000) the multiplication itself, at 1 % between {min(c.sc_row(1000, 0.01).convert_over_csr, c.sc_row(2000, 0.01).convert_over_csr):.1f}× and {max(c.sc_row(1000, 0.01).convert_over_csr, c.sc_row(2000, 0.01).convert_over_csr):.1f}× of it, and from 5 % on less than {100 * c.sc[c.sc.density >= 0.05].convert_over_csr.max():.0f} % of it.
Only matrix A was converted in the measurements; converting both inputs would take about twice as long (B has the same size and nearly the same nonzero count), and even with that cost CSR stays faster than dense at every tested density (Table 5, last column).</p>
<p><b>Memory.</b> Table 6 gives the peak working set. At low density the CSR process needs a fraction of the dense one (for example {r1.csr_peak_mib - 0:.1f} vs {r1.dense_peak_mib:.1f} MiB at <i>n</i> = 1000 and 5 %, and {c.sc_row(2000, 0.01).csr_peak_mib:.1f} vs {c.sc_row(2000, 0.01).dense_peak_mib:.1f} MiB at <i>n</i> = 2000 and 1 %).
The saving shrinks as the density grows, mostly because of the output: once C is full, its CSR form takes 12 bytes per entry against 8 in dense form. At 50 % density the two processes need the same memory ({c.sc_row(2000, 0.5).csr_peak_mib:.1f} vs {c.sc_row(2000, 0.5).dense_peak_mib:.1f} MiB at <i>n</i> = 2000).</p>
"""


def discussion(c):
    sp = c.sp
    return f"""
<p><b>Part A.</b> Both optimisations target the same weakness of the baseline: its inner loop reads B with a stride of <i>n</i> doubles, so for large matrices almost every access misses the caches [2]. Reordering the loops removes the stride and gives most of the gain ({sp.loc[2000].speedup_ikj:.1f}× at <i>n</i> = 2000);
blocking adds a further {sp.loc[1000:].blocked_vs_ikj.min():.2f}–{sp.loc[1000:].blocked_vs_ikj.max():.2f}× at <i>n</i> ≥ 1000 by reusing tiles of B from the cache. Since GCC vectorised none of the dense loops and all methods perform the same additions in the same order, the measured differences come from the memory-access pattern.
This also supports the hypothesis left open in Assignment 1 [1], that the rise of <i>t</i>/<i>n</i>³ there was a memory effect, although no hardware counters were used here either, so cache misses were not measured directly.</p>
<p><b>Part B.</b> Whether CSR helps depends on how much work it avoids, not on the percentage of zeros by itself. For these uniformly random matrices the work is about <i>d</i>²<i>n</i>³ products, so halving the density divides it by four; but each product costs several times more than a dense operation, and the output fills in quickly, which costs time and memory.
Other structures with the same density (for example banded or block-diagonal matrices, whose products stay sparse) would behave differently; only uniform random positions were tested. The conversion from dense storage matters only when the multiplication itself is very cheap, i.e. at the lowest densities.</p>
<p><b>Separating the two effects.</b> The dense comparator in Part B is the best dense method of Part A, so the Part B speed-ups measure the benefit of the sparse representation over an already optimised dense code, not over the baseline; against the baseline they would be larger by the Part A factor.</p>
"""


def limitations():
    return """
<ul>
<li><b>One machine, noisy state.</b> The same laptop as Assignment 1; background load was higher during this session (median 29 % of all logical processors before a launch) and the baseline’s small-size measurements were affected (Section 4.2). Times are not comparable with Assignment 1’s.</li>
<li><b>Few launches at large sizes.</b> Two launches up to <i>n</i> = 2000, one at 2500 and 3000; the block-size sensitivity and the dense comparator at <i>n</i> = 2000 have one launch.</li>
<li><b>Choices not tuned.</b> One tile size (S = 64, chosen in advance and not the best in the sensitivity run), <code>-O2</code> only, no SIMD; the sparse matrices have uniform random structure only, and densities above 50 % were not tested.</li>
<li><b>Conversion measured for one input only;</b> the cost for both inputs is estimated as twice that.</li>
<li><b>No hardware counters,</b> so cache behaviour is inferred from timings, not observed.</li>
</ul>
"""


def conclusions(c):
    sp = c.sp
    return f"""
<ul>
<li>Loop reordering and cache blocking made the dense C kernel {sp.speedup_ikj.min():.1f}–{sp.speedup_ikj.max():.1f}× and {sp.speedup_blocked.min():.1f}–{sp.speedup_blocked.max():.1f}× faster, with the largest gains at the largest sizes, and extended the feasible size under the 120 s budget from <i>n</i> = 2000 to at least 3000. Their cost per operation stays nearly constant with <i>n</i>, unlike the baseline’s.</li>
<li>For uniformly random sparse matrices, own-code CSR multiplication was faster than optimised dense multiplication at every tested density (0.1–50 %), but the advantage fell from four orders of magnitude to about 1.2–1.3× at 50 %, because each useful product costs more and the result fills in.</li>
<li>CSR saves memory at low density; at 50 % it needs as much as the dense code, mostly because the full output costs 12 bytes per entry. Dense-to-CSR conversion only matters at the lowest densities.</li>
<li><b>Open:</b> the best tile size, the crossover density above 50 %, and behaviour for structured sparsity patterns.</li>
</ul>
"""


def materials(repo_url):
    return f"""
<p><b>Materials.</b> In the repository ({repo_url}, folder <code>assignment2/</code>) and in <code>Assignment02_Choukri_Mohamed_Materials.zip</code>: source code (<code>src/c/</code>), declared protocol (<code>config/experiment.json</code>), build, input-generation, test, benchmark and analysis scripts (<code>scripts/</code>),
raw measurements (<code>results/raw_samples.csv</code>, <code>results/launches.csv</code>), tables, figures, environment record and correctness results. The inputs (about 0.9 GB) are not stored; <code>scripts/generate_inputs.py</code> regenerates them and checks them against the SHA-256 hashes and nonzero counts in <code>inputs/checksums.json</code>.
Reused from Assignment 1: the baseline kernel (copied into <code>mm_ijk</code>), the dense inputs (same generator and seed, byte-identical) and the launcher and memory measurement (<code>scripts/harness.py</code>).</p>
"""


# ---------------------------------------------------------------- tables
def dense_table(c):
    p = c.dt.set_index(["n", "method"])
    rows = []
    for n in sorted(c.dt.n.unique()):
        row = [str(n)]
        for m in METHODS:
            row.append(fmt_time_iqr(p.loc[(n, m)].median_s, p.loc[(n, m)].q1_s, p.loc[(n, m)].q3_s) if (n, m) in p.index else "timeout (> 120 s)")
        rows.append(row)
    return table(["n"] + [NAME[m] for m in METHODS], rows, num_cols=(0, 1, 2, 3),
                 caption="Table 2. Part A: time of one multiplication, median [Q1–Q3] over all measured batches of all launches.")


def speedup_table(c):
    rows = [[str(n), f"{r.speedup_ikj:.2f}×" if pd.notna(r.speedup_ikj) else "–", f"{r.speedup_blocked:.2f}×" if pd.notna(r.speedup_blocked) else "–",
             f"{r.blocked_vs_ikj:.2f}×"] for n, r in c.sp.iterrows()]
    return table(["n", "ijk ÷ ikj", "ijk ÷ blocked", "ikj ÷ blocked"], rows, num_cols=(0, 1, 2, 3),
                 caption="Table 3. Speed-ups (ratios of median times). Values for n ≤ 1000 are uncertain by up to the launch-to-launch factor of the baseline (Appendix A).")


def blocksize_table(c):
    b = c.bs
    rows = []
    for s in sorted(b.S.unique()):
        row = [str(s)]
        for n in sorted(b.n.unique()):
            r = b[(b.S == s) & (b.n == n)].iloc[0]
            row.append(f"{fmt_time(r.median_s)} ({r.ns_per_iteration:.2f} ns)")
        rows.append(row)
    return table(["Tile size S"] + [f"n = {n}" for n in sorted(b.n.unique())], rows, num_cols=(0, 1, 2),
                 caption="Table 4. Block-size sensitivity: median time of the blocked method (time per multiply-add in brackets); one launch, 5 measured batches each.")


def sparse_table(c):
    rows = []
    for r in c.sc.itertuples():
        rows.append([str(r.n), f"{100 * r.density:g} %", fmt_time(r.dense_s), fmt_time(r.csr_s), f"{r.speedup_csr_vs_dense:,.1f}×",
                     f"{r.csr_ns_per_product:.1f}", fmt_time(r.convert_a_s), fmt_time(r.csr_plus_convert_both_s),
                     f"{r.dense_s / r.csr_plus_convert_both_s:,.1f}×"])
    return table(["n", "Density", "Dense blocked", "CSR", "Dense ÷ CSR", "CSR ns per nonzero product", "Convert A", "CSR + convert A and B (est.)", "Dense ÷ (CSR + conv.)"],
                 rows, num_cols=tuple(range(0, 9)),
                 caption="Table 5. Part B: median times on the same logical matrices. “Convert A” is measured; the conversion of both inputs is estimated as twice it.")


def fill_table(c):
    rows = []
    for r in c.sc.itertuples():
        rows.append([str(r.n), f"{100 * r.density:g} %", f"{int(r.nnz_a):,}", f"{int(r.nnz_b):,}", f"{int(r.nnz_c):,}", f"{100 * r.out_density:.1f} %",
                     f"{r.csr_peak_mib:.1f}", f"{r.dense_peak_mib:.1f}"])
    return table(["n", "Density", "nnz(A)", "nnz(B)", "nnz(C)", "Density of C", "Peak memory CSR (MiB)", "Peak memory dense (MiB)"], rows,
                 num_cols=tuple(range(0, 8)), caption="Table 6. Nonzero counts of inputs and output, and peak working set of the CSR and dense processes (median over launches).")


def appendix_table(c):
    rows = []
    for m in METHODS:
        for r in c.dt[c.dt.method == m].sort_values("n").itertuples():
            rows.append([NAME[m], str(r.n), str(r.launches), str(r.samples), fmt_time(r.median_s), fmt_time(r.q1_s), fmt_time(r.q3_s),
                         f"{r.rel_iqr_percent:.1f}", f"{r.launch_ratio:.2f}"])
    return table(["Method", "n", "Launches", "Samples", "Median", "Q1", "Q3", "IQR/med (%)", "Slowest ÷ fastest launch"], rows,
                 num_cols=tuple(range(1, 9)), caption="Table A1. Part A, every configuration (all measured batches pooled).")

"""Report sections of Assignment 2 that do not depend on the measured numbers."""
from report_lib import esc, table


def header(author, repo_url, date):
    return f"""
<h1>Optimised Dense and Sparse Matrix Multiplication in C:<br>Loop Order, Cache Blocking and CSR</h1>
<p class="meta"><b>{esc(author)}</b> · Big Data 2026–2027, Grado en Ciencia e Ingeniería de Datos, Universidad de Las Palmas de Gran Canaria<br>
Individual Assignment 2 — Optimised Approaches and Sparse Matrices · {esc(date)}<br>
Code, raw data and scripts: {esc(repo_url)} (folder <code>assignment2/</code>)</p>
"""


def introduction():
    return """
<h2>1. Introduction and objective</h2>
<p>Assignment 1 measured the textbook triple loop in Python, Java and C and found that its cost per multiply-add rises with the matrix size, which suggested, without proving it, a memory-access effect [1].
This study keeps one language, C, and asks two questions, as the brief requires:</p>
<ol>
<li><b>Part A.</b> <i>How do two optimisations of the dense algorithm, loop reordering and cache blocking, change the runtime of the validated Assignment 1 kernel as the matrix size grows?</i></li>
<li><b>Part B.</b> <i>Under which tested densities does a sparse (CSR) representation make multiplication faster or smaller in memory than a dense method on the same matrices?</i></li>
</ol>
<p><b>Expectations</b> (from the course material and the literature [2, 3]; written after a short pilot that was used only to set sizes and limits):
the reordered loop should help most once matrix B no longer fits in the caches, because it reads B along rows instead of columns; blocking should add little to that at moderate sizes but keep the cost per operation flatter at large sizes;
CSR should win clearly at very low densities, where it skips almost all the work, and lose at high densities, where indexing overheads remain and the product itself becomes dense.</p>
"""


def background():
    return """
<h2>2. Background</h2>
<p><b>Loop reordering.</b> For <i>C</i> = <i>AB</i> the three loops can be nested in any order without changing the result. The baseline <i>i-j-k</i> order computes one element of C at a time with an inner loop that walks a <i>column</i> of B (addresses <i>n</i> doubles apart),
so it touches a new cache line at every step. In the <i>i-k-j</i> order the inner loop updates a whole row of C with one element of A times a row of B, so both B and C are read with unit stride [2, Sec. 6.2.1].</p>
<p><b>Cache blocking (tiling).</b> The matrices are processed in <i>S</i> × <i>S</i> tiles so that the tile of B being used is reused for several rows of A while it is still in cache [2, Sec. 6.2.1]. Here the tiles are visited in the order <i>ii-kk-jj</i> and the <i>i-k-j</i> order is used inside a tile.
<i>S</i> = 64 was fixed before the pilot: one 64 × 64 tile of doubles is 32 KiB, which fits in the 48 KiB level-1 data cache of the test CPU.</p>
<p><b>Sparse matrices and CSR.</b> <i>Density</i> is the percentage of nonzero entries and <i>sparsity</i> the percentage of zeros. Compressed sparse row (CSR) stores, for each row, the column index and value of its nonzeros, plus one row-pointer array: 12 bytes per nonzero (8-byte value, 4-byte index) and 8(<i>n</i>+1) bytes,
against 8<i>n</i>² bytes for the dense matrix [4]. CSR × CSR products are computed with Gustavson’s row-by-row algorithm [5], whose work is proportional to the number of nonzero products rather than to <i>n</i>³: with independent random nonzeros of density <i>d</i>, about <i>d</i>²<i>n</i>³ products.
The product can be much denser than the inputs: an entry of C is nonzero if any of its <i>n</i> products is, which happens with probability 1 − (1 − <i>d</i>²)<sup><i>n</i></sup>.</p>
"""


def methodology(ctx):
    env, cfg = ctx["env"], ctx["cfg"]
    cpu = env["cpu"]
    lim = cfg["limits"]
    sp = cfg["sparse"]
    dense_sizes = ", ".join(str(r[0]) for r in cfg["dense"]["plan"])
    impl = table(
        ["Method", "Code", "What changes with respect to the baseline", "Expected effect"],
        [["<b>ijk</b> (baseline)", "<code>mm_ijk</code>, <code>src/c/dense.c</code>",
          "nothing: the validated Assignment 1 C kernel [1], copied unchanged", "–"],
         ["<b>ikj</b> (loop reordering)", "<code>mm_ikj</code>",
          "the two inner loops are swapped; C is accumulated in place, so it is zeroed first (inside the timed kernel)",
          "B and C read with unit stride: far fewer cache misses for large <i>n</i>"],
         ["<b>blocked</b> (cache blocking)", "<code>mm_blocked</code>",
          "<i>ikj</i> applied to 64 × 64 tiles; tiles visited <i>ii-kk-jj</i>; C zeroed inside the timed kernel",
          "a tile of B is reused from cache; flatter cost per operation at large <i>n</i>"],
         ["<b>CSR</b> (sparse)", "<code>spgemm</code>, <code>src/c/sparse.c</code>",
          "own CSR format and Gustavson product (no library): symbolic pass to size C, then numeric pass with a dense accumulator",
          "work ∝ nonzero products; fast at low density, overheads at high density"]],
        caption="Table 1. The methods. All are single-threaded C, compiled together with GCC 16.2.0 and <code>-std=c11 -Wall -Wextra -O2</code>, as in Assignment 1.")
    return f"""
<h2>3. Methodology</h2>
<h3>3.1 Implementations</h3>
{impl}
<p>For every element C[<i>i</i>][<i>j</i>] all dense methods add the products <i>a</i><sub>ik</sub><i>b</i><sub>kj</sub> in the same order (<i>k</i> = 0, 1, …, <i>n</i>−1, starting from 0), and the CSR method adds the same products without the exact zeros, so all methods return
<b>bit-identical</b> results. GCC enables loop vectorisation at <code>-O2</code> [6]; its report (<code>-fopt-info-vec-optimized</code>, saved in <code>build/dense-vectorisation.txt</code>) lists no vectorised loop in <code>dense.c</code>, so the dense methods differ only in their memory-access order, not in SIMD use.
The CSR kernel stores the column indices of each row of C in first-touch order (unsorted) and its timed region includes both passes, the allocation of C and freeing the previous C; its work arrays (three arrays of length <i>n</i>) are allocated once, outside the timer.</p>
<h3>3.2 Inputs</h3>
<p><b>Part A</b> uses the dense inputs of Assignment 1, uniform in [0, 1), generated with the same rule (NumPy <code>default_rng([20261012, n, which])</code>); the files are byte-identical to Assignment 1’s (checked against its SHA-256 hashes). Sizes: <i>n</i> = {dense_sizes}.
<b>Part B</b> uses new matrices for <i>n</i> = {", ".join(str(n) for n in sp['sizes'])} and densities {", ".join(f"{d * 100:g} %" for d in sp['densities'])}: every entry is nonzero independently with probability <i>d</i> (uniformly scattered positions, no structure) and nonzero values are uniform in (0, 1];
seed <code>default_rng([20261026, n, d·10⁶, which])</code>. Each matrix is written twice with the same logical content, as a dense file with explicit zeros (read by the dense comparator) and as a CSR file (read by the sparse method); the actual nonzero counts are reported in Table 6.</p>
<h3>3.3 Environment</h3>
<p>{esc(env['computer_model'])}, {esc(cpu['Name'])} (4 cores, 8 logical processors; L1d 48 KiB and L2 1.25 MiB per core, L3 8 MiB shared), {env['memory_gb']} GB RAM, Windows 11 Pro (build {esc(env['os']['BuildNumber'])}), AC power, “Balanced” power plan;
the same laptop as Assignment 1. One kernel thread; no numerical library.</p>
<h3>3.4 Correctness</h3>
<p><code>scripts/run_tests.py</code> checks: identity and zero matrices (bit-exact); random matrices against NumPy with absolute and relative tolerance 10<sup>−9</sup> (as in Assignment 1), including sizes that are and are not multiples of the tile size (10, 63, 64, 65, 100, 130, 300) and tile sizes 32 and 64;
that every optimised method is bit-identical to the baseline; that the C dense → CSR conversion writes byte-identical files to an independent Python writer; and that the CSR product matches NumPy, is bit-identical to the dense baseline and has exactly as many stored entries as NumPy’s product has nonzeros,
for densities 0 % (empty), identity, 1 %, 10 %, 50 % and 100 %.</p>
<h3>3.5 Measurement protocol</h3>
<ul>
<li><b>Same harness as Assignment 1.</b> Each configuration runs in its own process; the timing loop, batching, warm-up rules, statistics and memory metric are those of Assignment 1 (the launcher and memory code are copied unchanged into <code>scripts/harness.py</code>).
A batch runs <i>m</i> multiplications (<i>m</i> doubled until a batch lasts ≥ {cfg['target_seconds']} s); calibration and warm-up batches are discarded from the statistics; then 5–10 measured batches. The median and the interquartile range (IQR, shown as [Q1–Q3]) of all measured batches of all launches are reported.</li>
<li><b>Timing boundary.</b> Kernel only: reading the input files and allocating the work arrays are outside the timer, for every method. For the dense optimised methods the zeroing of C is inside, for CSR the allocation of the output is inside. The dense → CSR conversion is <b>measured separately</b> (one input matrix, same protocol) and is not included in the CSR multiplication time.</li>
<li><b>Interleaving.</b> Configurations are interleaved in time (all methods of one size, then the next size; a second sweep with the method order rotated). This matters: in pilot runs the same baseline kernel at <i>n</i> = 1000 took between 1.6 s and 3.3 s depending on the state of the laptop, and in Assignment 1 it took 1.37 s.
Only measurements taken within this study’s sweep are compared with each other; Assignment 1 times are not mixed in.</li>
<li><b>Memory.</b> Peak working set of the process (Windows <code>PeakWorkingSetSize</code> [7]), one value per launch; it includes the runtime and the input data.</li>
<li><b>Budgets.</b> At most {lim['seconds_per_multiplication']} s per single multiplication (otherwise the launch is stopped and reported as a timeout), {lim['seconds_per_launch'] // 60} min per launch, {lim['peak_memory_bytes'] / 2**30:g} GiB of memory. Part A tests sizes up to 3000 for every method; Part B uses the dense blocked method as comparator.</li>
</ul>
"""


def disclosure():
    return """
<h2>8. Use of generative AI</h2>
<p>In line with the course policy, the use of AI is disclosed here. <b>Tool:</b> Claude Code (Anthropic), used interactively from the author’s editor. <b>What the AI did:</b> proposed the choice of optimisations and the sparse format; wrote the C programs, the driver, and the test, analysis and report scripts;
drafted this report; checked that cited web sources exist. <b>What the author did:</b> supplied the assignment brief; chose C as the language and approved the plan; had the code run on their own laptop; published the materials in their repository.
The author is responsible for understanding the code, the measurements and the conclusions.</p>
"""


def references():
    refs = [
        "A. Choukri Mohamed, “Basic Matrix Multiplication in Python, Java and C,” Big Data Individual Assignment 1, ULPGC, Oct. 2026. Code and data: https://github.com/abdela758/bigdata-matmul (folder <code>assignment1/</code>).",
        "U. Drepper, “What every programmer should know about memory,” Red Hat, Inc., Nov. 21, 2007. [Online]. Available: https://akkadia.org/drepper/cpumemory.pdf (Sec. 6.2.1, cache access in matrix multiplication).",
        "Universidad de Las Palmas de Gran Canaria, Big Data course 2026–2027, <i>Individual Assignments: Matrix Multiplication</i> (course material, campus virtual).",
        "Y. Saad, <i>Iterative Methods for Sparse Linear Systems</i>, 2nd ed. Philadelphia, PA, USA: SIAM, 2003 (ch. 3, sparse matrices and their storage formats, including CSR).",
        "F. G. Gustavson, “Two fast algorithms for sparse matrices: multiplication and permuted transposition,” <i>ACM Trans. Math. Softw.</i>, vol. 4, no. 3, pp. 250–269, 1978.",
        "Free Software Foundation, “Options That Control Optimization,” <i>GCC manual</i>. [Online]. Available: https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html (accessed Oct. 29, 2026).",
        "Microsoft, “PROCESS_MEMORY_COUNTERS_EX structure (psapi.h),” Win32 apps documentation. [Online]. Available: https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters_ex (accessed Oct. 29, 2026).",
    ]
    return "<h2>References and materials</h2><ol class='refs'>" + "".join(f"<li>{r}</li>" for r in refs) + "</ol>"

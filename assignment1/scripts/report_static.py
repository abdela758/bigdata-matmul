"""Report sections that do not depend on the measured numbers (introduction, background, method, disclosure, references).

Every figure quoted here comes from config/experiment.json, results/environment.json or build/build-info.json.
"""
from report_lib import esc, table

KERNELS = """<b>Python</b>
for i in range(n):
    for j in range(n):
        s = 0.0
        for k in range(n):
            s += a[i][k] * b[k][j]
        c[i][j] = s

<b>Java</b>
for (int i = 0; i &lt; n; i++)
    for (int j = 0; j &lt; n; j++) {
        double sum = 0.0;
        for (int k = 0; k &lt; n; k++) sum += a[i][k] * b[k][j];
        c[i][j] = sum;
    }

<b>C</b>
for (int i = 0; i &lt; n; i++)
    for (int j = 0; j &lt; n; j++) {
        double sum = 0.0;
        for (int k = 0; k &lt; n; k++) sum += a[(size_t)i*n + k] * b[(size_t)k*n + j];
        c[(size_t)i*n + j] = sum;
    }"""


def header(author, repo_url, date):
    return f"""
<h1>Basic Matrix Multiplication in Python, Java and C:<br>How Matrix Size Drives Runtime and Memory</h1>
<p class="meta"><b>{esc(author)}</b> · Big Data 2026–2027, Grado en Ciencia e Ingeniería de Datos, Universidad de Las Palmas de Gran Canaria<br>
Individual Assignment 1 — Basic Matrix Multiplication in Different Languages · {esc(date)}<br>
Code, raw data and scripts: {esc(repo_url)}</p>
"""


def introduction():
    return """
<h2>1. Introduction and objective</h2>
<p>Multiplying two dense <i>n</i> × <i>n</i> matrices with three nested loops needs <i>n</i>³ multiply-add operations and is the standard
starting point for talking about performance and scalability [1]. The algorithm is identical in every language, but the
cost of each operation is not: an interpreter, a just-in-time compiler and an ahead-of-time compiler turn the same source into very
different machine behaviour, and the way each language lays matrices out in memory decides how well the processor caches can help [2].</p>
<p><b>Research question.</b> <i>How does the matrix size <i>n</i> affect the runtime and the memory use of the textbook triple-loop
multiplication of dense <code>double</code> matrices implemented in Python, Java and C, under one fixed and documented set of conditions on one machine?</i>
The factor that is changed is <i>n</i> (and, as a second factor, the language); the outcomes are the time of one multiplication and the peak memory of the process.</p>
<p><b>Predictions.</b> They were written down from the algorithmic model and the course material before the final measurements were analysed, but after a small pilot run
(used to choose sizes and limits) had already shown that large sizes cost more per operation, so they are informed predictions, not blind ones:</p>
<ol>
<li><b>P1 – cubic growth.</b> While the working set stays in cache, the time of one multiplication grows like <i>n</i>³ in every language, so the
log-log slope is about 3 and the time per inner iteration, <i>t</i>/<i>n</i>³, is roughly constant. Small sizes may sit above that line because of fixed overheads.</li>
<li><b>P2 – constants differ by language.</b> The constant should be far larger in Python (interpreted bytecode, boxed floats) than in Java
or C; Java, after JIT compilation, should land in the same order of magnitude as C.</li>
<li><b>P3 – memory effects at large <i>n</i>.</b> In the <i>i-j-k</i> order the inner loop walks matrix B down a column, one cache line per iteration [2]. Once B (8<i>n</i>² bytes)
no longer fits in the 1.25 MiB L2 cache (<i>n</i> ≈ 400) and then in the 8 MiB L3 cache (<i>n</i> ≈ 1000), <i>t</i>/<i>n</i>³ should rise, in all three languages.</li>
<li><b>P4 – memory.</b> C needs about 24<i>n</i>² bytes for the three matrices plus a small runtime; Java the same plus a larger JVM baseline;
Python much more per element, because each element is a pointer to a separate 24-byte <code>float</code> object.</li>
</ol>
"""


def background():
    return """
<h2>2. Background</h2>
<p><b>Algorithm.</b> For square matrices <i>A</i> and <i>B</i>, <i>C</i><sub>ij</sub> = Σ<sub>k</sub> <i>A</i><sub>ik</sub> <i>B</i><sub>kj</sub>. Evaluating it directly takes
Θ(<i>n</i>³) arithmetic operations [1]. In the loop order <i>i-j-k</i> used here, the inner loop reads A along a row (consecutive addresses) and B along a column
(addresses <i>n</i> elements apart), which is the access pattern analysed by Drepper [2, Sec. 6.2.1].</p>
<p><b>Why the three languages should behave differently.</b> CPython executes bytecode in an interpreter and represents every number as an object, so a <code>list</code> of floats stores
pointers to separate <code>float</code> objects (24 bytes each in this interpreter). HotSpot starts in an interpreter, profiles the code and recompiles hot loops
with its tiered JIT compilers; a <code>double[][]</code> is an array of row arrays holding unboxed 8-byte values. A C compiler produces native code before the program runs,
and a <code>malloc</code>ed block holds the elements back to back. Compiler options matter: GCC enables, among others, loop interchange at <code>-O3</code> [5], which could silently change the loop order, so the main comparison uses <code>-O2</code>.</p>
<p><b>Floating point.</b> All three programs use IEEE-754 binary64 and the same operation order, so their results can be compared bit by bit. In general, different evaluation orders give slightly different
sums; for a dot product of length <i>n</i> computed by recursive summation the error is bounded by γ<sub>n</sub> Σ<sub>k</sub>|<i>a</i><sub>ik</sub><i>b</i><sub>kj</sub>| with γ<sub>n</sub> = <i>nu</i>/(1−<i>nu</i>) and unit roundoff <i>u</i> = 2<sup>−53</sup> [3]; this bound is used as the numerical tolerance in Section 3.4.</p>
"""


def methodology(ctx):
    env, build, cfg = ctx["env"], ctx["build"], ctx["cfg"]
    cpu = env["cpu"]
    lim = cfg["limits"]
    main = cfg["experiments"]["main"]["plan"]
    sizes = {lang: ", ".join(str(r[0]) for r in rows) for lang, rows in main.items()}
    impl = table(
        ["", "Python", "Java", "C"],
        [["Runtime / compiler", "CPython 3.12.0 (python.org build, MSC v.1935), standard library only",
          esc(build["java_version"].replace('"', "")) + " (HotSpot, tiered C1/C2 JIT, G1 GC)",
          esc(build["c_compiler_version"].replace("gcc.exe ", "GCC "))],
         ["Flags / settings", "default interpreter (no <code>-O</code>)", "<code>-Xmx4g</code>; all other JVM options default",
          "<code>-std=c11 -Wall -Wextra -O2</code>; no <code>-march</code>, no <code>-ffast-math</code>"],
         ["Matrix storage", "list of <i>n</i> row lists; every element is a pointer to a 24-byte <code>float</code> object",
          "<code>double[n][n]</code>: array of <i>n</i> row arrays of unboxed 8-byte values",
          "one <code>malloc</code>ed <code>double[n*n]</code> per matrix, row-major, 8 bytes per element"],
         ["Kernel timer", "<code>time.perf_counter_ns</code>", "<code>System.nanoTime</code>", "<code>QueryPerformanceCounter</code>"],
         ["Loop order, arithmetic", "<i>i-j-k</i>, local accumulator, <code>c[i][j]</code> assigned once", "same", "same"],
         ["Sizes <i>n</i> planned", esc(sizes["python"]), esc(sizes["java"]), esc(sizes["c"])]],
        caption="Table 1. The three implementations and what is fixed or different between them.")
    envt = table(
        ["Item", "Value"],
        [["Machine", f"{esc(env['computer_model'])}, Windows 11 Pro (build {esc(env['os']['BuildNumber'])})"],
         ["CPU", f"{esc(cpu['Name'])}: {cpu['NumberOfCores']} physical cores, {cpu['NumberOfLogicalProcessors']} logical processors; "
                 f"L2 {cpu['L2CacheSize'] // 1024} MiB in total (1.25 MiB per core), L3 {cpu['L3CacheSize'] // 1024} MiB shared"],
         ["RAM", f"{env['memory_gb']} GB"],
         ["Power", f"AC power, Windows power plan “{esc(env['power_plan']['name'])}” (Balanced); no change was made to frequency or affinity settings"],
         ["Threads used by the kernel", "1 in every language (no parallel code, no numerical library). The JVM additionally runs its own JIT-compiler and GC threads."],
         ["Analysis tools", "Python 3.13.9 (Anaconda) with NumPy 2.3.5, pandas 2.3.3, SciPy 1.16.3, Matplotlib 3.10.6"]],
        caption="Table 2. Hardware and software environment (recorded by <code>scripts/collect_environment.py</code> into <code>results/environment.json</code>).")
    return f"""
<h2>3. Methodology</h2>
<h3>3.1 Implementations</h3>
<p>The required algorithm, the straightforward triple loop with <i>O</i>(<i>n</i>³) work, is implemented in all three languages; no optimised library routine
replaces the loops (NumPy is used only to check results, Section 3.4). The loops are, literally:</p>
<pre>{KERNELS}</pre>
{impl}
<p>The accumulator is a local variable in all three languages so that every language performs the same additions in the same order. (The course’s JMH example accumulates directly into
<code>c[i][j]</code>; that variant is not used here.) The Python version is deliberately the unoptimised textbook loop, with no hoisting of row lookups.
The unavoidable differences are the storage layouts above, the fact that Python allocates one new <code>float</code> object per result element inside the timed kernel, and that the JVM compiles
code at run time and runs helper threads. Storage layout and compiler flags are not varied in this study; they appear only as possible explanations in Section 5.</p>
<h3>3.2 Inputs</h3>
<p>All implementations read the same files <code>A_n.bin</code> and <code>B_n.bin</code> (raw little-endian binary64, row-major, values uniform in [0, 1)). They are generated with NumPy’s
<code>default_rng([20261012, n, which])</code>, so each file is reproducible on its own, and their SHA-256 hashes are stored in <code>inputs/checksums.json</code>. Every language therefore multiplies
exactly the same logical matrices; no language uses its own random-number generator.</p>
<h3>3.3 Environment</h3>
{envt}
<h3>3.4 Correctness validation</h3>
<p>Before any benchmark, <code>scripts/run_tests.py</code> multiplies test matrices with each implementation and compares the result with references (Table 3 in Section 4.1 summarises the outcome). The checks are: identity and zero matrices
(bit-exact); hand-computed 2 × 2 and 3 × 3 products (bit-exact); random integer-valued matrices stored as doubles, where every partial sum is exactly representable so any summation order must agree with NumPy’s integer product (bit-exact);
random real matrices in [0, 1) and [−1, 1) and the shared benchmark inputs, compared (a) with an exact rational reference on 200 random entries and (b) with NumPy’s <code>A @ B</code> on all entries, against the bound γ<sub>n</sub> Σ<sub>k</sub>|<i>a</i><sub>ik</sub><i>b</i><sub>kj</sub>| of Section 2
(a result passes if the error is at most one bound against the exact value, and at most two bounds against NumPy, whose own rounding error is of the same size). Finally, the three implementations must return <b>bit-identical</b> matrices; this holds because they use the same
operation order and no fused multiply-add (GCC’s default for <code>-std=c11</code> is <code>-ffp-contract=off</code> [5]). The sum of all elements of C is also printed after every timed batch, which both proves that the computation ran and lets the checksums be compared across languages.</p>
<h3>3.5 Timing protocol</h3>
<ul>
<li><b>What the timer includes.</b> The multiplication kernel only. Input files are read, and the result matrix is allocated, before timing; the kernel assigns every element of C, so no initialisation is needed or timed. This is a kernel measurement, not an end-to-end one, and no end-to-end number is mixed with it.</li>
<li><b>Harness instead of JMH.</b> The course material suggests JMH, pytest-benchmark and <code>perf</code> [4, 10]. Here one small harness with the same batching, warm-up, repetition and memory-measurement rules is used for all three languages, because the comparison needs the same timing boundary everywhere
(<code>perf</code> is not available on Windows). Its behaviour is documented in this section and its raw output is kept; it was not cross-checked against JMH.</li>
<li><b>Clocks.</b> Monotonic timers in each language (Section 3.1). In CPython on Windows <code>perf_counter</code> is backed by <code>QueryPerformanceCounter</code> (resolution 100 ns on this machine) [7], the same source as the C program [8].</li>
<li><b>Batches.</b> To keep every timed interval much longer than the timer resolution, a <i>batch</i> executes <i>m</i> multiplications back to back and the program reports time per multiplication. <i>m</i> is found by doubling from 1 until a batch lasts at least {cfg['target_seconds']} s
(<i>m</i> = 1 whenever a single multiplication already takes longer). These <i>calibration</i> batches are also the first warm-up.</li>
<li><b>Warm-up and repetitions.</b> After calibration, a configuration-specific number of warm-up batches is run (0–5; see <code>config/experiment.json</code>), followed by 5–15 <i>measured</i> batches. Warm-up and calibration batches are kept in the raw file but never enter a statistic. Five or more measured batches are collected for every configuration.</li>
<li><b>Launches.</b> Each configuration is run in 1–3 separate processes (<i>launches</i>) so that process-level variation (JIT decisions, memory placement) is sampled, and the launches are interleaved in time: one sweep over all sizes and languages, then the next sweep with the language order rotated. Cheap configurations get three launches; the expensive ones, where one launch lasts minutes, get one or two, which is stated in Appendix A.</li>
<li><b>Summary statistics.</b> All measured batches of all successful launches of a configuration are pooled. The central value is the <b>median</b>; variability is the <b>interquartile range</b> (IQR, quartiles by linear interpolation), reported as [Q1–Q3]. All raw measurements are kept in <code>results/raw_samples.csv</code>, and no observation was removed.</li>
<li><b>Dead-code protection.</b> After each batch the sum of all elements of C is computed and printed; the multiplication function is not inlined in C (<code>noinline</code>) so that repeated calls in a batch cannot be merged.</li>
</ul>
<h3>3.6 Memory measurement</h3>
<p><b>Metric:</b> the <i>peak working set</i> of the whole process (<code>PeakWorkingSetSize</code> of <code>PROCESS_MEMORY_COUNTERS_EX</code> [9]) read by the driver with <code>GetProcessMemoryInfo</code> after the process ends, one value per launch; peak committed (private) bytes, <code>PeakPagefileUsage</code>, is recorded as well.
It includes the language runtime, the loaded input data, the result and all temporary data, so it is <b>not</b> the same quantity as managed-heap use or as 24<i>n</i>² bytes of array storage, and the figures must not be compared as if it were. The runtime baseline is estimated from the smallest size (<i>n</i> = 10, three matrices of 800 bytes).
Limits of the metric: working set counts resident pages only, the JVM reserves and touches memory in its own way, and a single peak per launch gives no information about when it occurred. CPU utilisation was not measured because it is not needed for the conclusions.</p>
<h3>3.7 Sizes, budgets and fixed settings</h3>
<p>The size sequence is roughly geometric (10, 20, 50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 2500, 3000 where feasible) and avoids powers of two, whose large strides can map matrix columns onto few cache sets (a cache-associativity effect that I did not test) and could confound the trend. It spans the three regimes of P3 (B inside L2, inside L3, larger than L3).
Budgets were declared before the final run: <b>at most {lim['seconds_per_multiplication']} s per single multiplication</b> (a launch that prints nothing for {lim['seconds_per_multiplication']} s is killed and recorded as a timeout), <b>at most {lim['seconds_per_launch'] // 60} min per launch</b> and <b>{lim['peak_memory_bytes'] / 2**30:g} GiB of memory</b> (also the JVM heap cap).
The largest size that met the limits is reported separately from the largest size attempted (Section 4.4); timeouts are reported as results at their limit and never given an invented time.
<b>Fixed in the main comparison:</b> machine and power state, input files, loop order, numeric type, accumulator style, single-threaded kernels, timing boundary, batch rule, limits and, per language, the compiler/runtime settings in Table 1.
For scale: three <i>n</i> × <i>n</i> binary64 matrices need 24<i>n</i>² bytes, 216 MB at <i>n</i> = 3000, far below the memory budget, so time, not memory, is the binding limit in this study.</p>
"""


def disclosure():
    return """
<h2>8. Use of generative AI</h2>
<p>In line with the course policy, the use of AI is disclosed here. <b>Tool:</b> Claude Code (Anthropic). <b>Purpose and affected parts:</b> planning the experimental design; writing the benchmark programs, the driver, the test and analysis scripts and the build scripts;
debugging the toolchain set-up on Windows; drafting the text of this report; checking that cited sources exist and say what they are cited for. <b>What the AI did not do:</b> no number in this report was produced or estimated by the AI; every result comes from running
the published code on the author’s machine, and the raw measurements are in the repository. The author is responsible for understanding the code, for the measurements and for the conclusions.</p>
"""


def references():
    refs = [
        "T. H. Cormen, C. E. Leiserson, R. L. Rivest and C. Stein, <i>Introduction to Algorithms</i>, 4th ed. Cambridge, MA, USA: MIT Press, 2022 (matrix multiplication, Θ(<i>n</i>³) for the direct method).",
        "U. Drepper, “What every programmer should know about memory,” Red Hat, Inc., Nov. 21, 2007. [Online]. Available: https://akkadia.org/drepper/cpumemory.pdf (Sec. 6.2.1, cache access in matrix multiplication).",
        "N. J. Higham, <i>Accuracy and Stability of Numerical Algorithms</i>, 2nd ed. Philadelphia, PA, USA: SIAM, 2002 (ch. 3, error analysis of inner products).",
        "OpenJDK, “Code Tools: jmh – Java Microbenchmark Harness.” [Online]. Available: https://openjdk.org/projects/code-tools/jmh (accessed Oct. 17, 2026).",
        "Free Software Foundation, “Options That Control Optimization,” <i>GCC manual</i>. [Online]. Available: https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html (accessed Oct. 17, 2026).",
        "Python Software Foundation, “Memory Management,” <i>Python/C API Reference Manual</i>. [Online]. Available: https://docs.python.org/3/c-api/memory.html (accessed Oct. 17, 2026).",
        "Python Software Foundation, “time — Time access and conversions,” <i>The Python Standard Library</i>. [Online]. Available: https://docs.python.org/3/library/time.html (accessed Oct. 17, 2026).",
        "Microsoft, “Acquiring high-resolution time stamps,” Win32 apps documentation. [Online]. Available: https://learn.microsoft.com/en-us/windows/win32/sysinfo/acquiring-high-resolution-time-stamps (accessed Oct. 17, 2026).",
        "Microsoft, “PROCESS_MEMORY_COUNTERS_EX structure (psapi.h),” Win32 apps documentation. [Online]. Available: https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters_ex (accessed Oct. 17, 2026).",
        "Universidad de Las Palmas de Gran Canaria, Big Data course 2026–2027, <i>Individual Assignments: Matrix Multiplication</i> and <i>Benchmarking Tools</i> (course material, campus virtual).",
    ]
    return "<h2>References</h2><ol class='refs'>" + "".join(f"<li>{r}</li>" for r in refs) + "</ol>"

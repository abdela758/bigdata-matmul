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
<p>Multiplying two dense <i>n</i> × <i>n</i> matrices with three nested loops needs <i>n</i>³ multiply-add operations and is the standard starting point for talking about performance and scalability [1].
The algorithm is the same in every language, but the cost of each operation is not: an interpreter, a just-in-time (JIT) compiler and an ahead-of-time compiler turn the same source into very different machine behaviour,
and the way each language lays a matrix out in memory decides how well the processor caches can help [2].</p>
<p><b>Research question.</b> <i>How does the matrix size <i>n</i> affect the runtime and the memory use of the textbook triple-loop multiplication of dense <code>double</code> matrices implemented in Python, Java and C,
under one fixed and documented set of conditions on one machine?</i> The factor that is changed is <i>n</i> (and, as a second factor, the language); the outcomes are the time of one multiplication and the peak memory of the process.</p>
<p><b>Expectations.</b> They were written down from the algorithmic model and the course material before the final measurements were analysed, but after a small pilot run (used to choose sizes and limits)
had already shown that large sizes cost more per operation, so they are informed expectations, not blind ones.</p>
<ol>
<li><b>E1 – cubic growth.</b> The time of one multiplication grows like <i>n</i>³, so the time per inner iteration, <i>t</i>/<i>n</i>³, should be roughly constant, at least for small sizes.</li>
<li><b>E2 – constants differ by language.</b> The constant should be far larger in Python than in Java or C; Java, once the JIT has compiled the loop, should be in the same order of magnitude as C.</li>
<li><b>E3 – memory effects.</b> In the <i>i-j-k</i> order the inner loop walks matrix B down a column, which uses the cache poorly [2]. Once B (8<i>n</i>² bytes) outgrows the 1.25 MiB L2 cache (<i>n</i> ≈ 400) and then the 8 MiB L3 cache (<i>n</i> ≈ 1000), <i>t</i>/<i>n</i>³ may rise.</li>
<li><b>E4 – memory.</b> Every process has a fixed runtime baseline, and on top of it the data grow with <i>n</i>²; Python, which stores each number as a separate object, should need much more per element than C.</li>
</ol>
"""


def background():
    return """
<h2>2. Background</h2>
<p><b>Algorithm.</b> For square matrices, <i>C</i><sub>ij</sub> = Σ<sub>k</sub> <i>A</i><sub>ik</sub> <i>B</i><sub>kj</sub>; evaluating it directly takes Θ(<i>n</i>³) operations [1].
In the loop order <i>i-j-k</i> used here, the inner loop reads A along a row (consecutive addresses) and B along a column (addresses <i>n</i> elements apart), the access pattern analysed by Drepper [2, Sec. 6.2.1].</p>
<p><b>The three languages.</b> CPython runs bytecode in an interpreter and stores every number as an object, so a list of floats holds pointers to separate 24-byte <code>float</code> objects. HotSpot (Java) starts in an interpreter,
profiles the code and recompiles hot loops with its JIT compilers; a <code>double[][]</code> is an array of row arrays of unboxed 8-byte values. A C compiler produces native code before the program runs and a <code>malloc</code>ed block stores the elements back to back.
GCC enables loop interchange at <code>-O3</code> [4], which could silently change the loop order, so the main comparison uses <code>-O2</code>. All three programs use IEEE-754 double precision and the same operation order.</p>
"""


def methodology(ctx):
    env, build, cfg = ctx["env"], ctx["build"], ctx["cfg"]
    cpu = env["cpu"]
    lim = cfg["limits"]
    main = cfg["experiments"]["main"]["plan"]
    sizes = {lang: ", ".join(str(r[0]) for r in rows) for lang, rows in main.items()}
    impl = table(
        ["", "Python", "Java", "C"],
        [["Runtime / compiler", "CPython 3.12.0 (python.org build), standard library only",
          esc(build["java_version"].replace('"', "")) + " (HotSpot, tiered JIT, G1 GC)",
          esc(build["c_compiler_version"].replace("gcc.exe ", "GCC "))],
         ["Flags / settings", "default interpreter", "<code>-Xmx4g</code>; other JVM options default",
          "<code>-std=c11 -Wall -Wextra -O2</code>"],
         ["Matrix storage", "list of <i>n</i> row lists; each element points to a 24-byte <code>float</code> object",
          "<code>double[n][n]</code>: <i>n</i> row arrays of unboxed 8-byte values",
          "one <code>double[n*n]</code> per matrix, row-major, 8 bytes per element"],
         ["Kernel timer", "<code>time.perf_counter_ns</code>", "<code>System.nanoTime</code>", "<code>QueryPerformanceCounter</code>"],
         ["Sizes <i>n</i> planned", esc(sizes["python"]), esc(sizes["java"]), esc(sizes["c"])]],
        caption="Table 1. The three implementations. Same in all three: <i>i-j-k</i> loop order, <code>double</code>, local accumulator, one kernel thread, same input files, same timing rules.")
    envt = table(
        ["Item", "Value"],
        [["Machine", f"{esc(env['computer_model'])}, Windows 11 Pro (build {esc(env['os']['BuildNumber'])})"],
         ["CPU", f"{esc(cpu['Name'])}: {cpu['NumberOfCores']} physical cores, {cpu['NumberOfLogicalProcessors']} logical processors; "
                 f"L2 1.25 MiB per core, L3 {cpu['L3CacheSize'] // 1024} MiB shared"],
         ["RAM / power", f"{env['memory_gb']} GB; AC power, Windows “Balanced” power plan, no changes to frequency or affinity settings"],
         ["Threads", "The kernel runs on one thread in every language (no parallel code, no numerical library); the JVM also runs its own JIT and GC threads."]],
        caption="Table 2. Hardware and software environment (<code>results/environment.json</code>).")
    return f"""
<h2>3. Methodology</h2>
<h3>3.1 Implementations</h3>
<p>The straightforward triple loop with <i>O</i>(<i>n</i>³) work is implemented in all three languages; no optimised library routine replaces the loops (NumPy is used only to check results). The loops are, literally:</p>
<pre>{KERNELS}</pre>
{impl}
<p>The accumulator is a local variable in all three languages, so each performs the same additions in the same order. The Python loop is deliberately the unoptimised textbook version. The unavoidable differences are the storage layouts in Table 1,
the fact that Python allocates one new <code>float</code> object per result element inside the timed kernel, and that the JVM compiles code at run time. Storage layout and compiler flags are not varied in this study.</p>
<h3>3.2 Inputs</h3>
<p>All implementations read the same files <code>A_n.bin</code> and <code>B_n.bin</code> (binary double precision, row-major, values uniform in [0, 1)), generated with NumPy’s <code>default_rng([20261012, n, which])</code>.
Each file is reproducible on its own, and its SHA-256 hash is stored in <code>inputs/checksums.json</code>, so every language multiplies exactly the same matrices.</p>
<h3>3.3 Environment</h3>
{envt}
<h3>3.4 Correctness validation</h3>
<p>Before any benchmark, <code>scripts/run_tests.py</code> multiplies test matrices with each implementation: identity and zero matrices, hand-computed 2 × 2 and 3 × 3 products, and random integer matrices stored as doubles (exact in any summation order) must match <b>bit for bit</b>.
Random real matrices (uniform in [0, 1) and [−1, 1)) and the shared benchmark inputs are compared with NumPy’s <code>A @ B</code> using an absolute and a relative tolerance of 10<sup>−9</sup> (<code>numpy.allclose</code>).
Double precision carries about 16 significant digits and summing a few hundred products only changes the last digits, so 10<sup>−9</sup> is strict enough to catch any real error (a wrong index gives errors of order 1) while allowing the different summation order of NumPy’s library.
Finally, the three implementations must return <b>bit-identical</b> matrices. The sum of all elements of C is also printed after every timed batch, which shows that the computation ran and lets results be compared across languages.</p>
<h3>3.5 Timing and memory protocol</h3>
<ul>
<li><b>What is timed.</b> The multiplication kernel only. Input files are read, and the result matrix allocated, before timing; the kernel assigns every element of C, so no initialisation is needed or timed. No end-to-end time is mixed with this kernel time.</li>
<li><b>Harness.</b> The course material suggests JMH, pytest-benchmark and <code>perf</code> [3, 8]. Here one small harness with the same rules is used for all three languages, so that the timing boundary and the memory measurement are identical everywhere (<code>perf</code> is not available on Windows). It was not cross-checked against JMH.
Timers are monotonic; in CPython on Windows <code>perf_counter</code> is backed by <code>QueryPerformanceCounter</code> (resolution 100 ns here) [5], the same source as the C program [6].</li>
<li><b>Batches.</b> To keep every timed interval much longer than the timer resolution, a <i>batch</i> runs <i>m</i> multiplications back to back and the program reports time per multiplication; <i>m</i> is found by doubling until a batch lasts at least {cfg['target_seconds']} s (<i>m</i> = 1 when one multiplication already takes longer).</li>
<li><b>Warm-up and repetitions.</b> The calibration batches plus 0–5 further warm-up batches (<code>config/experiment.json</code>) are run first and never enter a statistic; then 5–15 <i>measured</i> batches. Each configuration is run in 1–3 separate processes (<i>launches</i>), interleaved in time; cheap configurations get three launches, the expensive ones one or two (Appendix A).</li>
<li><b>Statistics.</b> All measured batches of all successful launches of a configuration are pooled. The central value is the <b>median</b>; variability is the <b>interquartile range</b> (IQR), shown as [Q1–Q3]. All raw measurements are kept in <code>results/raw_samples.csv</code>.</li>
<li><b>Memory.</b> The <i>peak working set</i> of the whole process (<code>PeakWorkingSetSize</code> [7]), read by the driver after the process ends; one value per launch. It includes the language runtime, so it is not the same as array size or Java heap use, and the figures must not be compared as if they were. The runtime baseline is estimated from the smallest size (<i>n</i> = 10).</li>
</ul>
<h3>3.6 Sizes and budgets</h3>
<p>Sizes grow roughly geometrically (10 … 3000, where feasible) and avoid powers of two. The budgets were declared before the final run: <b>at most {lim['seconds_per_multiplication']} s per single multiplication</b> (a launch that prints nothing for that long is stopped and recorded as a timeout),
<b>{lim['seconds_per_launch'] // 60} min per launch</b> and <b>{lim['peak_memory_bytes'] / 2**30:g} GiB of memory</b>. The largest size that met the budget is reported separately from the largest attempted, and timeouts are reported at their limit, never with an invented time.
Three <i>n</i> × <i>n</i> double matrices need 24<i>n</i>² bytes (216 MB at <i>n</i> = 3000), so time, not memory, is the limit that binds.</p>
"""


def disclosure():
    return """
<h2>8. Use of generative AI</h2>
<p>In line with the course policy, the use of AI is disclosed here. <b>Tool:</b> Claude Code (Anthropic), used interactively from the author’s editor. <b>What the AI did:</b> planned the experimental design;
wrote the benchmark programs, the driver, and the test, analysis and build scripts; debugged the toolchain set-up on Windows; drafted this report; checked that cited web sources exist.
<b>What the author did:</b> supplied the assignment brief and course material; set the scope of the study, including asking to drop extra experiments and to simplify the analysis and the correctness tolerance;
had the code run on their own laptop; created the GitHub repository and published the materials; chose the file names. The author is responsible for understanding the code, the measurements and the conclusions.</p>
"""


def references():
    refs = [
        "T. H. Cormen, C. E. Leiserson, R. L. Rivest and C. Stein, <i>Introduction to Algorithms</i>, 4th ed. Cambridge, MA, USA: MIT Press, 2022 (matrix multiplication, Θ(<i>n</i>³) for the direct method).",
        "U. Drepper, “What every programmer should know about memory,” Red Hat, Inc., Nov. 21, 2007. [Online]. Available: https://akkadia.org/drepper/cpumemory.pdf (Sec. 6.2.1, cache access in matrix multiplication).",
        "OpenJDK, “Code Tools: jmh – Java Microbenchmark Harness.” [Online]. Available: https://openjdk.org/projects/code-tools/jmh (accessed Oct. 17, 2026).",
        "Free Software Foundation, “Options That Control Optimization,” <i>GCC manual</i>. [Online]. Available: https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html (accessed Oct. 17, 2026).",
        "Python Software Foundation, “time — Time access and conversions,” <i>The Python Standard Library</i>. [Online]. Available: https://docs.python.org/3/library/time.html (accessed Oct. 17, 2026).",
        "Microsoft, “Acquiring high-resolution time stamps,” Win32 apps documentation. [Online]. Available: https://learn.microsoft.com/en-us/windows/win32/sysinfo/acquiring-high-resolution-time-stamps (accessed Oct. 17, 2026).",
        "Microsoft, “PROCESS_MEMORY_COUNTERS_EX structure (psapi.h),” Win32 apps documentation. [Online]. Available: https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters_ex (accessed Oct. 17, 2026).",
        "Universidad de Las Palmas de Gran Canaria, Big Data course 2026–2027, <i>Individual Assignments: Matrix Multiplication</i> and <i>Benchmarking Tools</i> (course material, campus virtual).",
    ]
    return "<h2>References and materials</h2><ol class='refs'>" + "".join(f"<li>{r}</li>" for r in refs) + "</ol>"

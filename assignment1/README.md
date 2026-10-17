# Assignment 1 – Basic matrix multiplication in Python, Java and C

Big Data 2026–2027, Grado en Ciencia e Ingeniería de Datos (ULPGC).
Question: *how does the matrix size n affect runtime and memory of the textbook triple-loop multiplication of dense n × n `double`
matrices in Python, Java and C, under one fixed, documented set of conditions on one machine?*
The report is `report/Assignment01_Choukri_Report.pdf`. This README says how to rebuild and re-run everything.

## Layout

```
src/python/matmul.py, src/java/MatMul.java, src/c/matmul.c   the three implementations (same CLI and output protocol)
config/experiment.json          declared protocol: sizes, launches, repetitions, warm-up, limits, JVM flags
scripts/build.ps1 | build.sh    compile C and Java, record the exact tools in build/build-info.json
scripts/generate_inputs.py      write / verify the shared input matrices (seeded; SHA-256 in inputs/checksums.json)
scripts/run_tests.py            correctness tests -> results/correctness.csv
scripts/collect_environment.py  hardware / software record -> results/environment.json
scripts/run_benchmarks.py       benchmark driver -> results/raw_samples.csv, results/launches.csv
scripts/analyze.py              tables (results/tables) and figures (results/figures)
scripts/build_report.py         HTML + PDF report (headless Edge) from the tables and figures
scripts/package_materials.py    dist/Assignment01_Choukri_Materials.zip
```

## Versions used for the reported results

| Component | Version |
|---|---|
| Machine | HP 250 G8, Intel Core i5-1135G7 (4 cores / 8 threads), 15.7 GB RAM, Windows 11 Pro 10.0.26200, AC power, "Balanced" plan |
| CPython (python.org build) | 3.12.0 – runs the Python kernel, standard library only |
| OpenJDK | 25.0.2 – runs the Java kernel with `-Xmx4g`, other JVM options default |
| GCC (WinLibs, MinGW-W64 UCRT) | 16.2.0 – `-std=c11 -Wall -Wextra -O2` (`winget install BrechtSanders.WinLibs.POSIX.UCRT`) |
| Anaconda Python | 3.13.9 – driver, tests, analysis: numpy 2.3.5, pandas 2.3.3, scipy 1.16.3, matplotlib 3.10.6 (`../requirements.txt`) |

The driver and the three programs need only the standard library, a JDK (17+) and a C compiler.

## Build and run (Windows PowerShell, from `assignment1/`)

```powershell
$kernelPython = 'C:\Users\<you>\AppData\Local\Programs\Python\Python312\python.exe'   # interpreter that runs the Python kernel
$py           = 'C:\Users\<you>\anaconda3\python.exe'                                  # driver / analysis (needs the requirements)

.\scripts\build.ps1                                            # 1. compile (finds gcc and a JDK, or set $env:CC / $env:JAVA_HOME)
& $py scripts\generate_inputs.py                               # 2. inputs\A_<n>.bin, B_<n>.bin (up to 144 MB for n = 3000)
& $py scripts\generate_inputs.py --check                       #    must print "OK: 26 files match inputs/checksums.json"
& $py scripts\run_tests.py --python $kernelPython              # 3. correctness: must end with "84/84 checks passed"
& $py scripts\collect_environment.py --python $kernelPython    # 4. environment record
& $py scripts\run_benchmarks.py --python $kernelPython         # 5. benchmarks, about 75 min (--resume continues an interrupted run)
& $py scripts\analyze.py                                       # 6. tables and figures
& $py scripts\build_report.py --repo-url <url>                 # 7. report\Assignment01_Choukri_Report.pdf
& $py scripts\package_materials.py                             # 8. dist\Assignment01_Choukri_Materials.zip
```

On Linux/macOS use `scripts/build.sh` for step 1 (not run on this machine) and the same Python commands.
`run_benchmarks.py --dry-run` prints the commands without running them; `--languages python c` / `--sizes 100 500` restrict a run.
Leave the machine idle and on AC power during step 5; the driver records the background CPU load before every launch.

Running one implementation by hand (arguments: n, A, B, warm-up batches, measured batches, target seconds per batch):

```powershell
build\c\matmul.exe bench 500 inputs\A_500.bin inputs\B_500.bin 3 15 0.2
java -Xmx4g -cp build\java MatMul bench 500 inputs\A_500.bin inputs\B_500.bin 3 15 0.2
& $kernelPython src\python\matmul.py bench 500 inputs\A_500.bin inputs\B_500.bin 3 15 0.2
```

## Shared inputs

`inputs/A_<n>.bin` and `B_<n>.bin` hold n·n little-endian float64 values in row-major order, uniform in [0, 1), generated with NumPy's
`default_rng([20261012, n, which])` (`which` = 0 for A, 1 for B). Every file is reproducible on its own; the SHA-256 of each is committed in
`inputs/checksums.json`. The `.bin` files themselves are git-ignored.

## Measurement protocol (details and justification are in the report, Section 3)

* Timed region: the multiplication kernel only (inputs read and result allocated beforehand). Monotonic timers: `perf_counter_ns`, `nanoTime`, `QueryPerformanceCounter`.
* A *batch* runs `m` multiplications back to back (`m` found by doubling until a batch lasts ≥ 0.2 s) and reports time per multiplication.
  Calibration and warm-up batches are stored (`phase` column) but never used in a statistic; 5–15 measured batches per launch; 1–3 launches per configuration, interleaved in time.
* Statistics: median and interquartile range of all measured batches of all successful launches.
* Memory: peak working set of the whole process (Windows `GetProcessMemoryInfo`), one value per launch; includes the language runtime.
  The Linux branch (`wait4`/`ru_maxrss`) is written but was not run.
* Limits: 120 s per single multiplication (a launch silent for 120 s is killed and recorded as a timeout), 30 min per launch, 4 GiB memory.

## Raw data

`results/raw_samples.csv`: `launch_id, experiment, sweep, language, variant, n, phase, index, inner, seconds_per_multiplication, checksum`.
`results/launches.csv`: one row per process (`status`, `peak_working_set_bytes`, `background_cpu_percent`, command, …). `variant` is always `main`/empty. Constant for every row, so not stored as columns: numeric type float64 (binary64) and one kernel thread (the JVM also runs its own JIT/GC threads).

## AI assistance

Claude Code (Anthropic) was used to plan the design, write the code, scripts and a first draft of the report, and to debug the build; all numbers come from running this code. See the report, Section 8.

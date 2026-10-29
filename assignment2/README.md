# Assignment 2 – Optimised dense and sparse matrix multiplication in C

Big Data 2026–2027, Grado en Ciencia e Ingeniería de Datos (ULPGC). Report: `report/Assignment02_Choukri_Mohamed_Report.pdf`.

* **Part A:** baseline `ijk` (the Assignment 1 C kernel, unchanged), loop reordering `ikj`, cache blocking (64 × 64 tiles), n = 100 … 3000.
* **Part B:** own CSR format and Gustavson CSR × CSR multiplication against dense blocked multiplication of the same matrices,
  n = 1000 and 2000, densities 0.1 / 1 / 5 / 10 / 25 / 50 %; dense → CSR conversion timed separately.

## Layout

```
src/c/dense.c                 ijk, ikj, blocked[S]   (verify / bench commands)
src/c/sparse.c                CSR read/write, Gustavson SpGEMM, dense -> CSR conversion (verify / bench / to-csr / convert)
config/experiment.json        declared protocol: sizes, densities, launches, repetitions, warm-up, limits, tile sizes
scripts/build.ps1 | build.sh  compile with GCC -O2; also saves GCC's vectorisation report in build/*-vectorisation.txt
scripts/generate_inputs.py    dense inputs (same rule as Assignment 1, byte-identical) and sparse inputs (.bin and .csr)
scripts/run_tests.py          115 correctness checks -> results/correctness.csv
scripts/collect_environment.py, scripts/run_benchmarks.py, scripts/harness.py (launcher + memory metric copied from Assignment 1)
scripts/analyze.py            tables (results/tables) and figures (results/figures)
scripts/build_report.py       HTML + PDF report;  scripts/package_materials.py -> dist/Assignment02_Choukri_Mohamed_Materials.zip
```

## Versions used

HP 250 G8, Intel Core i5-1135G7, 15.7 GB RAM, Windows 11 Pro (build 26300), AC power, "Balanced" plan.
GCC 16.2.0 (WinLibs, `winget install BrechtSanders.WinLibs.POSIX.UCRT`), flags `-std=c11 -Wall -Wextra -O2`.
Driver and analysis: Anaconda Python 3.13.9 with numpy 2.3.5, pandas 2.3.3, matplotlib 3.10.6 (`../requirements.txt`).

## Build and run (PowerShell, from `assignment2/`)

```powershell
$py = 'C:\Users\<you>\anaconda3\python.exe'
.\scripts\build.ps1                          # 1. compile dense.exe and sparse.exe
& $py scripts\generate_inputs.py             # 2. about 0.9 GB of inputs in inputs\
& $py scripts\generate_inputs.py --check     #    "OK (64 files compared)";  --check-a1 compares the dense files with Assignment 1
& $py scripts\run_tests.py                   # 3. must end with "115/115 checks passed"
& $py scripts\collect_environment.py         # 4. environment record
& $py scripts\run_benchmarks.py              # 5. about 40 min; --resume continues, --dry-run prints the commands
& $py scripts\analyze.py                     # 6. tables and figures
& $py scripts\build_report.py                # 7. report PDF (headless Edge)
& $py scripts\package_materials.py           # 8. materials zip
```

Leave the laptop idle and on AC power during step 5; a first attempt of the sweep was discarded because the machine was busy
(probably scanning the freshly generated inputs) and the sweep was restarted from scratch.

## Inputs

Dense: `A_<n>.bin`, `B_<n>.bin`, n·n float64 little-endian row-major, uniform [0, 1), `default_rng([20261012, n, which])`.
Sparse: `SA_<n>_d<ppm>.{bin,csr}` and `SB_…`: each entry nonzero with probability d (Bernoulli mask), values uniform (0, 1],
`default_rng([20261026, n, d·10^6, which])`. CSR file: int64 n, int64 nnz, int64 rowptr[n+1], int32 col[nnz], float64 val[nnz].
Hashes and exact nonzero counts: `inputs/checksums.json`. The input files themselves are git-ignored.

## Raw data

`results/raw_samples.csv`: `launch_id, experiment (dense|blocksize|sparse), sweep, method, n, density, phase, index, inner, seconds_per_multiplication, checksum`.
`results/launches.csv`: one row per process with status, peak working set, `nnz_a`, `nnz_b`, `nnz_c` (sparse runs), background CPU load and command.

## AI assistance

Claude Code (Anthropic) wrote the code and scripts and drafted the report; see the report, Section 8.

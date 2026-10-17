"""Benchmark driver for Assignment 1 (standard library only).

For every planned configuration (language, variant, n) it launches the
implementation as a separate process, reads the timing samples it prints,
measures the process's peak memory from outside, and appends everything to

    results/raw_samples.csv    one row per timed batch (warm-up included, flagged)
    results/launches.csv       one row per process launch (status, peak memory)

Limits come from config/experiment.json. A launch is killed and recorded as a
timeout when it prints nothing for `seconds_per_multiplication` seconds after
loading its inputs (so no single timed batch may exceed that limit) or when it
runs longer than `seconds_per_launch` in total. Nothing is ever back-filled for
a timed-out launch.

Peak memory metric: peak working set of the whole process (Windows
PeakWorkingSetSize via GetProcessMemoryInfo; Linux ru_maxrss from wait4),
plus peak private/committed bytes on Windows.

Examples
    python scripts/run_benchmarks.py --python C:/path/to/python.exe
    python scripts/run_benchmarks.py --resume              # continue an interrupted run
    python scripts/run_benchmarks.py --dry-run             # print the commands only
"""
import argparse
import csv
import datetime
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from common import RESULTS, ROOT, input_paths, kernel_command, load_config

LANG_ORDER = ["python", "java", "c"]
SAMPLE_FIELDS = ["launch_id", "experiment", "sweep", "language", "variant", "n", "phase", "index",
                 "inner", "seconds_per_multiplication", "checksum"]
LAUNCH_FIELDS = ["launch_id", "experiment", "sweep", "language", "variant", "n", "status", "exit_code",
                 "wall_seconds", "peak_working_set_bytes", "peak_private_bytes", "warmup_batches", "reps",
                 "target_seconds", "measured_samples", "background_cpu_percent", "started_utc", "command"]

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    class _PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t)]

    _gpmi = ctypes.WinDLL("kernel32", use_last_error=True).K32GetProcessMemoryInfo
    _gpmi.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
    _gpmi.restype = wintypes.BOOL


def background_cpu_percent(window=0.5):
    """System-wide CPU busy percentage over a short window, sampled before a launch.

    Recorded so that noisy launches can be explained afterwards. Returns None if unavailable.
    """
    def snapshot():
        if os.name == "nt":
            idle, kernel, user = (wintypes.FILETIME() for _ in range(3))
            ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
            as_int = lambda ft: (ft.dwHighDateTime << 32) | ft.dwLowDateTime  # noqa: E731
            return as_int(idle), as_int(kernel) + as_int(user)  # kernel time includes idle time
        with open("/proc/stat") as f:
            vals = [int(x) for x in f.readline().split()[1:]]
        return vals[3] + vals[4], sum(vals)

    try:
        i0, t0 = snapshot()
        time.sleep(window)
        i1, t1 = snapshot()
        return round(100.0 * (1 - (i1 - i0) / (t1 - t0)), 1) if t1 > t0 else None
    except Exception:  # noqa: BLE001 - diagnostic only
        return None


def finish_and_measure(proc):
    """Wait for the process and return (peak_working_set, peak_private) in bytes."""
    if os.name == "nt":
        proc.wait()
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        if not _gpmi(int(proc._handle), ctypes.byref(pmc), pmc.cb):  # handle stays valid after exit
            raise ctypes.WinError(ctypes.get_last_error())
        return int(pmc.PeakWorkingSetSize), int(pmc.PeakPagefileUsage)
    _, status, usage = os.wait4(proc.pid, 0)  # per-child rusage (not exercised on the Windows test machine)
    proc.returncode = os.waitstatus_to_exitcode(status)
    scale = 1 if sys.platform == "darwin" else 1024
    return int(usage.ru_maxrss * scale), ""


def run_launch(cmd, limits):
    """Run one process; return (status, exit_code, wall, peak_ws, peak_priv, sample_lines, stderr_tail)."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    q = queue.Queue()
    err = []

    def pump_out():
        for line in proc.stdout:
            q.put(line.rstrip("\n"))
        q.put(None)

    def pump_err():
        for line in proc.stderr:
            err.append(line)

    threading.Thread(target=pump_out, daemon=True).start()
    threading.Thread(target=pump_err, daemon=True).start()

    start = time.monotonic()
    last = start
    ready = False
    lines = []
    status = "ok"
    while True:
        try:
            line = q.get(timeout=1.0)
        except queue.Empty:
            line = ""
        now = time.monotonic()
        if line is None:
            break
        if line:
            lines.append(line)
            last = now
            if line.startswith("ready,"):
                ready = True
        allowed = limits["seconds_per_multiplication"] if ready else limits["seconds_to_load_inputs"]
        if now - last > allowed:
            status = "timeout_multiplication" if ready else "timeout_load"
            proc.kill()
            break
        if now - start > limits["seconds_per_launch"]:
            status = "timeout_launch"
            proc.kill()
            break
    peak_ws, peak_priv = finish_and_measure(proc)
    wall = time.monotonic() - start
    if status == "ok" and proc.returncode != 0:
        status = "error"
    if status != "ok":  # drain whatever was printed before the kill
        while True:
            try:
                item = q.get(timeout=0.2)
            except queue.Empty:
                break
            if item:
                lines.append(item)
    return status, proc.returncode, wall, peak_ws, peak_priv, lines, "".join(err)[-500:]


def jobs_for(cfg, experiment, only_languages, only_sizes, python, java):
    """Expand the experiment plan into (sweep, language, variant, n, reps, warmup) jobs."""
    exp = cfg["experiments"][experiment]
    entries = []  # (language, variant, n, launches, reps, warmup)
    for lang, rows in exp["plan"].items():
        for n, launches, reps, warmup in rows:
            entries.append((lang, None, n, launches, reps, warmup))
    if only_languages:
        entries = [e for e in entries if e[0] in only_languages]
    if only_sizes:
        entries = [e for e in entries if e[2] in only_sizes]
    max_sweeps = max((e[3] for e in entries), default=0)
    jobs = []
    for sweep in range(1, max_sweeps + 1):
        # rotate language order between sweeps so no language always runs first
        order = LANG_ORDER[(sweep - 1) % 3:] + LANG_ORDER[:(sweep - 1) % 3]
        for n in sorted({e[2] for e in entries}):
            for lang in order:
                for e in entries:
                    if e[0] == lang and e[2] == n and e[3] >= sweep:
                        jobs.append((sweep, *e[:3], e[4], e[5]))
    return jobs


def append_rows(path, fields, rows):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default="main", choices=["main"])
    ap.add_argument("--python", default=sys.executable, help="interpreter for the Python kernel")
    ap.add_argument("--java", default=None, help="java executable (default: the one recorded by build.ps1)")
    ap.add_argument("--languages", nargs="*", choices=LANG_ORDER)
    ap.add_argument("--sizes", nargs="*", type=int)
    ap.add_argument("--config", type=Path, default=None, help="alternative configuration file")
    ap.add_argument("--tag", default="", help="suffix for the result files, e.g. _pilot")
    ap.add_argument("--resume", action="store_true", help="skip jobs already present in launches.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text()) if args.config else load_config()
    limits = cfg["limits"]
    jobs = jobs_for(cfg, args.experiment, args.languages, args.sizes, args.python, args.java)
    RESULTS.mkdir(exist_ok=True)
    samples_path = RESULTS / f"raw_samples{args.tag}.csv"
    launches_path = RESULTS / f"launches{args.tag}.csv"

    done, next_id = set(), 1
    if launches_path.exists():
        with open(launches_path, newline="") as f:
            for row in csv.DictReader(f):
                next_id = max(next_id, int(row["launch_id"]) + 1)
                done.add((row["experiment"], int(row["sweep"]), row["language"], row["variant"], int(row["n"])))

    print(f"{len(jobs)} launches planned for experiment '{args.experiment}'")
    for i, (sweep, lang, variant, n, reps, warmup) in enumerate(jobs, 1):
        key = (args.experiment, sweep, lang, variant or "", n)
        if args.resume and key in done:
            continue
        cmd = kernel_command(lang, variant, python=args.python, java=args.java)
        a, b = input_paths(n)
        cmd += ["bench", str(n), a, b, str(warmup), str(reps), str(cfg["target_seconds"])]
        label = f"[{i}/{len(jobs)}] sweep {sweep} {lang}{'/' + variant if variant else ''} n={n}"
        if args.dry_run:
            print(label, " ".join(cmd))
            continue
        background = background_cpu_percent()
        started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        status, code, wall, peak_ws, peak_priv, lines, err = run_launch(cmd, limits)
        launch_id = next_id
        next_id += 1
        samples = []
        for line in lines:
            p = line.split(",")
            if p[0] == "sample":
                samples.append(dict(launch_id=launch_id, experiment=args.experiment, sweep=sweep, language=lang,
                                    variant=variant or "", n=n, phase=p[3], index=p[4], inner=p[5],
                                    seconds_per_multiplication=p[6], checksum=p[7]))
        append_rows(samples_path, SAMPLE_FIELDS, samples)
        measured = sum(1 for s in samples if s["phase"] == "measure")
        append_rows(launches_path, LAUNCH_FIELDS, [dict(
            launch_id=launch_id, experiment=args.experiment, sweep=sweep, language=lang, variant=variant or "",
            n=n, status=status, exit_code=code, wall_seconds=f"{wall:.3f}", peak_working_set_bytes=peak_ws,
            peak_private_bytes=peak_priv, warmup_batches=warmup, reps=reps, target_seconds=cfg["target_seconds"],
            measured_samples=measured, background_cpu_percent=background, started_utc=started,
            command=" ".join(cmd))])
        spm = [float(s["seconds_per_multiplication"]) for s in samples if s["phase"] == "measure"]
        med = sorted(spm)[len(spm) // 2] if spm else float("nan")
        print(f"{label}: {status} in {wall:.1f}s, median {med:.4g} s/mult, "
              f"peak WS {peak_ws / 2**20:.1f} MiB, bg CPU {background}%" + (f"  stderr: {err.strip()}" if err.strip() else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

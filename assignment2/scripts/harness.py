"""Process launcher and peak-memory measurement, copied unchanged from assignment1/scripts/run_benchmarks.py
(Assignment 1) so that Assignment 2 uses exactly the same timing boundary, timeout rule and memory metric.

run_launch(cmd, limits) -> (status, exit_code, wall_seconds, peak_working_set, peak_private, stdout_lines, stderr_tail)
"""
import os
import queue
import subprocess
import sys
import threading
import time

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



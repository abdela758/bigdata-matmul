"""Record the hardware / software environment of a benchmark session.

Writes results/environment.json. Standard library only; the hardware queries
use PowerShell (CIM) on Windows and /proc, lscpu on Linux.
"""
import argparse
import datetime
import json
import os
import platform
import re
import subprocess
import sys

from common import RESULTS, ROOT, build_info, load_config


def run(cmd, **kw):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, **kw)
        return (out.stdout or "") + (out.stderr or "")
    except Exception as exc:  # noqa: BLE001 - environment capture must never fail the run
        return f"unavailable: {exc}"


def ps_json(query):
    text = run(["powershell", "-NoProfile", "-Command", f"{query} | ConvertTo-Json -Compress"])
    try:
        return json.loads(text)
    except ValueError:
        return text.strip()


def collect(python_exe):
    env = {
        "collected_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    if os.name == "nt":
        env["cpu"] = ps_json("Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed,L2CacheSize,L3CacheSize")
        env["memory_gb"] = ps_json("[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)")
        env["computer_model"] = ps_json("(Get-CimInstance Win32_ComputerSystem).Model")
        env["os"] = ps_json("Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber")
        env["battery_status_2_means_AC"] = ps_json("Get-CimInstance Win32_Battery | Select-Object BatteryStatus,EstimatedChargeRemaining")
        plan = run(["powercfg", "/getactivescheme"])  # the label text is localised/OEM-encoded: keep GUID and name only
        guid = re.search(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", plan)
        name = re.search(r"\(([^)]*)\)", plan)
        env["power_plan"] = {"guid": guid.group(0) if guid else None, "name": name.group(1) if name else None}
    else:
        env["cpu"] = run(["lscpu"]).strip()
        env["memory_gb"] = run(["grep", "MemTotal", "/proc/meminfo"]).strip()
    env["python_kernel"] = {
        "executable": python_exe,
        "version": run([python_exe, "--version"]).strip(),
        "implementation_flags": "default interpreter flags (no -O, no JIT)",
    }
    env["driver_python"] = {"executable": sys.executable, "version": sys.version.split()[0]}
    try:
        env["build"] = build_info()
    except FileNotFoundError:
        env["build"] = "run scripts/build.ps1 first"
    cfg = load_config()
    env["java_flags"] = cfg["java_flags"]
    env["java_note"] = "default tiered JIT (C1+C2) and default G1 GC apart from the flags above"
    commit = run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]).strip()
    env["git_commit"] = commit if re.fullmatch(r"[0-9a-f]{7,40}", commit) else "not committed yet"
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable, help="interpreter used for the Python kernel")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    env = collect(args.python)
    (RESULTS / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    print("wrote results/environment.json")


if __name__ == "__main__":
    main()

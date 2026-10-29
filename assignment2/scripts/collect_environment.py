"""Record the hardware / software environment of the benchmark session -> results/environment.json
(same queries as Assignment 1's collect_environment.py, minus the Java and Python-kernel parts)."""
import datetime
import json
import platform
import re
import subprocess

from common import BUILD, RESULTS


def run(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (out.stdout or "") + (out.stderr or "")
    except Exception as exc:  # noqa: BLE001 - must never fail the run
        return f"unavailable: {exc}"


def ps_json(query):
    try:
        return json.loads(run(["powershell", "-NoProfile", "-Command", f"{query} | ConvertTo-Json -Compress"]))
    except ValueError:
        return None


def main():
    plan = run(["powercfg", "/getactivescheme"])
    name = re.search(r"\(([^)]*)\)", plan)
    env = {
        "collected_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "cpu": ps_json("Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed,L2CacheSize,L3CacheSize"),
        "memory_gb": ps_json("[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)"),
        "computer_model": ps_json("(Get-CimInstance Win32_ComputerSystem).Model"),
        "os": ps_json("Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber"),
        "battery_status_2_means_AC": ps_json("Get-CimInstance Win32_Battery | Select-Object BatteryStatus,EstimatedChargeRemaining"),
        "power_plan": name.group(1) if name else None,
        "build": json.loads((BUILD / "build-info.json").read_text(encoding="utf-8-sig")),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "environment.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    print("wrote results/environment.json")


if __name__ == "__main__":
    main()

"""Benchmark driver for Assignment 2 (standard library only).

Experiments (config/experiment.json):
  dense       Part A: methods ijk (baseline), ikj, blocked (S = block_size) on the uniform dense inputs
  blocksize   Part A: blocked method with several tile sizes S (sensitivity only)
  sparse      Part B: for every (n, density): the CSR multiplication, the dense comparator method on the
              same logical matrices (dense layout with zeros), and the dense -> CSR conversion of matrix A
Every configuration is a separate process launched through harness.run_launch (copied from Assignment 1:
same timeout rule and the same peak-working-set measurement). Launches are interleaved: sweep 1 runs
every configuration once, sweep 2 the ones with a second launch, with the method order rotated.

Writes results/raw_samples.csv (one row per timed batch, warm-up included and flagged) and
results/launches.csv (one row per process: status, peak memory, nonzero counts, command).

    python scripts/run_benchmarks.py --experiment dense blocksize sparse [--resume] [--dry-run]
"""
import argparse
import csv
import datetime
import sys

from common import DENSE, RESULTS, SPARSE, dense_inputs, load_config, sparse_inputs
from harness import background_cpu_percent, run_launch

SAMPLE_FIELDS = ["launch_id", "experiment", "sweep", "method", "n", "density", "phase", "index", "inner",
                 "seconds_per_multiplication", "checksum"]
LAUNCH_FIELDS = ["launch_id", "experiment", "sweep", "method", "n", "density", "status", "exit_code", "wall_seconds",
                 "peak_working_set_bytes", "peak_private_bytes", "nnz_a", "nnz_b", "nnz_c", "warmup_batches", "reps",
                 "measured_samples", "background_cpu_percent", "started_utc", "command"]


def rotate(seq, k):
    k %= len(seq)
    return seq[k:] + seq[:k]


def plan(cfg, experiments):
    """List of jobs: dicts with experiment, sweep, method, n, density, cmd, warmup, reps."""
    t = str(cfg["target_seconds"])
    jobs = []
    if "dense" in experiments:
        d = cfg["dense"]
        for sweep in (1, 2):
            for n, launches, reps, warmup in d["plan"]:
                if launches < sweep:
                    continue
                for m in rotate(d["methods"], sweep - 1):
                    kernel = m if m != "blocked" else f"blocked{cfg['block_size']}"
                    a, b = dense_inputs(n)
                    jobs.append(dict(experiment="dense", sweep=sweep, method=m, n=n, density=1.0, warmup=warmup, reps=reps,
                                     cmd=[DENSE, "bench", kernel, n, a, b, warmup, reps, t]))
    if "blocksize" in experiments:
        bsz = cfg["block_sizes"]
        for n in bsz["sizes"]:
            for s in bsz["values"]:
                a, b = dense_inputs(n)
                jobs.append(dict(experiment="blocksize", sweep=1, method=f"blocked{s}", n=n, density=1.0,
                                 warmup=bsz["warmup"], reps=bsz["reps"],
                                 cmd=[DENSE, "bench", f"blocked{s}", n, a, b, bsz["warmup"], bsz["reps"], t]))
    if "sparse" in experiments:
        sp = cfg["sparse"]
        comp = sp["dense_comparator"]
        kernel = comp if comp != "blocked" else f"blocked{cfg['block_size']}"
        for sweep in (1, 2):
            for n in sp["sizes"]:
                for dens in sp["densities"]:
                    sa, sb = sparse_inputs(n, dens, "csr")
                    da, db = sparse_inputs(n, dens, "bin")
                    w, r = sp["warmup"], sp["reps"]
                    cand = []
                    if sp["launches"] >= sweep:
                        cand.append(dict(method="csr", cmd=[SPARSE, "bench", n, sa, sb, w, r, t]))
                    if sp["dense_launches"][str(n)] >= sweep:
                        cand.append(dict(method=f"dense_{comp}", cmd=[DENSE, "bench", kernel, n, da, db, w, r, t]))
                    if sp["convert_launches"] >= sweep:
                        cand.append(dict(method="convert", cmd=[SPARSE, "convert", n, da, w, r, t]))
                    for c in rotate(cand, sweep - 1):
                        jobs.append(dict(experiment="sparse", sweep=sweep, n=n, density=dens, warmup=w, reps=r, **c))
    for j in jobs:
        j["cmd"] = [str(x) for x in j["cmd"]]
    return jobs


def append_rows(path, fields, rows):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", nargs="+", default=["dense", "blocksize", "sparse"], choices=["dense", "blocksize", "sparse"])
    ap.add_argument("--resume", action="store_true", help="skip jobs already present in launches.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    jobs = plan(cfg, args.experiment)
    RESULTS.mkdir(exist_ok=True)
    samples_path, launches_path = RESULTS / "raw_samples.csv", RESULTS / "launches.csv"
    done, next_id = set(), 1
    if launches_path.exists():
        with open(launches_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                next_id = max(next_id, int(row["launch_id"]) + 1)
                done.add((row["experiment"], int(row["sweep"]), row["method"], int(row["n"]), float(row["density"])))
    print(f"{len(jobs)} launches planned", flush=True)
    for i, j in enumerate(jobs, 1):
        key = (j["experiment"], j["sweep"], j["method"], j["n"], float(j["density"]))
        label = f"[{i}/{len(jobs)}] {j['experiment']} sweep {j['sweep']} {j['method']} n={j['n']} density={j['density']:g}"
        if args.resume and key in done:
            continue
        if args.dry_run:
            print(label, " ".join(j["cmd"]))
            continue
        bg = background_cpu_percent()
        started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        status, code, wall, ws, priv, lines, err = run_launch(j["cmd"], cfg["limits"])
        lid, next_id = next_id, next_id + 1
        info, samples = {}, []
        for line in lines:
            p = line.split(",")
            if p[0] == "info":
                info.update(kv.split("=") for kv in p[1:])
            elif p[0] == "sample":
                samples.append(dict(launch_id=lid, experiment=j["experiment"], sweep=j["sweep"], method=j["method"], n=j["n"],
                                    density=j["density"], phase=p[3], index=p[4], inner=p[5],
                                    seconds_per_multiplication=p[6], checksum=p[7]))
        append_rows(samples_path, SAMPLE_FIELDS, samples)
        measured = [float(s["seconds_per_multiplication"]) for s in samples if s["phase"] == "measure"]
        append_rows(launches_path, LAUNCH_FIELDS, [dict(
            launch_id=lid, experiment=j["experiment"], sweep=j["sweep"], method=j["method"], n=j["n"], density=j["density"],
            status=status, exit_code=code, wall_seconds=f"{wall:.3f}", peak_working_set_bytes=ws, peak_private_bytes=priv,
            nnz_a=info.get("nnzA", info.get("nnz", "")), nnz_b=info.get("nnzB", ""), nnz_c=info.get("nnzC", ""),
            warmup_batches=j["warmup"], reps=j["reps"], measured_samples=len(measured), background_cpu_percent=bg,
            started_utc=started, command=" ".join(j["cmd"]))])
        med = sorted(measured)[len(measured) // 2] if measured else float("nan")
        print(f"{label}: {status} in {wall:.1f}s, median {med:.4g} s, peak WS {ws / 2**20:.1f} MiB, bg {bg}%"
              + (f"  stderr: {err.strip()}" if err.strip() else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

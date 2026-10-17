"""Generate (or verify) the shared input matrices of Assignment 1.

For every size n two matrices are written as raw little-endian float64,
row-major, values uniform in [0, 1):

    inputs/A_<n>.bin   inputs/B_<n>.bin

All three language implementations read these exact files, so every method
multiplies the same logical matrices. A seed is derived from (SEED, n, which),
so every file is reproducible on its own. inputs/checksums.json stores the
SHA-256 of every file; `--check` verifies freshly generated files against it.

Requires NumPy (analysis environment, see requirements.txt).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SEED = 20261012  # first day of the submission window of Assignment 1
ROOT = Path(__file__).resolve().parent.parent


def planned_sizes(cfg):
    """Every n used in config/experiment.json."""
    return {row[0] for rows in cfg["experiments"]["main"]["plan"].values() for row in rows}


def matrix(n, which):
    rng = np.random.default_rng([SEED, n, which])
    return rng.random((n, n), dtype=np.float64)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="*", help="sizes to generate (default: all sizes in config/experiment.json)")
    ap.add_argument("--outdir", type=Path, default=ROOT / "inputs")
    ap.add_argument("--check", action="store_true", help="compare file hashes with inputs/checksums.json")
    args = ap.parse_args()

    cfg = json.loads((ROOT / "config" / "experiment.json").read_text())
    sizes = args.sizes or sorted(planned_sizes(cfg))
    args.outdir.mkdir(parents=True, exist_ok=True)

    sums_path = ROOT / "inputs" / "checksums.json"
    known = json.loads(sums_path.read_text()) if sums_path.exists() else {"files": {}}
    computed = {}
    for n in sizes:
        for which, name in ((0, "A"), (1, "B")):
            path = args.outdir / f"{name}_{n}.bin"
            if not path.exists():
                matrix(n, which).astype("<f8").tofile(path)
            computed[path.name] = sha256(path)
    if args.check:
        bad = [k for k, v in computed.items() if known["files"].get(k) != v]
        if bad:
            print("MISMATCH:", ", ".join(bad))
            return 1
        print(f"OK: {len(computed)} files match inputs/checksums.json")
        return 0
    known["generator"] = f"numpy {np.__version__}, default_rng([SEED, n, which]).random((n, n)), SEED={SEED}, uniform [0,1), float64 little-endian"
    known["files"].update(computed)
    known["files"] = dict(sorted(known["files"].items()))
    sums_path.write_text(json.dumps(known, indent=2) + "\n")
    print(f"wrote {len(computed)} files to {args.outdir} and updated {sums_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

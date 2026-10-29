"""Generate (or verify) the inputs of Assignment 2.

Part A (dense): inputs/A_<n>.bin, B_<n>.bin - n*n little-endian float64, row-major, uniform in [0, 1),
    numpy default_rng([20261012, n, which]). This is exactly the Assignment 1 rule, so the files are
    byte-identical to Assignment 1's inputs (checked with --check-a1).
Part B (sparse): for every (n, density) two matrices SA, SB. Each entry is nonzero independently with
    probability `density` (uniform random positions, Bernoulli mask); nonzero values are uniform in (0, 1].
    default_rng([20261026, n, density_ppm, which]). Written twice with the same logical content:
    .bin (dense layout with explicit zeros, read by the dense method) and .csr (read by the sparse method;
    format described in src/c/sparse.c).
inputs/checksums.json stores SHA-256 hashes and the exact nonzero counts.
"""
import argparse
import hashlib
import json
import sys

import numpy as np

from common import INPUTS, ROOT, dense_inputs, load_config, sparse_inputs

SEED_A, SEED_B = 20261012, 20261026


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csr(path, m):
    n = m.shape[0]
    rows, cols = np.nonzero(m)  # row-major order, columns increasing inside a row
    rowptr = np.zeros(n + 1, dtype="<i8")
    np.cumsum(np.bincount(rows, minlength=n), out=rowptr[1:])
    with open(path, "wb") as f:
        np.array([n, len(cols)], dtype="<i8").tofile(f)
        rowptr.tofile(f)
        cols.astype("<i4").tofile(f)
        m[rows, cols].astype("<f8").tofile(f)
    return len(cols)


def sparse_matrix(n, d, which):
    rng = np.random.default_rng([SEED_B, n, round(d * 1_000_000), which])
    mask = rng.random((n, n)) < d
    vals = 1.0 - rng.random((n, n))  # (0, 1]: a stored value is never zero
    return np.where(mask, vals, 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare hashes with inputs/checksums.json")
    ap.add_argument("--check-a1", action="store_true", help="compare Part A files with Assignment 1's checksums")
    args = ap.parse_args()
    cfg = load_config()
    INPUTS.mkdir(exist_ok=True)
    files, nnz = {}, {}
    for n in sorted({row[0] for row in cfg["dense"]["plan"]}):
        for which, path in enumerate(dense_inputs(n)):
            if not (INPUTS / path).exists():
                np.random.default_rng([SEED_A, n, which]).random((n, n), dtype=np.float64).astype("<f8").tofile(path)
            files[path.split("\\")[-1].split("/")[-1]] = sha256(path)
    for n in cfg["sparse"]["sizes"]:
        for d in cfg["sparse"]["densities"]:
            for which, (pbin, pcsr) in enumerate(zip(sparse_inputs(n, d, "bin"), sparse_inputs(n, d, "csr"))):
                m = sparse_matrix(n, d, which)
                m.astype("<f8").tofile(pbin)
                nz = write_csr(pcsr, m)
                for p in (pbin, pcsr):
                    files[p.replace("\\", "/").split("/")[-1]] = sha256(p)
                nnz[pcsr.replace("\\", "/").split("/")[-1]] = nz
    sums = INPUTS / "checksums.json"
    if args.check or args.check_a1:
        ref = json.loads(sums.read_text())["files"] if args.check else \
            json.loads((ROOT.parent / "assignment1" / "inputs" / "checksums.json").read_text())["files"]
        names = [k for k in files if k in ref] if args.check_a1 else list(files)
        bad = [k for k in names if ref.get(k) != files[k]]
        print(f"{'MISMATCH: ' + ', '.join(bad) if bad else 'OK'} ({len(names)} files compared)")
        return 1 if bad else 0
    sums.write_text(json.dumps({"generator": "see scripts/generate_inputs.py docstring; numpy " + np.__version__,
                                "files": dict(sorted(files.items())), "nnz": dict(sorted(nnz.items()))}, indent=2) + "\n")
    print(f"wrote {len(files)} files, hashes in {sums}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Correctness tests for Assignment 2 (needs NumPy). Writes results/correctness.csv; exit code 1 on any failure.

Dense (ijk, ikj, blocked with S = 64 and S = 32):
  identity / zero matrices bit-exact; random matrices vs NumPy A @ B with numpy.allclose(rtol=atol=1e-9)
  (same tolerance as Assignment 1); sizes that are and are not multiples of the block size; and the three
  methods must be bit-identical to the ijk baseline (same addition order for every C[i][j]).
Sparse (CSR):
  dense -> CSR conversion in C must produce byte-identical files to the Python writer;
  products for densities 0 (empty), identity, 1 %, 10 %, 50 % and 100 % compared with NumPy (same tolerance)
  and bit-identical to the dense ijk result; nnz(C) must equal the number of nonzeros of NumPy's product
  (all values are positive, so no cancellation can create a zero).
"""
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from common import DENSE, RESULTS, SPARSE
from generate_inputs import write_csr

TOL = 1e-9
METHODS = ["ijk", "ikj", "blocked", "blocked32"]
rows = []


def record(test, method, n, ok, detail):
    rows.append(dict(test=test, method=method, n=n, passed=int(ok), detail=detail))
    print(f"{'PASS' if ok else 'FAIL'}  {test:<26} {method:<10} n={n:<4} {detail}")


def run(cmd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr}")


def read_csr(path):
    with open(path, "rb") as f:
        n, nnz = np.fromfile(f, "<i8", 2)
        rowptr = np.fromfile(f, "<i8", n + 1)
        col = np.fromfile(f, "<i4", nnz)
        val = np.fromfile(f, "<f8", nnz)
    m = np.zeros((n, n))
    for i in range(n):
        m[i, col[rowptr[i]:rowptr[i + 1]]] = val[rowptr[i]:rowptr[i + 1]]
    return m, int(nnz)


def close(C, ref):
    d = float(np.abs(C - ref).max()) if C.size else 0.0
    return bool(np.allclose(C, ref, rtol=TOL, atol=TOL)), f"max |C - NumPy| = {d:.2e}"


def main():
    rng = np.random.default_rng(11)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def dense_case(name, A, B, expected=None):
            n = A.shape[0]
            a, b = tmp / "A.bin", tmp / "B.bin"
            A.astype("<f8").tofile(a); B.astype("<f8").tofile(b)
            out = {}
            for m in METHODS:
                run([DENSE, "verify", m, n, a, b, tmp / "C.bin"])
                out[m] = np.fromfile(tmp / "C.bin", "<f8").reshape(n, n)
                if expected is not None:
                    ok = np.array_equal(out[m], expected); record(name, m, n, ok, "bit-exact" if ok else "mismatch")
                else:
                    record(name, m, n, *close(out[m], A @ B))
            same = all(np.array_equal(out["ijk"], out[m]) for m in METHODS[1:])
            record(name + "_same_as_ijk", "all", n, same, "bit-identical to the baseline" if same else "differs")
            return out["ijk"]

        for n in (1, 7, 33):
            X = rng.random((n, n))
            dense_case("identity", np.eye(n), X, X)
            dense_case("zero", np.zeros((n, n)), X, np.zeros((n, n)))
        for n in (10, 63, 64, 65, 100, 130, 300):  # around and away from the block sizes 32 and 64
            dense_case("random_dense", rng.random((n, n)), rng.uniform(-1, 1, (n, n)))

        def sparse_case(name, A, B):
            n = A.shape[0]
            for tag, M in (("A", A), ("B", B)):
                M.astype("<f8").tofile(tmp / f"{tag}.bin")
                write_csr(tmp / f"{tag}_py.csr", M)
                run([SPARSE, "to-csr", n, tmp / f"{tag}.bin", tmp / f"{tag}_c.csr"])
                same = (tmp / f"{tag}_py.csr").read_bytes() == (tmp / f"{tag}_c.csr").read_bytes()
                record(name + "_conversion", "csr", n, same, f"matrix {tag}: C conversion " + ("= Python writer" if same else "DIFFERS"))
            run([SPARSE, "verify", n, tmp / "A_py.csr", tmp / "B_py.csr", tmp / "C.csr"])
            C, nnz = read_csr(tmp / "C.csr")
            ref = A @ B
            record(name, "csr", n, *close(C, ref))
            ok = nnz == int(np.count_nonzero(ref))
            record(name + "_nnz", "csr", n, ok, f"nnz(C) = {nnz}, NumPy = {np.count_nonzero(ref)}")
            run([DENSE, "verify", "ijk", n, tmp / "A.bin", tmp / "B.bin", tmp / "Cd.bin"])
            Cd = np.fromfile(tmp / "Cd.bin", "<f8").reshape(n, n)
            ok = np.array_equal(C, Cd)
            record(name + "_same_as_dense", "csr", n, ok, "bit-identical to dense ijk" if ok else "differs")

        def sp(n, d):
            mask = rng.random((n, n)) < d
            return np.where(mask, 1.0 - rng.random((n, n)), 0.0)

        sparse_case("sparse_empty", np.zeros((50, 50)), sp(50, 0.1))
        sparse_case("sparse_identity", np.eye(60), sp(60, 0.2))
        for n in (50, 200):
            for d in (0.01, 0.1, 0.5, 1.0):
                sparse_case(f"sparse_d{d:g}", sp(n, d), sp(n, d))

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "correctness.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["test", "method", "n", "passed", "detail"])
        w.writeheader(); w.writerows(rows)
    failed = [r for r in rows if not r["passed"]]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

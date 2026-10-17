"""Correctness tests for the three implementations of Assignment 1.

Needs NumPy (analysis environment). Every implementation is run through its `verify` command on the same
matrices; the result C = A x B is read back and compared with references. Writes results/correctness.csv and
exits non-zero if any check fails.

Checks
  identity / zero   I x X = X x I = X and 0 x X = 0, bit-exact
  known answers     hand-computed 2x2 and 3x3 products, bit-exact
  integer-valued    random integers 0..9 stored as doubles: every partial sum is exactly representable, so any
                    summation order gives the same answer; compared bit-exact with NumPy's int64 product
  random reals      uniform [0,1) and [-1,1), and the shared benchmark inputs: compared with NumPy's A @ B using
                    numpy.allclose(rtol=1e-9, atol=1e-9). binary64 carries about 16 significant digits and summing
                    n <= 300 products only changes the last digits, so 1e-9 is strict but leaves room for the
                    different summation order of NumPy's BLAS
  cross-language    the three result files must be bit-identical (same operation order, no fused multiply-add)
"""
import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from common import RESULTS, input_paths, kernel_command

RTOL = ATOL = 1e-9
LANGS = ["c", "java", "python"]


def run_verify(lang, n, a, b, out, python):
    cmd = kernel_command(lang, python=python) + ["verify", str(n), str(a), str(b), str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{lang} verify failed: {r.stderr.strip()}")
    return np.fromfile(out, dtype="<f8").reshape(n, n)


def save(path, m):
    np.ascontiguousarray(m, dtype="<f8").tofile(path)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", default=sys.executable, help="interpreter for the Python kernel")
    args = ap.parse_args()

    rows = []

    def record(test, lang, n, passed, detail):
        rows.append(dict(test=test, language=lang, n=n, passed=int(passed), detail=detail))
        print(f"{'PASS' if passed else 'FAIL'}  {test:<28} {lang:<7} n={n:<4} {detail}")

    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        results = {}  # (test, n) -> {lang: matrix} for the cross-language check

        def case(test, A, B, check):
            n = A.shape[0]
            a, b = save(tmp / f"{test}_A.bin", A), save(tmp / f"{test}_B.bin", B)
            for lang in LANGS:
                C = run_verify(lang, n, a, b, tmp / f"{test}_{lang}_C.bin", args.python)
                results.setdefault((test, n), {})[lang] = C
                ok, detail = check(C)
                record(test, lang, n, ok, detail)

        def exact(expected):
            return lambda C: (np.array_equal(C, expected), "bit-exact" if np.array_equal(C, expected)
                              else f"max |diff| = {np.abs(C - expected).max():.3g}")

        def close_to_numpy(A, B):
            ref = A @ B
            def check(C):
                diff = np.abs(C - ref)
                ok = bool(np.allclose(C, ref, rtol=RTOL, atol=ATOL))
                return ok, f"max |C - NumPy| = {diff.max():.2e} (allowed {ATOL:g} + {RTOL:g}·|NumPy|)"
            return check

        # identity / zero
        for n in (1, 7, 33):
            X = rng.random((n, n))
            I = np.eye(n)
            case("identity_left", I, X, exact(X))
            case("identity_right", X, I, exact(X))
            case("zero", np.zeros((n, n)), X, exact(np.zeros((n, n))))

        # known answers
        A2 = np.array([[1.0, 2.0], [3.0, 4.0]]); B2 = np.array([[5.0, 6.0], [7.0, 8.0]])
        case("known_2x2", A2, B2, exact(np.array([[19.0, 22.0], [43.0, 50.0]])))
        A3 = np.array([[2.0, 0.0, 1.0], [1.0, 3.0, 2.0], [0.0, 1.0, 4.0]])
        B3 = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [5.0, 0.0, 1.0]])
        case("known_3x3", A3, B3, exact(np.array([[7.0, 4.0, 1.0], [11.0, 5.0, 11.0], [20.0, 1.0, 7.0]])))

        # integer-valued matrices: exact for every summation order
        for n in (50, 100):
            Ai = rng.integers(0, 10, (n, n)); Bi = rng.integers(0, 10, (n, n))
            case("integer_valued", Ai.astype(float), Bi.astype(float), exact((Ai @ Bi).astype(float)))

        # random reals against NumPy
        for name, lo, hi in (("random_01", 0.0, 1.0), ("random_pm1", -1.0, 1.0)):
            for n in (10, 100, 200):
                A = rng.uniform(lo, hi, (n, n)); B = rng.uniform(lo, hi, (n, n))
                case(name, A, B, close_to_numpy(A, B))

        # the shared benchmark inputs themselves
        for n in (100, 300):
            a, b = input_paths(n)
            A = np.fromfile(a, dtype="<f8").reshape(n, n); B = np.fromfile(b, dtype="<f8").reshape(n, n)
            for lang in LANGS:
                C = run_verify(lang, n, a, b, tmp / f"shared_{lang}.bin", args.python)
                results.setdefault(("shared_inputs", n), {})[lang] = C
                ok, detail = close_to_numpy(A, B)(C)
                record("shared_inputs", lang, n, ok, detail)

        # cross-language bit-identity
        for (test, n), per_lang in sorted(results.items()):
            same = all(np.array_equal(per_lang[LANGS[0]], per_lang[l]) for l in LANGS[1:])
            record("cross_language_bit_identical", "all", n, same, f"test case '{test}'")

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "correctness.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["test", "language", "n", "passed", "detail"])
        w.writeheader(); w.writerows(rows)
    failed = [r for r in rows if not r["passed"]]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

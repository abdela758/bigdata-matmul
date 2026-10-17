"""Assignment 1 - basic dense matrix multiplication in pure Python.

Straightforward i-j-k triple loop on n x n matrices of float, stored as a list
of n row lists (each element is a boxed Python float). Standard library only:
no NumPy and no hoisting of row lookups, on purpose, so the kernel is the
literal textbook loop.

    python matmul.py verify <n> <A.bin> <B.bin> <C.bin>
    python matmul.py bench  <n> <A.bin> <B.bin> <warmup_batches> <reps> <target_seconds>

The bench protocol is the same as in the C and Java programs
(see README.md, "Measurement protocol").
"""
import sys
from array import array
from time import perf_counter_ns


def multiply(a, b, c, n):
    """The kernel under test."""
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += a[i][k] * b[k][j]
            c[i][j] = s


def read_matrix(path, n):
    rows = []
    with open(path, "rb") as f:
        for _ in range(n):
            buf = f.read(8 * n)
            if len(buf) != 8 * n:
                raise SystemExit(f"{path}: expected {8 * n * n} bytes")
            row = array("d")
            row.frombytes(buf)
            rows.append(row.tolist())
        if f.read(1):
            raise SystemExit(f"{path}: file longer than n*n doubles")
    return rows


def write_matrix(path, m):
    with open(path, "wb") as f:
        for row in m:
            f.write(array("d", row).tobytes())


def checksum(c, n):
    """Row-major sequential sum: same order in all three languages."""
    s = 0.0
    for row in c:
        for x in row:
            s += x
    return s


def run_batch(a, b, c, n, inner):
    t0 = perf_counter_ns()
    for _ in range(inner):
        multiply(a, b, c, n)
    t1 = perf_counter_ns()
    return (t1 - t0) * 1e-9


def emit(phase, index, n, inner, batch_seconds, chk):
    print(f"sample,python,{n},{phase},{index},{inner},{batch_seconds / inner:.9e},{chk:.17g}", flush=True)


def bench(n, pa, pb, warmup, reps, target):
    a = read_matrix(pa, n)
    b = read_matrix(pb, n)
    c = [[0.0] * n for _ in range(n)]
    print(f"ready,python,{n}", flush=True)

    # calibration ("autorange"): double the batch size until one batch lasts
    # at least `target` seconds; these runs also act as the first warm-up.
    inner = 1
    idx = 0
    while True:
        t = run_batch(a, b, c, n, inner)
        emit("calibration", idx, n, inner, t, checksum(c, n))
        idx += 1
        if t >= target:
            break
        inner *= 2
    for w in range(warmup):
        t = run_batch(a, b, c, n, inner)
        emit("warmup", w, n, inner, t, checksum(c, n))
    for r in range(reps):
        t = run_batch(a, b, c, n, inner)
        emit("measure", r, n, inner, t, checksum(c, n))
    print(f"done,python,{n}", flush=True)


def main(argv):
    if len(argv) == 6 and argv[1] == "verify":
        n = int(argv[2])
        a = read_matrix(argv[3], n)
        b = read_matrix(argv[4], n)
        c = [[0.0] * n for _ in range(n)]
        multiply(a, b, c, n)
        write_matrix(argv[5], c)
        return 0
    if len(argv) == 8 and argv[1] == "bench":
        bench(int(argv[2]), argv[3], argv[4], int(argv[5]), int(argv[6]), float(argv[7]))
        return 0
    sys.stderr.write(
        "usage:\n  matmul.py verify <n> <A.bin> <B.bin> <C.bin>\n"
        "  matmul.py bench  <n> <A.bin> <B.bin> <warmup_batches> <reps> <target_seconds>\n"
    )
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv))

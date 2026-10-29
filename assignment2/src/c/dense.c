/*
 * Assignment 2, Part A - dense matrix multiplication: baseline and two optimisations.
 *
 * Matrices are n x n double, one contiguous row-major block each (element (i,j) is a[i*n + j]),
 * exactly as in Assignment 1 (assignment1/src/c/matmul.c).
 *
 *   ijk       baseline: the validated Assignment 1 kernel, unchanged.
 *   ikj       loop reordering: the inner loop walks a row of B and a row of C (unit stride)
 *             instead of a column of B (stride n).
 *   blockedS  cache blocking with S x S tiles (default S = 64, e.g. "blocked" or "blocked32"),
 *             i-k-j order inside each tile, so a tile of B is reused while it is in cache.
 *
 * For every element C[i][j] all three add the products a[i][k]*b[k][j] in the same order
 * (k = 0, 1, ..., n-1, starting from 0.0), so with -O2 (no fused multiply-add, no reassociation)
 * they return bit-identical results.
 *
 *   dense verify <method> <n> <A.bin> <B.bin> <C.bin>
 *   dense bench  <method> <n> <A.bin> <B.bin> <warmup_batches> <reps> <target_seconds>
 * The bench protocol and output lines are the same as in Assignment 1.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
static double now_seconds(void) {
    static LARGE_INTEGER freq;
    LARGE_INTEGER t;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart;
}
#else
#include <time.h>
static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
#endif

#if defined(__GNUC__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

/* Baseline: Assignment 1 kernel, unchanged. Writes every element of C. */
NOINLINE static void mm_ijk(const double *a, const double *b, double *c, int n, int bs) {
    (void)bs;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += a[(size_t)i * n + k] * b[(size_t)k * n + j];
            }
            c[(size_t)i * n + j] = sum;
        }
    }
}

/* Loop reordering. C is accumulated in place, so it is zeroed first (inside the timed kernel). */
NOINLINE static void mm_ikj(const double *a, const double *b, double *c, int n, int bs) {
    (void)bs;
    memset(c, 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        double *ci = c + (size_t)i * n;
        for (int k = 0; k < n; k++) {
            double aik = a[(size_t)i * n + k];
            const double *bk = b + (size_t)k * n;
            for (int j = 0; j < n; j++) {
                ci[j] += aik * bk[j];
            }
        }
    }
}

/* Cache blocking: tiles of bs x bs; the k-tiles are visited in increasing order, so each
 * C[i][j] still receives its products in increasing k. C is zeroed inside the timed kernel. */
NOINLINE static void mm_blocked(const double *a, const double *b, double *c, int n, int bs) {
    memset(c, 0, (size_t)n * n * sizeof(double));
    for (int ii = 0; ii < n; ii += bs) {
        int iend = ii + bs < n ? ii + bs : n;
        for (int kk = 0; kk < n; kk += bs) {
            int kend = kk + bs < n ? kk + bs : n;
            for (int jj = 0; jj < n; jj += bs) {
                int jend = jj + bs < n ? jj + bs : n;
                for (int i = ii; i < iend; i++) {
                    double *ci = c + (size_t)i * n;
                    for (int k = kk; k < kend; k++) {
                        double aik = a[(size_t)i * n + k];
                        const double *bk = b + (size_t)k * n;
                        for (int j = jj; j < jend; j++) {
                            ci[j] += aik * bk[j];
                        }
                    }
                }
            }
        }
    }
}

typedef void (*kernel_fn)(const double *, const double *, double *, int, int);

static kernel_fn parse_method(const char *m, int *bs) {
    *bs = 64;
    if (strcmp(m, "ijk") == 0) return mm_ijk;
    if (strcmp(m, "ikj") == 0) return mm_ikj;
    if (strncmp(m, "blocked", 7) == 0) {
        if (m[7] != '\0') *bs = atoi(m + 7);
        if (*bs <= 0) { fprintf(stderr, "bad block size in %s\n", m); exit(64); }
        return mm_blocked;
    }
    fprintf(stderr, "unknown method %s (ijk | ikj | blocked[S])\n", m);
    exit(64);
}

static double *read_matrix(const char *path, int n) {
    size_t count = (size_t)n * (size_t)n;
    double *m = malloc(count * sizeof(double));
    if (!m) { fprintf(stderr, "out of memory for %s\n", path); exit(2); }
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    if (fread(m, sizeof(double), count, f) != count) { fprintf(stderr, "%s: expected %zu doubles\n", path, count); exit(2); }
    if (fgetc(f) != EOF) { fprintf(stderr, "%s: file longer than n*n doubles\n", path); exit(2); }
    fclose(f);
    return m;
}

static double checksum(const double *c, int n) {
    double s = 0.0;
    size_t count = (size_t)n * (size_t)n;
    for (size_t i = 0; i < count; i++) s += c[i];
    return s;
}

static double run_batch(kernel_fn f, const double *a, const double *b, double *c, int n, int bs, long inner) {
    double t0 = now_seconds();
    for (long r = 0; r < inner; r++) f(a, b, c, n, bs);
    return now_seconds() - t0;
}

static void emit(const char *phase, int index, int n, long inner, double batch_seconds, double chk) {
    printf("sample,c,%d,%s,%d,%ld,%.9e,%.17g\n", n, phase, index, inner, batch_seconds / (double)inner, chk);
    fflush(stdout);
}

int main(int argc, char **argv) {
    if (argc == 7 && strcmp(argv[1], "verify") == 0) {
        int bs, n = atoi(argv[3]);
        kernel_fn f = parse_method(argv[2], &bs);
        double *a = read_matrix(argv[4], n), *b = read_matrix(argv[5], n);
        double *c = malloc((size_t)n * n * sizeof(double));
        f(a, b, c, n, bs);
        FILE *out = fopen(argv[6], "wb");
        if (!out) { fprintf(stderr, "cannot write %s\n", argv[6]); return 2; }
        fwrite(c, sizeof(double), (size_t)n * n, out);
        fclose(out);
        return 0;
    }
    if (argc == 9 && strcmp(argv[1], "bench") == 0) {
        int bs, n = atoi(argv[3]), warmup = atoi(argv[6]), reps = atoi(argv[7]);
        double target = atof(argv[8]);
        kernel_fn f = parse_method(argv[2], &bs);
        double *a = read_matrix(argv[4], n), *b = read_matrix(argv[5], n);
        double *c = malloc((size_t)n * n * sizeof(double));
        if (!c) { fprintf(stderr, "out of memory\n"); return 2; }
        memset(c, 0, (size_t)n * n * sizeof(double));
        printf("ready,c,%d\n", n);
        fflush(stdout);
        long inner = 1;
        int idx = 0;
        for (;;) {  /* calibration: double the batch until it lasts >= target seconds */
            double t = run_batch(f, a, b, c, n, bs, inner);
            emit("calibration", idx++, n, inner, t, checksum(c, n));
            if (t >= target) break;
            inner *= 2;
        }
        for (int w = 0; w < warmup; w++) emit("warmup", w, n, inner, run_batch(f, a, b, c, n, bs, inner), checksum(c, n));
        for (int r = 0; r < reps; r++) emit("measure", r, n, inner, run_batch(f, a, b, c, n, bs, inner), checksum(c, n));
        printf("done,c,%d\n", n);
        fflush(stdout);
        return 0;
    }
    fprintf(stderr, "usage:\n  %s verify <ijk|ikj|blocked[S]> <n> <A.bin> <B.bin> <C.bin>\n"
                    "  %s bench  <ijk|ikj|blocked[S]> <n> <A.bin> <B.bin> <warmup_batches> <reps> <target_seconds>\n",
            argv[0], argv[0]);
    return 64;
}

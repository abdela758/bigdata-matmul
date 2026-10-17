/*
 * Assignment 1 - basic dense matrix multiplication in C.
 *
 * Straightforward i-j-k triple loop on n x n matrices of double, stored as one
 * contiguous row-major block per matrix (element (i,j) is a[i*n + j]).
 *
 *   matmul verify <n> <A.bin> <B.bin> <C.bin>
 *       one multiplication, result written as raw little-endian float64
 *   matmul bench  <n> <A.bin> <B.bin> <warmup_batches> <reps> <target_seconds>
 *       timing protocol shared with the Java and Python programs
 *       (see README.md, "Measurement protocol")
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

/* The kernel under test. noinline keeps the compiler from fusing repeated
 * calls inside a timed batch into one computation. */
#if defined(__GNUC__)
__attribute__((noinline))
#endif
static void multiply(const double *a, const double *b, double *c, int n) {
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

static double *read_matrix(const char *path, int n) {
    size_t count = (size_t)n * (size_t)n;
    double *m = malloc(count * sizeof(double));
    if (!m) { fprintf(stderr, "out of memory for %s\n", path); exit(2); }
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    if (fread(m, sizeof(double), count, f) != count) {
        fprintf(stderr, "%s: expected %zu doubles\n", path, count);
        exit(2);
    }
    int extra = fgetc(f);
    if (extra != EOF) { fprintf(stderr, "%s: file longer than n*n doubles\n", path); exit(2); }
    fclose(f);
    return m;
}

/* Row-major sequential sum: same order in all three languages. */
static double checksum(const double *c, int n) {
    double s = 0.0;
    size_t count = (size_t)n * (size_t)n;
    for (size_t i = 0; i < count; i++) s += c[i];
    return s;
}

static double run_batch(const double *a, const double *b, double *c, int n, long inner) {
    double t0 = now_seconds();
    for (long r = 0; r < inner; r++) multiply(a, b, c, n);
    double t1 = now_seconds();
    return t1 - t0;
}

static void emit(const char *phase, int index, int n, long inner, double batch_seconds, double chk) {
    printf("sample,c,%d,%s,%d,%ld,%.9e,%.17g\n", n, phase, index, inner,
           batch_seconds / (double)inner, chk);
    fflush(stdout);
}

static int cmd_verify(int n, const char *pa, const char *pb, const char *pc) {
    double *a = read_matrix(pa, n), *b = read_matrix(pb, n);
    double *c = malloc((size_t)n * n * sizeof(double));
    multiply(a, b, c, n);
    FILE *f = fopen(pc, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", pc); return 2; }
    fwrite(c, sizeof(double), (size_t)n * n, f);
    fclose(f);
    return 0;
}

static int cmd_bench(int n, const char *pa, const char *pb, int warmup, int reps, double target) {
    double *a = read_matrix(pa, n), *b = read_matrix(pb, n);
    double *c = malloc((size_t)n * n * sizeof(double));
    if (!c) { fprintf(stderr, "out of memory\n"); return 2; }
    memset(c, 0, (size_t)n * n * sizeof(double));
    printf("ready,c,%d\n", n);
    fflush(stdout);

    /* calibration ("autorange"): double the batch size until one batch lasts
     * at least target seconds; these runs also act as the first warm-up. */
    long inner = 1;
    int idx = 0;
    for (;;) {
        double t = run_batch(a, b, c, n, inner);
        emit("calibration", idx++, n, inner, t, checksum(c, n));
        if (t >= target) break;
        inner *= 2;
    }
    for (int w = 0; w < warmup; w++) {
        double t = run_batch(a, b, c, n, inner);
        emit("warmup", w, n, inner, t, checksum(c, n));
    }
    for (int r = 0; r < reps; r++) {
        double t = run_batch(a, b, c, n, inner);
        emit("measure", r, n, inner, t, checksum(c, n));
    }
    printf("done,c,%d\n", n);
    fflush(stdout);
    free(a); free(b); free(c);
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 6 && strcmp(argv[1], "verify") == 0)
        return cmd_verify(atoi(argv[2]), argv[3], argv[4], argv[5]);
    if (argc == 8 && strcmp(argv[1], "bench") == 0)
        return cmd_bench(atoi(argv[2]), argv[3], argv[4], atoi(argv[5]), atoi(argv[6]), atof(argv[7]));
    fprintf(stderr,
            "usage:\n  %s verify <n> <A.bin> <B.bin> <C.bin>\n"
            "  %s bench  <n> <A.bin> <B.bin> <warmup_batches> <reps> <target_seconds>\n",
            argv[0], argv[0]);
    return 64;
}

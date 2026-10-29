/*
 * Assignment 2, Part B - sparse matrix multiplication in CSR format (own code, no library).
 *
 * CSR (compressed sparse row): for an n x n matrix with nnz nonzeros,
 *   rowptr[n+1]  (int64)  row i occupies positions rowptr[i] .. rowptr[i+1]-1
 *   col[nnz]     (int32)  column of each stored value
 *   val[nnz]     (double) the value
 * 8 + 12 bytes per nonzero plus 8(n+1) bytes, against 8n^2 bytes for the dense matrix.
 * File format (.csr): int64 n, int64 nnz, rowptr, col, val, all little-endian.
 *
 * Product C = A x B with Gustavson's row-by-row algorithm, in two passes:
 *   1. symbolic: count the distinct columns of each row of C (marker array) -> rowptr of C
 *   2. allocate col/val of C exactly, then numeric: for each row i, for each nonzero A[i][k]
 *      (increasing k), add A[i][k] * B[k][j] into a dense accumulator acc[j].
 * Each C[i][j] receives its nonzero products in increasing k, starting from 0.0, i.e. the same
 * additions as the dense i-j-k kernel minus the exact zeros, so results are bit-identical to it.
 * Column indices inside a row of C are stored in first-touch order (not sorted).
 * The timed kernel includes both passes, the allocation of C and freeing the previous C.
 *
 *   sparse verify  <n> <A.csr> <B.csr> <C.csr>                     multiply and write C
 *   sparse bench   <n> <A.csr> <B.csr> <warmup> <reps> <target_s>  time the multiplication
 *   sparse to-csr  <n> <A.bin> <A.csr>                              convert a dense file (for tests)
 *   sparse convert <n> <A.bin> <warmup> <reps> <target_s>           time dense -> CSR conversion
 */
#include <stdint.h>
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

typedef struct {
    int n;
    int64_t nnz;
    int64_t *rowptr;
    int32_t *col;
    double *val;
} csr;

static void *xmalloc(size_t bytes) {
    void *p = malloc(bytes ? bytes : 1);
    if (!p) { fprintf(stderr, "out of memory (%zu bytes)\n", bytes); exit(2); }
    return p;
}

static void csr_free(csr *m) {
    free(m->rowptr); free(m->col); free(m->val);
    m->rowptr = NULL; m->col = NULL; m->val = NULL; m->nnz = 0;
}

static csr csr_read(const char *path, int n) {
    csr m = {0};
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    int64_t hdr[2];
    if (fread(hdr, sizeof(int64_t), 2, f) != 2 || hdr[0] != n) { fprintf(stderr, "%s: bad header\n", path); exit(2); }
    m.n = n; m.nnz = hdr[1];
    m.rowptr = xmalloc((size_t)(n + 1) * sizeof(int64_t));
    m.col = xmalloc((size_t)m.nnz * sizeof(int32_t));
    m.val = xmalloc((size_t)m.nnz * sizeof(double));
    if (fread(m.rowptr, sizeof(int64_t), (size_t)n + 1, f) != (size_t)n + 1 ||
        fread(m.col, sizeof(int32_t), (size_t)m.nnz, f) != (size_t)m.nnz ||
        fread(m.val, sizeof(double), (size_t)m.nnz, f) != (size_t)m.nnz) { fprintf(stderr, "%s: short file\n", path); exit(2); }
    if (fgetc(f) != EOF) { fprintf(stderr, "%s: trailing data\n", path); exit(2); }
    fclose(f);
    return m;
}

static void csr_write(const char *path, const csr *m) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); exit(2); }
    int64_t hdr[2] = {m->n, m->nnz};
    fwrite(hdr, sizeof(int64_t), 2, f);
    fwrite(m->rowptr, sizeof(int64_t), (size_t)m->n + 1, f);
    fwrite(m->col, sizeof(int32_t), (size_t)m->nnz, f);
    fwrite(m->val, sizeof(double), (size_t)m->nnz, f);
    fclose(f);
}

static double *dense_read(const char *path, int n) {
    size_t count = (size_t)n * (size_t)n;
    double *m = xmalloc(count * sizeof(double));
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    if (fread(m, sizeof(double), count, f) != count) { fprintf(stderr, "%s: expected %zu doubles\n", path, count); exit(2); }
    fclose(f);
    return m;
}

/* Dense row-major -> CSR (two passes: count, fill). The previous content of *out is freed. */
NOINLINE static void dense_to_csr(const double *d, int n, csr *out) {
    csr_free(out);
    out->n = n;
    out->rowptr = xmalloc((size_t)(n + 1) * sizeof(int64_t));
    int64_t nnz = 0;
    out->rowptr[0] = 0;
    for (int i = 0; i < n; i++) {
        const double *row = d + (size_t)i * n;
        for (int j = 0; j < n; j++) nnz += (row[j] != 0.0);
        out->rowptr[i + 1] = nnz;
    }
    out->nnz = nnz;
    out->col = xmalloc((size_t)nnz * sizeof(int32_t));
    out->val = xmalloc((size_t)nnz * sizeof(double));
    int64_t p = 0;
    for (int i = 0; i < n; i++) {
        const double *row = d + (size_t)i * n;
        for (int j = 0; j < n; j++) {
            if (row[j] != 0.0) { out->col[p] = j; out->val[p] = row[j]; p++; }
        }
    }
}

/* Work arrays of length n, allocated once per process (outside the timed region). */
typedef struct {
    double *acc;
    int32_t *mark;   /* mark[j] == i + 1 when column j was already touched in row i */
    int32_t *list;   /* columns touched in the current row, in first-touch order */
} workspace;

NOINLINE static void spgemm(const csr *a, const csr *b, csr *c, workspace *w) {
    int n = a->n;
    csr_free(c);
    c->n = n;
    c->rowptr = xmalloc((size_t)(n + 1) * sizeof(int64_t));
    for (int j = 0; j < n; j++) w->mark[j] = 0;

    /* pass 1: symbolic */
    int64_t nnz = 0;
    c->rowptr[0] = 0;
    for (int i = 0; i < n; i++) {
        int32_t tag = i + 1;
        for (int64_t p = a->rowptr[i]; p < a->rowptr[i + 1]; p++) {
            int k = a->col[p];
            for (int64_t q = b->rowptr[k]; q < b->rowptr[k + 1]; q++) {
                int j = b->col[q];
                if (w->mark[j] != tag) { w->mark[j] = tag; nnz++; }
            }
        }
        c->rowptr[i + 1] = nnz;
    }
    c->nnz = nnz;
    c->col = xmalloc((size_t)nnz * sizeof(int32_t));
    c->val = xmalloc((size_t)nnz * sizeof(double));

    /* pass 2: numeric (tags continue from n + 1 so pass-1 marks never match) */
    for (int i = 0; i < n; i++) {
        int32_t tag = n + i + 1;
        int len = 0;
        for (int64_t p = a->rowptr[i]; p < a->rowptr[i + 1]; p++) {
            int k = a->col[p];
            double aik = a->val[p];
            for (int64_t q = b->rowptr[k]; q < b->rowptr[k + 1]; q++) {
                int j = b->col[q];
                if (w->mark[j] != tag) { w->mark[j] = tag; w->acc[j] = 0.0; w->list[len++] = j; }
                w->acc[j] += aik * b->val[q];
            }
        }
        int64_t base = c->rowptr[i];
        for (int t = 0; t < len; t++) {
            int j = w->list[t];
            c->col[base + t] = j;
            c->val[base + t] = w->acc[j];
        }
    }
}

static double csr_checksum(const csr *m) {
    double s = 0.0;
    for (int64_t p = 0; p < m->nnz; p++) s += m->val[p];
    return s;
}

static void emit(const char *phase, int index, int n, long inner, double batch_seconds, double chk) {
    printf("sample,c,%d,%s,%d,%ld,%.9e,%.17g\n", n, phase, index, inner, batch_seconds / (double)inner, chk);
    fflush(stdout);
}

/* Shared calibration / warm-up / measurement loop; op(ctx) performs one operation. */
typedef double (*batch_fn)(void *ctx, long inner);

static void bench_loop(int n, int warmup, int reps, double target, batch_fn run, void *ctx, double (*chk)(void *)) {
    long inner = 1;
    int idx = 0;
    for (;;) {
        double t = run(ctx, inner);
        emit("calibration", idx++, n, inner, t, chk(ctx));
        if (t >= target) break;
        inner *= 2;
    }
    for (int w = 0; w < warmup; w++) emit("warmup", w, n, inner, run(ctx, inner), chk(ctx));
    for (int r = 0; r < reps; r++) emit("measure", r, n, inner, run(ctx, inner), chk(ctx));
}

typedef struct { csr a, b, c; workspace w; } mult_ctx;

static double mult_batch(void *p, long inner) {
    mult_ctx *x = p;
    double t0 = now_seconds();
    for (long r = 0; r < inner; r++) spgemm(&x->a, &x->b, &x->c, &x->w);
    return now_seconds() - t0;
}
static double mult_chk(void *p) { return csr_checksum(&((mult_ctx *)p)->c); }

typedef struct { double *d; int n; csr out; } conv_ctx;

static double conv_batch(void *p, long inner) {
    conv_ctx *x = p;
    double t0 = now_seconds();
    for (long r = 0; r < inner; r++) dense_to_csr(x->d, x->n, &x->out);
    return now_seconds() - t0;
}
static double conv_chk(void *p) { return csr_checksum(&((conv_ctx *)p)->out); }

static workspace ws_alloc(int n) {
    workspace w;
    w.acc = xmalloc((size_t)n * sizeof(double));
    w.mark = xmalloc((size_t)n * sizeof(int32_t));
    w.list = xmalloc((size_t)n * sizeof(int32_t));
    return w;
}

int main(int argc, char **argv) {
    if (argc == 6 && strcmp(argv[1], "verify") == 0) {
        int n = atoi(argv[2]);
        mult_ctx x = {0};
        x.a = csr_read(argv[3], n); x.b = csr_read(argv[4], n); x.w = ws_alloc(n);
        spgemm(&x.a, &x.b, &x.c, &x.w);
        csr_write(argv[5], &x.c);
        return 0;
    }
    if (argc == 8 && strcmp(argv[1], "bench") == 0) {
        int n = atoi(argv[2]);
        mult_ctx x = {0};
        x.a = csr_read(argv[3], n); x.b = csr_read(argv[4], n); x.w = ws_alloc(n);
        spgemm(&x.a, &x.b, &x.c, &x.w);  /* untimed: only to report nnz(C) before timing starts */
        printf("info,nnzA=%lld,nnzB=%lld,nnzC=%lld\n", (long long)x.a.nnz, (long long)x.b.nnz, (long long)x.c.nnz);
        printf("ready,c,%d\n", n);
        fflush(stdout);
        bench_loop(n, atoi(argv[5]), atoi(argv[6]), atof(argv[7]), mult_batch, &x, mult_chk);
        printf("done,c,%d\n", n);
        return 0;
    }
    if (argc == 5 && strcmp(argv[1], "to-csr") == 0) {
        int n = atoi(argv[2]);
        csr out = {0};
        double *d = dense_read(argv[3], n);
        dense_to_csr(d, n, &out);
        csr_write(argv[4], &out);
        return 0;
    }
    if (argc == 7 && strcmp(argv[1], "convert") == 0) {
        conv_ctx x = {0};
        x.n = atoi(argv[2]);
        x.d = dense_read(argv[3], x.n);
        printf("ready,c,%d\n", x.n);
        fflush(stdout);
        bench_loop(x.n, atoi(argv[4]), atoi(argv[5]), atof(argv[6]), conv_batch, &x, conv_chk);
        printf("info,nnz=%lld\n", (long long)x.out.nnz);
        printf("done,c,%d\n", x.n);
        return 0;
    }
    fprintf(stderr, "usage:\n  %s verify <n> <A.csr> <B.csr> <C.csr>\n  %s bench <n> <A.csr> <B.csr> <warmup> <reps> <target_s>\n"
                    "  %s to-csr <n> <A.bin> <A.csr>\n  %s convert <n> <A.bin> <warmup> <reps> <target_s>\n",
            argv[0], argv[0], argv[0], argv[0]);
    return 64;
}

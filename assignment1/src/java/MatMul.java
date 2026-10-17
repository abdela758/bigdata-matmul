import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Locale;

/**
 * Assignment 1 - basic dense matrix multiplication in Java.
 *
 * Straightforward i-j-k triple loop on n x n matrices of double, stored as
 * double[n][n] (an array of n row arrays, each row contiguous).
 *
 *   java MatMul verify <n> <A.bin> <B.bin> <C.bin>
 *   java MatMul bench  <n> <A.bin> <B.bin> <warmup_batches> <reps> <target_seconds>
 *
 * The bench protocol is the same as in the C and Python programs
 * (see README.md, "Measurement protocol").
 */
public final class MatMul {

    /** The kernel under test. */
    static void multiply(double[][] a, double[][] b, double[][] c, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += a[i][k] * b[k][j];
                }
                c[i][j] = sum;
            }
        }
    }

    static double[][] readMatrix(String file, int n) throws IOException {
        double[][] m = new double[n][n];
        try (FileChannel ch = FileChannel.open(Path.of(file), StandardOpenOption.READ)) {
            if (ch.size() != 8L * n * n) {
                throw new IOException(file + ": expected " + (8L * n * n) + " bytes, found " + ch.size());
            }
            ByteBuffer buf = ByteBuffer.allocate(8 * n).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < n; i++) {
                buf.clear();
                while (buf.hasRemaining()) {
                    if (ch.read(buf) < 0) throw new EOFException(file);
                }
                buf.flip();
                buf.asDoubleBuffer().get(m[i]);
            }
        }
        return m;
    }

    static void writeMatrix(String file, double[][] m, int n) throws IOException {
        try (FileChannel ch = FileChannel.open(Path.of(file), StandardOpenOption.CREATE,
                StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
            ByteBuffer buf = ByteBuffer.allocate(8 * n).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < n; i++) {
                buf.clear();
                buf.asDoubleBuffer().put(m[i]);
                buf.limit(8 * n);
                while (buf.hasRemaining()) ch.write(buf);
            }
        }
    }

    /** Row-major sequential sum: same order in all three languages. */
    static double checksum(double[][] c, int n) {
        double s = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) s += c[i][j];
        }
        return s;
    }

    static double runBatch(double[][] a, double[][] b, double[][] c, int n, long inner) {
        long t0 = System.nanoTime();
        for (long r = 0; r < inner; r++) multiply(a, b, c, n);
        long t1 = System.nanoTime();
        return (t1 - t0) * 1e-9;
    }

    static void emit(String phase, int index, int n, long inner, double batchSeconds, double chk) {
        System.out.println(String.format(Locale.ROOT, "sample,java,%d,%s,%d,%d,%.9e,%.17g",
                n, phase, index, inner, batchSeconds / inner, chk));
        System.out.flush();
    }

    static void bench(int n, String pa, String pb, int warmup, int reps, double target) throws IOException {
        double[][] a = readMatrix(pa, n);
        double[][] b = readMatrix(pb, n);
        double[][] c = new double[n][n];
        System.out.println("ready,java," + n);
        System.out.flush();

        // calibration ("autorange"): double the batch size until one batch lasts
        // at least `target` seconds; these runs also act as the first warm-up.
        long inner = 1;
        int idx = 0;
        while (true) {
            double t = runBatch(a, b, c, n, inner);
            emit("calibration", idx++, n, inner, t, checksum(c, n));
            if (t >= target) break;
            inner *= 2;
        }
        for (int w = 0; w < warmup; w++) {
            double t = runBatch(a, b, c, n, inner);
            emit("warmup", w, n, inner, t, checksum(c, n));
        }
        for (int r = 0; r < reps; r++) {
            double t = runBatch(a, b, c, n, inner);
            emit("measure", r, n, inner, t, checksum(c, n));
        }
        System.out.println("done,java," + n);
        System.out.flush();
    }

    public static void main(String[] args) throws IOException {
        if (args.length == 5 && args[0].equals("verify")) {
            int n = Integer.parseInt(args[1]);
            double[][] a = readMatrix(args[2], n);
            double[][] b = readMatrix(args[3], n);
            double[][] c = new double[n][n];
            multiply(a, b, c, n);
            writeMatrix(args[4], c, n);
        } else if (args.length == 7 && args[0].equals("bench")) {
            bench(Integer.parseInt(args[1]), args[2], args[3], Integer.parseInt(args[4]),
                    Integer.parseInt(args[5]), Double.parseDouble(args[6]));
        } else {
            System.err.println("usage:\n  MatMul verify <n> <A.bin> <B.bin> <C.bin>\n"
                    + "  MatMul bench  <n> <A.bin> <B.bin> <warmup_batches> <reps> <target_seconds>");
            System.exit(64);
        }
    }
}

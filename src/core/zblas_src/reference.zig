// zblas/src/reference.zig
// Naive reference implementations for correctness verification
//
// These implementations are simple and correct, but not optimized.
// Used for testing the optimized kernels.

const std = @import("std");

/// Reference SGEMM: C = alpha * A * B + beta * C
/// A is M x K, B is K x N, C is M x N (all row-major)
///
/// This is the naive O(M*N*K) triple-loop implementation.
/// Correct but slow - used only for testing.
pub fn sgemm_reference(
    m: usize,
    n: usize,
    k: usize,
    A: []const f32,
    lda: usize,
    B: []const f32,
    ldb: usize,
    C: []f32,
    ldc: usize,
    alpha: f32,
    beta: f32,
) void {
    // Scale C by beta first
    for (0..m) |i| {
        for (0..n) |j| {
            C[i * ldc + j] *= beta;
        }
    }

    // C += alpha * A * B
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0.0;
            for (0..k) |kk| {
                sum += A[i * lda + kk] * B[kk * ldb + j];
            }
            C[i * ldc + j] += alpha * sum;
        }
    }
}

/// Reference SGEMM with contiguous arrays (lda=K, ldb=N, ldc=N)
pub fn sgemm_reference_simple(
    m: usize,
    n: usize,
    k: usize,
    A: []const f32,
    B: []const f32,
    C: []f32,
    alpha: f32,
    beta: f32,
) void {
    sgemm_reference(m, n, k, A, k, B, n, C, n, alpha, beta);
}

/// Reference SGEMV: y = alpha * A * x + beta * y
/// A is M x N (row-major), x is N, y is M
pub fn sgemv_reference(
    m: usize,
    n: usize,
    A: []const f32,
    lda: usize,
    x: []const f32,
    y: []f32,
    alpha: f32,
    beta: f32,
) void {
    for (0..m) |i| {
        var sum: f32 = 0.0;
        for (0..n) |j| {
            sum += A[i * lda + j] * x[j];
        }
        y[i] = alpha * sum + beta * y[i];
    }
}

/// Reference SGEMV with contiguous array (lda=N)
pub fn sgemv_reference_simple(
    m: usize,
    n: usize,
    A: []const f32,
    x: []const f32,
    y: []f32,
    alpha: f32,
    beta: f32,
) void {
    sgemv_reference(m, n, A, n, x, y, alpha, beta);
}

/// Reference SAXPY: y = alpha * x + y
pub fn saxpy_reference(n: usize, alpha: f32, x: []const f32, y: []f32) void {
    for (0..n) |i| {
        y[i] += alpha * x[i];
    }
}

/// Reference SDOT: result = x · y
pub fn sdot_reference(n: usize, x: []const f32, y: []const f32) f32 {
    var sum: f32 = 0.0;
    for (0..n) |i| {
        sum += x[i] * y[i];
    }
    return sum;
}

/// Reference SSCAL: x = alpha * x
pub fn sscal_reference(n: usize, alpha: f32, x: []f32) void {
    for (0..n) |i| {
        x[i] *= alpha;
    }
}

// ============================================================================
// Tests
// ============================================================================

test "reference sgemm 2x2" {
    // A = [1, 2; 3, 4], B = [5, 6; 7, 8]
    // C = A * B = [19, 22; 43, 50]
    const A = [_]f32{ 1, 2, 3, 4 };
    const B = [_]f32{ 5, 6, 7, 8 };
    var C = [_]f32{ 0, 0, 0, 0 };

    sgemm_reference_simple(2, 2, 2, &A, &B, &C, 1.0, 0.0);

    try std.testing.expectApproxEqRel(C[0], 19.0, 1e-5);
    try std.testing.expectApproxEqRel(C[1], 22.0, 1e-5);
    try std.testing.expectApproxEqRel(C[2], 43.0, 1e-5);
    try std.testing.expectApproxEqRel(C[3], 50.0, 1e-5);
}

test "reference sgemm with alpha and beta" {
    const A = [_]f32{ 1, 2, 3, 4 };
    const B = [_]f32{ 5, 6, 7, 8 };
    var C = [_]f32{ 1, 1, 1, 1 };

    // C = 2.0 * A * B + 0.5 * C
    sgemm_reference_simple(2, 2, 2, &A, &B, &C, 2.0, 0.5);

    // Expected: 2*[19,22;43,50] + 0.5*[1,1;1,1] = [38.5, 44.5; 86.5, 100.5]
    try std.testing.expectApproxEqRel(C[0], 38.5, 1e-5);
    try std.testing.expectApproxEqRel(C[1], 44.5, 1e-5);
    try std.testing.expectApproxEqRel(C[2], 86.5, 1e-5);
    try std.testing.expectApproxEqRel(C[3], 100.5, 1e-5);
}

test "reference sgemv" {
    // A = [1, 2, 3; 4, 5, 6], x = [1, 2, 3]
    // y = A * x = [14, 32]
    const A = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const x = [_]f32{ 1, 2, 3 };
    var y = [_]f32{ 0, 0 };

    sgemv_reference_simple(2, 3, &A, &x, &y, 1.0, 0.0);

    try std.testing.expectApproxEqRel(y[0], 14.0, 1e-5);
    try std.testing.expectApproxEqRel(y[1], 32.0, 1e-5);
}

test "reference sdot" {
    const x = [_]f32{ 1, 2, 3 };
    const y = [_]f32{ 4, 5, 6 };
    const result = sdot_reference(3, &x, &y);
    try std.testing.expectApproxEqRel(result, 32.0, 1e-5); // 1*4 + 2*5 + 3*6
}

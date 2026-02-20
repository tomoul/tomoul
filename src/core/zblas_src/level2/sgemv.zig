// zblas/src/level2/sgemv.zig
// Single-precision General Matrix-Vector Multiply (SGEMV)
//
// y = alpha * A * x + beta * y
// A is [M x N], x is [N], y is [M]

const std = @import("std");
const config = @import("../config.zig");
const reference = @import("../reference.zig");

/// SGEMV: y = alpha * A * x + beta * y
/// A is [M x N] row-major, x is [N], y is [M]
pub fn sgemv(
    M: usize,
    N: usize,
    A: []const f32,
    x: []const f32,
    y: []f32,
    alpha: f32,
    beta: f32,
) void {
    // Validate input sizes
    if (M == 0 or N == 0) return;
    std.debug.assert(A.len >= M * N);
    std.debug.assert(x.len >= N);
    std.debug.assert(y.len >= M);

    // For now, use reference implementation
    // TODO Phase 5: Add SIMD-optimized version
    if (N < config.MIN_OPTIMIZED_SIZE) {
        reference.sgemv_reference_simple(M, N, A, x, y, alpha, beta);
        return;
    }

    // Use SIMD-optimized path
    sgemvOptimized(M, N, A, N, x, y, alpha, beta);
}

/// SGEMV with leading dimension parameter - optimized with multi-row processing
fn sgemvOptimized(
    M: usize,
    N: usize,
    A: []const f32,
    lda: usize,
    x: []const f32,
    y: []f32,
    alpha: f32,
    beta: f32,
) void {
    const VEC_WIDTH = comptime config.getVectorWidth();
    const Vec = @Vector(VEC_WIDTH, f32);

    // Scale y by beta with SIMD
    if (beta == 0.0) {
        @memset(y[0..M], 0.0);
    } else if (beta != 1.0) {
        const beta_vec: Vec = @splat(beta);
        var i: usize = 0;
        while (i + VEC_WIDTH <= M) : (i += VEC_WIDTH) {
            const y_vec: Vec = y[i..][0..VEC_WIDTH].*;
            y[i..][0..VEC_WIDTH].* = y_vec * beta_vec;
        }
        while (i < M) : (i += 1) {
            y[i] *= beta;
        }
    }

    // If alpha is zero, we're done
    if (alpha == 0.0) return;

    // Process multiple rows at once for better cache utilization
    // x vector is reused across rows, so keeping it in cache helps
    const ROWS_PER_ITER = 4;
    var i: usize = 0;

    while (i + ROWS_PER_ITER <= M) : (i += ROWS_PER_ITER) {
        var sum0: f32 = 0.0;
        var sum1: f32 = 0.0;
        var sum2: f32 = 0.0;
        var sum3: f32 = 0.0;

        var j: usize = 0;

        // Vectorized inner loop - process VEC_WIDTH elements at a time
        if (VEC_WIDTH > 1 and N >= VEC_WIDTH) {
            var sum0_vec: Vec = @splat(0.0);
            var sum1_vec: Vec = @splat(0.0);
            var sum2_vec: Vec = @splat(0.0);
            var sum3_vec: Vec = @splat(0.0);

            while (j + VEC_WIDTH <= N) : (j += VEC_WIDTH) {
                const x_vec: Vec = x[j..][0..VEC_WIDTH].*;

                const a0: Vec = A[(i + 0) * lda + j ..][0..VEC_WIDTH].*;
                const a1: Vec = A[(i + 1) * lda + j ..][0..VEC_WIDTH].*;
                const a2: Vec = A[(i + 2) * lda + j ..][0..VEC_WIDTH].*;
                const a3: Vec = A[(i + 3) * lda + j ..][0..VEC_WIDTH].*;

                sum0_vec += a0 * x_vec;
                sum1_vec += a1 * x_vec;
                sum2_vec += a2 * x_vec;
                sum3_vec += a3 * x_vec;
            }

            // Reduce vectors to scalars
            sum0 = @reduce(.Add, sum0_vec);
            sum1 = @reduce(.Add, sum1_vec);
            sum2 = @reduce(.Add, sum2_vec);
            sum3 = @reduce(.Add, sum3_vec);
        }

        // Scalar tail
        while (j < N) : (j += 1) {
            const x_val = x[j];
            sum0 += A[(i + 0) * lda + j] * x_val;
            sum1 += A[(i + 1) * lda + j] * x_val;
            sum2 += A[(i + 2) * lda + j] * x_val;
            sum3 += A[(i + 3) * lda + j] * x_val;
        }

        y[i + 0] += alpha * sum0;
        y[i + 1] += alpha * sum1;
        y[i + 2] += alpha * sum2;
        y[i + 3] += alpha * sum3;
    }

    // Handle remaining rows (< 4)
    while (i < M) : (i += 1) {
        var sum: f32 = 0.0;
        var j: usize = 0;

        if (VEC_WIDTH > 1 and N >= VEC_WIDTH) {
            var sum_vec: Vec = @splat(0.0);

            while (j + VEC_WIDTH <= N) : (j += VEC_WIDTH) {
                const a_vec: Vec = A[i * lda + j ..][0..VEC_WIDTH].*;
                const x_vec: Vec = x[j..][0..VEC_WIDTH].*;
                sum_vec += a_vec * x_vec;
            }

            sum = @reduce(.Add, sum_vec);
        }

        while (j < N) : (j += 1) {
            sum += A[i * lda + j] * x[j];
        }

        y[i] += alpha * sum;
    }
}

// ============================================================================
// Tests
// ============================================================================

test "sgemv basic" {
    // A = [1, 2, 3; 4, 5, 6], x = [1, 2, 3]
    // y = A * x = [14, 32]
    const A = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const x = [_]f32{ 1, 2, 3 };
    var y = [_]f32{ 0, 0 };

    sgemv(2, 3, &A, &x, &y, 1.0, 0.0);

    try std.testing.expectApproxEqRel(y[0], 14.0, 1e-5);
    try std.testing.expectApproxEqRel(y[1], 32.0, 1e-5);
}

test "sgemv with alpha beta" {
    const A = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const x = [_]f32{ 1, 2, 3 };
    var y = [_]f32{ 1, 1 };

    // y = 2.0 * A * x + 0.5 * y = 2*[14, 32] + 0.5*[1, 1] = [28.5, 64.5]
    sgemv(2, 3, &A, &x, &y, 2.0, 0.5);

    try std.testing.expectApproxEqRel(y[0], 28.5, 1e-5);
    try std.testing.expectApproxEqRel(y[1], 64.5, 1e-5);
}

test "sgemv empty" {
    var y = [_]f32{ 1, 2 };
    sgemv(0, 3, &[_]f32{}, &[_]f32{}, &y, 1.0, 0.0);
    // Should not modify y
    try std.testing.expectEqual(y[0], 1.0);
}

test "sgemv large" {
    // Test with size > MIN_OPTIMIZED_SIZE to exercise SIMD path
    const N = 64;
    const M = 4;
    var A: [M * N]f32 = undefined;
    var x: [N]f32 = undefined;
    var y: [M]f32 = undefined;
    var y_ref: [M]f32 = undefined;

    // Initialize with simple pattern
    for (0..M) |i| {
        for (0..N) |j| {
            A[i * N + j] = @as(f32, @floatFromInt(i + j));
        }
    }
    for (0..N) |j| {
        x[j] = @as(f32, @floatFromInt(j));
    }
    @memset(&y, 0.0);
    @memset(&y_ref, 0.0);

    // Compute with optimized path
    sgemv(M, N, &A, &x, &y, 1.0, 0.0);

    // Compute with reference
    reference.sgemv_reference_simple(M, N, &A, &x, &y_ref, 1.0, 0.0);

    // Compare
    for (0..M) |i| {
        try std.testing.expectApproxEqRel(y[i], y_ref[i], 1e-5);
    }
}

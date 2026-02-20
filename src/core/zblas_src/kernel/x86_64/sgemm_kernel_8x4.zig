// zblas/src/kernel/x86_64/sgemm_kernel_8x4.zig
// AVX2-optimized 8×4 micro-kernel for x86-64 (OpenBLAS-style)
//
// This kernel uses the same MR×NR dimensions as OpenBLAS Haswell:
//   - MR = 8 (rows of A/C per kernel call)
//   - NR = 4 (columns of B/C per kernel call)
//
// Benefits over 8×8:
//   - Uses 4 accumulator vectors instead of 8 (half the register pressure)
//   - Each A load is reused 4 times vs 8 times, but with less register spilling
//   - Better suited for Intel's FMA latency/throughput balance
//   - Matches OpenBLAS's tuned Haswell parameters
//
// Register allocation:
//   c0-c3: 4 accumulator vectors (8 floats each = 4 columns of 8 rows)
//   Remaining YMM registers for A loads, B broadcasts, and temporaries

const std = @import("std");

/// Micro-kernel register tile dimensions
pub const mr = 8;
pub const nr = 4;

const MR = mr;
const NR = nr;

const Vec8 = @Vector(8, f32);

/// 8×4 Micro-kernel: C[8×4] += alpha * A_packed[8×k] * B_packed[k×4]
///
/// This computes the outer product accumulation for an 8×4 tile of C.
/// Uses Zig's @Vector which compiles to AVX/AVX2 instructions on x86-64.
///
/// Parameters:
///   k: Reduction dimension (number of A columns / B rows)
///   A: Packed A panel pointer (MR=8 elements per k iteration)
///   B: Packed B panel pointer (NR=4 elements per k iteration)
///   C: Output matrix pointer (row-major)
///   ldc: Leading dimension of C (stride between rows)
///   alpha: Scalar multiplier applied to A*B before adding to C
///
pub fn kernel(
    k: usize,
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    ldc: usize,
    alpha: f32,
) void {
    // 4 accumulator vectors for 8×4 output block
    // Each vector holds one column of the C tile (8 rows)
    var c0: Vec8 = @splat(0.0);
    var c1: Vec8 = @splat(0.0);
    var c2: Vec8 = @splat(0.0);
    var c3: Vec8 = @splat(0.0);

    // Main reduction loop - unroll by 4 for better pipelining
    var kk: usize = 0;
    const k_unroll = k & ~@as(usize, 3); // Round down to multiple of 4

    while (kk < k_unroll) : (kk += 4) {
        // Prefetch next blocks (2-4 cache lines ahead)
        @prefetch(A + (kk + 16) * MR, .{ .locality = 3, .cache = .data });
        @prefetch(B + (kk + 16) * NR, .{ .locality = 3, .cache = .data });

        // Unrolled iterations - process 4 k values per loop
        inline for (0..4) |u| {
            const idx = kk + u;

            // Load 8 elements from packed A (column of 8×k panel)
            const a_vec: Vec8 = A[idx * MR ..][0..8].*;

            // Load 4 elements from packed B and broadcast each
            const b_ptr = B + idx * NR;
            const b0: Vec8 = @splat(b_ptr[0]);
            const b1: Vec8 = @splat(b_ptr[1]);
            const b2: Vec8 = @splat(b_ptr[2]);
            const b3: Vec8 = @splat(b_ptr[3]);

            // Accumulate: c[col] += a_vec * b[col]
            c0 += a_vec * b0;
            c1 += a_vec * b1;
            c2 += a_vec * b2;
            c3 += a_vec * b3;
        }
    }

    // Handle remaining iterations (k mod 4)
    while (kk < k) : (kk += 1) {
        const a_vec: Vec8 = A[kk * MR ..][0..8].*;
        const b_ptr = B + kk * NR;

        c0 += a_vec * @as(Vec8, @splat(b_ptr[0]));
        c1 += a_vec * @as(Vec8, @splat(b_ptr[1]));
        c2 += a_vec * @as(Vec8, @splat(b_ptr[2]));
        c3 += a_vec * @as(Vec8, @splat(b_ptr[3]));
    }

    // Apply alpha and store to C
    // Note: C is stored column by column, each column is 8 consecutive elements
    // C layout: row-major, so C[i,j] = C[i * ldc + j]
    // We computed 8 rows × 4 columns, stored by columns in accumulators
    // Store column 0: C[0..8, 0]
    // Store column 1: C[0..8, 1]
    // etc.
    // For row-major C, we need to scatter the results

    // Extract each element and add to C (row-major layout)
    // c0[i] = result for C[i, 0], c1[i] = result for C[i, 1], etc.
    inline for (0..8) |row| {
        const c_ptr = C + row * ldc;
        c_ptr[0] += alpha * c0[row];
        c_ptr[1] += alpha * c1[row];
        c_ptr[2] += alpha * c2[row];
        c_ptr[3] += alpha * c3[row];
    }
}

/// Alternative: Vectorized store for aligned, contiguous output
/// This version assumes C columns are contiguous (transposed output)
/// Used when we can rearrange the output layout
pub fn kernelTransposedStore(
    k: usize,
    A: [*]const f32,
    B: [*]const f32,
    C: [*]f32,
    ldc: usize,
    alpha: f32,
) void {
    var c0: Vec8 = @splat(0.0);
    var c1: Vec8 = @splat(0.0);
    var c2: Vec8 = @splat(0.0);
    var c3: Vec8 = @splat(0.0);

    var kk: usize = 0;
    const k_unroll = k & ~@as(usize, 3);

    while (kk < k_unroll) : (kk += 4) {
        @prefetch(A + (kk + 16) * MR, .{ .locality = 3, .cache = .data });
        @prefetch(B + (kk + 16) * NR, .{ .locality = 3, .cache = .data });

        inline for (0..4) |u| {
            const idx = kk + u;
            const a_vec: Vec8 = A[idx * MR ..][0..8].*;
            const b_ptr = B + idx * NR;

            c0 += a_vec * @as(Vec8, @splat(b_ptr[0]));
            c1 += a_vec * @as(Vec8, @splat(b_ptr[1]));
            c2 += a_vec * @as(Vec8, @splat(b_ptr[2]));
            c3 += a_vec * @as(Vec8, @splat(b_ptr[3]));
        }
    }

    while (kk < k) : (kk += 1) {
        const a_vec: Vec8 = A[kk * MR ..][0..8].*;
        const b_ptr = B + kk * NR;

        c0 += a_vec * @as(Vec8, @splat(b_ptr[0]));
        c1 += a_vec * @as(Vec8, @splat(b_ptr[1]));
        c2 += a_vec * @as(Vec8, @splat(b_ptr[2]));
        c3 += a_vec * @as(Vec8, @splat(b_ptr[3]));
    }

    const alpha_vec: Vec8 = @splat(alpha);

    // Column-major store (if C is column-major)
    const c0_ptr: *[8]f32 = @ptrCast(C);
    const c1_ptr: *[8]f32 = @ptrCast(C + ldc);
    const c2_ptr: *[8]f32 = @ptrCast(C + 2 * ldc);
    const c3_ptr: *[8]f32 = @ptrCast(C + 3 * ldc);

    c0_ptr.* = @as(Vec8, c0_ptr.*) + c0 * alpha_vec;
    c1_ptr.* = @as(Vec8, c1_ptr.*) + c1 * alpha_vec;
    c2_ptr.* = @as(Vec8, c2_ptr.*) + c2 * alpha_vec;
    c3_ptr.* = @as(Vec8, c3_ptr.*) + c3 * alpha_vec;
}

// ============================================================================
// Tests
// ============================================================================

test "kernel 8x4 identity" {
    // A = 8x4 portion of identity, B = 4x4 sequential values
    // We need K=4 for this test

    // Pack A: column-major within each MR group (identity-like)
    var packed_a: [4 * 8]f32 = undefined;
    for (0..4) |k_idx| {
        for (0..8) |i| {
            // A[i, k] = 1 if i == k, else 0
            packed_a[k_idx * MR + i] = if (i == k_idx) 1.0 else 0.0;
        }
    }

    // B = sequential values 1..16 (4x4)
    var packed_b: [4 * 4]f32 = undefined;
    for (0..16) |i| {
        packed_b[i] = @floatFromInt(i + 1);
    }

    var C = [_]f32{0} ** (8 * 4);

    // Use row-major C with ldc=4
    kernel(4, &packed_a, &packed_b, &C, 4, 1.0);

    // First 4 rows should match B (transposed interpretation)
    // C[i, j] for i < 4 should be B[i, j] = (i*4 + j + 1)
    for (0..4) |row| {
        for (0..4) |col| {
            const expected: f32 = @floatFromInt(row * 4 + col + 1);
            try std.testing.expectApproxEqRel(C[row * 4 + col], expected, 1e-5);
        }
    }

    // Rows 4-7 should be zero (no contribution from identity part)
    for (4..8) |row| {
        for (0..4) |col| {
            try std.testing.expectApproxEqRel(C[row * 4 + col], 0.0, 1e-5);
        }
    }
}

test "kernel 8x4 all ones" {
    // A = all 1s (8 rows, k columns), B = all 1s (k rows, 4 columns)
    // Result: each C element = k (sum of k ones)

    const k_test: usize = 8;
    const packed_a = [_]f32{1.0} ** (k_test * 8);
    const packed_b = [_]f32{1.0} ** (k_test * 4);
    var C = [_]f32{0} ** (8 * 4);

    kernel(k_test, &packed_a, &packed_b, &C, 4, 1.0);

    // Each element should be k_test (8.0)
    for (0..32) |i| {
        try std.testing.expectApproxEqRel(C[i], @as(f32, @floatFromInt(k_test)), 1e-5);
    }
}

test "kernel 8x4 with alpha" {
    const k_test: usize = 4;
    const packed_a = [_]f32{1.0} ** (k_test * 8);
    const packed_b = [_]f32{1.0} ** (k_test * 4);
    var C = [_]f32{0} ** (8 * 4);

    kernel(k_test, &packed_a, &packed_b, &C, 4, 2.0);

    // Each element should be k_test * 2.0 = 8.0
    for (0..32) |i| {
        try std.testing.expectApproxEqRel(C[i], @as(f32, @floatFromInt(k_test * 2)), 1e-5);
    }
}

test "kernel 8x4 accumulates" {
    const k_test: usize = 4;
    const packed_a = [_]f32{1.0} ** (k_test * 8);
    const packed_b = [_]f32{1.0} ** (k_test * 4);
    var C = [_]f32{10.0} ** (8 * 4);

    kernel(k_test, &packed_a, &packed_b, &C, 4, 1.0);

    // Each element should be 10 + k_test = 14.0
    for (0..32) |i| {
        try std.testing.expectApproxEqRel(C[i], 14.0, 1e-5);
    }
}

test "kernel 8x4 k=1" {
    // Test with k=1 (no unrolling)
    const packed_a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const packed_b = [_]f32{ 1, 2, 3, 4 };
    var C = [_]f32{0} ** (8 * 4);

    kernel(1, &packed_a, &packed_b, &C, 4, 1.0);

    // C[i, j] = A[i] * B[j] = (i+1) * (j+1)
    for (0..8) |row| {
        for (0..4) |col| {
            const expected: f32 = @floatFromInt((row + 1) * (col + 1));
            try std.testing.expectApproxEqRel(C[row * 4 + col], expected, 1e-5);
        }
    }
}

test "kernel 8x4 k=5 (cleanup)" {
    // Test cleanup loop (k=5 = 4 unrolled + 1 cleanup)
    const packed_a = [_]f32{1.0} ** (5 * 8);
    const packed_b = [_]f32{1.0} ** (5 * 4);
    var C = [_]f32{0} ** (8 * 4);

    kernel(5, &packed_a, &packed_b, &C, 4, 1.0);

    // Each element should be 5.0
    for (0..32) |i| {
        try std.testing.expectApproxEqRel(C[i], 5.0, 1e-5);
    }
}

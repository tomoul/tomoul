// zblas/src/kernel/x86_64/sgemm_kernel_fused.zig
// AVX2-optimized fused GEMM kernel for x86-64
//
// This kernel fuses B packing with computation, eliminating the need for
// a separate packed B buffer. Instead of pre-packing B, we load B elements
// directly from the original matrix during computation.
//
// Memory flow comparison:
//   Standard:  Read B -> Pack B -> Read packed B -> Compute
//   Fused:     Read B -> Compute immediately
//
// This eliminates one full memory pass over B, reducing memory traffic by ~33%
// for memory-bound workloads.

const std = @import("std");

/// Micro-kernel register tile dimensions
/// Using 8x8 to match the standard kernel for consistency
pub const mr = 8;
pub const nr = 8;

const MR = mr;
const NR = nr;

const Vec8 = @Vector(8, f32);

/// Fused 8×8 Micro-kernel: C[8×8] += alpha * A_packed[8×k] * B_original[k×ldb]
///
/// This kernel reads B directly from the original matrix (not pre-packed).
/// A must still be pre-packed for optimal performance (A is reused across N).
///
/// Parameters:
///   k: Reduction dimension (number of A columns / B rows)
///   A: Packed A panel pointer (MR=8 elements per k iteration)
///   B: Original B matrix pointer (NOT packed), points to first column of block
///   ldb: Leading dimension of B (stride between rows)
///   C: Output matrix pointer (row-major)
///   ldc: Leading dimension of C (stride between rows)
///   alpha: Scalar multiplier applied to A*B before adding to C
///
pub fn kernelFused(
    k: usize,
    A: [*]const f32,
    B: [*]const f32,
    ldb: usize,
    C: [*]f32,
    ldc: usize,
    alpha: f32,
) void {
    // 8 accumulator vectors for 8×8 output block
    // Each vector holds one row of the C tile
    var c0: Vec8 = @splat(0.0);
    var c1: Vec8 = @splat(0.0);
    var c2: Vec8 = @splat(0.0);
    var c3: Vec8 = @splat(0.0);
    var c4: Vec8 = @splat(0.0);
    var c5: Vec8 = @splat(0.0);
    var c6: Vec8 = @splat(0.0);
    var c7: Vec8 = @splat(0.0);

    // Main reduction loop - unroll by 4 for better pipelining
    var kk: usize = 0;
    const k_unroll = k & ~@as(usize, 3); // Round down to multiple of 4

    while (kk < k_unroll) : (kk += 4) {
        // Prefetch next B rows (4-8 rows ahead for memory latency hiding)
        @prefetch(B + (kk + 8) * ldb, .{ .locality = 3, .cache = .data });
        @prefetch(A + (kk + 16) * MR, .{ .locality = 3, .cache = .data });

        // Unrolled iterations - process 4 k values per loop
        inline for (0..4) |u| {
            const idx = kk + u;

            // Load 8 elements from B row directly (fused packing)
            // B[idx, col] = B[idx * ldb + col] for cols 0..7
            const b_vec: Vec8 = B[idx * ldb ..][0..8].*;

            // Load 8 elements from packed A and broadcast-multiply-accumulate
            // A is packed as: a[0,k], a[1,k], ..., a[7,k] for each k
            const a_ptr = A + idx * MR;

            c0 += @as(Vec8, @splat(a_ptr[0])) * b_vec;
            c1 += @as(Vec8, @splat(a_ptr[1])) * b_vec;
            c2 += @as(Vec8, @splat(a_ptr[2])) * b_vec;
            c3 += @as(Vec8, @splat(a_ptr[3])) * b_vec;
            c4 += @as(Vec8, @splat(a_ptr[4])) * b_vec;
            c5 += @as(Vec8, @splat(a_ptr[5])) * b_vec;
            c6 += @as(Vec8, @splat(a_ptr[6])) * b_vec;
            c7 += @as(Vec8, @splat(a_ptr[7])) * b_vec;
        }
    }

    // Handle remaining iterations (k mod 4)
    while (kk < k) : (kk += 1) {
        const b_vec: Vec8 = B[kk * ldb ..][0..8].*;
        const a_ptr = A + kk * MR;

        c0 += @as(Vec8, @splat(a_ptr[0])) * b_vec;
        c1 += @as(Vec8, @splat(a_ptr[1])) * b_vec;
        c2 += @as(Vec8, @splat(a_ptr[2])) * b_vec;
        c3 += @as(Vec8, @splat(a_ptr[3])) * b_vec;
        c4 += @as(Vec8, @splat(a_ptr[4])) * b_vec;
        c5 += @as(Vec8, @splat(a_ptr[5])) * b_vec;
        c6 += @as(Vec8, @splat(a_ptr[6])) * b_vec;
        c7 += @as(Vec8, @splat(a_ptr[7])) * b_vec;
    }

    // Apply alpha and store to C
    const alpha_vec: Vec8 = @splat(alpha);

    // Load existing C, add scaled contribution, store back
    const c0_ptr: *[8]f32 = @ptrCast(C);
    const c1_ptr: *[8]f32 = @ptrCast(C + ldc);
    const c2_ptr: *[8]f32 = @ptrCast(C + 2 * ldc);
    const c3_ptr: *[8]f32 = @ptrCast(C + 3 * ldc);
    const c4_ptr: *[8]f32 = @ptrCast(C + 4 * ldc);
    const c5_ptr: *[8]f32 = @ptrCast(C + 5 * ldc);
    const c6_ptr: *[8]f32 = @ptrCast(C + 6 * ldc);
    const c7_ptr: *[8]f32 = @ptrCast(C + 7 * ldc);

    c0_ptr.* = @as(Vec8, c0_ptr.*) + c0 * alpha_vec;
    c1_ptr.* = @as(Vec8, c1_ptr.*) + c1 * alpha_vec;
    c2_ptr.* = @as(Vec8, c2_ptr.*) + c2 * alpha_vec;
    c3_ptr.* = @as(Vec8, c3_ptr.*) + c3 * alpha_vec;
    c4_ptr.* = @as(Vec8, c4_ptr.*) + c4 * alpha_vec;
    c5_ptr.* = @as(Vec8, c5_ptr.*) + c5 * alpha_vec;
    c6_ptr.* = @as(Vec8, c6_ptr.*) + c6 * alpha_vec;
    c7_ptr.* = @as(Vec8, c7_ptr.*) + c7 * alpha_vec;
}

/// Beta=0 optimized fused kernel: C[8×8] = alpha * A_packed[8×k] * B_original[k×ldb]
///
/// This is the fast path for the common case where beta=0 (overwrite C).
/// Avoids reading C before writing, reducing memory traffic.
///
pub fn kernelFusedBetaZero(
    k: usize,
    A: [*]const f32,
    B: [*]const f32,
    ldb: usize,
    C: [*]f32,
    ldc: usize,
    alpha: f32,
) void {
    var c0: Vec8 = @splat(0.0);
    var c1: Vec8 = @splat(0.0);
    var c2: Vec8 = @splat(0.0);
    var c3: Vec8 = @splat(0.0);
    var c4: Vec8 = @splat(0.0);
    var c5: Vec8 = @splat(0.0);
    var c6: Vec8 = @splat(0.0);
    var c7: Vec8 = @splat(0.0);

    var kk: usize = 0;
    const k_unroll = k & ~@as(usize, 3);

    while (kk < k_unroll) : (kk += 4) {
        @prefetch(B + (kk + 8) * ldb, .{ .locality = 3, .cache = .data });
        @prefetch(A + (kk + 16) * MR, .{ .locality = 3, .cache = .data });

        inline for (0..4) |u| {
            const idx = kk + u;
            const b_vec: Vec8 = B[idx * ldb ..][0..8].*;
            const a_ptr = A + idx * MR;

            c0 += @as(Vec8, @splat(a_ptr[0])) * b_vec;
            c1 += @as(Vec8, @splat(a_ptr[1])) * b_vec;
            c2 += @as(Vec8, @splat(a_ptr[2])) * b_vec;
            c3 += @as(Vec8, @splat(a_ptr[3])) * b_vec;
            c4 += @as(Vec8, @splat(a_ptr[4])) * b_vec;
            c5 += @as(Vec8, @splat(a_ptr[5])) * b_vec;
            c6 += @as(Vec8, @splat(a_ptr[6])) * b_vec;
            c7 += @as(Vec8, @splat(a_ptr[7])) * b_vec;
        }
    }

    while (kk < k) : (kk += 1) {
        const b_vec: Vec8 = B[kk * ldb ..][0..8].*;
        const a_ptr = A + kk * MR;

        c0 += @as(Vec8, @splat(a_ptr[0])) * b_vec;
        c1 += @as(Vec8, @splat(a_ptr[1])) * b_vec;
        c2 += @as(Vec8, @splat(a_ptr[2])) * b_vec;
        c3 += @as(Vec8, @splat(a_ptr[3])) * b_vec;
        c4 += @as(Vec8, @splat(a_ptr[4])) * b_vec;
        c5 += @as(Vec8, @splat(a_ptr[5])) * b_vec;
        c6 += @as(Vec8, @splat(a_ptr[6])) * b_vec;
        c7 += @as(Vec8, @splat(a_ptr[7])) * b_vec;
    }

    // Direct store without reading C (beta=0 fast path)
    const alpha_vec: Vec8 = @splat(alpha);

    const c0_ptr: *[8]f32 = @ptrCast(C);
    const c1_ptr: *[8]f32 = @ptrCast(C + ldc);
    const c2_ptr: *[8]f32 = @ptrCast(C + 2 * ldc);
    const c3_ptr: *[8]f32 = @ptrCast(C + 3 * ldc);
    const c4_ptr: *[8]f32 = @ptrCast(C + 4 * ldc);
    const c5_ptr: *[8]f32 = @ptrCast(C + 5 * ldc);
    const c6_ptr: *[8]f32 = @ptrCast(C + 6 * ldc);
    const c7_ptr: *[8]f32 = @ptrCast(C + 7 * ldc);

    // No read from C - direct write
    c0_ptr.* = c0 * alpha_vec;
    c1_ptr.* = c1 * alpha_vec;
    c2_ptr.* = c2 * alpha_vec;
    c3_ptr.* = c3 * alpha_vec;
    c4_ptr.* = c4 * alpha_vec;
    c5_ptr.* = c5 * alpha_vec;
    c6_ptr.* = c6 * alpha_vec;
    c7_ptr.* = c7 * alpha_vec;
}

/// Ultimate fast path: alpha=1.0, beta=0.0 (most common in inference)
/// C[8×8] = A_packed[8×k] * B_original[k×ldb]
///
pub fn kernelFusedFast(
    k: usize,
    A: [*]const f32,
    B: [*]const f32,
    ldb: usize,
    C: [*]f32,
    ldc: usize,
) void {
    var c0: Vec8 = @splat(0.0);
    var c1: Vec8 = @splat(0.0);
    var c2: Vec8 = @splat(0.0);
    var c3: Vec8 = @splat(0.0);
    var c4: Vec8 = @splat(0.0);
    var c5: Vec8 = @splat(0.0);
    var c6: Vec8 = @splat(0.0);
    var c7: Vec8 = @splat(0.0);

    var kk: usize = 0;
    const k_unroll = k & ~@as(usize, 3);

    while (kk < k_unroll) : (kk += 4) {
        @prefetch(B + (kk + 8) * ldb, .{ .locality = 3, .cache = .data });
        @prefetch(A + (kk + 16) * MR, .{ .locality = 3, .cache = .data });

        inline for (0..4) |u| {
            const idx = kk + u;
            const b_vec: Vec8 = B[idx * ldb ..][0..8].*;
            const a_ptr = A + idx * MR;

            c0 += @as(Vec8, @splat(a_ptr[0])) * b_vec;
            c1 += @as(Vec8, @splat(a_ptr[1])) * b_vec;
            c2 += @as(Vec8, @splat(a_ptr[2])) * b_vec;
            c3 += @as(Vec8, @splat(a_ptr[3])) * b_vec;
            c4 += @as(Vec8, @splat(a_ptr[4])) * b_vec;
            c5 += @as(Vec8, @splat(a_ptr[5])) * b_vec;
            c6 += @as(Vec8, @splat(a_ptr[6])) * b_vec;
            c7 += @as(Vec8, @splat(a_ptr[7])) * b_vec;
        }
    }

    while (kk < k) : (kk += 1) {
        const b_vec: Vec8 = B[kk * ldb ..][0..8].*;
        const a_ptr = A + kk * MR;

        c0 += @as(Vec8, @splat(a_ptr[0])) * b_vec;
        c1 += @as(Vec8, @splat(a_ptr[1])) * b_vec;
        c2 += @as(Vec8, @splat(a_ptr[2])) * b_vec;
        c3 += @as(Vec8, @splat(a_ptr[3])) * b_vec;
        c4 += @as(Vec8, @splat(a_ptr[4])) * b_vec;
        c5 += @as(Vec8, @splat(a_ptr[5])) * b_vec;
        c6 += @as(Vec8, @splat(a_ptr[6])) * b_vec;
        c7 += @as(Vec8, @splat(a_ptr[7])) * b_vec;
    }

    // Direct store - no alpha multiplication, no C read
    const c0_ptr: *[8]f32 = @ptrCast(C);
    const c1_ptr: *[8]f32 = @ptrCast(C + ldc);
    const c2_ptr: *[8]f32 = @ptrCast(C + 2 * ldc);
    const c3_ptr: *[8]f32 = @ptrCast(C + 3 * ldc);
    const c4_ptr: *[8]f32 = @ptrCast(C + 4 * ldc);
    const c5_ptr: *[8]f32 = @ptrCast(C + 5 * ldc);
    const c6_ptr: *[8]f32 = @ptrCast(C + 6 * ldc);
    const c7_ptr: *[8]f32 = @ptrCast(C + 7 * ldc);

    c0_ptr.* = c0;
    c1_ptr.* = c1;
    c2_ptr.* = c2;
    c3_ptr.* = c3;
    c4_ptr.* = c4;
    c5_ptr.* = c5;
    c6_ptr.* = c6;
    c7_ptr.* = c7;
}

// ============================================================================
// Tests
// ============================================================================

test "kernelFused identity" {
    // A = 8x8 identity (packed), B = 8x8 sequential values
    // C should equal B after multiplication

    // Pack A: identity matrix in packed format
    var packed_a: [8 * 8]f32 = undefined;
    for (0..8) |k_idx| {
        for (0..8) |i| {
            packed_a[k_idx * MR + i] = if (i == k_idx) 1.0 else 0.0;
        }
    }

    // B = sequential values 1..64 in row-major
    var B: [8 * 8]f32 = undefined;
    for (0..64) |i| {
        B[i] = @floatFromInt(i + 1);
    }

    var C = [_]f32{0} ** 64;

    kernelFused(8, &packed_a, &B, 8, &C, 8, 1.0);

    // C should equal B
    for (0..8) |row| {
        for (0..8) |col| {
            const expected: f32 = @floatFromInt(row * 8 + col + 1);
            try std.testing.expectApproxEqRel(C[row * 8 + col], expected, 1e-5);
        }
    }
}

test "kernelFused all ones" {
    // A = all 1s, B = all 1s
    // A*B = 8 (each element is sum of 8 ones)

    const packed_a = [_]f32{1.0} ** 64;
    const B = [_]f32{1.0} ** 64;
    var C = [_]f32{0} ** 64;

    kernelFused(8, &packed_a, &B, 8, &C, 8, 1.0);

    for (0..64) |i| {
        try std.testing.expectApproxEqRel(C[i], 8.0, 1e-5);
    }
}

test "kernelFused with alpha" {
    const packed_a = [_]f32{1.0} ** 64;
    const B = [_]f32{1.0} ** 64;
    var C = [_]f32{0} ** 64;

    kernelFused(8, &packed_a, &B, 8, &C, 8, 2.0);

    for (0..64) |i| {
        try std.testing.expectApproxEqRel(C[i], 16.0, 1e-5);
    }
}

test "kernelFused accumulates to C" {
    // Start with C = all 10s
    var packed_a: [8 * 8]f32 = undefined;
    for (0..8) |k_idx| {
        for (0..8) |i| {
            packed_a[k_idx * MR + i] = if (k_idx == 0 and i == 0) 1.0 else 0.0;
        }
    }

    var B: [8 * 8]f32 = undefined;
    for (0..8) |j| {
        B[j] = 1.0; // Only first row of B is 1s
    }
    @memset(B[8..], 0.0);

    var C = [_]f32{10.0} ** 64;

    kernelFused(8, &packed_a, &B, 8, &C, 8, 1.0);

    // First row gets +1, rest stays 10
    for (0..8) |j| {
        try std.testing.expectApproxEqRel(C[j], 11.0, 1e-5);
    }
    for (8..64) |i| {
        try std.testing.expectApproxEqRel(C[i], 10.0, 1e-5);
    }
}

test "kernelFusedBetaZero" {
    const packed_a = [_]f32{1.0} ** 64;
    const B = [_]f32{1.0} ** 64;
    var C = [_]f32{999.0} ** 64; // Should be overwritten

    kernelFusedBetaZero(8, &packed_a, &B, 8, &C, 8, 1.0);

    // C should be 8.0, not 999+8
    for (0..64) |i| {
        try std.testing.expectApproxEqRel(C[i], 8.0, 1e-5);
    }
}

test "kernelFusedFast" {
    const packed_a = [_]f32{1.0} ** 64;
    const B = [_]f32{1.0} ** 64;
    var C = [_]f32{999.0} ** 64;

    kernelFusedFast(8, &packed_a, &B, 8, &C, 8);

    for (0..64) |i| {
        try std.testing.expectApproxEqRel(C[i], 8.0, 1e-5);
    }
}

test "kernelFused k=3 (non-multiple of 4)" {
    const packed_a = [_]f32{1.0} ** (3 * 8);
    var B: [3 * 8]f32 = undefined;
    for (0..3) |row| {
        for (0..8) |col| {
            B[row * 8 + col] = 1.0;
        }
    }
    var C = [_]f32{0} ** 64;

    kernelFused(3, &packed_a, &B, 8, &C, 8, 1.0);

    for (0..64) |i| {
        try std.testing.expectApproxEqRel(C[i], 3.0, 1e-5);
    }
}

test "kernelFused with non-contiguous B (ldb > 8)" {
    // B has stride 16 (extra padding between rows)
    const packed_a = [_]f32{1.0} ** 64;
    var B: [8 * 16]f32 = [_]f32{0} ** (8 * 16);

    // Set first 8 cols of each row to 1.0
    for (0..8) |row| {
        for (0..8) |col| {
            B[row * 16 + col] = 1.0;
        }
    }

    var C = [_]f32{0} ** 64;

    kernelFused(8, &packed_a, &B, 16, &C, 8, 1.0);

    for (0..64) |i| {
        try std.testing.expectApproxEqRel(C[i], 8.0, 1e-5);
    }
}

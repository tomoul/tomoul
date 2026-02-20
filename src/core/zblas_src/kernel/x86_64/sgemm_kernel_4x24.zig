// zblas/src/kernel/x86_64/sgemm_kernel_4x24.zig
// AVX2-optimized 4×24 micro-kernel for x86-64
//
// This kernel processes 4 rows × 24 columns = 96 elements per micro-kernel call.
// Uses 12 YMM registers for accumulators (4 rows × 3 vectors of 8 floats each).
// Leaves 4 YMM registers for A broadcasts and B loads.
//
// This is the same tile size as Tomoul's ops.zig fallback, which performs well
// for transformer workloads (whisper, llama, etc.)

const std = @import("std");

/// Micro-kernel register tile dimensions
pub const mr = 4;
pub const nr = 24;

const MR = mr;
const NR = nr;
const VEC_WIDTH = 8;

const Vec8 = @Vector(8, f32);

/// 4×24 Micro-kernel: C[4×24] += alpha * A_packed[4×k] * B_packed[k×24]
///
/// This computes the outer product accumulation for a 4×24 tile of C.
/// Uses 12 AVX registers for accumulators, processes 3 vectors of B per iteration.
///
/// Parameters:
///   k: Reduction dimension (number of A columns / B rows)
///   A: Packed A panel pointer (MR=4 elements per k iteration)
///   B: Packed B panel pointer (NR=24 elements per k iteration)
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
    // 12 accumulator vectors for 4×24 output block
    // Each row has 3 vectors (24 floats)
    var c00: Vec8 = @splat(0.0);
    var c01: Vec8 = @splat(0.0);
    var c02: Vec8 = @splat(0.0);
    var c10: Vec8 = @splat(0.0);
    var c11: Vec8 = @splat(0.0);
    var c12: Vec8 = @splat(0.0);
    var c20: Vec8 = @splat(0.0);
    var c21: Vec8 = @splat(0.0);
    var c22: Vec8 = @splat(0.0);
    var c30: Vec8 = @splat(0.0);
    var c31: Vec8 = @splat(0.0);
    var c32: Vec8 = @splat(0.0);

    // Main reduction loop
    var kk: usize = 0;
    while (kk < k) : (kk += 1) {
        // Prefetch next iteration
        if (kk + 8 < k) {
            @prefetch(A + (kk + 8) * MR, .{ .locality = 3, .cache = .data });
            @prefetch(B + (kk + 8) * NR, .{ .locality = 3, .cache = .data });
        }

        // Load 4 elements from packed A and broadcast
        const a_ptr = A + kk * MR;
        const a0: Vec8 = @splat(a_ptr[0]);
        const a1: Vec8 = @splat(a_ptr[1]);
        const a2: Vec8 = @splat(a_ptr[2]);
        const a3: Vec8 = @splat(a_ptr[3]);

        // Load 3 vectors (24 elements) from packed B
        const b_ptr = B + kk * NR;
        const b0: Vec8 = b_ptr[0..8].*;
        const b1: Vec8 = b_ptr[8..16].*;
        const b2: Vec8 = b_ptr[16..24].*;

        // Accumulate outer products
        c00 += a0 * b0;
        c01 += a0 * b1;
        c02 += a0 * b2;
        c10 += a1 * b0;
        c11 += a1 * b1;
        c12 += a1 * b2;
        c20 += a2 * b0;
        c21 += a2 * b1;
        c22 += a2 * b2;
        c30 += a3 * b0;
        c31 += a3 * b1;
        c32 += a3 * b2;
    }

    // Apply alpha and store to C
    const alpha_vec: Vec8 = @splat(alpha);

    // Row 0
    const c0_base = C;
    const c0_v0: *[8]f32 = @ptrCast(c0_base);
    const c0_v1: *[8]f32 = @ptrCast(c0_base + 8);
    const c0_v2: *[8]f32 = @ptrCast(c0_base + 16);
    c0_v0.* = @as(Vec8, c0_v0.*) + c00 * alpha_vec;
    c0_v1.* = @as(Vec8, c0_v1.*) + c01 * alpha_vec;
    c0_v2.* = @as(Vec8, c0_v2.*) + c02 * alpha_vec;

    // Row 1
    const c1_base = C + ldc;
    const c1_v0: *[8]f32 = @ptrCast(c1_base);
    const c1_v1: *[8]f32 = @ptrCast(c1_base + 8);
    const c1_v2: *[8]f32 = @ptrCast(c1_base + 16);
    c1_v0.* = @as(Vec8, c1_v0.*) + c10 * alpha_vec;
    c1_v1.* = @as(Vec8, c1_v1.*) + c11 * alpha_vec;
    c1_v2.* = @as(Vec8, c1_v2.*) + c12 * alpha_vec;

    // Row 2
    const c2_base = C + 2 * ldc;
    const c2_v0: *[8]f32 = @ptrCast(c2_base);
    const c2_v1: *[8]f32 = @ptrCast(c2_base + 8);
    const c2_v2: *[8]f32 = @ptrCast(c2_base + 16);
    c2_v0.* = @as(Vec8, c2_v0.*) + c20 * alpha_vec;
    c2_v1.* = @as(Vec8, c2_v1.*) + c21 * alpha_vec;
    c2_v2.* = @as(Vec8, c2_v2.*) + c22 * alpha_vec;

    // Row 3
    const c3_base = C + 3 * ldc;
    const c3_v0: *[8]f32 = @ptrCast(c3_base);
    const c3_v1: *[8]f32 = @ptrCast(c3_base + 8);
    const c3_v2: *[8]f32 = @ptrCast(c3_base + 16);
    c3_v0.* = @as(Vec8, c3_v0.*) + c30 * alpha_vec;
    c3_v1.* = @as(Vec8, c3_v1.*) + c31 * alpha_vec;
    c3_v2.* = @as(Vec8, c3_v2.*) + c32 * alpha_vec;
}

// ============================================================================
// Tests
// ============================================================================

test "kernel 4x24 identity" {
    // Simple test: A = ones, B = ones, result should be k
    const k_dim = 8;
    var packed_a: [k_dim * MR]f32 = undefined;
    var packed_b: [k_dim * NR]f32 = undefined;
    @memset(&packed_a, 1.0);
    @memset(&packed_b, 1.0);

    var C: [4 * 24]f32 = undefined;
    @memset(&C, 0.0);

    kernel(k_dim, &packed_a, &packed_b, &C, 24, 1.0);

    // Each element should be k_dim
    for (0..4) |row| {
        for (0..24) |col| {
            try std.testing.expectApproxEqRel(C[row * 24 + col], @as(f32, k_dim), 1e-5);
        }
    }
}

test "kernel 4x24 with alpha" {
    const k_dim = 4;
    var packed_a: [k_dim * MR]f32 = undefined;
    var packed_b: [k_dim * NR]f32 = undefined;
    @memset(&packed_a, 1.0);
    @memset(&packed_b, 1.0);

    var C: [4 * 24]f32 = undefined;
    @memset(&C, 0.0);

    kernel(k_dim, &packed_a, &packed_b, &C, 24, 2.0);

    // Each element should be k_dim * alpha = 4 * 2 = 8
    for (0..96) |i| {
        try std.testing.expectApproxEqRel(C[i], 8.0, 1e-5);
    }
}

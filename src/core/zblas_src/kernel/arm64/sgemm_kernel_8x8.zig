// zblas/src/kernel/arm64/sgemm_kernel_8x8.zig
// ARM64 NEON-optimized 8×8 micro-kernel
//
// This is the innermost computation of GEMM on ARM64. It computes an 8×8 block
// of C by accumulating outer products of vectors from A and B.
//
// ARM64 NEON has 32 128-bit vector registers (v0-v31), each holding 4 floats.
// We use 16 registers for accumulators (8 rows × 2 vectors per row = 64 floats).
//
// Register allocation:
//   c00-c71: 16 accumulator vectors (64 floats total)
//   Remaining 16 registers for A broadcasts, B loads, and temporaries

const std = @import("std");

/// Micro-kernel register tile dimensions
pub const mr = 8;
pub const nr = 8;

const MR = mr;
const NR = nr;

// NEON uses 128-bit vectors (4 floats)
const Vec4 = @Vector(4, f32);

/// 8×8 Micro-kernel: C[8×8] += alpha * A_packed[8×k] * B_packed[k×8]
///
/// This computes the outer product accumulation for an 8×8 tile of C.
/// Uses Zig's @Vector which compiles to NEON instructions on ARM64.
///
/// ARM64 has 32 NEON registers, so we can keep all 16 accumulators in registers:
///   - 16 vectors for C (8 rows × 2 vectors per row = 64 floats)
///   - Remaining registers for A broadcasts, B loads, and temporaries
///
/// Parameters:
///   k: Reduction dimension (number of A columns / B rows)
///   A: Packed A panel pointer (MR=8 elements per k iteration)
///   B: Packed B panel pointer (NR=8 elements per k iteration)
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
    // 16 accumulators: 2 Vec4 per row × 8 rows
    // Row 0: c00 (cols 0-3), c01 (cols 4-7)
    // Row 1: c10, c11
    // ... etc
    var c00: Vec4 = @splat(0.0);
    var c01: Vec4 = @splat(0.0);
    var c10: Vec4 = @splat(0.0);
    var c11: Vec4 = @splat(0.0);
    var c20: Vec4 = @splat(0.0);
    var c21: Vec4 = @splat(0.0);
    var c30: Vec4 = @splat(0.0);
    var c31: Vec4 = @splat(0.0);
    var c40: Vec4 = @splat(0.0);
    var c41: Vec4 = @splat(0.0);
    var c50: Vec4 = @splat(0.0);
    var c51: Vec4 = @splat(0.0);
    var c60: Vec4 = @splat(0.0);
    var c61: Vec4 = @splat(0.0);
    var c70: Vec4 = @splat(0.0);
    var c71: Vec4 = @splat(0.0);

    // Main reduction loop - unroll by 4 for better pipelining
    var kk: usize = 0;
    const k_unroll = k & ~@as(usize, 3); // Round down to multiple of 4

    while (kk < k_unroll) : (kk += 4) {
        // Prefetch next blocks
        @prefetch(A + (kk + 16) * MR, .{ .locality = 2, .cache = .data });
        @prefetch(B + (kk + 16) * NR, .{ .locality = 2, .cache = .data });

        // Unrolled iterations - process 4 k values per loop
        inline for (0..4) |u| {
            const idx = kk + u;

            // Load B row: 8 elements = 2 Vec4
            const b0: Vec4 = B[idx * NR ..][0..4].*;
            const b1: Vec4 = B[idx * NR + 4 ..][0..4].*;

            // A column: 8 elements (broadcast each)
            const a_ptr = A + idx * MR;

            const a0: Vec4 = @splat(a_ptr[0]);
            const a1: Vec4 = @splat(a_ptr[1]);
            const a2: Vec4 = @splat(a_ptr[2]);
            const a3: Vec4 = @splat(a_ptr[3]);
            const a4: Vec4 = @splat(a_ptr[4]);
            const a5: Vec4 = @splat(a_ptr[5]);
            const a6: Vec4 = @splat(a_ptr[6]);
            const a7: Vec4 = @splat(a_ptr[7]);

            // FMA accumulation - each row accumulates a_i * B
            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
            c20 += a2 * b0;
            c21 += a2 * b1;
            c30 += a3 * b0;
            c31 += a3 * b1;
            c40 += a4 * b0;
            c41 += a4 * b1;
            c50 += a5 * b0;
            c51 += a5 * b1;
            c60 += a6 * b0;
            c61 += a6 * b1;
            c70 += a7 * b0;
            c71 += a7 * b1;
        }
    }

    // Handle remaining iterations (k mod 4)
    while (kk < k) : (kk += 1) {
        // Load B row: 8 elements = 2 Vec4
        const b0: Vec4 = B[kk * NR ..][0..4].*;
        const b1: Vec4 = B[kk * NR + 4 ..][0..4].*;

        // A column: 8 elements (broadcast each)
        const a_ptr = A + kk * MR;

        const a0: Vec4 = @splat(a_ptr[0]);
        const a1: Vec4 = @splat(a_ptr[1]);
        const a2: Vec4 = @splat(a_ptr[2]);
        const a3: Vec4 = @splat(a_ptr[3]);
        const a4: Vec4 = @splat(a_ptr[4]);
        const a5: Vec4 = @splat(a_ptr[5]);
        const a6: Vec4 = @splat(a_ptr[6]);
        const a7: Vec4 = @splat(a_ptr[7]);

        c00 += a0 * b0;
        c01 += a0 * b1;
        c10 += a1 * b0;
        c11 += a1 * b1;
        c20 += a2 * b0;
        c21 += a2 * b1;
        c30 += a3 * b0;
        c31 += a3 * b1;
        c40 += a4 * b0;
        c41 += a4 * b1;
        c50 += a5 * b0;
        c51 += a5 * b1;
        c60 += a6 * b0;
        c61 += a6 * b1;
        c70 += a7 * b0;
        c71 += a7 * b1;
    }

    // Apply alpha and store to C
    const alpha_vec: Vec4 = @splat(alpha);

    // Load existing C, add scaled contribution, store back
    // Row 0
    const c0_ptr0: *[4]f32 = @ptrCast(C);
    const c0_ptr1: *[4]f32 = @ptrCast(C + 4);
    c0_ptr0.* = @as(Vec4, c0_ptr0.*) + c00 * alpha_vec;
    c0_ptr1.* = @as(Vec4, c0_ptr1.*) + c01 * alpha_vec;

    // Row 1
    const c1_ptr0: *[4]f32 = @ptrCast(C + ldc);
    const c1_ptr1: *[4]f32 = @ptrCast(C + ldc + 4);
    c1_ptr0.* = @as(Vec4, c1_ptr0.*) + c10 * alpha_vec;
    c1_ptr1.* = @as(Vec4, c1_ptr1.*) + c11 * alpha_vec;

    // Row 2
    const c2_ptr0: *[4]f32 = @ptrCast(C + 2 * ldc);
    const c2_ptr1: *[4]f32 = @ptrCast(C + 2 * ldc + 4);
    c2_ptr0.* = @as(Vec4, c2_ptr0.*) + c20 * alpha_vec;
    c2_ptr1.* = @as(Vec4, c2_ptr1.*) + c21 * alpha_vec;

    // Row 3
    const c3_ptr0: *[4]f32 = @ptrCast(C + 3 * ldc);
    const c3_ptr1: *[4]f32 = @ptrCast(C + 3 * ldc + 4);
    c3_ptr0.* = @as(Vec4, c3_ptr0.*) + c30 * alpha_vec;
    c3_ptr1.* = @as(Vec4, c3_ptr1.*) + c31 * alpha_vec;

    // Row 4
    const c4_ptr0: *[4]f32 = @ptrCast(C + 4 * ldc);
    const c4_ptr1: *[4]f32 = @ptrCast(C + 4 * ldc + 4);
    c4_ptr0.* = @as(Vec4, c4_ptr0.*) + c40 * alpha_vec;
    c4_ptr1.* = @as(Vec4, c4_ptr1.*) + c41 * alpha_vec;

    // Row 5
    const c5_ptr0: *[4]f32 = @ptrCast(C + 5 * ldc);
    const c5_ptr1: *[4]f32 = @ptrCast(C + 5 * ldc + 4);
    c5_ptr0.* = @as(Vec4, c5_ptr0.*) + c50 * alpha_vec;
    c5_ptr1.* = @as(Vec4, c5_ptr1.*) + c51 * alpha_vec;

    // Row 6
    const c6_ptr0: *[4]f32 = @ptrCast(C + 6 * ldc);
    const c6_ptr1: *[4]f32 = @ptrCast(C + 6 * ldc + 4);
    c6_ptr0.* = @as(Vec4, c6_ptr0.*) + c60 * alpha_vec;
    c6_ptr1.* = @as(Vec4, c6_ptr1.*) + c61 * alpha_vec;

    // Row 7
    const c7_ptr0: *[4]f32 = @ptrCast(C + 7 * ldc);
    const c7_ptr1: *[4]f32 = @ptrCast(C + 7 * ldc + 4);
    c7_ptr0.* = @as(Vec4, c7_ptr0.*) + c70 * alpha_vec;
    c7_ptr1.* = @as(Vec4, c7_ptr1.*) + c71 * alpha_vec;
}

// ============================================================================
// Tests
// ============================================================================

test "kernel 8x8 identity" {
    // A = 8x8 identity, B = 8x8 sequential values
    // C should equal B after multiplication

    // Pack A: column-major within each MR group (identity matrix)
    var packed_a: [8 * 8]f32 = undefined;
    for (0..8) |k_idx| {
        for (0..8) |i| {
            packed_a[k_idx * MR + i] = if (i == k_idx) 1.0 else 0.0;
        }
    }

    // B = sequential values 1..64
    var packed_b: [8 * 8]f32 = undefined;
    for (0..64) |i| {
        packed_b[i] = @floatFromInt(i + 1);
    }

    var C = [_]f32{0} ** 64;

    kernel(8, &packed_a, &packed_b, &C, 8, 1.0);

    // C should equal B (reshaped to 8x8)
    for (0..8) |row| {
        for (0..8) |col| {
            const expected: f32 = @floatFromInt(row * 8 + col + 1);
            try std.testing.expectApproxEqRel(C[row * 8 + col], expected, 1e-5);
        }
    }
}

test "kernel 8x8 all ones" {
    // A = all 1s, B = all 1s
    // A*B = 8 (each element is sum of 8 ones)

    const packed_a = [_]f32{1.0} ** 64;
    const packed_b = [_]f32{1.0} ** 64;
    var C = [_]f32{0} ** 64;

    kernel(8, &packed_a, &packed_b, &C, 8, 1.0);

    // Each element should be 8.0
    for (0..64) |i| {
        try std.testing.expectApproxEqRel(C[i], 8.0, 1e-5);
    }
}

test "kernel 8x8 with alpha" {
    // Same as above but with alpha = 2.0
    const packed_a = [_]f32{1.0} ** 64;
    const packed_b = [_]f32{1.0} ** 64;
    var C = [_]f32{0} ** 64;

    kernel(8, &packed_a, &packed_b, &C, 8, 2.0);

    // Each element should be 16.0 (8 * 2)
    for (0..64) |i| {
        try std.testing.expectApproxEqRel(C[i], 16.0, 1e-5);
    }
}

test "kernel 8x8 accumulates to C" {
    // Test that kernel adds to existing C values
    var packed_a: [8 * 8]f32 = undefined;
    for (0..8) |k_idx| {
        for (0..8) |i| {
            packed_a[k_idx * MR + i] = if (k_idx == 0 and i == 0) 1.0 else 0.0;
        }
    }

    var packed_b: [8 * 8]f32 = undefined;
    for (0..8) |j| {
        packed_b[0 * NR + j] = 1.0; // Only first row of B is 1s
    }
    @memset(packed_b[8..], 0.0);

    // Start with C = all 10s
    var C = [_]f32{10.0} ** 64;

    kernel(8, &packed_a, &packed_b, &C, 8, 1.0);

    // First row gets +1 to each element, rest stays 10
    for (0..8) |j| {
        try std.testing.expectApproxEqRel(C[j], 11.0, 1e-5);
    }
    for (8..64) |i| {
        try std.testing.expectApproxEqRel(C[i], 10.0, 1e-5);
    }
}

test "kernel 8x8 k=1" {
    // Test with k=1 (no unrolling)
    const packed_a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const packed_b = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1 };
    var C = [_]f32{0} ** 64;

    kernel(1, &packed_a, &packed_b, &C, 8, 1.0);

    // Row i should be all a[i] values
    for (0..8) |row| {
        const expected: f32 = @floatFromInt(row + 1);
        for (0..8) |col| {
            try std.testing.expectApproxEqRel(C[row * 8 + col], expected, 1e-5);
        }
    }
}

test "kernel 8x8 k=3 (non-multiple of 4)" {
    // Test cleanup loop with k not divisible by 4
    const packed_a = [_]f32{1.0} ** (3 * 8);
    const packed_b = [_]f32{1.0} ** (3 * 8);
    var C = [_]f32{0} ** 64;

    kernel(3, &packed_a, &packed_b, &C, 8, 1.0);

    // Each element should be 3.0
    for (0..64) |i| {
        try std.testing.expectApproxEqRel(C[i], 3.0, 1e-5);
    }
}

test "kernel 8x8 k=5 (cleanup after unroll)" {
    // Test 4 unrolled + 1 cleanup
    const packed_a = [_]f32{1.0} ** (5 * 8);
    const packed_b = [_]f32{1.0} ** (5 * 8);
    var C = [_]f32{0} ** 64;

    kernel(5, &packed_a, &packed_b, &C, 8, 1.0);

    // Each element should be 5.0
    for (0..64) |i| {
        try std.testing.expectApproxEqRel(C[i], 5.0, 1e-5);
    }
}

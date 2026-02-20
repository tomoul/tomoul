// zblas/src/kernel/x86_64/sgemm_kernel_8x8.zig
// AVX2-optimized 8×8 micro-kernel for x86-64
//
// This is the innermost computation of GEMM on x86-64. It computes an 8×8 block
// of C by accumulating outer products of 8-element vectors from A and B.
//
// Register allocation:
//   c0-c7: 8 accumulator vectors (64 floats total = 8 YMM registers)
//   Remaining 8 YMM registers for A broadcasts, B loads, and temporaries

const std = @import("std");

/// Micro-kernel register tile dimensions
pub const mr = 8;
pub const nr = 8;

const MR = mr;
const NR = nr;

const Vec8 = @Vector(8, f32);

/// 8×8 Micro-kernel: C[8×8] += alpha * A_packed[8×k] * B_packed[k×8]
///
/// This computes the outer product accumulation for an 8×8 tile of C.
/// Uses Zig's @Vector which compiles to AVX/AVX2 instructions on x86-64.
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
        // Prefetch next blocks (2-4 cache lines ahead)
        @prefetch(A + (kk + 16) * MR, .{ .locality = 3, .cache = .data });
        @prefetch(B + (kk + 16) * NR, .{ .locality = 3, .cache = .data });

        // Unrolled iterations - process 4 k values per loop
        inline for (0..4) |u| {
            const idx = kk + u;

            // Load 8 elements from packed B (row of k×8 panel)
            const b_vec: Vec8 = B[idx * NR ..][0..8].*;

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
        const b_vec: Vec8 = B[kk * NR ..][0..8].*;
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

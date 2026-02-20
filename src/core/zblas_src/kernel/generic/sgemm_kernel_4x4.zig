// zblas/src/kernel/generic/sgemm_kernel_4x4.zig
// Portable 4×4 micro-kernel using Zig vectors
//
// This is the innermost computation of GEMM. It computes a 4×4 block of C
// by accumulating outer products of 4-element vectors from A and B.
//
// The micro-kernel assumes packed input data for sequential memory access.

const std = @import("std");

/// Micro-kernel register tile dimensions
pub const mr = 4;
pub const nr = 4;

const MR = mr;
const NR = nr;

/// 4×4 Micro-kernel: C[4×4] += alpha * A_packed[4×k] * B_packed[k×4]
///
/// This computes the outer product accumulation for a 4×4 tile of C.
/// Uses Zig's @Vector for portable SIMD across architectures.
///
/// Parameters:
///   k: Reduction dimension (number of A columns / B rows)
///   A: Packed A panel pointer (MR elements per k iteration)
///   B: Packed B panel pointer (NR elements per k iteration)
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
    const Vec4 = @Vector(4, f32);

    // 4 accumulator vectors = 16 elements (4×4 block)
    // Each vector holds one row of the C tile
    var c0: Vec4 = @splat(0.0);
    var c1: Vec4 = @splat(0.0);
    var c2: Vec4 = @splat(0.0);
    var c3: Vec4 = @splat(0.0);

    // Reduction loop over k
    // Each iteration loads 4 elements from A (one column of the 4×k panel)
    // and 4 elements from B (one row of the k×4 panel), then computes
    // their outer product and accumulates into C
    for (0..k) |kk| {
        // Load 4 elements from packed A (column kk of 4×k panel)
        const a: Vec4 = A[kk * MR ..][0..MR].*;

        // Load 4 elements from packed B (row kk of k×4 panel)
        const b: Vec4 = B[kk * NR ..][0..NR].*;

        // Outer product: c[i][j] += a[i] * b[j]
        // Each a[i] is broadcast and multiplied with entire b vector
        c0 += @as(Vec4, @splat(a[0])) * b;
        c1 += @as(Vec4, @splat(a[1])) * b;
        c2 += @as(Vec4, @splat(a[2])) * b;
        c3 += @as(Vec4, @splat(a[3])) * b;
    }

    // Apply alpha and accumulate to C
    const alpha_vec: Vec4 = @splat(alpha);

    // Load existing C values, add our contribution, store back
    // C = C + alpha * (A * B)
    const c0_ptr: *[4]f32 = @ptrCast(C);
    const c1_ptr: *[4]f32 = @ptrCast(C + ldc);
    const c2_ptr: *[4]f32 = @ptrCast(C + 2 * ldc);
    const c3_ptr: *[4]f32 = @ptrCast(C + 3 * ldc);

    c0_ptr.* = @as(Vec4, c0_ptr.*) + c0 * alpha_vec;
    c1_ptr.* = @as(Vec4, c1_ptr.*) + c1 * alpha_vec;
    c2_ptr.* = @as(Vec4, c2_ptr.*) + c2 * alpha_vec;
    c3_ptr.* = @as(Vec4, c3_ptr.*) + c3 * alpha_vec;
}

// ============================================================================
// Tests
// ============================================================================

test "kernel 4x4 identity" {
    // A = 4x4 identity, B = 4x4 values, C should equal B after multiplication
    // Pack A: column-major within each MR group
    // For identity matrix packed: each k iteration has only one non-zero
    const packed_a = [_]f32{
        // k=0: col 0 of A
        1, 0, 0, 0,
        // k=1: col 1 of A
        0, 1, 0, 0,
        // k=2: col 2 of A
        0, 0, 1, 0,
        // k=3: col 3 of A
        0, 0, 0, 1,
    };

    // B = [1,2,3,4; 5,6,7,8; 9,10,11,12; 13,14,15,16]
    // Pack B: row-major within each NR group
    const packed_b = [_]f32{
        1,  2,  3,  4, // k=0
        5,  6,  7,  8, // k=1
        9,  10, 11, 12, // k=2
        13, 14, 15, 16, // k=3
    };

    var C = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    kernel(4, &packed_a, &packed_b, &C, 4, 1.0);

    // C should equal B
    const expected = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    for (0..16) |i| {
        try std.testing.expectApproxEqRel(C[i], expected[i], 1e-5);
    }
}

test "kernel 4x4 basic multiply" {
    // A = [1,0,0,0; 0,1,0,0; 0,0,1,0; 0,0,0,1] but let's use simpler values
    // A = [[1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1]] (all ones)
    // B = [[1,2,3,4], [1,2,3,4], [1,2,3,4], [1,2,3,4]]
    // A*B = [[4,8,12,16], [4,8,12,16], [4,8,12,16], [4,8,12,16]]

    const packed_a = [_]f32{
        // k=0 through k=3, each column of A is all 1s
        1, 1, 1, 1, // k=0
        1, 1, 1, 1, // k=1
        1, 1, 1, 1, // k=2
        1, 1, 1, 1, // k=3
    };

    const packed_b = [_]f32{
        1, 2, 3, 4, // k=0
        1, 2, 3, 4, // k=1
        1, 2, 3, 4, // k=2
        1, 2, 3, 4, // k=3
    };

    var C = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    kernel(4, &packed_a, &packed_b, &C, 4, 1.0);

    // Each row should be [4, 8, 12, 16]
    const expected = [_]f32{ 4, 8, 12, 16, 4, 8, 12, 16, 4, 8, 12, 16, 4, 8, 12, 16 };
    for (0..16) |i| {
        try std.testing.expectApproxEqRel(C[i], expected[i], 1e-5);
    }
}

test "kernel with alpha" {
    // Same as above but with alpha = 2.0
    const packed_a = [_]f32{
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    };

    const packed_b = [_]f32{
        1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
    };

    var C = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    kernel(4, &packed_a, &packed_b, &C, 4, 2.0);

    // Each row should be [8, 16, 24, 32] (doubled)
    const expected = [_]f32{ 8, 16, 24, 32, 8, 16, 24, 32, 8, 16, 24, 32, 8, 16, 24, 32 };
    for (0..16) |i| {
        try std.testing.expectApproxEqRel(C[i], expected[i], 1e-5);
    }
}

test "kernel accumulates to C" {
    // Test that kernel adds to existing C values
    const packed_a = [_]f32{
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    const packed_b = [_]f32{
        1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    // Start with C = all 10s
    var C = [_]f32{ 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10 };

    kernel(4, &packed_a, &packed_b, &C, 4, 1.0);

    // First row gets +1 to each element, rest stays 10
    try std.testing.expectApproxEqRel(C[0], 11.0, 1e-5);
    try std.testing.expectApproxEqRel(C[1], 11.0, 1e-5);
    try std.testing.expectApproxEqRel(C[2], 11.0, 1e-5);
    try std.testing.expectApproxEqRel(C[3], 11.0, 1e-5);
    try std.testing.expectApproxEqRel(C[4], 10.0, 1e-5);
}

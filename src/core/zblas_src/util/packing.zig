// zblas/src/util/packing.zig
// Matrix packing for cache-blocked GEMM
//
// Packing rearranges matrix data so micro-kernels access memory sequentially,
// maximizing cache utilization and avoiding TLB misses.
//
// Architecture-specific implementations:
// - x86_64: Uses AVX2 vector loads/stores for 8-wide packing
// - ARM64: Uses NEON vector loads/stores for 8-wide packing
// - Generic: Scalar fallback for other architectures

const std = @import("std");
const builtin = @import("builtin");

// Architecture-specific packing implementations
const has_arch_packing = builtin.cpu.arch == .x86_64 or builtin.cpu.arch == .aarch64;

const arch_packing_x86_64 = if (builtin.cpu.arch == .x86_64) @import("packing_x86_64.zig") else struct {};
const arch_packing_arm64 = if (builtin.cpu.arch == .aarch64) @import("packing_arm64.zig") else struct {};

/// Pack A into MR-wide column panels for cache-efficient access.
///
/// A is [m × k] in row-major, packed becomes [⌈m/MR⌉ * MR × k] with
/// elements rearranged so each MR×k panel is contiguous.
///
/// Memory layout after packing (for MR=4):
///   Panel 0: [a00 a10 a20 a30] [a01 a11 a21 a31] ... [a0,k-1 a1,k-1 a2,k-1 a3,k-1]
///   Panel 1: [a40 a50 a60 a70] [a41 a51 a61 a71] ...
///
/// This allows the micro-kernel to load MR elements with a single vector load
/// for each column of the A panel.
///
/// Parameters:
///   A: Source matrix pointer (row-major)
///   lda: Leading dimension of A (stride between rows)
///   packed: Destination buffer for packed data
///   m: Number of rows to pack
///   k: Number of columns to pack
///   mr: Micro-kernel row count (panel width)
///
pub fn packA(
    A: []const f32,
    lda: usize,
    dest: []f32,
    m: usize,
    k: usize,
    mr: usize,
) void {
    // Use architecture-optimized version for MR=8
    if (has_arch_packing and mr == 8) {
        if (builtin.cpu.arch == .x86_64) {
            arch_packing_x86_64.packA8(A.ptr, lda, dest.ptr, m, k);
        } else if (builtin.cpu.arch == .aarch64) {
            arch_packing_arm64.packA8(A.ptr, lda, dest.ptr, m, k);
        }
        return;
    }

    // Generic fallback implementation
    packAGeneric(A, lda, dest, m, k, mr);
}

/// Generic (scalar) packA implementation
fn packAGeneric(
    A: []const f32,
    lda: usize,
    dest: []f32,
    m: usize,
    k: usize,
    mr: usize,
) void {
    var dest_idx: usize = 0;

    // Process full MR-wide panels
    var i: usize = 0;
    while (i + mr <= m) : (i += mr) {
        // For each column in the panel
        for (0..k) |kk| {
            // Pack MR elements from column kk of rows i..i+mr
            for (0..mr) |ii| {
                dest[dest_idx] = A[(i + ii) * lda + kk];
                dest_idx += 1;
            }
        }
    }

    // Handle remaining rows (partial panel)
    const remaining = m - i;
    if (remaining > 0) {
        for (0..k) |kk| {
            // Pack actual remaining elements
            for (0..remaining) |ii| {
                dest[dest_idx] = A[(i + ii) * lda + kk];
                dest_idx += 1;
            }
            // Zero-pad to full MR width
            for (remaining..mr) |_| {
                dest[dest_idx] = 0.0;
                dest_idx += 1;
            }
        }
    }
}

/// Pack B into NR-wide row panels for cache-efficient access.
///
/// B is [k × n] in row-major, packed becomes [k × ⌈n/NR⌉ * NR] with
/// elements rearranged so each k×NR panel is contiguous.
///
/// Memory layout after packing (for NR=4):
///   Panel 0: [b00 b01 b02 b03] [b10 b11 b12 b13] ... [bk-1,0 bk-1,1 bk-1,2 bk-1,3]
///   Panel 1: [b04 b05 b06 b07] [b14 b15 b16 b17] ...
///
/// This allows the micro-kernel to load NR elements with a single vector load
/// for each row of the B panel.
///
/// Parameters:
///   B: Source matrix pointer (row-major)
///   ldb: Leading dimension of B (stride between rows)
///   packed: Destination buffer for packed data
///   k: Number of rows to pack
///   n: Number of columns to pack
///   nr: Micro-kernel column count (panel width)
///
pub fn packB(
    B: []const f32,
    ldb: usize,
    dest: []f32,
    k: usize,
    n: usize,
    nr: usize,
) void {
    // Use architecture-optimized version for NR=8
    if (has_arch_packing and nr == 8) {
        if (builtin.cpu.arch == .x86_64) {
            arch_packing_x86_64.packB8(B.ptr, ldb, dest.ptr, k, n);
        } else if (builtin.cpu.arch == .aarch64) {
            arch_packing_arm64.packB8(B.ptr, ldb, dest.ptr, k, n);
        }
        return;
    }

    // Generic fallback implementation
    packBGeneric(B, ldb, dest, k, n, nr);
}

/// Generic (scalar) packB implementation
fn packBGeneric(
    B: []const f32,
    ldb: usize,
    dest: []f32,
    k: usize,
    n: usize,
    nr: usize,
) void {
    var dest_idx: usize = 0;

    // Process full NR-wide panels
    var j: usize = 0;
    while (j + nr <= n) : (j += nr) {
        // For each row in the panel
        for (0..k) |kk| {
            // Pack NR elements from row kk, columns j..j+nr
            for (0..nr) |jj| {
                dest[dest_idx] = B[kk * ldb + (j + jj)];
                dest_idx += 1;
            }
        }
    }

    // Handle remaining columns (partial panel)
    const remaining = n - j;
    if (remaining > 0) {
        for (0..k) |kk| {
            // Pack actual remaining elements
            for (0..remaining) |jj| {
                dest[dest_idx] = B[kk * ldb + (j + jj)];
                dest_idx += 1;
            }
            // Zero-pad to full NR width
            for (remaining..nr) |_| {
                dest[dest_idx] = 0.0;
                dest_idx += 1;
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "packA basic 4x4" {
    // A = [1, 2, 3, 4;
    //      5, 6, 7, 8;
    //      9, 10, 11, 12;
    //      13, 14, 15, 16]
    const A = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    var result: [16]f32 = undefined;

    packA(&A, 4, &result, 4, 4, 4);

    // After packing with MR=4:
    // Column 0: [1, 5, 9, 13]
    // Column 1: [2, 6, 10, 14]
    // Column 2: [3, 7, 11, 15]
    // Column 3: [4, 8, 12, 16]
    const expected = [_]f32{ 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16 };

    for (0..16) |i| {
        try std.testing.expectApproxEqRel(result[i], expected[i], 1e-5);
    }
}

test "packA with partial panel" {
    // A = [1, 2;
    //      3, 4;
    //      5, 6] (3 rows, 2 cols, MR=4)
    const A = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var result: [8]f32 = undefined;

    packA(&A, 2, &result, 3, 2, 4);

    // After packing with MR=4:
    // Column 0: [1, 3, 5, 0] (zero-padded)
    // Column 1: [2, 4, 6, 0]
    const expected = [_]f32{ 1, 3, 5, 0, 2, 4, 6, 0 };

    for (0..8) |i| {
        try std.testing.expectApproxEqRel(result[i], expected[i], 1e-5);
    }
}

test "packB basic 4x4" {
    // B = [1, 2, 3, 4;
    //      5, 6, 7, 8;
    //      9, 10, 11, 12;
    //      13, 14, 15, 16]
    const B = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    var result: [16]f32 = undefined;

    packB(&B, 4, &result, 4, 4, 4);

    // After packing with NR=4:
    // Row 0: [1, 2, 3, 4]
    // Row 1: [5, 6, 7, 8]
    // Row 2: [9, 10, 11, 12]
    // Row 3: [13, 14, 15, 16]
    const expected = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };

    for (0..16) |i| {
        try std.testing.expectApproxEqRel(result[i], expected[i], 1e-5);
    }
}

test "packB with partial panel" {
    // B = [1, 2, 3;
    //      4, 5, 6] (2 rows, 3 cols, NR=4)
    const B = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var result: [8]f32 = undefined;

    packB(&B, 3, &result, 2, 3, 4);

    // After packing with NR=4:
    // Row 0: [1, 2, 3, 0] (zero-padded)
    // Row 1: [4, 5, 6, 0]
    const expected = [_]f32{ 1, 2, 3, 0, 4, 5, 6, 0 };

    for (0..8) |i| {
        try std.testing.expectApproxEqRel(result[i], expected[i], 1e-5);
    }
}

test "packA with lda larger than k" {
    // A stored with lda=5 but only using k=3 columns
    // A = [1, 2, 3, _, _;
    //      4, 5, 6, _, _] (2 rows)
    const A = [_]f32{ 1, 2, 3, 0, 0, 4, 5, 6, 0, 0 };
    var result: [12]f32 = undefined;

    packA(&A, 5, &result, 2, 3, 4);

    // After packing with MR=4 (2 rows + 2 zeros):
    // Column 0: [1, 4, 0, 0]
    // Column 1: [2, 5, 0, 0]
    // Column 2: [3, 6, 0, 0]
    const expected = [_]f32{ 1, 4, 0, 0, 2, 5, 0, 0, 3, 6, 0, 0 };

    for (0..12) |i| {
        try std.testing.expectApproxEqRel(result[i], expected[i], 1e-5);
    }
}

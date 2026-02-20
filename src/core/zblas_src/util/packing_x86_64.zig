// zblas/src/util/packing_x86_64.zig
// x86-64 AVX2-optimized packing routines for GEMM
//
// Packing can be 20-30% of GEMM time. These routines use:
// - Multi-row/column processing (8 at a time)
// - SIMD vector loads/stores
// - Prefetching for next blocks
// - Register blocking to maximize cache line utilization

const std = @import("std");

const Vec8 = @Vector(8, f32);
const Vec4 = @Vector(4, f32);

/// Pack 8 rows of A into column-major panel format (MR=8)
/// Source: row-major A[m × k], stride lda
/// Dest: column-major packed[k × 8] for each panel
///
/// Memory layout after packing:
///   For each k: [a[0,k], a[1,k], a[2,k], a[3,k], a[4,k], a[5,k], a[6,k], a[7,k]]
///
/// This allows the micro-kernel to load 8 consecutive floats per k iteration.
pub fn packA8(
    src: [*]const f32,
    lda: usize,
    dest: [*]f32,
    m: usize,
    k: usize,
) void {
    const MR = 8;
    var panel_offset: usize = 0;

    // Process full 8-row panels
    var row: usize = 0;
    while (row + MR <= m) : (row += MR) {
        // 8 row pointers for current panel
        const r0 = src + (row + 0) * lda;
        const r1 = src + (row + 1) * lda;
        const r2 = src + (row + 2) * lda;
        const r3 = src + (row + 3) * lda;
        const r4 = src + (row + 4) * lda;
        const r5 = src + (row + 5) * lda;
        const r6 = src + (row + 6) * lda;
        const r7 = src + (row + 7) * lda;

        var col: usize = 0;

        // Process 8 columns at a time for best cache utilization
        // This loads 8x8 blocks and stores them transposed
        while (col + 8 <= k) : (col += 8) {
            // Prefetch next cache lines
            @prefetch(r0 + col + 64, .{ .locality = 3, .cache = .data });
            @prefetch(r4 + col + 64, .{ .locality = 3, .cache = .data });

            // Load 8 elements from each of 8 rows
            const v0: Vec8 = r0[col..][0..8].*;
            const v1: Vec8 = r1[col..][0..8].*;
            const v2: Vec8 = r2[col..][0..8].*;
            const v3: Vec8 = r3[col..][0..8].*;
            const v4: Vec8 = r4[col..][0..8].*;
            const v5: Vec8 = r5[col..][0..8].*;
            const v6: Vec8 = r6[col..][0..8].*;
            const v7: Vec8 = r7[col..][0..8].*;

            // Transpose 8x8 and store
            // After transpose, column c of input becomes row c of output
            // We want: dest[col+c][row+r] = src[row+r][col+c]
            // In packed format: dest[(col+c)*8 + r] = src[row+r][col+c]
            const base = panel_offset + col * MR;

            // Store transposed - column 0 from all 8 rows
            dest[base + 0] = v0[0];
            dest[base + 1] = v1[0];
            dest[base + 2] = v2[0];
            dest[base + 3] = v3[0];
            dest[base + 4] = v4[0];
            dest[base + 5] = v5[0];
            dest[base + 6] = v6[0];
            dest[base + 7] = v7[0];

            // Column 1
            dest[base + 8] = v0[1];
            dest[base + 9] = v1[1];
            dest[base + 10] = v2[1];
            dest[base + 11] = v3[1];
            dest[base + 12] = v4[1];
            dest[base + 13] = v5[1];
            dest[base + 14] = v6[1];
            dest[base + 15] = v7[1];

            // Column 2
            dest[base + 16] = v0[2];
            dest[base + 17] = v1[2];
            dest[base + 18] = v2[2];
            dest[base + 19] = v3[2];
            dest[base + 20] = v4[2];
            dest[base + 21] = v5[2];
            dest[base + 22] = v6[2];
            dest[base + 23] = v7[2];

            // Column 3
            dest[base + 24] = v0[3];
            dest[base + 25] = v1[3];
            dest[base + 26] = v2[3];
            dest[base + 27] = v3[3];
            dest[base + 28] = v4[3];
            dest[base + 29] = v5[3];
            dest[base + 30] = v6[3];
            dest[base + 31] = v7[3];

            // Column 4
            dest[base + 32] = v0[4];
            dest[base + 33] = v1[4];
            dest[base + 34] = v2[4];
            dest[base + 35] = v3[4];
            dest[base + 36] = v4[4];
            dest[base + 37] = v5[4];
            dest[base + 38] = v6[4];
            dest[base + 39] = v7[4];

            // Column 5
            dest[base + 40] = v0[5];
            dest[base + 41] = v1[5];
            dest[base + 42] = v2[5];
            dest[base + 43] = v3[5];
            dest[base + 44] = v4[5];
            dest[base + 45] = v5[5];
            dest[base + 46] = v6[5];
            dest[base + 47] = v7[5];

            // Column 6
            dest[base + 48] = v0[6];
            dest[base + 49] = v1[6];
            dest[base + 50] = v2[6];
            dest[base + 51] = v3[6];
            dest[base + 52] = v4[6];
            dest[base + 53] = v5[6];
            dest[base + 54] = v6[6];
            dest[base + 55] = v7[6];

            // Column 7
            dest[base + 56] = v0[7];
            dest[base + 57] = v1[7];
            dest[base + 58] = v2[7];
            dest[base + 59] = v3[7];
            dest[base + 60] = v4[7];
            dest[base + 61] = v5[7];
            dest[base + 62] = v6[7];
            dest[base + 63] = v7[7];
        }

        // Process 4 columns at a time
        while (col + 4 <= k) : (col += 4) {
            const base = panel_offset + col * MR;

            // Load 4 elements from each row
            const v0: Vec4 = r0[col..][0..4].*;
            const v1: Vec4 = r1[col..][0..4].*;
            const v2: Vec4 = r2[col..][0..4].*;
            const v3: Vec4 = r3[col..][0..4].*;
            const v4: Vec4 = r4[col..][0..4].*;
            const v5: Vec4 = r5[col..][0..4].*;
            const v6: Vec4 = r6[col..][0..4].*;
            const v7: Vec4 = r7[col..][0..4].*;

            // Store transposed - 4 columns
            // Column 0
            dest[base + 0] = v0[0];
            dest[base + 1] = v1[0];
            dest[base + 2] = v2[0];
            dest[base + 3] = v3[0];
            dest[base + 4] = v4[0];
            dest[base + 5] = v5[0];
            dest[base + 6] = v6[0];
            dest[base + 7] = v7[0];

            // Column 1
            dest[base + 8] = v0[1];
            dest[base + 9] = v1[1];
            dest[base + 10] = v2[1];
            dest[base + 11] = v3[1];
            dest[base + 12] = v4[1];
            dest[base + 13] = v5[1];
            dest[base + 14] = v6[1];
            dest[base + 15] = v7[1];

            // Column 2
            dest[base + 16] = v0[2];
            dest[base + 17] = v1[2];
            dest[base + 18] = v2[2];
            dest[base + 19] = v3[2];
            dest[base + 20] = v4[2];
            dest[base + 21] = v5[2];
            dest[base + 22] = v6[2];
            dest[base + 23] = v7[2];

            // Column 3
            dest[base + 24] = v0[3];
            dest[base + 25] = v1[3];
            dest[base + 26] = v2[3];
            dest[base + 27] = v3[3];
            dest[base + 28] = v4[3];
            dest[base + 29] = v5[3];
            dest[base + 30] = v6[3];
            dest[base + 31] = v7[3];
        }

        // Handle remaining columns (< 4) one at a time
        while (col < k) : (col += 1) {
            const base = panel_offset + col * MR;
            dest[base + 0] = r0[col];
            dest[base + 1] = r1[col];
            dest[base + 2] = r2[col];
            dest[base + 3] = r3[col];
            dest[base + 4] = r4[col];
            dest[base + 5] = r5[col];
            dest[base + 6] = r6[col];
            dest[base + 7] = r7[col];
        }

        panel_offset += k * MR;
    }

    // Handle remaining rows (< 8) with zero padding
    const remaining = m - row;
    if (remaining > 0) {
        for (0..k) |col| {
            const base = panel_offset + col * MR;

            // Pack actual remaining elements
            for (0..remaining) |ii| {
                dest[base + ii] = src[(row + ii) * lda + col];
            }
            // Zero-pad to full MR width
            for (remaining..MR) |ii| {
                dest[base + ii] = 0.0;
            }
        }
    }
}

/// Pack 8 columns of B into row-major panel format (NR=8)
/// Source: row-major B[k × n], stride ldb
/// Dest: row-major packed[k × 8] for each panel
///
/// Memory layout after packing:
///   For each k: [b[k,0], b[k,1], b[k,2], b[k,3], b[k,4], b[k,5], b[k,6], b[k,7]]
///
/// This is simpler than packA since B is already row-major and we just need
/// to extract 8-column panels. Each row becomes a contiguous Vec8.
pub fn packB8(
    src: [*]const f32,
    ldb: usize,
    dest: [*]f32,
    k: usize,
    n: usize,
) void {
    const NR = 8;
    var panel_offset: usize = 0;

    // Process full 8-column panels
    var col: usize = 0;
    while (col + NR <= n) : (col += NR) {
        // Process 4 rows at a time for better cache utilization
        var row: usize = 0;
        while (row + 4 <= k) : (row += 4) {
            const base = panel_offset + row * NR;

            // Prefetch next rows
            @prefetch(src + (row + 8) * ldb + col, .{ .locality = 3, .cache = .data });

            // Load 8 consecutive elements from each of 4 rows
            const v0: Vec8 = src[(row + 0) * ldb + col ..][0..8].*;
            const v1: Vec8 = src[(row + 1) * ldb + col ..][0..8].*;
            const v2: Vec8 = src[(row + 2) * ldb + col ..][0..8].*;
            const v3: Vec8 = src[(row + 3) * ldb + col ..][0..8].*;

            // Store directly - already in the right order
            dest[base ..][0..8].* = v0;
            dest[base + 8 ..][0..8].* = v1;
            dest[base + 16 ..][0..8].* = v2;
            dest[base + 24 ..][0..8].* = v3;
        }

        // Handle remaining rows
        while (row < k) : (row += 1) {
            const base = panel_offset + row * NR;
            const v: Vec8 = src[row * ldb + col ..][0..8].*;
            dest[base ..][0..8].* = v;
        }

        panel_offset += k * NR;
    }

    // Handle remaining columns (< 8) with zero padding
    const remaining = n - col;
    if (remaining > 0) {
        for (0..k) |row| {
            const base = panel_offset + row * NR;

            // Pack actual remaining elements
            for (0..remaining) |jj| {
                dest[base + jj] = src[row * ldb + col + jj];
            }
            // Zero-pad to full NR width
            for (remaining..NR) |jj| {
                dest[base + jj] = 0.0;
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "packA8 basic 8x8" {
    // A = 8x8 sequential values
    var A: [64]f32 = undefined;
    for (0..64) |i| {
        A[i] = @floatFromInt(i + 1);
    }

    var result: [64]f32 = undefined;
    packA8(&A, 8, &result, 8, 8);

    // After packing with MR=8:
    // Column 0: [1, 9, 17, 25, 33, 41, 49, 57]
    // Column 1: [2, 10, 18, 26, 34, 42, 50, 58]
    // etc.
    for (0..8) |col| {
        for (0..8) |row| {
            const expected: f32 = @floatFromInt(row * 8 + col + 1);
            try std.testing.expectApproxEqRel(result[col * 8 + row], expected, 1e-5);
        }
    }
}

test "packA8 with partial rows" {
    // A = 5x4 matrix (5 rows, 4 columns)
    var A: [20]f32 = undefined;
    for (0..20) |i| {
        A[i] = @floatFromInt(i + 1);
    }

    // Need space for 8x4 (padded)
    var result: [32]f32 = undefined;
    packA8(&A, 4, &result, 5, 4);

    // After packing with MR=8, 5 rows + 3 zero-padded:
    // Column 0: [1, 5, 9, 13, 17, 0, 0, 0]
    // Column 1: [2, 6, 10, 14, 18, 0, 0, 0]
    // etc.
    for (0..4) |col| {
        for (0..5) |row| {
            const expected: f32 = @floatFromInt(row * 4 + col + 1);
            try std.testing.expectApproxEqRel(result[col * 8 + row], expected, 1e-5);
        }
        // Check zero padding
        for (5..8) |row| {
            try std.testing.expectApproxEqRel(result[col * 8 + row], 0.0, 1e-5);
        }
    }
}

test "packB8 basic 8x8" {
    // B = 8x8 sequential values
    var B: [64]f32 = undefined;
    for (0..64) |i| {
        B[i] = @floatFromInt(i + 1);
    }

    var result: [64]f32 = undefined;
    packB8(&B, 8, &result, 8, 8);

    // After packing with NR=8 (B is row-major, pack extracts 8-col panels):
    // Row 0: [1, 2, 3, 4, 5, 6, 7, 8]
    // Row 1: [9, 10, 11, 12, 13, 14, 15, 16]
    // etc.
    // Since we're packing first 8 columns (all columns), output equals input
    for (0..64) |i| {
        const expected: f32 = @floatFromInt(i + 1);
        try std.testing.expectApproxEqRel(result[i], expected, 1e-5);
    }
}

test "packB8 with partial columns" {
    // B = 4x5 matrix (4 rows, 5 columns)
    var B: [20]f32 = undefined;
    for (0..20) |i| {
        B[i] = @floatFromInt(i + 1);
    }

    // Need space for 4x8 (padded)
    var result: [32]f32 = undefined;
    packB8(&B, 5, &result, 4, 5);

    // After packing with NR=8, 5 cols + 3 zero-padded:
    // Row 0: [1, 2, 3, 4, 5, 0, 0, 0]
    // Row 1: [6, 7, 8, 9, 10, 0, 0, 0]
    // etc.
    for (0..4) |row| {
        for (0..5) |col| {
            const expected: f32 = @floatFromInt(row * 5 + col + 1);
            try std.testing.expectApproxEqRel(result[row * 8 + col], expected, 1e-5);
        }
        // Check zero padding
        for (5..8) |col| {
            try std.testing.expectApproxEqRel(result[row * 8 + col], 0.0, 1e-5);
        }
    }
}

test "packA8 matches generic packing" {
    const packing = @import("packing.zig");

    // Random-ish test matrix
    var A: [256]f32 = undefined;
    for (0..256) |i| {
        A[i] = @as(f32, @floatFromInt((i * 7 + 13) % 100)) / 10.0;
    }

    var result_generic: [256]f32 = undefined;
    var result_x86: [256]f32 = undefined;

    packing.packA(&A, 16, &result_generic, 16, 16, 8);
    packA8(&A, 16, &result_x86, 16, 16);

    for (0..256) |i| {
        try std.testing.expectApproxEqRel(result_x86[i], result_generic[i], 1e-5);
    }
}

test "packB8 matches generic packing" {
    const packing = @import("packing.zig");

    // Random-ish test matrix
    var B: [256]f32 = undefined;
    for (0..256) |i| {
        B[i] = @as(f32, @floatFromInt((i * 7 + 13) % 100)) / 10.0;
    }

    var result_generic: [256]f32 = undefined;
    var result_x86: [256]f32 = undefined;

    packing.packB(&B, 16, &result_generic, 16, 16, 8);
    packB8(&B, 16, &result_x86, 16, 16);

    for (0..256) |i| {
        try std.testing.expectApproxEqRel(result_x86[i], result_generic[i], 1e-5);
    }
}

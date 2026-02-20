// zblas/src/level3/sgemm.zig
// Single-precision General Matrix Multiply (SGEMM)
//
// This is the main SGEMM dispatcher that selects between:
// - Reference implementation (for small matrices or testing)
// - Cache-blocked implementation with optimized micro-kernels
//
// Phase 2: Cache-blocked implementation with 4x4 generic micro-kernel

const std = @import("std");
const builtin = @import("builtin");
const config = @import("../config.zig");
const reference = @import("../reference.zig");
const packing = @import("../util/packing.zig");

// Select micro-kernel based on architecture
const micro_kernel = switch (builtin.cpu.arch) {
    .x86_64 => @import("../kernel/x86_64/sgemm_kernel_8x8.zig"),
    .aarch64 => @import("../kernel/arm64/sgemm_kernel_8x8.zig"),
    else => @import("../kernel/generic/sgemm_kernel_4x4.zig"),
};

// Fused kernel for x86_64 (packs B on-the-fly during computation)
const fused_kernel_impl = @import("../kernel/x86_64/sgemm_kernel_fused.zig");
const has_fused_kernel = builtin.cpu.arch == .x86_64;

// Fast 4x24 kernel for direct (non-packed) path on x86_64
const fast_kernel = switch (builtin.cpu.arch) {
    .x86_64 => @import("../kernel/x86_64/sgemm_kernel_4x24.zig"),
    else => null,
};

// Re-export Transpose type from parent
pub const Transpose = @import("../zblas.zig").Transpose;

/// SGEMM: C = alpha * A * B + beta * C
/// A is [M x K], B is [K x N], C is [M x N], all row-major
pub inline fn sgemm(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    B: []const f32,
    C: []f32,
    alpha: f32,
    beta: f32,
) void {
    // Validate input sizes
    if (M == 0 or N == 0 or K == 0) return;
    std.debug.assert(A.len >= M * K);
    std.debug.assert(B.len >= K * N);
    std.debug.assert(C.len >= M * N);

    // Small matrices: use simple reference implementation
    if (M * N * K < config.MIN_OPTIMIZED_SIZE * config.MIN_OPTIMIZED_SIZE * config.MIN_OPTIMIZED_SIZE) {
        reference.sgemm_reference_simple(M, N, K, A, B, C, alpha, beta);
        return;
    }

    // Algorithm selection based on matrix size and shape
    // Key insight from benchmarks:
    // - Direct path works well for small/medium matrices (≤512 max dim)
    // - Blocked path is much faster for larger matrices due to cache reuse
    // - Skinny matrices (M or N < 64) benefit from simpler row-by-row approach
    const max_dim = @max(M, @max(N, K));
    const min_dim = @min(M, @min(N, K));

    // Use direct path for small matrices or very skinny matrices
    // where packing overhead outweighs cache benefits
    const use_direct = blk: {
        // Very small: direct is always better (no packing overhead)
        if (max_dim <= 128) break :blk true;

        // Skinny matrices: direct is better (less data reuse to amortize packing)
        if (min_dim <= 32) break :blk true;

        // Medium matrices: use direct if total work is small enough
        // that packing overhead dominates
        const work = M * N * K;
        if (work <= 256 * 256 * 256) break :blk true;

        // Larger matrices: blocked path with packing wins
        break :blk false;
    };

    if (use_direct) {
        sgemmDirect(M, N, K, A, K, B, N, C, N, alpha, beta);
        return;
    }

    // Larger matrices: use cache-blocked implementation with packing
    sgemmOptimized(M, N, K, A, K, B, N, C, N, alpha, beta);
}

/// Direct SGEMM without packing - faster for medium matrices
/// Uses 4x24 micro-kernel on x86_64, similar to Tomoul's ops.zig fallback
inline fn sgemmDirect(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    lda: usize,
    B: []const f32,
    ldb: usize,
    C: []f32,
    ldc: usize,
    alpha: f32,
    beta: f32,
) void {
    // FAST PATH: alpha=1.0, beta=0.0 (common in inference)
    // Avoids read-modify-write, just direct stores
    if (alpha == 1.0 and beta == 0.0) {
        sgemmDirectFast(M, N, K, A, lda, B, ldb, C, ldc);
        return;
    }

    const VEC_WIDTH = 8;
    const Vec = @Vector(VEC_WIDTH, f32);

    // Scale C by beta first
    if (beta == 0.0) {
        for (0..M) |i| {
            @memset(C[i * ldc ..][0..N], 0.0);
        }
    } else if (beta != 1.0) {
        const beta_vec: Vec = @splat(beta);
        for (0..M) |i| {
            var j: usize = 0;
            while (j + VEC_WIDTH <= N) : (j += VEC_WIDTH) {
                const c_vec: Vec = C[i * ldc + j ..][0..VEC_WIDTH].*;
                C[i * ldc + j ..][0..VEC_WIDTH].* = c_vec * beta_vec;
            }
            while (j < N) : (j += 1) {
                C[i * ldc + j] *= beta;
            }
        }
    }

    if (alpha == 0.0) return;

    // Process 4 rows at a time, 24 columns at a time (like Tomoul fallback)
    const MR = 4;
    const NR = 24;

    var i: usize = 0;
    while (i + MR <= M) : (i += MR) {
        var j: usize = 0;

        // Main loop: 4x24 tiles
        while (j + NR <= N) : (j += NR) {
            // Accumulators for 4x24 block (12 vectors)
            var c00: Vec = @splat(0.0);
            var c01: Vec = @splat(0.0);
            var c02: Vec = @splat(0.0);
            var c10: Vec = @splat(0.0);
            var c11: Vec = @splat(0.0);
            var c12: Vec = @splat(0.0);
            var c20: Vec = @splat(0.0);
            var c21: Vec = @splat(0.0);
            var c22: Vec = @splat(0.0);
            var c30: Vec = @splat(0.0);
            var c31: Vec = @splat(0.0);
            var c32: Vec = @splat(0.0);

            // Reduction over K
            for (0..K) |kk| {
                // Broadcast A elements
                const a0: Vec = @splat(A[(i + 0) * lda + kk]);
                const a1: Vec = @splat(A[(i + 1) * lda + kk]);
                const a2: Vec = @splat(A[(i + 2) * lda + kk]);
                const a3: Vec = @splat(A[(i + 3) * lda + kk]);

                // Load B vectors
                const b_base = kk * ldb + j;
                const b0: Vec = B[b_base ..][0..VEC_WIDTH].*;
                const b1: Vec = B[b_base + VEC_WIDTH ..][0..VEC_WIDTH].*;
                const b2: Vec = B[b_base + 2 * VEC_WIDTH ..][0..VEC_WIDTH].*;

                // Accumulate
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

            // Store with alpha scaling
            const alpha_vec: Vec = @splat(alpha);
            const c00_ptr = C[(i + 0) * ldc + j ..][0..VEC_WIDTH];
            const c01_ptr = C[(i + 0) * ldc + j + VEC_WIDTH ..][0..VEC_WIDTH];
            const c02_ptr = C[(i + 0) * ldc + j + 2 * VEC_WIDTH ..][0..VEC_WIDTH];
            const c10_ptr = C[(i + 1) * ldc + j ..][0..VEC_WIDTH];
            const c11_ptr = C[(i + 1) * ldc + j + VEC_WIDTH ..][0..VEC_WIDTH];
            const c12_ptr = C[(i + 1) * ldc + j + 2 * VEC_WIDTH ..][0..VEC_WIDTH];
            const c20_ptr = C[(i + 2) * ldc + j ..][0..VEC_WIDTH];
            const c21_ptr = C[(i + 2) * ldc + j + VEC_WIDTH ..][0..VEC_WIDTH];
            const c22_ptr = C[(i + 2) * ldc + j + 2 * VEC_WIDTH ..][0..VEC_WIDTH];
            const c30_ptr = C[(i + 3) * ldc + j ..][0..VEC_WIDTH];
            const c31_ptr = C[(i + 3) * ldc + j + VEC_WIDTH ..][0..VEC_WIDTH];
            const c32_ptr = C[(i + 3) * ldc + j + 2 * VEC_WIDTH ..][0..VEC_WIDTH];

            c00_ptr.* = @as(Vec, c00_ptr.*) + c00 * alpha_vec;
            c01_ptr.* = @as(Vec, c01_ptr.*) + c01 * alpha_vec;
            c02_ptr.* = @as(Vec, c02_ptr.*) + c02 * alpha_vec;
            c10_ptr.* = @as(Vec, c10_ptr.*) + c10 * alpha_vec;
            c11_ptr.* = @as(Vec, c11_ptr.*) + c11 * alpha_vec;
            c12_ptr.* = @as(Vec, c12_ptr.*) + c12 * alpha_vec;
            c20_ptr.* = @as(Vec, c20_ptr.*) + c20 * alpha_vec;
            c21_ptr.* = @as(Vec, c21_ptr.*) + c21 * alpha_vec;
            c22_ptr.* = @as(Vec, c22_ptr.*) + c22 * alpha_vec;
            c30_ptr.* = @as(Vec, c30_ptr.*) + c30 * alpha_vec;
            c31_ptr.* = @as(Vec, c31_ptr.*) + c31 * alpha_vec;
            c32_ptr.* = @as(Vec, c32_ptr.*) + c32 * alpha_vec;
        }

        // Handle remaining columns (8 at a time)
        while (j + VEC_WIDTH <= N) : (j += VEC_WIDTH) {
            var c0: Vec = @splat(0.0);
            var c1: Vec = @splat(0.0);
            var c2: Vec = @splat(0.0);
            var c3: Vec = @splat(0.0);

            for (0..K) |kk| {
                const b_vec: Vec = B[kk * ldb + j ..][0..VEC_WIDTH].*;
                c0 += @as(Vec, @splat(A[(i + 0) * lda + kk])) * b_vec;
                c1 += @as(Vec, @splat(A[(i + 1) * lda + kk])) * b_vec;
                c2 += @as(Vec, @splat(A[(i + 2) * lda + kk])) * b_vec;
                c3 += @as(Vec, @splat(A[(i + 3) * lda + kk])) * b_vec;
            }

            const alpha_vec: Vec = @splat(alpha);
            const c0_ptr = C[(i + 0) * ldc + j ..][0..VEC_WIDTH];
            const c1_ptr = C[(i + 1) * ldc + j ..][0..VEC_WIDTH];
            const c2_ptr = C[(i + 2) * ldc + j ..][0..VEC_WIDTH];
            const c3_ptr = C[(i + 3) * ldc + j ..][0..VEC_WIDTH];
            c0_ptr.* = @as(Vec, c0_ptr.*) + c0 * alpha_vec;
            c1_ptr.* = @as(Vec, c1_ptr.*) + c1 * alpha_vec;
            c2_ptr.* = @as(Vec, c2_ptr.*) + c2 * alpha_vec;
            c3_ptr.* = @as(Vec, c3_ptr.*) + c3 * alpha_vec;
        }

        // Scalar tail
        while (j < N) : (j += 1) {
            var c0: f32 = 0.0;
            var c1: f32 = 0.0;
            var c2: f32 = 0.0;
            var c3: f32 = 0.0;
            for (0..K) |kk| {
                const b_val = B[kk * ldb + j];
                c0 += A[(i + 0) * lda + kk] * b_val;
                c1 += A[(i + 1) * lda + kk] * b_val;
                c2 += A[(i + 2) * lda + kk] * b_val;
                c3 += A[(i + 3) * lda + kk] * b_val;
            }
            C[(i + 0) * ldc + j] += alpha * c0;
            C[(i + 1) * ldc + j] += alpha * c1;
            C[(i + 2) * ldc + j] += alpha * c2;
            C[(i + 3) * ldc + j] += alpha * c3;
        }
    }

    // Handle remaining rows (< 4)
    while (i < M) : (i += 1) {
        var j: usize = 0;

        while (j + VEC_WIDTH <= N) : (j += VEC_WIDTH) {
            var c0: Vec = @splat(0.0);
            for (0..K) |kk| {
                const a_val: Vec = @splat(A[i * lda + kk]);
                const b_vec: Vec = B[kk * ldb + j ..][0..VEC_WIDTH].*;
                c0 += a_val * b_vec;
            }
            const alpha_vec: Vec = @splat(alpha);
            const c0_ptr = C[i * ldc + j ..][0..VEC_WIDTH];
            c0_ptr.* = @as(Vec, c0_ptr.*) + c0 * alpha_vec;
        }

        while (j < N) : (j += 1) {
            var c0: f32 = 0.0;
            for (0..K) |kk| {
                c0 += A[i * lda + kk] * B[kk * ldb + j];
            }
            C[i * ldc + j] += alpha * c0;
        }
    }
}

/// Fast path SGEMM for alpha=1.0, beta=0.0 (inference workloads)
/// Direct stores without read-modify-write - significantly faster for memory-bound ops
inline fn sgemmDirectFast(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    lda: usize,
    B: []const f32,
    ldb: usize,
    C: []f32,
    ldc: usize,
) void {
    const VEC_WIDTH = 8;
    const Vec = @Vector(VEC_WIDTH, f32);

    const MR = 4;
    const NR = 24;

    var i: usize = 0;
    while (i + MR <= M) : (i += MR) {
        var j: usize = 0;

        // Main loop: 4x24 tiles
        while (j + NR <= N) : (j += NR) {
            var c00: Vec = @splat(0.0);
            var c01: Vec = @splat(0.0);
            var c02: Vec = @splat(0.0);
            var c10: Vec = @splat(0.0);
            var c11: Vec = @splat(0.0);
            var c12: Vec = @splat(0.0);
            var c20: Vec = @splat(0.0);
            var c21: Vec = @splat(0.0);
            var c22: Vec = @splat(0.0);
            var c30: Vec = @splat(0.0);
            var c31: Vec = @splat(0.0);
            var c32: Vec = @splat(0.0);

            for (0..K) |kk| {
                const a0: Vec = @splat(A[(i + 0) * lda + kk]);
                const a1: Vec = @splat(A[(i + 1) * lda + kk]);
                const a2: Vec = @splat(A[(i + 2) * lda + kk]);
                const a3: Vec = @splat(A[(i + 3) * lda + kk]);

                const b_base = kk * ldb + j;
                const b0: Vec = B[b_base ..][0..VEC_WIDTH].*;
                const b1: Vec = B[b_base + VEC_WIDTH ..][0..VEC_WIDTH].*;
                const b2: Vec = B[b_base + 2 * VEC_WIDTH ..][0..VEC_WIDTH].*;

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

            // FAST: Direct store without read (no alpha multiply, no add to existing C)
            C[(i + 0) * ldc + j ..][0..VEC_WIDTH].* = c00;
            C[(i + 0) * ldc + j + VEC_WIDTH ..][0..VEC_WIDTH].* = c01;
            C[(i + 0) * ldc + j + 2 * VEC_WIDTH ..][0..VEC_WIDTH].* = c02;
            C[(i + 1) * ldc + j ..][0..VEC_WIDTH].* = c10;
            C[(i + 1) * ldc + j + VEC_WIDTH ..][0..VEC_WIDTH].* = c11;
            C[(i + 1) * ldc + j + 2 * VEC_WIDTH ..][0..VEC_WIDTH].* = c12;
            C[(i + 2) * ldc + j ..][0..VEC_WIDTH].* = c20;
            C[(i + 2) * ldc + j + VEC_WIDTH ..][0..VEC_WIDTH].* = c21;
            C[(i + 2) * ldc + j + 2 * VEC_WIDTH ..][0..VEC_WIDTH].* = c22;
            C[(i + 3) * ldc + j ..][0..VEC_WIDTH].* = c30;
            C[(i + 3) * ldc + j + VEC_WIDTH ..][0..VEC_WIDTH].* = c31;
            C[(i + 3) * ldc + j + 2 * VEC_WIDTH ..][0..VEC_WIDTH].* = c32;
        }

        // Handle remaining columns (8 at a time)
        while (j + VEC_WIDTH <= N) : (j += VEC_WIDTH) {
            var c0: Vec = @splat(0.0);
            var c1: Vec = @splat(0.0);
            var c2: Vec = @splat(0.0);
            var c3: Vec = @splat(0.0);

            for (0..K) |kk| {
                const b_vec: Vec = B[kk * ldb + j ..][0..VEC_WIDTH].*;
                c0 += @as(Vec, @splat(A[(i + 0) * lda + kk])) * b_vec;
                c1 += @as(Vec, @splat(A[(i + 1) * lda + kk])) * b_vec;
                c2 += @as(Vec, @splat(A[(i + 2) * lda + kk])) * b_vec;
                c3 += @as(Vec, @splat(A[(i + 3) * lda + kk])) * b_vec;
            }

            C[(i + 0) * ldc + j ..][0..VEC_WIDTH].* = c0;
            C[(i + 1) * ldc + j ..][0..VEC_WIDTH].* = c1;
            C[(i + 2) * ldc + j ..][0..VEC_WIDTH].* = c2;
            C[(i + 3) * ldc + j ..][0..VEC_WIDTH].* = c3;
        }

        // Scalar tail
        while (j < N) : (j += 1) {
            var c0: f32 = 0.0;
            var c1: f32 = 0.0;
            var c2: f32 = 0.0;
            var c3: f32 = 0.0;
            for (0..K) |kk| {
                const b_val = B[kk * ldb + j];
                c0 += A[(i + 0) * lda + kk] * b_val;
                c1 += A[(i + 1) * lda + kk] * b_val;
                c2 += A[(i + 2) * lda + kk] * b_val;
                c3 += A[(i + 3) * lda + kk] * b_val;
            }
            C[(i + 0) * ldc + j] = c0;
            C[(i + 1) * ldc + j] = c1;
            C[(i + 2) * ldc + j] = c2;
            C[(i + 3) * ldc + j] = c3;
        }
    }

    // Handle remaining rows (< 4)
    while (i < M) : (i += 1) {
        var j: usize = 0;

        while (j + VEC_WIDTH <= N) : (j += VEC_WIDTH) {
            var c0: Vec = @splat(0.0);
            for (0..K) |kk| {
                const a_val: Vec = @splat(A[i * lda + kk]);
                const b_vec: Vec = B[kk * ldb + j ..][0..VEC_WIDTH].*;
                c0 += a_val * b_vec;
            }
            C[i * ldc + j ..][0..VEC_WIDTH].* = c0;
        }

        while (j < N) : (j += 1) {
            var c0: f32 = 0.0;
            for (0..K) |kk| {
                c0 += A[i * lda + kk] * B[kk * ldb + j];
            }
            C[i * ldc + j] = c0;
        }
    }
}

/// SGEMM with leading dimension parameters - cache-blocked implementation
fn sgemmOptimized(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    lda: usize,
    B: []const f32,
    ldb: usize,
    C: []f32,
    ldc: usize,
    alpha: f32,
    beta: f32,
) void {
    // Cache blocking parameters
    const MC = config.MC;
    const KC = config.KC;
    const NC = config.NC;
    const MR = micro_kernel.mr;
    const NR = micro_kernel.nr;

    // Allocate packing buffers
    // For x86_64: MC=256, KC=512, NC=4096
    //   packed_a: 256*512 = 128KB
    //   packed_b: 512*4096 = 2MB (too large for stack)
    // Use static thread-local buffers to avoid allocation overhead
    const PackedA = struct {
        threadlocal var buf: [MC * KC]f32 = undefined;
    };
    const PackedB = struct {
        threadlocal var buf: [KC * NC]f32 = undefined;
    };
    const packed_a: *[MC * KC]f32 = &PackedA.buf;
    const packed_b: *[KC * NC]f32 = &PackedB.buf;

    // Scale C by beta first (applied once before all accumulation)
    if (beta == 0.0) {
        for (0..M) |i| {
            @memset(C[i * ldc ..][0..N], 0.0);
        }
    } else if (beta != 1.0) {
        for (0..M) |i| {
            for (0..N) |j| {
                C[i * ldc + j] *= beta;
            }
        }
    }

    // Main blocking loops (GotoBLAS algorithm)
    // Loop order: jc (NC blocks), pc (KC blocks), ic (MC blocks)
    var jc: usize = 0;
    while (jc < N) : (jc += NC) {
        const jb = @min(NC, N - jc); // Actual columns in this block

        var pc: usize = 0;
        while (pc < K) : (pc += KC) {
            const pb = @min(KC, K - pc); // Actual K in this block

            // Pack B panel [pb × jb] -> packed_b
            // B block starts at B[pc * ldb + jc]
            packing.packB(
                B[pc * ldb + jc ..],
                ldb,
                packed_b,
                pb,
                jb,
                NR,
            );

            var ic: usize = 0;
            while (ic < M) : (ic += MC) {
                const ib = @min(MC, M - ic); // Actual rows in this block

                // Pack A block [ib × pb] -> packed_a
                // A block starts at A[ic * lda + pc]
                packing.packA(
                    A[ic * lda + pc ..],
                    lda,
                    packed_a,
                    ib,
                    pb,
                    MR,
                );

                // Compute C[ic:ic+ib, jc:jc+jb] += packed_A * packed_B
                computeBlock(
                    packed_a,
                    packed_b,
                    C[ic * ldc + jc ..],
                    ldc,
                    ib,
                    jb,
                    pb,
                    alpha,
                );
            }
        }
    }
}

/// Compute a block of C using packed A and B matrices
fn computeBlock(
    packed_a: []const f32,
    packed_b: []const f32,
    C: []f32,
    ldc: usize,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
) void {
    const MR = micro_kernel.mr;
    const NR = micro_kernel.nr;

    // Process NR×MR micro-tiles
    var jr: usize = 0;
    while (jr < n) : (jr += NR) {
        const nr_actual = @min(NR, n - jr);

        var ir: usize = 0;
        while (ir < m) : (ir += MR) {
            const mr_actual = @min(MR, m - ir);

            if (mr_actual == MR and nr_actual == NR) {
                // Full micro-kernel (4×4)
                // A panel for row ir is at packed_a[ir * k ..]
                // B panel for col jr is at packed_b[jr * k ..]
                micro_kernel.kernel(
                    k,
                    @ptrCast(packed_a.ptr + ir * k),
                    @ptrCast(packed_b.ptr + jr * k),
                    @ptrCast(C.ptr + ir * ldc + jr),
                    ldc,
                    alpha,
                );
            } else {
                // Edge case: use scalar for partial tiles
                scalarMicroKernel(
                    packed_a[ir * k ..],
                    packed_b[jr * k ..],
                    C[ir * ldc + jr ..],
                    ldc,
                    mr_actual,
                    nr_actual,
                    k,
                    alpha,
                    MR,
                    NR,
                );
            }
        }
    }
}

/// Scalar micro-kernel for edge cases (partial tiles)
fn scalarMicroKernel(
    A: []const f32,
    B: []const f32,
    C: []f32,
    ldc: usize,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    mr: usize,
    nr: usize,
) void {
    // A is packed: for each kk, elements [kk*mr .. kk*mr + m] contain the m rows
    // B is packed: for each kk, elements [kk*nr .. kk*nr + n] contain the n cols
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0.0;
            for (0..k) |kk| {
                // A[i] at position kk is at A[kk * mr + i]
                // B[j] at position kk is at B[kk * nr + j]
                sum += A[kk * mr + i] * B[kk * nr + j];
            }
            C[i * ldc + j] += alpha * sum;
        }
    }
}

// ============================================================================
// Fused SGEMM Implementation (Phase 10)
// ============================================================================
//
// Fused GEMM eliminates the need for a separate packed B buffer by reading
// B directly during computation. This reduces memory traffic significantly:
//
// Standard flow:  Read B -> Pack B to buffer -> Read packed B -> Compute
// Fused flow:     Read B -> Compute immediately
//
// Benefits:
// - Eliminates one full memory pass over B
// - Reduces memory bandwidth by ~33% for memory-bound workloads
// - Smaller working set (no packed_b buffer needed)
//
// Trade-offs:
// - B access pattern is less regular (row-major vs packed)
// - May be slower if B is not in cache (less prefetch-friendly)
// - Only works when B is not transposed
//

/// Fused SGEMM: C = alpha*A*B + beta*C with fused B packing
/// Only packs A, reads B directly during computation.
///
/// This is optimized for the NN (no transpose) case and works best when:
/// - Matrix is large enough that packing overhead matters
/// - B is accessed once (not reused across multiple A blocks)
/// - Memory bandwidth is the bottleneck
pub fn sgemmFused(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    lda: usize,
    B: []const f32,
    ldb: usize,
    C: []f32,
    ldc: usize,
    alpha: f32,
    beta: f32,
) void {
    // Only available on x86_64 with fused kernel
    if (!has_fused_kernel) {
        // Fallback to standard optimized path
        sgemmOptimized(M, N, K, A, lda, B, ldb, C, ldc, alpha, beta);
        return;
    }

    // Small matrices: use reference
    if (M * N * K < config.MIN_OPTIMIZED_SIZE * config.MIN_OPTIMIZED_SIZE * config.MIN_OPTIMIZED_SIZE) {
        reference.sgemm_reference_simple(M, N, K, A, B, C, alpha, beta);
        return;
    }

    // Fast path selection based on alpha/beta
    const use_fast_path = (alpha == 1.0 and beta == 0.0);
    const use_beta_zero = (beta == 0.0);

    // Cache blocking parameters (use config.* directly for buffer sizes)
    const MC = config.MC;
    const KC = config.KC;
    const NC = config.NC;
    const MR = fused_kernel_impl.mr;

    // Only need packed A buffer (no packed B!)
    const PackedA = struct {
        threadlocal var buf: [config.MC * config.KC]f32 = undefined;
    };
    const packed_a: *[config.MC * config.KC]f32 = &PackedA.buf;

    // Scale C by beta first (if not using fast path which overwrites)
    if (!use_fast_path) {
        if (beta == 0.0) {
            for (0..M) |i| {
                @memset(C[i * ldc ..][0..N], 0.0);
            }
        } else if (beta != 1.0) {
            for (0..M) |i| {
                for (0..N) |j| {
                    C[i * ldc + j] *= beta;
                }
            }
        }
    }

    // Main blocking loops (GotoBLAS algorithm with fused B)
    var jc: usize = 0;
    while (jc < N) : (jc += NC) {
        const jb = @min(NC, N - jc);

        var pc: usize = 0;
        while (pc < K) : (pc += KC) {
            const pb = @min(KC, K - pc);

            // NO B packing here - we read B directly in the kernel!

            var ic: usize = 0;
            while (ic < M) : (ic += MC) {
                const ib = @min(MC, M - ic);

                // Pack A block [ib × pb]
                packing.packA(
                    A[ic * lda + pc ..],
                    lda,
                    packed_a,
                    ib,
                    pb,
                    MR,
                );

                // Compute with fused B packing
                // Pass pointer to B[pc * ldb + jc] (start of B panel)
                computeBlockFused(
                    packed_a,
                    B[pc * ldb + jc ..].ptr,
                    ldb,
                    C[ic * ldc + jc ..].ptr,
                    ldc,
                    ib,
                    jb,
                    pb,
                    alpha,
                    use_fast_path,
                    use_beta_zero,
                    pc == 0, // first_k_block: only first iteration should overwrite
                );
            }
        }
    }
}

/// Compute a block using fused kernel (reads B directly)
fn computeBlockFused(
    packed_a: []const f32,
    B: [*]const f32,
    ldb: usize,
    C: [*]f32,
    ldc: usize,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    use_fast_path: bool,
    use_beta_zero: bool,
    first_k_block: bool,
) void {
    if (!has_fused_kernel) return;

    const MR = fused_kernel_impl.mr;
    const NR = fused_kernel_impl.nr;

    var jr: usize = 0;
    while (jr < n) : (jr += NR) {
        const nr_actual = @min(NR, n - jr);

        var ir: usize = 0;
        while (ir < m) : (ir += MR) {
            const mr_actual = @min(MR, m - ir);

            if (mr_actual == MR and nr_actual == NR) {
                // Full 8x8 micro-kernel with fused B access
                // B pointer: B + jr (offset to correct column)
                const b_ptr = B + jr;
                const c_ptr = C + ir * ldc + jr;

                if (use_fast_path and first_k_block) {
                    // Ultimate fast path: alpha=1, beta=0, first k block
                    fused_kernel_impl.kernelFusedFast(k, @ptrCast(packed_a.ptr + ir * k), b_ptr, ldb, c_ptr, ldc);
                } else if (use_beta_zero and first_k_block) {
                    // Beta=0 fast path (first k block only)
                    fused_kernel_impl.kernelFusedBetaZero(k, @ptrCast(packed_a.ptr + ir * k), b_ptr, ldb, c_ptr, ldc, alpha);
                } else {
                    // General case: accumulate to existing C
                    fused_kernel_impl.kernelFused(k, @ptrCast(packed_a.ptr + ir * k), b_ptr, ldb, c_ptr, ldc, alpha);
                }
            } else {
                // Edge case: scalar fallback for partial tiles
                scalarMicroKernelFused(
                    packed_a[ir * k ..],
                    B + jr,
                    ldb,
                    C + ir * ldc + jr,
                    ldc,
                    mr_actual,
                    nr_actual,
                    k,
                    alpha,
                    MR,
                );
            }
        }
    }
}

/// Scalar micro-kernel for fused edge cases
fn scalarMicroKernelFused(
    A: []const f32,
    B: [*]const f32,
    ldb: usize,
    C: [*]f32,
    ldc: usize,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    mr: usize,
) void {
    // A is packed: for each kk, elements [kk*mr .. kk*mr + m] contain the m rows
    // B is original: B[kk * ldb + j] for row kk, col j
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0.0;
            for (0..k) |kk| {
                sum += A[kk * mr + i] * B[kk * ldb + j];
            }
            C[i * ldc + j] += alpha * sum;
        }
    }
}

/// SGEMM with transpose options
pub fn sgemmTranspose(
    transA: Transpose,
    transB: Transpose,
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    lda: usize,
    B: []const f32,
    ldb: usize,
    C: []f32,
    ldc: usize,
    alpha: f32,
    beta: f32,
) void {
    // Handle transpose cases
    // For now, implement NN case and error on others
    // TODO: Full transpose support in Phase 2+

    if (transA == .NoTrans and transB == .NoTrans) {
        // NN case: straightforward
        sgemmOptimized(M, N, K, A, lda, B, ldb, C, ldc, alpha, beta);
    } else if (transA == .Trans and transB == .NoTrans) {
        // TN case: A is transposed
        sgemmTN(M, N, K, A, lda, B, ldb, C, ldc, alpha, beta);
    } else if (transA == .NoTrans and transB == .Trans) {
        // NT case: B is transposed
        sgemmNT(M, N, K, A, lda, B, ldb, C, ldc, alpha, beta);
    } else {
        // TT case: both transposed
        sgemmTT(M, N, K, A, lda, B, ldb, C, ldc, alpha, beta);
    }
}

/// SGEMM with A transposed (TN case)
fn sgemmTN(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    lda: usize,
    B: []const f32,
    ldb: usize,
    C: []f32,
    ldc: usize,
    alpha: f32,
    beta: f32,
) void {
    // Scale C by beta
    for (0..M) |i| {
        for (0..N) |j| {
            C[i * ldc + j] *= beta;
        }
    }

    // C += alpha * A^T * B
    // A^T[i,k] = A[k,i], so A is K x M and we read A[k * lda + i]
    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0.0;
            for (0..K) |k| {
                sum += A[k * lda + i] * B[k * ldb + j];
            }
            C[i * ldc + j] += alpha * sum;
        }
    }
}

/// SGEMM with B transposed (NT case)
fn sgemmNT(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    lda: usize,
    B: []const f32,
    ldb: usize,
    C: []f32,
    ldc: usize,
    alpha: f32,
    beta: f32,
) void {
    // Scale C by beta
    for (0..M) |i| {
        for (0..N) |j| {
            C[i * ldc + j] *= beta;
        }
    }

    // C += alpha * A * B^T
    // B^T[k,j] = B[j,k], so B is N x K and we read B[j * ldb + k]
    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0.0;
            for (0..K) |k| {
                sum += A[i * lda + k] * B[j * ldb + k];
            }
            C[i * ldc + j] += alpha * sum;
        }
    }
}

/// SGEMM with both transposed (TT case)
fn sgemmTT(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    lda: usize,
    B: []const f32,
    ldb: usize,
    C: []f32,
    ldc: usize,
    alpha: f32,
    beta: f32,
) void {
    // Scale C by beta
    for (0..M) |i| {
        for (0..N) |j| {
            C[i * ldc + j] *= beta;
        }
    }

    // C += alpha * A^T * B^T
    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0.0;
            for (0..K) |k| {
                sum += A[k * lda + i] * B[j * ldb + k];
            }
            C[i * ldc + j] += alpha * sum;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "sgemm NN" {
    const A = [_]f32{ 1, 2, 3, 4 };
    const B = [_]f32{ 5, 6, 7, 8 };
    var C = [_]f32{ 0, 0, 0, 0 };

    sgemm(2, 2, 2, &A, &B, &C, 1.0, 0.0);

    try std.testing.expectApproxEqRel(C[0], 19.0, 1e-5);
    try std.testing.expectApproxEqRel(C[1], 22.0, 1e-5);
    try std.testing.expectApproxEqRel(C[2], 43.0, 1e-5);
    try std.testing.expectApproxEqRel(C[3], 50.0, 1e-5);
}

test "sgemm TN" {
    // A^T = [1, 3; 2, 4] (original A = [1, 2; 3, 4])
    // B = [5, 6; 7, 8]
    // A^T * B = [26, 30; 38, 44]
    const A = [_]f32{ 1, 2, 3, 4 }; // stored as 2x2, will be transposed
    const B = [_]f32{ 5, 6, 7, 8 };
    var C = [_]f32{ 0, 0, 0, 0 };

    sgemmTranspose(.Trans, .NoTrans, 2, 2, 2, &A, 2, &B, 2, &C, 2, 1.0, 0.0);

    try std.testing.expectApproxEqRel(C[0], 26.0, 1e-5);
    try std.testing.expectApproxEqRel(C[1], 30.0, 1e-5);
    try std.testing.expectApproxEqRel(C[2], 38.0, 1e-5);
    try std.testing.expectApproxEqRel(C[3], 44.0, 1e-5);
}

test "sgemm NT" {
    // A = [1, 2; 3, 4]
    // B^T = [5, 7; 6, 8] (original B = [5, 6; 7, 8])
    // A * B^T = [17, 23; 39, 53]
    const A = [_]f32{ 1, 2, 3, 4 };
    const B = [_]f32{ 5, 6, 7, 8 };
    var C = [_]f32{ 0, 0, 0, 0 };

    sgemmTranspose(.NoTrans, .Trans, 2, 2, 2, &A, 2, &B, 2, &C, 2, 1.0, 0.0);

    try std.testing.expectApproxEqRel(C[0], 17.0, 1e-5);
    try std.testing.expectApproxEqRel(C[1], 23.0, 1e-5);
    try std.testing.expectApproxEqRel(C[2], 39.0, 1e-5);
    try std.testing.expectApproxEqRel(C[3], 53.0, 1e-5);
}

test "sgemm empty" {
    var C = [_]f32{ 1, 2, 3, 4 };
    sgemm(0, 2, 2, &[_]f32{}, &[_]f32{}, &C, 1.0, 0.0);
    // Should not modify C
    try std.testing.expectEqual(C[0], 1.0);
}

// zblas/src/level3/sgemm_parallel.zig
// Parallel SGEMM implementation with thread pool and size thresholds
//
// Key lessons from Phase 5 benchmarks:
// - Multi-threaded OpenBLAS was 63% SLOWER than single-threaded on Whisper
// - Thread spawn/join overhead dominates for small/medium matrices
// - Only parallelize when FLOPS > 50M and M > 512
//
// This implementation:
// 1. Uses aggressive size thresholds to avoid overhead on small matrices
// 2. Splits the M dimension (rows of C) across threads
// 3. Packs B once on main thread, shares read-only across workers
// 4. Each thread has its own A packing buffer (no contention)

const std = @import("std");
const builtin = @import("builtin");
const config = @import("../config.zig");
const packing = @import("../util/packing.zig");

// Import the appropriate kernel based on architecture
const micro_kernel = switch (builtin.cpu.arch) {
    .x86_64 => @import("../kernel/x86_64/sgemm_kernel_8x8.zig"),
    .aarch64 => @import("../kernel/arm64/sgemm_kernel_8x8.zig"),
    else => @import("../kernel/generic/sgemm_kernel_4x4.zig"),
};

const Allocator = std.mem.Allocator;

// Maximum threads supported (avoid large stack allocations)
const MAX_THREADS = 16;

// =============================================================================
// Size Thresholds (Critical for avoiding thread overhead!)
// =============================================================================

/// Minimum FLOPS to consider parallelization (2*M*N*K)
/// Based on Phase 5 findings: thread overhead ~1ms, need 10ms+ compute
const MIN_FLOPS_FOR_PARALLEL: usize = 50_000_000; // 50M FLOPS

/// Minimum FLOPS per thread to be worthwhile
const MIN_FLOPS_PER_THREAD: usize = 25_000_000; // 25M FLOPS

/// Minimum rows to enable parallelization (need enough to split)
const MIN_ROWS_FOR_PARALLEL: usize = 512;

/// Minimum rows per thread (should be at least MC for efficient blocking)
const MIN_ROWS_PER_THREAD: usize = config.MC;

// =============================================================================
// Thread Work Structure
// =============================================================================

/// Work item for a single thread
const GemmWork = struct {
    // Input matrices (shared, read-only)
    A: []const f32,
    lda: usize,
    B: []const f32,
    ldb: usize,

    // Output matrix (each thread writes to different rows - no contention)
    C: []f32,
    ldc: usize,

    // Full dimensions
    M: usize,
    N: usize,
    K: usize,

    // Row range for this thread [row_start, row_end)
    row_start: usize,
    row_end: usize,

    // Scalars
    alpha: f32,
    beta: f32,

    // Thread-local A packing buffer
    packed_a: []f32,

    // Shared, pre-packed B buffer (read-only)
    packed_b: []const f32,
};

// =============================================================================
// Worker Function
// =============================================================================

/// Worker function executed by each thread
fn gemmWorker(work: *GemmWork) void {
    const MC = config.MC;
    const KC = config.KC;
    const NC = config.NC;
    const MR = micro_kernel.mr;

    const m = work.row_end - work.row_start;
    if (m == 0) return;

    // Scale this thread's portion of C by beta
    if (work.beta == 0.0) {
        for (work.row_start..work.row_end) |i| {
            @memset(work.C[i * work.ldc ..][0..work.N], 0.0);
        }
    } else if (work.beta != 1.0) {
        for (work.row_start..work.row_end) |i| {
            for (0..work.N) |j| {
                work.C[i * work.ldc + j] *= work.beta;
            }
        }
    }

    if (work.alpha == 0.0) return;

    // Main blocking loops - iterate over this thread's rows only
    var jc: usize = 0;
    while (jc < work.N) : (jc += NC) {
        const jb = @min(NC, work.N - jc);

        var pc: usize = 0;
        while (pc < work.K) : (pc += KC) {
            const pb = @min(KC, work.K - pc);

            // Calculate offset into pre-packed B
            // B is packed in NC×KC panels, row-major order
            const b_panel_idx = (jc / NC);
            const k_panel_idx = (pc / KC);
            const num_k_panels = (work.K + KC - 1) / KC;
            const packed_b_offset = (b_panel_idx * num_k_panels + k_panel_idx) * KC * NC;
            const packed_b = work.packed_b[packed_b_offset..];

            var ic: usize = work.row_start;
            while (ic < work.row_end) : (ic += MC) {
                const ib = @min(MC, work.row_end - ic);

                // Pack A block [ib × pb] -> thread-local buffer
                packing.packA(
                    work.A[ic * work.lda + pc ..],
                    work.lda,
                    work.packed_a,
                    ib,
                    pb,
                    MR,
                );

                // Compute C[ic:ic+ib, jc:jc+jb] += alpha * packed_A * packed_B
                computeBlock(
                    work.packed_a,
                    packed_b,
                    work.C[ic * work.ldc + jc ..],
                    work.ldc,
                    ib,
                    jb,
                    pb,
                    work.alpha,
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

    var jr: usize = 0;
    while (jr < n) : (jr += NR) {
        const nr_actual = @min(NR, n - jr);

        var ir: usize = 0;
        while (ir < m) : (ir += MR) {
            const mr_actual = @min(MR, m - ir);

            if (mr_actual == MR and nr_actual == NR) {
                // Full micro-kernel
                micro_kernel.kernel(
                    k,
                    @ptrCast(packed_a.ptr + ir * k),
                    @ptrCast(packed_b.ptr + jr * k),
                    @ptrCast(C.ptr + ir * ldc + jr),
                    ldc,
                    alpha,
                );
            } else {
                // Edge case: scalar fallback for partial tiles
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
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0.0;
            for (0..k) |kk| {
                sum += A[kk * mr + i] * B[kk * nr + j];
            }
            C[i * ldc + j] += alpha * sum;
        }
    }
}

// =============================================================================
// Public API
// =============================================================================

/// Check if a matrix size should be parallelized
/// Based on Phase 5 findings - threading hurts small/medium matrices!
pub fn shouldParallelize(M: usize, N: usize, K: usize) bool {
    const flops = 2 * M * N * K;
    return flops >= MIN_FLOPS_FOR_PARALLEL and M >= MIN_ROWS_FOR_PARALLEL;
}

/// Get optimal thread count based on problem size and available cores
/// CONSERVATIVE: Based on Phase 5 findings that threading overhead is significant
pub fn getOptimalThreadCount(M: usize, N: usize, K: usize) usize {
    const flops = 2 * M * N * K;
    const cpu_count = std.Thread.getCpuCount() catch 1;

    // Don't parallelize if problem is too small
    if (!shouldParallelize(M, N, K)) {
        return 1;
    }

    // Limit threads by available work
    const max_threads_by_work = flops / MIN_FLOPS_PER_THREAD;
    const max_threads_by_rows = M / MIN_ROWS_PER_THREAD;

    return @max(1, @min(cpu_count, @min(max_threads_by_work, @min(max_threads_by_rows, MAX_THREADS))));
}

/// Parallel SGEMM: C = alpha * A * B + beta * C
///
/// Automatically determines optimal thread count based on problem size.
/// Falls back to single-threaded for small matrices where threading would hurt.
///
/// Parameters:
///   - allocator: Memory allocator for packing buffers
///   - M, N, K: Matrix dimensions (A is MxK, B is KxN, C is MxN)
///   - A, B: Input matrices in row-major order
///   - C: Output matrix in row-major order
///   - alpha, beta: Scalar multipliers
pub fn sgemmParallelAuto(
    allocator: Allocator,
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    B: []const f32,
    C: []f32,
    alpha: f32,
    beta: f32,
) !void {
    const num_threads = getOptimalThreadCount(M, N, K);
    try sgemmParallel(allocator, num_threads, M, N, K, A, K, B, N, C, N, alpha, beta);
}

/// Parallel SGEMM with explicit thread count
///
/// Splits the M dimension across threads. Each thread writes to different
/// rows of C, so there's no write contention.
///
/// Parameters:
///   - allocator: Memory allocator for packing buffers
///   - num_threads: Number of threads to use (will be reduced if problem too small)
///   - M, N, K: Matrix dimensions
///   - A: Input matrix A [M x K]
///   - lda: Leading dimension of A
///   - B: Input matrix B [K x N]
///   - ldb: Leading dimension of B
///   - C: Output matrix C [M x N]
///   - ldc: Leading dimension of C
///   - alpha, beta: Scalar multipliers
pub fn sgemmParallel(
    allocator: Allocator,
    num_threads: usize,
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
) !void {
    // Validate inputs
    if (M == 0 or N == 0 or K == 0) return;

    const MC = config.MC;
    const KC = config.KC;
    const NC = config.NC;
    const NR = micro_kernel.nr;

    // Calculate effective thread count (may be reduced for small problems)
    var effective_threads = @min(num_threads, MAX_THREADS);
    effective_threads = @min(effective_threads, M / MIN_ROWS_PER_THREAD);
    effective_threads = @max(1, effective_threads);

    // For single thread or tiny problems, use existing single-threaded path
    if (effective_threads <= 1) {
        @import("sgemm.zig").sgemm(M, N, K, A, B, C, alpha, beta);
        return;
    }

    // Allocate shared B packing buffer
    const num_n_blocks = (N + NC - 1) / NC;
    const num_k_blocks = (K + KC - 1) / KC;
    const packed_b_size = num_n_blocks * num_k_blocks * KC * NC;
    const packed_b = try allocator.alloc(f32, packed_b_size);
    defer allocator.free(packed_b);

    // Pack B once on main thread (shared read-only across all workers)
    var b_offset: usize = 0;
    var jc: usize = 0;
    while (jc < N) : (jc += NC) {
        const jb = @min(NC, N - jc);
        var pc: usize = 0;
        while (pc < K) : (pc += KC) {
            const pb = @min(KC, K - pc);
            packing.packB(B[pc * ldb + jc ..], ldb, packed_b[b_offset..], pb, jb, NR);
            b_offset += KC * NC; // Fixed stride for indexing
        }
    }

    // Allocate per-thread A packing buffers
    var packed_a_buffers: [MAX_THREADS][]f32 = undefined;
    var alloc_count: usize = 0;
    errdefer {
        for (0..alloc_count) |t| {
            allocator.free(packed_a_buffers[t]);
        }
    }

    for (0..effective_threads) |t| {
        packed_a_buffers[t] = try allocator.alloc(f32, MC * KC);
        alloc_count += 1;
    }
    defer {
        for (0..effective_threads) |t| {
            allocator.free(packed_a_buffers[t]);
        }
    }

    // Create work items with row distribution
    var work_items: [MAX_THREADS]GemmWork = undefined;
    const rows_per_thread = M / effective_threads;
    const extra_rows = M % effective_threads;

    var row_start: usize = 0;
    for (0..effective_threads) |t| {
        // Distribute extra rows to first threads
        const rows = rows_per_thread + @as(usize, if (t < extra_rows) 1 else 0);
        work_items[t] = .{
            .A = A,
            .lda = lda,
            .B = B,
            .ldb = ldb,
            .C = C,
            .ldc = ldc,
            .M = M,
            .N = N,
            .K = K,
            .row_start = row_start,
            .row_end = row_start + rows,
            .alpha = alpha,
            .beta = beta,
            .packed_a = packed_a_buffers[t],
            .packed_b = packed_b,
        };
        row_start += rows;
    }

    // Spawn worker threads
    var threads: [MAX_THREADS]?std.Thread = [_]?std.Thread{null} ** MAX_THREADS;
    var spawn_count: usize = 0;

    // Spawn threads 1..N-1 (thread 0 will run on main)
    for (1..effective_threads) |t| {
        threads[t] = std.Thread.spawn(.{}, gemmWorker, .{&work_items[t]}) catch null;
        if (threads[t] != null) {
            spawn_count += 1;
        } else {
            // Thread spawn failed - run this work on main thread later
        }
    }

    // Run first work item on main thread
    gemmWorker(&work_items[0]);

    // Run any work items whose threads failed to spawn
    for (1..effective_threads) |t| {
        if (threads[t] == null) {
            gemmWorker(&work_items[t]);
        }
    }

    // Wait for all spawned threads
    for (1..effective_threads) |t| {
        if (threads[t]) |thread| {
            thread.join();
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

test "shouldParallelize thresholds" {
    const testing = std.testing;

    // Small matrix - should NOT parallelize
    try testing.expect(!shouldParallelize(100, 100, 100)); // 2M FLOPS

    // Medium matrix - still no (25M FLOPS, but M too small)
    try testing.expect(!shouldParallelize(100, 500, 250));

    // Large matrix - should parallelize
    try testing.expect(shouldParallelize(1024, 1024, 1024)); // 2B FLOPS, M=1024

    // Edge case: high FLOPS but few rows
    try testing.expect(!shouldParallelize(256, 4096, 4096)); // 137M FLOPS but M=256 < 512
}

test "getOptimalThreadCount" {
    const testing = std.testing;

    // Small matrix - 1 thread
    try testing.expectEqual(@as(usize, 1), getOptimalThreadCount(100, 100, 100));

    // Large matrix - should get multiple threads
    const threads = getOptimalThreadCount(2048, 2048, 2048);
    try testing.expect(threads >= 1);
    try testing.expect(threads <= MAX_THREADS);
}

test "parallel sgemm correctness small" {
    const allocator = std.testing.allocator;
    const N = 64;

    var A: [N * N]f32 = undefined;
    var B: [N * N]f32 = undefined;
    var C_parallel: [N * N]f32 = undefined;
    var C_single: [N * N]f32 = undefined;

    // Initialize with deterministic values
    for (0..N * N) |i| {
        A[i] = @as(f32, @floatFromInt(i % 100)) / 100.0;
        B[i] = @as(f32, @floatFromInt((i * 7) % 100)) / 100.0;
    }
    @memset(&C_parallel, 0.0);
    @memset(&C_single, 0.0);

    // Single-threaded reference
    @import("sgemm.zig").sgemm(N, N, N, &A, &B, &C_single, 1.0, 0.0);

    // Parallel with 4 threads (will fall back to single for small N)
    try sgemmParallel(allocator, 4, N, N, N, &A, N, &B, N, &C_parallel, N, 1.0, 0.0);

    // Compare results
    for (0..N * N) |i| {
        try std.testing.expectApproxEqRel(C_parallel[i], C_single[i], 1e-4);
    }
}

test "parallel sgemm correctness large" {
    const allocator = std.testing.allocator;
    const N = 256;

    const A = try allocator.alloc(f32, N * N);
    defer allocator.free(A);
    const B = try allocator.alloc(f32, N * N);
    defer allocator.free(B);
    const C_parallel = try allocator.alloc(f32, N * N);
    defer allocator.free(C_parallel);
    const C_single = try allocator.alloc(f32, N * N);
    defer allocator.free(C_single);

    // Initialize
    for (0..N * N) |i| {
        A[i] = @as(f32, @floatFromInt(i % 100)) / 100.0;
        B[i] = @as(f32, @floatFromInt((i * 7) % 100)) / 100.0;
    }
    @memset(C_parallel, 0.0);
    @memset(C_single, 0.0);

    // Single-threaded reference
    @import("sgemm.zig").sgemm(N, N, N, A, B, C_single, 1.0, 0.0);

    // Parallel with 4 threads
    try sgemmParallel(allocator, 4, N, N, N, A, N, B, N, C_parallel, N, 1.0, 0.0);

    // Compare
    for (0..N * N) |i| {
        try std.testing.expectApproxEqRel(C_parallel[i], C_single[i], 1e-4);
    }
}

test "parallel sgemm with beta" {
    const allocator = std.testing.allocator;
    const N = 128;

    var A: [N * N]f32 = undefined;
    var B: [N * N]f32 = undefined;
    var C_parallel: [N * N]f32 = undefined;
    var C_single: [N * N]f32 = undefined;

    for (0..N * N) |i| {
        A[i] = @as(f32, @floatFromInt(i % 50)) / 50.0;
        B[i] = @as(f32, @floatFromInt((i * 3) % 50)) / 50.0;
        C_parallel[i] = @as(f32, @floatFromInt((i * 11) % 50)) / 50.0;
        C_single[i] = C_parallel[i];
    }

    // Test with beta = 0.5
    @import("sgemm.zig").sgemm(N, N, N, &A, &B, &C_single, 2.0, 0.5);
    try sgemmParallel(allocator, 4, N, N, N, &A, N, &B, N, &C_parallel, N, 2.0, 0.5);

    for (0..N * N) |i| {
        try std.testing.expectApproxEqRel(C_parallel[i], C_single[i], 1e-4);
    }
}

test "parallel sgemm various thread counts" {
    const allocator = std.testing.allocator;
    const N = 128;
    const thread_counts = [_]usize{ 1, 2, 3, 4, 7, 8 };

    var A: [N * N]f32 = undefined;
    var B: [N * N]f32 = undefined;
    var C_ref: [N * N]f32 = undefined;

    for (0..N * N) |i| {
        A[i] = @as(f32, @floatFromInt(i % 100)) / 100.0;
        B[i] = @as(f32, @floatFromInt((i * 7) % 100)) / 100.0;
    }
    @memset(&C_ref, 0.0);

    // Get reference result
    @import("sgemm.zig").sgemm(N, N, N, &A, &B, &C_ref, 1.0, 0.0);

    // Test each thread count
    for (thread_counts) |num_threads| {
        var C_test: [N * N]f32 = undefined;
        @memset(&C_test, 0.0);

        try sgemmParallel(allocator, num_threads, N, N, N, &A, N, &B, N, &C_test, N, 1.0, 0.0);

        for (0..N * N) |i| {
            try std.testing.expectApproxEqRel(C_test[i], C_ref[i], 1e-4);
        }
    }
}

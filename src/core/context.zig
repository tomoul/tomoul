const std = @import("std");
const builtin = @import("builtin");

/// Execution Context for parallel inference
///
/// The Context struct holds shared resources for inference:
/// - Thread pool for parallel operations (native targets only)
/// - Allocator for memory management
/// - Per-thread scratch buffers to avoid allocator contention
///
/// Usage:
///   var ctx = try Context.init(allocator, 8); // 8 threads
///   defer ctx.deinit();
///   var model = try Model.init(&ctx, weights_path);
///   const output = try model.infer(input);
pub const Context = struct {
    /// Base allocator for all operations
    allocator: std.mem.Allocator,

    /// Thread pool for parallel operations (null on WASM or single-threaded mode)
    thread_pool: if (supportsThreading()) ?std.Thread.Pool else void,

    /// Number of worker threads (1 = single-threaded)
    num_threads: usize,

    /// Per-thread scratch buffers for temporary allocations
    /// Avoids allocator contention in parallel sections
    scratch_buffers: if (supportsThreading()) []ThreadLocalScratch else void,

    /// Minimum rows for parallel matmul (below this, single-thread is faster)
    min_parallel_rows: usize,

    const Self = @This();

    /// Check if the current target supports threading
    pub fn supportsThreading() bool {
        // WASM does not support std.Thread easily
        // Also disable for freestanding targets
        const cpu_arch = builtin.cpu.arch;
        const is_wasm = cpu_arch == .wasm32 or cpu_arch == .wasm64;
        return !is_wasm and builtin.os.tag != .freestanding;
    }

    /// Initialize execution context
    ///
    /// Parameters:
    /// - allocator: Memory allocator for all operations
    /// - num_threads: Number of worker threads (null = auto-detect CPU cores)
    ///
    /// On WASM or unsupported targets, creates a single-threaded context.
    pub fn init(allocator: std.mem.Allocator, num_threads: ?usize) !Self {
        if (comptime !supportsThreading()) {
            // Single-threaded mode for WASM/freestanding
            return Self{
                .allocator = allocator,
                .thread_pool = {},
                .num_threads = 1,
                .scratch_buffers = {},
                .min_parallel_rows = std.math.maxInt(usize), // Never parallelize
            };
        }

        // Native threading mode
        // Cap threads for single-sequence inference: more threads = more overhead
        // with no benefit when memory-bandwidth bound. 4-6 threads is optimal.
        const detected = num_threads orelse detectCpuCount();
        const thread_count = @min(detected, MAX_THREADS_SINGLE_SEQUENCE);

        if (thread_count <= 1) {
            // Single-threaded mode (no pool overhead)
            return Self{
                .allocator = allocator,
                .thread_pool = null,
                .num_threads = 1,
                .scratch_buffers = &[_]ThreadLocalScratch{},
                .min_parallel_rows = std.math.maxInt(usize),
            };
        }

        // Multi-threaded mode: initialize thread pool
        var pool: std.Thread.Pool = undefined;
        try pool.init(.{
            .allocator = allocator,
            .n_jobs = thread_count,
        });
        errdefer pool.deinit();

        // Allocate per-thread scratch buffers
        const scratch_buffers = try allocator.alloc(ThreadLocalScratch, thread_count);
        errdefer allocator.free(scratch_buffers);

        // Initialize each scratch buffer
        for (scratch_buffers, 0..) |*scratch, i| {
            scratch.* = try ThreadLocalScratch.init(allocator, DEFAULT_SCRATCH_SIZE);
            errdefer {
                for (scratch_buffers[0..i]) |*s| s.deinit();
            }
        }

        return Self{
            .allocator = allocator,
            .thread_pool = pool,
            .num_threads = thread_count,
            .scratch_buffers = scratch_buffers,
            .min_parallel_rows = DEFAULT_MIN_PARALLEL_ROWS,
        };
    }

    /// Initialize with explicit single-threaded mode
    /// Useful for testing or when threading overhead is undesirable
    pub fn initSingleThreaded(allocator: std.mem.Allocator) Self {
        if (comptime !supportsThreading()) {
            return Self{
                .allocator = allocator,
                .thread_pool = {},
                .num_threads = 1,
                .scratch_buffers = {},
                .min_parallel_rows = std.math.maxInt(usize),
            };
        }

        return Self{
            .allocator = allocator,
            .thread_pool = null,
            .num_threads = 1,
            .scratch_buffers = &[_]ThreadLocalScratch{},
            .min_parallel_rows = std.math.maxInt(usize),
        };
    }

    /// Initialize for multi-threaded execution WITHOUT thread pool
    /// Uses direct thread spawning in parallel ops (no pool overhead or shutdown issues)
    pub fn initMultiThreaded(allocator: std.mem.Allocator, num_threads: ?usize) Self {
        if (comptime !supportsThreading()) {
            return Self{
                .allocator = allocator,
                .thread_pool = {},
                .num_threads = 1,
                .scratch_buffers = {},
                .min_parallel_rows = std.math.maxInt(usize),
            };
        }

        const detected = num_threads orelse detectCpuCount();
        const thread_count = @min(detected, MAX_THREADS_SINGLE_SEQUENCE);

        return Self{
            .allocator = allocator,
            .thread_pool = null, // No pool - direct spawning
            .num_threads = thread_count,
            .scratch_buffers = &[_]ThreadLocalScratch{},
            .min_parallel_rows = DEFAULT_MIN_PARALLEL_ROWS,
        };
    }

    /// Clean up resources
    pub fn deinit(self: *Self) void {
        if (comptime !supportsThreading()) {
            return;
        }

        // Free scratch buffers
        if (self.scratch_buffers.len > 0) {
            for (self.scratch_buffers) |*scratch| {
                scratch.deinit();
            }
            self.allocator.free(self.scratch_buffers);
        }

        // Shut down thread pool
        if (self.thread_pool) |*pool| {
            pool.deinit();
        }
    }

    /// Check if parallel execution is available
    pub fn isParallel(self: *const Self) bool {
        if (comptime !supportsThreading()) {
            return false;
        }
        // Check num_threads > 1 (works with both pool-based and direct spawning)
        return self.num_threads > 1;
    }

    /// Check if a workload should be parallelized based on size
    pub fn shouldParallelize(self: *const Self, rows: usize) bool {
        return self.isParallel() and rows >= self.min_parallel_rows;
    }

    /// Get scratch buffer for a specific thread
    /// Thread ID should be obtained from the thread pool task context
    pub fn getScratch(self: *Self, thread_id: usize) ?*ThreadLocalScratch {
        if (comptime !supportsThreading()) {
            return null;
        }

        if (thread_id >= self.scratch_buffers.len) {
            return null;
        }
        return &self.scratch_buffers[thread_id];
    }

    /// Get the thread pool (for spawning parallel tasks)
    pub fn getPool(self: *Self) ?*std.Thread.Pool {
        if (comptime !supportsThreading()) {
            return null;
        }
        if (self.thread_pool) |*pool| {
            return pool;
        }
        return null;
    }

    /// Detect number of CPU cores available
    fn detectCpuCount() usize {
        if (comptime !supportsThreading()) {
            return 1;
        }
        return std.Thread.getCpuCount() catch 4; // Default to 4 if detection fails
    }
};

/// Per-thread scratch buffer for temporary allocations
///
/// Each thread gets its own scratch buffer to avoid lock contention
/// on the main allocator during parallel operations.
pub const ThreadLocalScratch = struct {
    /// Underlying memory buffer
    buffer: []u8,

    /// Fixed buffer allocator over the scratch space
    fba: std.heap.FixedBufferAllocator,

    /// Parent allocator (for freeing the buffer)
    parent_allocator: std.mem.Allocator,

    const Self = @This();

    /// Initialize scratch buffer with given size
    pub fn init(parent: std.mem.Allocator, size: usize) !Self {
        const buffer = try parent.alloc(u8, size);
        errdefer parent.free(buffer);

        return Self{
            .buffer = buffer,
            .fba = std.heap.FixedBufferAllocator.init(buffer),
            .parent_allocator = parent,
        };
    }

    /// Free the scratch buffer
    pub fn deinit(self: *Self) void {
        self.parent_allocator.free(self.buffer);
    }

    /// Reset the scratch buffer for reuse
    /// Call this between operations to reclaim memory
    pub fn reset(self: *Self) void {
        self.fba.reset();
    }

    /// Get an allocator backed by this scratch buffer
    pub fn allocator(self: *Self) std.mem.Allocator {
        return self.fba.allocator();
    }

    /// Get remaining capacity in bytes
    pub fn remainingCapacity(self: *const Self) usize {
        return self.buffer.len - self.fba.end_index;
    }
};

/// Default scratch buffer size per thread (4MB)
/// Sized for XLM-RoBERTa intermediate tensors: [seq_len, hidden_dim] = [512, 1024] = 2MB
const DEFAULT_SCRATCH_SIZE: usize = 4 * 1024 * 1024;

/// Minimum rows for parallel matmul (below this, single-thread is faster due to overhead)
/// Set conservatively high to avoid thread pool contention on small workloads.
/// For transformer models, this means parallelism kicks in for seq_len >= 64.
const DEFAULT_MIN_PARALLEL_ROWS: usize = 64;

/// Maximum threads for single-sequence inference
/// More threads = more overhead with no benefit when memory-bandwidth bound.
/// 4 threads is optimal for typical transformer workloads on CPU.
const MAX_THREADS_SINGLE_SEQUENCE: usize = 4;

// ============================================================================
// Tests
// ============================================================================

test "context single-threaded initialization" {
    const allocator = std.testing.allocator;

    var ctx = Context.initSingleThreaded(allocator);
    defer ctx.deinit();

    try std.testing.expectEqual(@as(usize, 1), ctx.num_threads);
    try std.testing.expect(!ctx.isParallel());
    try std.testing.expect(!ctx.shouldParallelize(100));
}

test "scratch buffer allocation and reset" {
    const allocator = std.testing.allocator;

    var scratch = try ThreadLocalScratch.init(allocator, 1024);
    defer scratch.deinit();

    // Allocate some memory
    const initial_capacity = scratch.remainingCapacity();
    const data = try scratch.allocator().alloc(u8, 100);
    _ = data;

    try std.testing.expect(scratch.remainingCapacity() < initial_capacity);

    // Reset should restore capacity
    scratch.reset();
    try std.testing.expectEqual(initial_capacity, scratch.remainingCapacity());
}

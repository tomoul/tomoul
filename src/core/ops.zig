const std = @import("std");
const builtin = @import("builtin");

// Support both module imports (Wasm build) and relative imports (native build)
const tensor_import = @import("tensor.zig");
const Tensor = tensor_import.Tensor;
const TensorError = tensor_import.TensorError;

// Build options for BLAS support
const build_options = @import("build_options");
const use_blas: bool = if (@hasDecl(build_options, "use_blas")) build_options.use_blas else false;
pub const use_zblas: bool = if (@hasDecl(build_options, "use_zblas")) build_options.use_zblas else false;

// BLAS modules (only used when enabled)
const blas = if (use_blas) @import("blas.zig") else undefined;
const zblas = if (use_zblas) @import("zblas") else undefined;

// Context for parallel execution
const context_import = @import("context.zig");
pub const Context = context_import.Context;

/// Global thread pool for parallel operations
/// Set via initGlobalContext() at startup, used by matmul automatically
var global_context: ?*Context = null;

/// Initialize the global execution context for parallel operations
/// Call this once at startup before using matmul
pub fn initGlobalContext(ctx: *Context) void {
    global_context = ctx;
}

/// Deinitialize the global execution context
pub fn deinitGlobalContext() void {
    global_context = null;
}

/// Operations error types
pub const OpsError = error{
    ShapeMismatch,
    OutOfBounds,
    InvalidShape,
    OutOfMemory,
};

// ============================================================================
// Shape Validation Helpers
// ============================================================================

/// Check if two shapes are identical
pub fn shapesMatch(a: []const usize, b: []const usize) bool {
    if (a.len != b.len) {
        return false;
    }
    for (a, b) |dim_a, dim_b| {
        if (dim_a != dim_b) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// Element-wise Operations (Allocating)
// ============================================================================

/// Element-wise addition: C = A + B
/// Allocates a new tensor for the result.
pub fn add(allocator: std.mem.Allocator, a: *const Tensor, b: *const Tensor) !Tensor {
    if (!shapesMatch(a.shape, b.shape)) {
        return OpsError.ShapeMismatch;
    }

    var result = try Tensor.init(allocator, a.shape);
    errdefer result.deinit();

    for (result.data, a.data, b.data) |*r, val_a, val_b| {
        r.* = val_a + val_b;
    }

    return result;
}

/// Element-wise subtraction: C = A - B
/// Allocates a new tensor for the result.
pub fn sub(allocator: std.mem.Allocator, a: *const Tensor, b: *const Tensor) !Tensor {
    if (!shapesMatch(a.shape, b.shape)) {
        return OpsError.ShapeMismatch;
    }

    var result = try Tensor.init(allocator, a.shape);
    errdefer result.deinit();

    for (result.data, a.data, b.data) |*r, val_a, val_b| {
        r.* = val_a - val_b;
    }

    return result;
}

/// Element-wise multiplication (Hadamard product): C = A * B
/// Allocates a new tensor for the result.
pub fn mul(allocator: std.mem.Allocator, a: *const Tensor, b: *const Tensor) !Tensor {
    if (!shapesMatch(a.shape, b.shape)) {
        return OpsError.ShapeMismatch;
    }

    var result = try Tensor.init(allocator, a.shape);
    errdefer result.deinit();

    for (result.data, a.data, b.data) |*r, val_a, val_b| {
        r.* = val_a * val_b;
    }

    return result;
}

/// Element-wise division: C = A / B
/// Allocates a new tensor for the result.
pub fn div(allocator: std.mem.Allocator, a: *const Tensor, b: *const Tensor) !Tensor {
    if (!shapesMatch(a.shape, b.shape)) {
        return OpsError.ShapeMismatch;
    }

    var result = try Tensor.init(allocator, a.shape);
    errdefer result.deinit();

    for (result.data, a.data, b.data) |*r, val_a, val_b| {
        r.* = val_a / val_b;
    }

    return result;
}

/// Scalar multiplication: C = A * scalar
/// Allocates a new tensor for the result.
pub fn scale(allocator: std.mem.Allocator, a: *const Tensor, scalar: f32) !Tensor {
    var result = try Tensor.init(allocator, a.shape);
    errdefer result.deinit();

    for (result.data, a.data) |*r, val| {
        r.* = val * scalar;
    }

    return result;
}

/// Scalar addition: C = A + scalar
/// Allocates a new tensor for the result.
pub fn addScalar(allocator: std.mem.Allocator, a: *const Tensor, scalar: f32) !Tensor {
    var result = try Tensor.init(allocator, a.shape);
    errdefer result.deinit();

    for (result.data, a.data) |*r, val| {
        r.* = val + scalar;
    }

    return result;
}

// ============================================================================
// In-place Element-wise Operations
// ============================================================================

/// In-place addition: A += B
pub fn addInPlace(a: *Tensor, b: *const Tensor) OpsError!void {
    if (!shapesMatch(a.shape, b.shape)) {
        return OpsError.ShapeMismatch;
    }

    for (a.data, b.data) |*val_a, val_b| {
        val_a.* += val_b;
    }
}

/// In-place subtraction: A -= B
pub fn subInPlace(a: *Tensor, b: *const Tensor) OpsError!void {
    if (!shapesMatch(a.shape, b.shape)) {
        return OpsError.ShapeMismatch;
    }

    for (a.data, b.data) |*val_a, val_b| {
        val_a.* -= val_b;
    }
}

/// In-place element-wise multiplication: A *= B
pub fn mulInPlace(a: *Tensor, b: *const Tensor) OpsError!void {
    if (!shapesMatch(a.shape, b.shape)) {
        return OpsError.ShapeMismatch;
    }

    for (a.data, b.data) |*val_a, val_b| {
        val_a.* *= val_b;
    }
}

/// In-place scalar multiplication: A *= scalar
pub fn scaleInPlace(a: *Tensor, scalar: f32) void {
    for (a.data) |*val| {
        val.* *= scalar;
    }
}

/// In-place scalar addition: A += scalar
pub fn addScalarInPlace(a: *Tensor, scalar: f32) void {
    for (a.data) |*val| {
        val.* += scalar;
    }
}

/// Negate all elements in place: A = -A
pub fn negateInPlace(a: *Tensor) void {
    for (a.data) |*val| {
        val.* = -val.*;
    }
}

// ============================================================================
// Matrix Operations
// ============================================================================

/// Matrix multiplication: C = A @ B
/// For A with shape [M, K] and B with shape [K, N], result has shape [M, N].
/// Uses BLAS when available (-Dblas=true), otherwise SIMD-optimized Zig.
/// Automatically uses multithreading if global context is set and workload is large enough.
pub fn matmul(allocator: std.mem.Allocator, a: *const Tensor, b: *const Tensor) !Tensor {
    // Validate: both must be 2D matrices
    if (a.shape.len != 2 or b.shape.len != 2) {
        return OpsError.InvalidShape;
    }

    const m = a.shape[0]; // rows of A
    const k_a = a.shape[1]; // cols of A
    const k_b = b.shape[0]; // rows of B
    const n = b.shape[1]; // cols of B

    // Validate: A columns must match B rows
    if (k_a != k_b) {
        return OpsError.ShapeMismatch;
    }

    const k = k_a;

    // Check if we should use parallel execution (when global context is set)
    // Only parallelize for large enough matrices (m >= min_parallel_rows)
    if (global_context) |ctx| {
        if (ctx.shouldParallelize(m)) {
            return matmulParallel(allocator, ctx, a, b);
        }
    }

    // Result shape: [M, N]
    var result_shape = [_]usize{ m, n };
    var result = try Tensor.init(allocator, &result_shape);
    errdefer result.deinit();

    // Use BLAS if available, otherwise zblas, otherwise fall back to pure Zig SIMD
    if (use_blas) {
        // OpenBLAS path: cblas_sgemm(C = alpha*A*B + beta*C)
        blas.sgemm(m, n, k, a.data, b.data, result.data, 1.0, 0.0);
        return result;
    } else if (use_zblas) {
        // zblas path: pure Zig optimized SGEMM
        zblas.sgemm(m, n, k, a.data, b.data, result.data, 1.0, 0.0);
        return result;
    }

    // Fallback pure Zig path: SIMD-optimized implementation (for benchmarking)
    // Initialize result to zero
    @memset(result.data, 0.0);

    // SIMD vector width (8 floats for AVX/AVX2)
    const VEC_WIDTH = 8;
    const Vec = @Vector(VEC_WIDTH, f32);

    // Register blocking: process MR rows of A at once
    // This keeps MR accumulators in registers, reducing memory traffic
    const MR = 4; // Number of rows to process together
    const NR = 24; // Number of columns per micro-kernel (3 vectors)

    // Main loop with register blocking
    var i: usize = 0;
    while (i + MR <= m) : (i += MR) {
        var j: usize = 0;

        // Vectorized columns (process NR columns at a time)
        while (j + NR <= n) : (j += NR) {
            // Accumulators for MR x NR block (kept in registers)
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

            // Reduction over K dimension
            for (0..k) |kk| {
                // Load 4 elements from column kk of A (broadcast each)
                const a0: Vec = @splat(a.data[(i + 0) * k + kk]);
                const a1: Vec = @splat(a.data[(i + 1) * k + kk]);
                const a2: Vec = @splat(a.data[(i + 2) * k + kk]);
                const a3: Vec = @splat(a.data[(i + 3) * k + kk]);

                // Load 3 vectors (24 elements) from row kk of B
                const b_base = kk * n + j;
                const b0: Vec = b.data[b_base ..][0..VEC_WIDTH].*;
                const b1: Vec = b.data[b_base + VEC_WIDTH ..][0..VEC_WIDTH].*;
                const b2: Vec = b.data[b_base + 2 * VEC_WIDTH ..][0..VEC_WIDTH].*;

                // Accumulate: C[i, j] += A[i, k] * B[k, j]
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

            // Store results
            result.data[(i + 0) * n + j ..][0..VEC_WIDTH].* = c00;
            result.data[(i + 0) * n + j + VEC_WIDTH ..][0..VEC_WIDTH].* = c01;
            result.data[(i + 0) * n + j + 2 * VEC_WIDTH ..][0..VEC_WIDTH].* = c02;
            result.data[(i + 1) * n + j ..][0..VEC_WIDTH].* = c10;
            result.data[(i + 1) * n + j + VEC_WIDTH ..][0..VEC_WIDTH].* = c11;
            result.data[(i + 1) * n + j + 2 * VEC_WIDTH ..][0..VEC_WIDTH].* = c12;
            result.data[(i + 2) * n + j ..][0..VEC_WIDTH].* = c20;
            result.data[(i + 2) * n + j + VEC_WIDTH ..][0..VEC_WIDTH].* = c21;
            result.data[(i + 2) * n + j + 2 * VEC_WIDTH ..][0..VEC_WIDTH].* = c22;
            result.data[(i + 3) * n + j ..][0..VEC_WIDTH].* = c30;
            result.data[(i + 3) * n + j + VEC_WIDTH ..][0..VEC_WIDTH].* = c31;
            result.data[(i + 3) * n + j + 2 * VEC_WIDTH ..][0..VEC_WIDTH].* = c32;
        }

        // Handle remaining columns with smaller vectors
        while (j + VEC_WIDTH <= n) : (j += VEC_WIDTH) {
            var c0: Vec = @splat(0.0);
            var c1: Vec = @splat(0.0);
            var c2: Vec = @splat(0.0);
            var c3: Vec = @splat(0.0);

            for (0..k) |kk| {
                const b_vec: Vec = b.data[kk * n + j ..][0..VEC_WIDTH].*;
                c0 += @as(Vec, @splat(a.data[(i + 0) * k + kk])) * b_vec;
                c1 += @as(Vec, @splat(a.data[(i + 1) * k + kk])) * b_vec;
                c2 += @as(Vec, @splat(a.data[(i + 2) * k + kk])) * b_vec;
                c3 += @as(Vec, @splat(a.data[(i + 3) * k + kk])) * b_vec;
            }

            result.data[(i + 0) * n + j ..][0..VEC_WIDTH].* = c0;
            result.data[(i + 1) * n + j ..][0..VEC_WIDTH].* = c1;
            result.data[(i + 2) * n + j ..][0..VEC_WIDTH].* = c2;
            result.data[(i + 3) * n + j ..][0..VEC_WIDTH].* = c3;
        }

        // Scalar tail for remaining columns
        while (j < n) : (j += 1) {
            var c0: f32 = 0.0;
            var c1: f32 = 0.0;
            var c2: f32 = 0.0;
            var c3: f32 = 0.0;
            for (0..k) |kk| {
                const b_val = b.data[kk * n + j];
                c0 += a.data[(i + 0) * k + kk] * b_val;
                c1 += a.data[(i + 1) * k + kk] * b_val;
                c2 += a.data[(i + 2) * k + kk] * b_val;
                c3 += a.data[(i + 3) * k + kk] * b_val;
            }
            result.data[(i + 0) * n + j] = c0;
            result.data[(i + 1) * n + j] = c1;
            result.data[(i + 2) * n + j] = c2;
            result.data[(i + 3) * n + j] = c3;
        }
    }

    // Handle remaining rows (when M is not divisible by MR)
    while (i < m) : (i += 1) {
        var j: usize = 0;
        while (j + VEC_WIDTH <= n) : (j += VEC_WIDTH) {
            var c: Vec = @splat(0.0);
            for (0..k) |kk| {
                const a_vec: Vec = @splat(a.data[i * k + kk]);
                const b_vec: Vec = b.data[kk * n + j ..][0..VEC_WIDTH].*;
                c += a_vec * b_vec;
            }
            result.data[i * n + j ..][0..VEC_WIDTH].* = c;
        }
        while (j < n) : (j += 1) {
            var c: f32 = 0.0;
            for (0..k) |kk| {
                c += a.data[i * k + kk] * b.data[kk * n + j];
            }
            result.data[i * n + j] = c;
        }
    }

    return result;
}

/// Parallel matrix multiplication: C = A @ B
/// Splits M dimension across threads for parallel execution.
/// Falls back to single-threaded matmul if Context is not parallel or workload is small.
///
/// Parameters:
/// - allocator: Memory allocator for result tensor
/// - ctx: Execution context with thread pool
/// - a: Left matrix [M, K]
/// - b: Right matrix [K, N]
///
/// Returns: Result matrix [M, N]
pub fn matmulParallel(allocator: std.mem.Allocator, ctx: *Context, a: *const Tensor, b: *const Tensor) !Tensor {
    _ = ctx; // Parallel context not used - zblas handles parallelism internally

    // Validate: both must be 2D matrices
    if (a.shape.len != 2 or b.shape.len != 2) {
        return OpsError.InvalidShape;
    }

    const m = a.shape[0]; // rows of A
    const k_a = a.shape[1]; // cols of A
    const k_b = b.shape[0]; // rows of B
    const n = b.shape[1]; // cols of B

    // Validate: A columns must match B rows
    if (k_a != k_b) {
        return OpsError.ShapeMismatch;
    }

    const k = k_a;

    // Result shape: [M, N]
    var result_shape = [_]usize{ m, n };
    var result = try Tensor.init(allocator, &result_shape);
    errdefer result.deinit();

    // Use zblas if available (it handles parallelism internally via sgemmParallel)
    // Otherwise fall back to the pure Zig SIMD implementation
    if (use_zblas) {
        // zblas path: pure Zig optimized SGEMM
        zblas.sgemm(m, n, k, a.data, b.data, result.data, 1.0, 0.0);
        return result;
    } else if (use_blas) {
        blas.sgemm(m, n, k, a.data, b.data, result.data, 1.0, 0.0);
        return result;
    }

    // Fallback: pure Zig SIMD (no parallelism in this path)
    @memset(result.data, 0.0);
    matmulRowRange(a.data, b.data, result.data, k, n, 0, m);

    return result;
}

/// Task function for parallel matmul work (used by spawnWg)
fn matmulRowRangeTask(
    a_data: []const f32,
    b_data: []const f32,
    result_data: []f32,
    k: usize,
    n: usize,
    row_start: usize,
    row_end: usize,
) void {
    matmulRowRange(a_data, b_data, result_data, k, n, row_start, row_end);
}

/// Compute a range of rows for matmul: result[row_start:row_end, :] = a[row_start:row_end, :] @ b
/// This is the inner kernel shared by both single-threaded and parallel matmul.
fn matmulRowRange(
    a_data: []const f32,
    b_data: []const f32,
    result_data: []f32,
    k: usize,
    n: usize,
    row_start: usize,
    row_end: usize,
) void {
    // SIMD vector width (process 8 floats at once)
    const VEC_WIDTH = 8;
    const Vec = @Vector(VEC_WIDTH, f32);

    // Process each row in this thread's range
    for (row_start..row_end) |i| {
        var j: usize = 0;

        // Vectorized columns
        while (j + VEC_WIDTH <= n) : (j += VEC_WIDTH) {
            var acc: Vec = @splat(0.0);
            for (0..k) |kk| {
                const a_val: Vec = @splat(a_data[i * k + kk]);
                const b_vec: Vec = b_data[kk * n + j ..][0..VEC_WIDTH].*;
                acc += a_val * b_vec;
            }
            result_data[i * n + j ..][0..VEC_WIDTH].* = acc;
        }

        // Scalar tail
        while (j < n) : (j += 1) {
            var acc: f32 = 0.0;
            for (0..k) |kk| {
                acc += a_data[i * k + kk] * b_data[kk * n + j];
            }
            result_data[i * n + j] = acc;
        }
    }
}

/// Transpose a 2D matrix: B = A^T
/// For A with shape [M, N], result has shape [N, M].
/// Required for PyTorch weight compatibility (nn.Linear stores [out, in]).
pub fn transpose(allocator: std.mem.Allocator, t: *const Tensor) !Tensor {
    // Validate: must be 2D matrix
    if (t.shape.len != 2) {
        return OpsError.InvalidShape;
    }

    const m = t.shape[0]; // rows of input
    const n = t.shape[1]; // cols of input

    // Result shape: [N, M] (flipped)
    var result_shape = [_]usize{ n, m };
    var result = try Tensor.init(allocator, &result_shape);
    errdefer result.deinit();

    // Transpose: result[j, i] = t[i, j]
    for (0..m) |i| {
        for (0..n) |j| {
            // t[i, j] is at index i * n + j
            // result[j, i] is at index j * m + i
            result.data[j * m + i] = t.data[i * n + j];
        }
    }

    return result;
}

/// Matrix-vector multiplication: y = A @ x
/// For A with shape [M, N] and x with shape [N], result has shape [M].
pub fn matvec(allocator: std.mem.Allocator, a: *const Tensor, x: *const Tensor) !Tensor {
    // Validate shapes
    if (a.shape.len != 2 or x.shape.len != 1) {
        return OpsError.InvalidShape;
    }

    const m = a.shape[0];
    const n = a.shape[1];

    if (x.shape[0] != n) {
        return OpsError.ShapeMismatch;
    }

    var result_shape = [_]usize{m};
    var result = try Tensor.init(allocator, &result_shape);
    errdefer result.deinit();

    for (0..m) |i| {
        var dot: f32 = 0.0;
        for (0..n) |j| {
            dot += a.data[i * n + j] * x.data[j];
        }
        result.data[i] = dot;
    }

    return result;
}

// ============================================================================
// Activation Functions (Allocating)
// ============================================================================

/// ReLU activation: max(0, x)
/// Allocates a new tensor for the result.
pub fn relu(allocator: std.mem.Allocator, a: *const Tensor) !Tensor {
    var result = try Tensor.init(allocator, a.shape);
    errdefer result.deinit();

    for (result.data, a.data) |*r, val| {
        r.* = @max(0.0, val);
    }

    return result;
}

/// Sigmoid activation: 1 / (1 + exp(-x))
/// Allocates a new tensor for the result.
/// Critical for VAD output layer.
pub fn sigmoid(allocator: std.mem.Allocator, a: *const Tensor) !Tensor {
    var result = try Tensor.init(allocator, a.shape);
    errdefer result.deinit();

    for (result.data, a.data) |*r, val| {
        r.* = 1.0 / (1.0 + @exp(-val));
    }

    return result;
}

/// Tanh activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
/// Allocates a new tensor for the result.
/// Critical for LSTM internal state.
pub fn tanh(allocator: std.mem.Allocator, a: *const Tensor) !Tensor {
    var result = try Tensor.init(allocator, a.shape);
    errdefer result.deinit();

    for (result.data, a.data) |*r, val| {
        r.* = std.math.tanh(val);
    }

    return result;
}

// ============================================================================
// Activation Functions (In-place)
// ============================================================================

/// In-place ReLU: x = max(0, x)
pub fn reluInPlace(a: *Tensor) void {
    for (a.data) |*val| {
        val.* = @max(0.0, val.*);
    }
}

/// In-place Sigmoid: x = 1 / (1 + exp(-x))
pub fn sigmoidInPlace(a: *Tensor) void {
    for (a.data) |*val| {
        val.* = 1.0 / (1.0 + @exp(-val.*));
    }
}

/// In-place Tanh: x = tanh(x)
pub fn tanhInPlace(a: *Tensor) void {
    for (a.data) |*val| {
        val.* = std.math.tanh(val.*);
    }
}

// ============================================================================
// Reduction Operations
// ============================================================================

/// Sum all elements in the tensor
pub fn sum(a: *const Tensor) f32 {
    var total: f32 = 0.0;
    for (a.data) |val| {
        total += val;
    }
    return total;
}

/// Find maximum element
pub fn max(a: *const Tensor) f32 {
    if (a.data.len == 0) return 0.0;
    var result = a.data[0];
    for (a.data[1..]) |val| {
        if (val > result) {
            result = val;
        }
    }
    return result;
}

/// Find minimum element
pub fn min(a: *const Tensor) f32 {
    if (a.data.len == 0) return 0.0;
    var result = a.data[0];
    for (a.data[1..]) |val| {
        if (val < result) {
            result = val;
        }
    }
    return result;
}

/// Calculate mean of all elements
pub fn mean(a: *const Tensor) f32 {
    if (a.data.len == 0) return 0.0;
    return sum(a) / @as(f32, @floatFromInt(a.data.len));
}

// ============================================================================
// Tests
// ============================================================================

test "element-wise add" {
    const allocator = std.testing.allocator;
    var shape = [_]usize{ 2, 2 };

    var a = try Tensor.init(allocator, &shape);
    defer a.deinit();
    a.fill(2.0);

    var b = try Tensor.init(allocator, &shape);
    defer b.deinit();
    b.fill(3.0);

    var c = try add(allocator, &a, &b);
    defer c.deinit();

    // 2 + 3 = 5
    try std.testing.expectEqual(@as(f32, 5.0), try c.get(0));
    try std.testing.expectEqual(@as(f32, 5.0), try c.get(1));
    try std.testing.expectEqual(@as(f32, 5.0), try c.get(2));
    try std.testing.expectEqual(@as(f32, 5.0), try c.get(3));
}

test "element-wise sub" {
    const allocator = std.testing.allocator;
    var shape = [_]usize{ 2, 2 };

    var a = try Tensor.init(allocator, &shape);
    defer a.deinit();
    a.fill(5.0);

    var b = try Tensor.init(allocator, &shape);
    defer b.deinit();
    b.fill(3.0);

    var c = try sub(allocator, &a, &b);
    defer c.deinit();

    // 5 - 3 = 2
    try std.testing.expectEqual(@as(f32, 2.0), try c.get(0));
}

test "element-wise mul" {
    const allocator = std.testing.allocator;
    var shape = [_]usize{ 2, 2 };

    var a = try Tensor.init(allocator, &shape);
    defer a.deinit();
    a.fill(2.0);

    var b = try Tensor.init(allocator, &shape);
    defer b.deinit();
    b.fill(3.0);

    var c = try mul(allocator, &a, &b);
    defer c.deinit();

    // 2 * 3 = 6
    try std.testing.expectEqual(@as(f32, 6.0), try c.get(0));
}

test "scalar operations" {
    const allocator = std.testing.allocator;
    var shape = [_]usize{4};

    var a = try Tensor.init(allocator, &shape);
    defer a.deinit();
    a.fill(2.0);

    // Scale: 2 * 3 = 6
    var scaled = try scale(allocator, &a, 3.0);
    defer scaled.deinit();
    try std.testing.expectEqual(@as(f32, 6.0), try scaled.get(0));

    // Add scalar: 2 + 10 = 12
    var added = try addScalar(allocator, &a, 10.0);
    defer added.deinit();
    try std.testing.expectEqual(@as(f32, 12.0), try added.get(0));
}

test "in-place operations" {
    const allocator = std.testing.allocator;
    var shape = [_]usize{ 2, 2 };

    var a = try Tensor.init(allocator, &shape);
    defer a.deinit();
    a.fill(2.0);

    var b = try Tensor.init(allocator, &shape);
    defer b.deinit();
    b.fill(3.0);

    // In-place add: 2 + 3 = 5
    try addInPlace(&a, &b);
    try std.testing.expectEqual(@as(f32, 5.0), try a.get(0));

    // In-place scale: 5 * 2 = 10
    scaleInPlace(&a, 2.0);
    try std.testing.expectEqual(@as(f32, 10.0), try a.get(0));
}

test "shape mismatch error" {
    const allocator = std.testing.allocator;

    var shape_a = [_]usize{ 2, 2 };
    var a = try Tensor.init(allocator, &shape_a);
    defer a.deinit();

    var shape_b = [_]usize{ 2, 3 };
    var b = try Tensor.init(allocator, &shape_b);
    defer b.deinit();

    // Should return ShapeMismatch error
    const result = add(allocator, &a, &b);
    try std.testing.expectError(OpsError.ShapeMismatch, result);
}

test "reduction operations" {
    const allocator = std.testing.allocator;
    var shape = [_]usize{4};
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    var tensor = try Tensor.initWithData(allocator, &shape, &data);
    defer tensor.deinit();

    // Sum: 1 + 2 + 3 + 4 = 10
    try std.testing.expectEqual(@as(f32, 10.0), sum(&tensor));

    // Mean: 10 / 4 = 2.5
    try std.testing.expectEqual(@as(f32, 2.5), mean(&tensor));

    // Max: 4
    try std.testing.expectEqual(@as(f32, 4.0), max(&tensor));

    // Min: 1
    try std.testing.expectEqual(@as(f32, 1.0), min(&tensor));
}

// ============================================================================
// Tests: Matrix Multiplication & Activations
// ============================================================================

test "transpose 2x3 -> 3x2" {
    const allocator = std.testing.allocator;

    // Matrix A (2x3):
    // [ 1, 2, 3 ]
    // [ 4, 5, 6 ]
    var shape_a = [_]usize{ 2, 3 };
    const data_a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var a = try Tensor.initWithData(allocator, &shape_a, &data_a);
    defer a.deinit();

    // Transpose: A^T (3x2)
    // [ 1, 4 ]
    // [ 2, 5 ]
    // [ 3, 6 ]
    var a_t = try transpose(allocator, &a);
    defer a_t.deinit();

    // Verify shape
    try std.testing.expectEqual(@as(usize, 2), a_t.shape.len);
    try std.testing.expectEqual(@as(usize, 3), a_t.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), a_t.shape[1]);

    // Verify data (row-major order)
    // Row 0: [1, 4]
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), a_t.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), a_t.data[1], 0.001);
    // Row 1: [2, 5]
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), a_t.data[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), a_t.data[3], 0.001);
    // Row 2: [3, 6]
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), a_t.data[4], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), a_t.data[5], 0.001);
}

test "transpose then matmul (PyTorch Linear compatibility)" {
    const allocator = std.testing.allocator;

    // Simulating PyTorch nn.Linear weight shape [out_features, in_features]
    // Weight W (2x3) - 2 output features, 3 input features
    var shape_w = [_]usize{ 2, 3 };
    const data_w = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var weight = try Tensor.initWithData(allocator, &shape_w, &data_w);
    defer weight.deinit();

    // Input X (1x3) - batch of 1, 3 input features
    var shape_x = [_]usize{ 1, 3 };
    const data_x = [_]f32{ 1, 1, 1 };
    var input = try Tensor.initWithData(allocator, &shape_x, &data_x);
    defer input.deinit();

    // Transpose weight: W^T (3x2)
    var weight_t = try transpose(allocator, &weight);
    defer weight_t.deinit();

    // Y = X @ W^T -> (1x3) @ (3x2) = (1x2)
    var output = try matmul(allocator, &input, &weight_t);
    defer output.deinit();

    // Expected: [1*1+1*2+1*3, 1*4+1*5+1*6] = [6, 15]
    try std.testing.expectEqual(@as(usize, 2), output.shape.len);
    try std.testing.expectEqual(@as(usize, 1), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), output.shape[1]);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), output.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 15.0), output.data[1], 0.001);
}

test "matmul 2x3 @ 3x2" {
    const allocator = std.testing.allocator;

    // Matrix A (2x3):
    // [ 1, 2, 3 ]
    // [ 4, 5, 6 ]
    var shape_a = [_]usize{ 2, 3 };
    const data_a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var a = try Tensor.initWithData(allocator, &shape_a, &data_a);
    defer a.deinit();

    // Matrix B (3x2):
    // [ 7,  8 ]
    // [ 9, 10 ]
    // [11, 12 ]
    var shape_b = [_]usize{ 3, 2 };
    const data_b = [_]f32{ 7, 8, 9, 10, 11, 12 };
    var b = try Tensor.initWithData(allocator, &shape_b, &data_b);
    defer b.deinit();

    // Expected C (2x2):
    // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    // C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    // C[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    // C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    var c = try matmul(allocator, &a, &b);
    defer c.deinit();

    try std.testing.expectEqual(@as(usize, 2), c.shape.len);
    try std.testing.expectEqual(@as(usize, 2), c.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), c.shape[1]);

    try std.testing.expectApproxEqAbs(@as(f32, 58.0), c.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 64.0), c.data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 139.0), c.data[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 154.0), c.data[3], 0.001);
}

test "matmul shape mismatch" {
    const allocator = std.testing.allocator;

    // A is 2x3, B is 2x2 (incompatible: A cols != B rows)
    var shape_a = [_]usize{ 2, 3 };
    var a = try Tensor.init(allocator, &shape_a);
    defer a.deinit();

    var shape_b = [_]usize{ 2, 2 };
    var b = try Tensor.init(allocator, &shape_b);
    defer b.deinit();

    const result = matmul(allocator, &a, &b);
    try std.testing.expectError(OpsError.ShapeMismatch, result);
}

test "matmul identity" {
    const allocator = std.testing.allocator;

    // A (2x2)
    var shape_a = [_]usize{ 2, 2 };
    const data_a = [_]f32{ 1, 2, 3, 4 };
    var a = try Tensor.initWithData(allocator, &shape_a, &data_a);
    defer a.deinit();

    // Identity matrix I (2x2)
    var shape_i = [_]usize{ 2, 2 };
    const data_i = [_]f32{ 1, 0, 0, 1 };
    var identity = try Tensor.initWithData(allocator, &shape_i, &data_i);
    defer identity.deinit();

    // A @ I = A
    var result = try matmul(allocator, &a, &identity);
    defer result.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), result.data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), result.data[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), result.data[3], 0.001);
}

test "matvec" {
    const allocator = std.testing.allocator;

    // Matrix A (2x3):
    // [ 1, 2, 3 ]
    // [ 4, 5, 6 ]
    var shape_a = [_]usize{ 2, 3 };
    const data_a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var a = try Tensor.initWithData(allocator, &shape_a, &data_a);
    defer a.deinit();

    // Vector x (3):
    // [ 1, 2, 3 ]
    var shape_x = [_]usize{3};
    const data_x = [_]f32{ 1, 2, 3 };
    var x = try Tensor.initWithData(allocator, &shape_x, &data_x);
    defer x.deinit();

    // Expected y (2):
    // y[0] = 1*1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
    // y[1] = 4*1 + 5*2 + 6*3 = 4 + 10 + 18 = 32
    var y = try matvec(allocator, &a, &x);
    defer y.deinit();

    try std.testing.expectEqual(@as(usize, 1), y.shape.len);
    try std.testing.expectEqual(@as(usize, 2), y.shape[0]);
    try std.testing.expectApproxEqAbs(@as(f32, 14.0), y.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), y.data[1], 0.001);
}

test "relu activation" {
    const allocator = std.testing.allocator;

    var shape = [_]usize{5};
    const data = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    var input = try Tensor.initWithData(allocator, &shape, &data);
    defer input.deinit();

    var output = try relu(allocator, &input);
    defer output.deinit();

    // Expected: [0, 0, 0, 1, 2]
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output.data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output.data[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output.data[3], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), output.data[4], 0.001);
}

test "sigmoid activation" {
    const allocator = std.testing.allocator;

    var shape = [_]usize{3};
    const data = [_]f32{ -1.0, 0.0, 1.0 };
    var input = try Tensor.initWithData(allocator, &shape, &data);
    defer input.deinit();

    var output = try sigmoid(allocator, &input);
    defer output.deinit();

    // sigmoid(-1) = 1/(1+e^1) ≈ 0.2689
    // sigmoid(0) = 1/(1+e^0) = 0.5
    // sigmoid(1) = 1/(1+e^-1) ≈ 0.7311
    try std.testing.expectApproxEqAbs(@as(f32, 0.2689), output.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), output.data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7311), output.data[2], 0.001);
}

test "tanh activation" {
    const allocator = std.testing.allocator;

    var shape = [_]usize{3};
    const data = [_]f32{ -1.0, 0.0, 1.0 };
    var input = try Tensor.initWithData(allocator, &shape, &data);
    defer input.deinit();

    var output = try tanh(allocator, &input);
    defer output.deinit();

    // tanh(-1) ≈ -0.7616
    // tanh(0) = 0
    // tanh(1) ≈ 0.7616
    try std.testing.expectApproxEqAbs(@as(f32, -0.7616), output.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output.data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7616), output.data[2], 0.001);
}

test "in-place activations" {
    const allocator = std.testing.allocator;

    // Test in-place ReLU
    var shape = [_]usize{3};
    const data = [_]f32{ -1.0, 0.0, 1.0 };

    var t1 = try Tensor.initWithData(allocator, &shape, &data);
    defer t1.deinit();
    reluInPlace(&t1);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), t1.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), t1.data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), t1.data[2], 0.001);

    // Test in-place Sigmoid
    var t2 = try Tensor.initWithData(allocator, &shape, &data);
    defer t2.deinit();
    sigmoidInPlace(&t2);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2689), t2.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), t2.data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7311), t2.data[2], 0.001);

    // Test in-place Tanh
    var t3 = try Tensor.initWithData(allocator, &shape, &data);
    defer t3.deinit();
    tanhInPlace(&t3);
    try std.testing.expectApproxEqAbs(@as(f32, -0.7616), t3.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), t3.data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7616), t3.data[2], 0.001);
}

// ============================================================================
// LSTM Operations
// ============================================================================

/// LSTM State holds hidden state (h) and cell state (c)
pub const LSTMState = struct {
    h: Tensor, // Hidden state [batch, hidden_size] or [hidden_size]
    c: Tensor, // Cell state [batch, hidden_size] or [hidden_size]
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Initialize LSTM state with zeros
    pub fn init(allocator: std.mem.Allocator, hidden_size: usize) !Self {
        var h_shape = [_]usize{hidden_size};
        var h = try Tensor.init(allocator, &h_shape);
        errdefer h.deinit();

        var c_shape = [_]usize{hidden_size};
        var c = try Tensor.init(allocator, &c_shape);
        errdefer c.deinit();

        return Self{
            .h = h,
            .c = c,
            .allocator = allocator,
        };
    }

    /// Free LSTM state memory
    pub fn deinit(self: *Self) void {
        self.h.deinit();
        self.c.deinit();
    }

    /// Reset state to zeros
    pub fn reset(self: *Self) void {
        self.h.fill(0.0);
        self.c.fill(0.0);
    }
};

/// LSTM Weights (packed as [4*hidden, input] and [4*hidden, hidden])
/// PyTorch packs gates as: input, forget, cell, output
pub const LSTMWeights = struct {
    weight_ih: Tensor, // [4*hidden, input]
    weight_hh: Tensor, // [4*hidden, hidden]
    bias_ih: Tensor, // [4*hidden]
    bias_hh: Tensor, // [4*hidden]
};

/// LSTM Cell forward pass
/// Computes one step of LSTM given input x and previous state
/// Updates state in place with new h and c values
///
/// LSTM equations:
///   i = sigmoid(W_ii @ x + b_ii + W_hi @ h + b_hi)  # input gate
///   f = sigmoid(W_if @ x + b_if + W_hf @ h + b_hf)  # forget gate
///   g = tanh(W_ig @ x + b_ig + W_hg @ h + b_hg)     # cell gate
///   o = sigmoid(W_io @ x + b_io + W_ho @ h + b_ho)  # output gate
///   c_new = f * c + i * g
///   h_new = o * tanh(c_new)
pub fn lstmCell(
    allocator: std.mem.Allocator,
    x: *const Tensor,
    state: *LSTMState,
    weights: *const LSTMWeights,
) !void {
    const hidden_size = state.h.data.len;

    // Compute input-hidden contribution: W_ih @ x + b_ih
    // W_ih is [4*hidden, input], x is [input] -> result is [4*hidden]
    var ih_gates = try matvec(allocator, &weights.weight_ih, x);
    defer ih_gates.deinit();

    // Add input bias
    for (ih_gates.data, weights.bias_ih.data) |*g, b| {
        g.* += b;
    }

    // Compute hidden-hidden contribution: W_hh @ h + b_hh
    var hh_gates = try matvec(allocator, &weights.weight_hh, &state.h);
    defer hh_gates.deinit();

    // Add hidden bias
    for (hh_gates.data, weights.bias_hh.data) |*g, b| {
        g.* += b;
    }

    // Combine: gates = ih_gates + hh_gates
    for (ih_gates.data, hh_gates.data) |*ih, hh| {
        ih.* += hh;
    }

    // Split into 4 gates and apply activations
    // Gates are packed as [i, f, g, o], each of size hidden_size
    const i_start: usize = 0;
    const f_start: usize = hidden_size;
    const g_start: usize = 2 * hidden_size;
    const o_start: usize = 3 * hidden_size;

    // Apply activations in place on the combined gates tensor
    // i = sigmoid(gates[0:hidden])
    for (ih_gates.data[i_start .. i_start + hidden_size]) |*val| {
        val.* = 1.0 / (1.0 + @exp(-val.*));
    }
    // f = sigmoid(gates[hidden:2*hidden])
    for (ih_gates.data[f_start .. f_start + hidden_size]) |*val| {
        val.* = 1.0 / (1.0 + @exp(-val.*));
    }
    // g = tanh(gates[2*hidden:3*hidden])
    for (ih_gates.data[g_start .. g_start + hidden_size]) |*val| {
        val.* = std.math.tanh(val.*);
    }
    // o = sigmoid(gates[3*hidden:4*hidden])
    for (ih_gates.data[o_start .. o_start + hidden_size]) |*val| {
        val.* = 1.0 / (1.0 + @exp(-val.*));
    }

    // Compute new cell state: c_new = f * c + i * g
    for (state.c.data, 0..) |*c_val, idx| {
        const f_val = ih_gates.data[f_start + idx];
        const i_val = ih_gates.data[i_start + idx];
        const g_val = ih_gates.data[g_start + idx];
        c_val.* = f_val * c_val.* + i_val * g_val;
    }

    // Compute new hidden state: h_new = o * tanh(c_new)
    for (state.h.data, 0..) |*h_val, idx| {
        const o_val = ih_gates.data[o_start + idx];
        const c_val = state.c.data[idx];
        h_val.* = o_val * std.math.tanh(c_val);
    }
}

// ============================================================================
// Conv1D Operations
// ============================================================================

/// 1D Convolution operation
/// Input: [in_channels, width]
/// Weight: [out_channels, in_channels, kernel_size]
/// Output: [out_channels, output_width]
/// where output_width = (width + 2*padding - kernel_size) / stride + 1
pub fn conv1d(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    weight: *const Tensor,
    bias: ?*const Tensor,
    stride: usize,
    padding: usize,
) !Tensor {
    // Validate shapes
    if (input.shape.len != 2 or weight.shape.len != 3) {
        return OpsError.InvalidShape;
    }

    const in_channels = input.shape[0];
    const in_width = input.shape[1];
    const out_channels = weight.shape[0];
    const weight_in_channels = weight.shape[1];
    const kernel_size = weight.shape[2];

    if (in_channels != weight_in_channels) {
        return OpsError.ShapeMismatch;
    }

    // Compute output width
    const padded_width = in_width + 2 * padding;
    if (padded_width < kernel_size) {
        return OpsError.InvalidShape;
    }
    const out_width = (padded_width - kernel_size) / stride + 1;

    // Allocate output tensor [out_channels, out_width]
    var out_shape = [_]usize{ out_channels, out_width };
    var output = try Tensor.init(allocator, &out_shape);
    errdefer output.deinit();

    // Naive convolution implementation
    for (0..out_channels) |oc| {
        for (0..out_width) |ow| {
            var sum_val: f32 = 0.0;

            for (0..in_channels) |ic| {
                for (0..kernel_size) |k| {
                    // Calculate input position (with padding consideration)
                    const in_pos_signed: i64 = @as(i64, @intCast(ow * stride + k)) - @as(i64, @intCast(padding));

                    if (in_pos_signed >= 0 and in_pos_signed < @as(i64, @intCast(in_width))) {
                        const in_pos: usize = @intCast(in_pos_signed);
                        // input[ic, in_pos]
                        const in_val = input.data[ic * in_width + in_pos];
                        // weight[oc, ic, k]
                        const w_val = weight.data[oc * (in_channels * kernel_size) + ic * kernel_size + k];
                        sum_val += in_val * w_val;
                    }
                }
            }

            // Add bias if provided
            if (bias) |b| {
                sum_val += b.data[oc];
            }

            // output[oc, ow]
            output.data[oc * out_width + ow] = sum_val;
        }
    }

    return output;
}

/// Slice a 1D tensor from start to end (exclusive)
/// Allocates a new tensor with the sliced data
pub fn slice1d(allocator: std.mem.Allocator, t: *const Tensor, start: usize, end: usize) !Tensor {
    if (t.shape.len != 1) {
        return OpsError.InvalidShape;
    }
    if (start >= end or end > t.data.len) {
        return OpsError.OutOfBounds;
    }

    const len = end - start;
    var shape = [_]usize{len};
    var result = try Tensor.init(allocator, &shape);
    errdefer result.deinit();

    @memcpy(result.data, t.data[start..end]);
    return result;
}

/// Add a 1D bias to each channel of a 2D tensor
/// input: [channels, width], bias: [channels]
/// Modifies input in place
pub fn addBias2d(input: *Tensor, bias: *const Tensor) OpsError!void {
    if (input.shape.len != 2 or bias.shape.len != 1) {
        return OpsError.InvalidShape;
    }
    if (input.shape[0] != bias.shape[0]) {
        return OpsError.ShapeMismatch;
    }

    const channels = input.shape[0];
    const width = input.shape[1];

    for (0..channels) |c| {
        const bias_val = bias.data[c];
        for (0..width) |w| {
            input.data[c * width + w] += bias_val;
        }
    }
}

// ============================================================================
// Tests: LSTM and Conv1D
// ============================================================================

test "lstm state init and reset" {
    const allocator = std.testing.allocator;

    var state = try LSTMState.init(allocator, 8);
    defer state.deinit();

    // State should be initialized to zeros
    for (state.h.data) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
    }
    for (state.c.data) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
    }

    // Modify state
    state.h.fill(1.0);
    state.c.fill(2.0);

    // Reset should zero it out
    state.reset();
    for (state.h.data) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
    }
    for (state.c.data) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
    }
}

test "lstm cell forward" {
    const allocator = std.testing.allocator;

    const input_size: usize = 4;
    const hidden_size: usize = 8;

    // Initialize state
    var state = try LSTMState.init(allocator, hidden_size);
    defer state.deinit();

    // Create weights with small values
    var weight_ih_shape = [_]usize{ 4 * hidden_size, input_size };
    var weight_ih = try Tensor.init(allocator, &weight_ih_shape);
    defer weight_ih.deinit();
    weight_ih.fill(0.1);

    var weight_hh_shape = [_]usize{ 4 * hidden_size, hidden_size };
    var weight_hh = try Tensor.init(allocator, &weight_hh_shape);
    defer weight_hh.deinit();
    weight_hh.fill(0.1);

    var bias_shape = [_]usize{4 * hidden_size};
    var bias_ih = try Tensor.init(allocator, &bias_shape);
    defer bias_ih.deinit();

    var bias_hh = try Tensor.init(allocator, &bias_shape);
    defer bias_hh.deinit();

    const weights = LSTMWeights{
        .weight_ih = weight_ih,
        .weight_hh = weight_hh,
        .bias_ih = bias_ih,
        .bias_hh = bias_hh,
    };

    // Create input
    var input_shape = [_]usize{input_size};
    var x = try Tensor.init(allocator, &input_shape);
    defer x.deinit();
    x.fill(1.0);

    // Run LSTM cell
    try lstmCell(allocator, &x, &state, &weights);

    // Hidden state should be non-zero after processing
    var h_sum: f32 = 0;
    for (state.h.data) |v| h_sum += @abs(v);
    try std.testing.expect(h_sum > 0.0);

    // Cell state should also be non-zero
    var c_sum: f32 = 0;
    for (state.c.data) |v| c_sum += @abs(v);
    try std.testing.expect(c_sum > 0.0);

    // Run another step - state should change
    const old_h0 = state.h.data[0];
    try lstmCell(allocator, &x, &state, &weights);
    try std.testing.expect(state.h.data[0] != old_h0);
}

test "conv1d basic" {
    const allocator = std.testing.allocator;

    // Input: 2 channels, width 5
    var input_shape = [_]usize{ 2, 5 };
    var input = try Tensor.init(allocator, &input_shape);
    defer input.deinit();
    // Channel 0: [1, 2, 3, 4, 5]
    // Channel 1: [1, 1, 1, 1, 1]
    input.data[0] = 1.0;
    input.data[1] = 2.0;
    input.data[2] = 3.0;
    input.data[3] = 4.0;
    input.data[4] = 5.0;
    input.data[5] = 1.0;
    input.data[6] = 1.0;
    input.data[7] = 1.0;
    input.data[8] = 1.0;
    input.data[9] = 1.0;

    // Weight: 1 output channel, 2 input channels, kernel size 3
    var weight_shape = [_]usize{ 1, 2, 3 };
    var weight = try Tensor.init(allocator, &weight_shape);
    defer weight.deinit();
    // All weights = 1.0
    weight.fill(1.0);

    // Bias
    var bias_shape = [_]usize{1};
    var bias = try Tensor.init(allocator, &bias_shape);
    defer bias.deinit();
    bias.data[0] = 0.5;

    // Conv with stride=1, padding=0
    // Output width = (5 + 0 - 3) / 1 + 1 = 3
    var output = try conv1d(allocator, &input, &weight, &bias, 1, 0);
    defer output.deinit();

    // Verify output shape [1, 3]
    try std.testing.expectEqual(@as(usize, 2), output.shape.len);
    try std.testing.expectEqual(@as(usize, 1), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 3), output.shape[1]);

    // Position 0: (1+2+3) + (1+1+1) + 0.5 = 6 + 3 + 0.5 = 9.5
    try std.testing.expectApproxEqAbs(@as(f32, 9.5), output.data[0], 0.001);

    // Position 1: (2+3+4) + (1+1+1) + 0.5 = 9 + 3 + 0.5 = 12.5
    try std.testing.expectApproxEqAbs(@as(f32, 12.5), output.data[1], 0.001);

    // Position 2: (3+4+5) + (1+1+1) + 0.5 = 12 + 3 + 0.5 = 15.5
    try std.testing.expectApproxEqAbs(@as(f32, 15.5), output.data[2], 0.001);
}

test "conv1d with padding" {
    const allocator = std.testing.allocator;

    // Input: 1 channel, width 3
    var input_shape = [_]usize{ 1, 3 };
    var input = try Tensor.init(allocator, &input_shape);
    defer input.deinit();
    input.data[0] = 1.0;
    input.data[1] = 2.0;
    input.data[2] = 3.0;

    // Weight: 1 output channel, 1 input channel, kernel size 3
    var weight_shape = [_]usize{ 1, 1, 3 };
    var weight = try Tensor.init(allocator, &weight_shape);
    defer weight.deinit();
    weight.fill(1.0);

    // Conv with padding=1 (same padding for kernel 3)
    // Output width = (3 + 2 - 3) / 1 + 1 = 3
    var output = try conv1d(allocator, &input, &weight, null, 1, 1);
    defer output.deinit();

    // Verify output shape [1, 3]
    try std.testing.expectEqual(@as(usize, 3), output.shape[1]);

    // Position 0: 0 + 1 + 2 = 3 (left pad is 0)
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), output.data[0], 0.001);

    // Position 1: 1 + 2 + 3 = 6
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), output.data[1], 0.001);

    // Position 2: 2 + 3 + 0 = 5 (right pad is 0)
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), output.data[2], 0.001);
}

test "slice1d" {
    const allocator = std.testing.allocator;

    var shape = [_]usize{10};
    var t = try Tensor.init(allocator, &shape);
    defer t.deinit();
    for (t.data, 0..) |*v, i| {
        v.* = @floatFromInt(i);
    }

    // Slice [2:5]
    var sliced = try slice1d(allocator, &t, 2, 5);
    defer sliced.deinit();

    try std.testing.expectEqual(@as(usize, 3), sliced.data.len);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), sliced.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), sliced.data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), sliced.data[2], 0.001);
}

// ============================================================================
// Transformer Primitives
// ============================================================================

/// Embedding lookup: convert token IDs to vectors
/// input_ids: array of token indices
/// weight: [vocab_size, embedding_dim] embedding table
/// Returns: [input_len, embedding_dim] tensor
pub fn embedding(
    allocator: std.mem.Allocator,
    input_ids: []const u32,
    weight: *const Tensor,
) !Tensor {
    if (weight.shape.len != 2) {
        return OpsError.InvalidShape;
    }

    const vocab_size = weight.shape[0];
    const embed_dim = weight.shape[1];
    const seq_len = input_ids.len;

    var result_shape = [_]usize{ seq_len, embed_dim };
    var result = try Tensor.init(allocator, &result_shape);
    errdefer result.deinit();

    for (input_ids, 0..) |token_id, i| {
        if (token_id >= vocab_size) return OpsError.OutOfBounds;

        // Copy row from embedding table
        const src_start = token_id * embed_dim;
        const dst_start = i * embed_dim;
        @memcpy(
            result.data[dst_start..][0..embed_dim],
            weight.data[src_start..][0..embed_dim],
        );
    }

    return result;
}

/// Layer Normalization
/// Normalizes across the last dimension (features)
/// y = (x - mean) / sqrt(var + eps) * gamma + beta
/// SIMD-optimized for AVX (8-wide vectors)
pub fn layerNorm(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    gamma: *const Tensor,
    beta: *const Tensor,
    epsilon: f32,
) !Tensor {
    var result = try input.clone(allocator);
    errdefer result.deinit();

    // Use the optimized in-place version (which does all validation)
    try layerNormInPlace(&result, gamma, beta, epsilon);

    return result;
}

/// In-place layer normalization
/// SIMD-optimized for AVX (8-wide vectors)
pub fn layerNormInPlace(
    input: *Tensor,
    gamma: *const Tensor,
    beta: *const Tensor,
    epsilon: f32,
) OpsError!void {
    if (input.shape.len != 2) {
        return OpsError.InvalidShape;
    }

    const seq_len = input.shape[0];
    const hidden_dim = input.shape[1];

    if (gamma.shape.len != 1 or gamma.shape[0] != hidden_dim) {
        return OpsError.ShapeMismatch;
    }
    if (beta.shape.len != 1 or beta.shape[0] != hidden_dim) {
        return OpsError.ShapeMismatch;
    }

    const VEC_WIDTH = 8;
    const Vec = @Vector(VEC_WIDTH, f32);
    const hidden_dim_f: f32 = @floatFromInt(hidden_dim);

    for (0..seq_len) |row| {
        const row_start = row * hidden_dim;
        const row_data = input.data[row_start..][0..hidden_dim];

        // Calculate mean using SIMD
        var mean_val: f32 = 0.0;
        if (hidden_dim >= VEC_WIDTH) {
            var sum_vec: Vec = @splat(0.0);
            var i: usize = 0;
            while (i + VEC_WIDTH <= hidden_dim) : (i += VEC_WIDTH) {
                const v: Vec = row_data[i..][0..VEC_WIDTH].*;
                sum_vec += v;
            }
            mean_val = @reduce(.Add, sum_vec);
            // Handle remainder
            while (i < hidden_dim) : (i += 1) {
                mean_val += row_data[i];
            }
        } else {
            for (row_data) |v| mean_val += v;
        }
        mean_val /= hidden_dim_f;

        // Calculate variance using SIMD
        var variance: f32 = 0.0;
        const mean_vec: Vec = @splat(mean_val);
        if (hidden_dim >= VEC_WIDTH) {
            var var_vec: Vec = @splat(0.0);
            var i: usize = 0;
            while (i + VEC_WIDTH <= hidden_dim) : (i += VEC_WIDTH) {
                const v: Vec = row_data[i..][0..VEC_WIDTH].*;
                const diff = v - mean_vec;
                var_vec += diff * diff;
            }
            variance = @reduce(.Add, var_vec);
            // Handle remainder
            while (i < hidden_dim) : (i += 1) {
                const diff = row_data[i] - mean_val;
                variance += diff * diff;
            }
        } else {
            for (row_data) |v| {
                const diff = v - mean_val;
                variance += diff * diff;
            }
        }
        variance /= hidden_dim_f;

        // Normalize, scale, and shift using SIMD
        const inv_std: f32 = 1.0 / @sqrt(variance + epsilon);
        const inv_std_vec: Vec = @splat(inv_std);

        if (hidden_dim >= VEC_WIDTH) {
            var i: usize = 0;
            while (i + VEC_WIDTH <= hidden_dim) : (i += VEC_WIDTH) {
                const v: Vec = row_data[i..][0..VEC_WIDTH].*;
                const gamma_v: Vec = gamma.data[i..][0..VEC_WIDTH].*;
                const beta_v: Vec = beta.data[i..][0..VEC_WIDTH].*;
                const normalized = (v - mean_vec) * inv_std_vec;
                row_data[i..][0..VEC_WIDTH].* = normalized * gamma_v + beta_v;
            }
            // Handle remainder
            while (i < hidden_dim) : (i += 1) {
                const normalized = (row_data[i] - mean_val) * inv_std;
                row_data[i] = normalized * gamma.data[i] + beta.data[i];
            }
        } else {
            for (row_data, 0..) |*v, i| {
                const normalized = (v.* - mean_val) * inv_std;
                v.* = normalized * gamma.data[i] + beta.data[i];
            }
        }
    }
}

/// GELU activation (Gaussian Error Linear Unit) - Tanh Approximation
/// Used by GPT-2, BERT/DistilBERT
/// Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
/// SIMD-optimized for AVX (8-wide vectors)
pub fn gelu(tensor: *Tensor) void {
    const sqrt_2_over_pi: f32 = 0.7978845608; // sqrt(2/pi)
    const coeff: f32 = 0.044715;

    const VEC_WIDTH = 8;
    const Vec = @Vector(VEC_WIDTH, f32);
    const len = tensor.data.len;

    const sqrt_vec: Vec = @splat(sqrt_2_over_pi);
    const coeff_vec: Vec = @splat(coeff);
    const half_vec: Vec = @splat(0.5);
    const one_vec: Vec = @splat(1.0);

    var i: usize = 0;
    while (i + VEC_WIDTH <= len) : (i += VEC_WIDTH) {
        const x: Vec = tensor.data[i..][0..VEC_WIDTH].*;
        const x2 = x * x;
        const x3 = x2 * x;
        const inner = sqrt_vec * (x + coeff_vec * x3);
        // Fully vectorized tanh approximation (rational Padé, max error ~3e-7)
        const tanh_v = tanhApproxVec(inner);
        tensor.data[i..][0..VEC_WIDTH].* = half_vec * x * (one_vec + tanh_v);
    }
    // Handle remainder
    while (i < len) : (i += 1) {
        const x = tensor.data[i];
        const x3 = x * x * x;
        const inner = sqrt_2_over_pi * (x + coeff * x3);
        tensor.data[i] = 0.5 * x * (1.0 + std.math.tanh(inner));
    }
}

/// Fully vectorized tanh approximation using rational Padé approximant.
/// tanh(x) ≈ x * (27 + x²) / (27 + 9*x²)  for |x| ≤ ~4.5
/// Clamped to ±1 for large inputs. Max error ~3e-7 in the GELU operating range.
fn tanhApproxVec(x: @Vector(8, f32)) @Vector(8, f32) {
    const Vec = @Vector(8, f32);
    const ones: Vec = @splat(1.0);
    const neg_ones: Vec = @splat(-1.0);
    const c27: Vec = @splat(27.0);
    const c9: Vec = @splat(9.0);

    const x2 = x * x;
    const num = x * (c27 + x2);
    const den = c27 + c9 * x2;
    const result = num / den;

    // Clamp to [-1, 1] for large |x|
    return @min(ones, @max(neg_ones, result));
}

/// Approximate error function (erf) using Abramowitz and Stegun approximation
/// Maximum error: 1.5e-7
fn erf(x: f32) f32 {
    // Constants for the approximation
    const a1: f32 = 0.254829592;
    const a2: f32 = -0.284496736;
    const a3: f32 = 1.421413741;
    const a4: f32 = -1.453152027;
    const a5: f32 = 1.061405429;
    const p: f32 = 0.3275911;

    // Save the sign of x
    const sign: f32 = if (x < 0) -1.0 else 1.0;
    const abs_x = @abs(x);

    // A&S formula 7.1.26
    const t = 1.0 / (1.0 + p * abs_x);
    const t2 = t * t;
    const t3 = t2 * t;
    const t4 = t3 * t;
    const t5 = t4 * t;

    const y = 1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * @exp(-abs_x * abs_x);

    return sign * y;
}

/// Exact GELU activation using error function
/// Used by Whisper and other models that use approximate='none'
/// Formula: x * 0.5 * (1 + erf(x / sqrt(2)))
/// SIMD-optimized for AVX (8-wide vectors)
pub fn geluExact(tensor: *Tensor) void {
    const inv_sqrt_2: f32 = 0.7071067811865476; // 1/sqrt(2)

    const VEC_WIDTH = 8;
    const Vec = @Vector(VEC_WIDTH, f32);
    const len = tensor.data.len;

    const inv_sqrt_2_vec: Vec = @splat(inv_sqrt_2);
    const half_vec: Vec = @splat(0.5);
    const one_vec: Vec = @splat(1.0);

    var i: usize = 0;
    while (i + VEC_WIDTH <= len) : (i += VEC_WIDTH) {
        const x: Vec = tensor.data[i..][0..VEC_WIDTH].*;
        const scaled = x * inv_sqrt_2_vec;
        // Vectorized erf
        const erf_v = Vec{
            erf(scaled[0]), erf(scaled[1]), erf(scaled[2]), erf(scaled[3]),
            erf(scaled[4]), erf(scaled[5]), erf(scaled[6]), erf(scaled[7]),
        };
        tensor.data[i..][0..VEC_WIDTH].* = x * half_vec * (one_vec + erf_v);
    }
    // Handle remainder
    while (i < len) : (i += 1) {
        const x = tensor.data[i];
        const erf_val = erf(x * inv_sqrt_2);
        tensor.data[i] = x * 0.5 * (1.0 + erf_val);
    }
}

/// Allocating version of GELU (tanh approximation)
pub fn geluAlloc(allocator: std.mem.Allocator, tensor: *const Tensor) !Tensor {
    var result = try tensor.clone(allocator);
    gelu(&result);
    return result;
}

/// Allocating version of exact GELU
pub fn geluExactAlloc(allocator: std.mem.Allocator, tensor: *const Tensor) !Tensor {
    var result = try tensor.clone(allocator);
    geluExact(&result);
    return result;
}

/// Softmax along last dimension (rows)
/// Uses stable softmax: subtract max before exp to prevent overflow
/// SIMD-optimized for AVX (8-wide vectors)
pub fn softmax(tensor: *Tensor) void {
    if (tensor.shape.len != 2) {
        return; // Only support 2D tensors for now
    }

    const rows = tensor.shape[0];
    const cols = tensor.shape[1];

    const VEC_WIDTH = 8;
    const Vec = @Vector(VEC_WIDTH, f32);

    for (0..rows) |row| {
        const row_start = row * cols;
        const row_data = tensor.data[row_start..][0..cols];

        // Find max for numerical stability using SIMD
        var max_val: f32 = row_data[0];

        if (cols >= VEC_WIDTH) {
            var max_vec: Vec = @splat(-std.math.inf(f32));
            var i: usize = 0;
            while (i + VEC_WIDTH <= cols) : (i += VEC_WIDTH) {
                const v: Vec = row_data[i..][0..VEC_WIDTH].*;
                max_vec = @max(max_vec, v);
            }
            // Reduce vector to scalar
            max_val = @reduce(.Max, max_vec);
            // Handle remainder
            while (i < cols) : (i += 1) {
                if (row_data[i] > max_val) max_val = row_data[i];
            }
        } else {
            for (row_data[1..]) |v| {
                if (v > max_val) max_val = v;
            }
        }

        // Compute exp(x - max) and sum using SIMD
        var sum_val: f32 = 0.0;
        const max_vec: Vec = @splat(max_val);

        if (cols >= VEC_WIDTH) {
            var sum_vec: Vec = @splat(0.0);
            var i: usize = 0;
            while (i + VEC_WIDTH <= cols) : (i += VEC_WIDTH) {
                const v: Vec = row_data[i..][0..VEC_WIDTH].*;
                const shifted = v - max_vec;
                // Vectorized exp
                const exp_v = Vec{
                    @exp(shifted[0]), @exp(shifted[1]), @exp(shifted[2]), @exp(shifted[3]),
                    @exp(shifted[4]), @exp(shifted[5]), @exp(shifted[6]), @exp(shifted[7]),
                };
                row_data[i..][0..VEC_WIDTH].* = exp_v;
                sum_vec += exp_v;
            }
            sum_val = @reduce(.Add, sum_vec);
            // Handle remainder
            while (i < cols) : (i += 1) {
                row_data[i] = @exp(row_data[i] - max_val);
                sum_val += row_data[i];
            }
        } else {
            for (row_data) |*v| {
                v.* = @exp(v.* - max_val);
                sum_val += v.*;
            }
        }

        // Normalize using SIMD
        const inv_sum: Vec = @splat(1.0 / sum_val);
        if (cols >= VEC_WIDTH) {
            var i: usize = 0;
            while (i + VEC_WIDTH <= cols) : (i += VEC_WIDTH) {
                const v: Vec = row_data[i..][0..VEC_WIDTH].*;
                row_data[i..][0..VEC_WIDTH].* = v * inv_sum;
            }
            const inv_sum_scalar = 1.0 / sum_val;
            while (i < cols) : (i += 1) {
                row_data[i] *= inv_sum_scalar;
            }
        } else {
            for (row_data) |*v| {
                v.* /= sum_val;
            }
        }
    }
}

/// Allocating softmax
pub fn softmaxAlloc(allocator: std.mem.Allocator, tensor: *const Tensor) !Tensor {
    var result = try tensor.clone(allocator);
    softmax(&result);
    return result;
}

/// Add bias to each row of a 2D tensor
/// input: [seq_len, hidden_dim], bias: [hidden_dim]
/// Modifies input in place
pub fn addBiasInPlace(input: *Tensor, bias: *const Tensor) OpsError!void {
    if (input.shape.len != 2 or bias.shape.len != 1) {
        return OpsError.InvalidShape;
    }

    const seq_len = input.shape[0];
    const hidden_dim = input.shape[1];

    if (bias.shape[0] != hidden_dim) {
        return OpsError.ShapeMismatch;
    }

    const VEC_WIDTH = 8;
    const Vec = @Vector(VEC_WIDTH, f32);

    for (0..seq_len) |row| {
        const row_start = row * hidden_dim;
        var col: usize = 0;
        while (col + VEC_WIDTH <= hidden_dim) : (col += VEC_WIDTH) {
            const inp: Vec = input.data[row_start + col ..][0..VEC_WIDTH].*;
            const b: Vec = bias.data[col..][0..VEC_WIDTH].*;
            input.data[row_start + col ..][0..VEC_WIDTH].* = inp + b;
        }
        while (col < hidden_dim) : (col += 1) {
            input.data[row_start + col] += bias.data[col];
        }
    }
}

/// Slice columns from a 2D tensor
/// Returns a new tensor with columns [start, end)
pub fn sliceColumns(
    allocator: std.mem.Allocator,
    t: *const Tensor,
    start: usize,
    end: usize,
) !Tensor {
    if (t.shape.len != 2) {
        return OpsError.InvalidShape;
    }
    if (start >= end or end > t.shape[1]) {
        return OpsError.OutOfBounds;
    }

    const rows = t.shape[0];
    const orig_cols = t.shape[1];
    const new_cols = end - start;

    var out_shape = [_]usize{ rows, new_cols };
    var result = try Tensor.init(allocator, &out_shape);
    errdefer result.deinit();

    for (0..rows) |row| {
        const src_row_start = row * orig_cols + start;
        const dst_row_start = row * new_cols;
        @memcpy(
            result.data[dst_row_start..][0..new_cols],
            t.data[src_row_start..][0..new_cols],
        );
    }

    return result;
}

/// Concatenate multiple tensors along columns (axis 1)
/// All tensors must have the same number of rows
pub fn concatColumns(
    allocator: std.mem.Allocator,
    tensors: []const Tensor,
) !Tensor {
    if (tensors.len == 0) {
        return OpsError.InvalidShape;
    }

    // Verify all tensors are 2D with same number of rows
    const rows = tensors[0].shape[0];
    var total_cols: usize = 0;

    for (tensors) |t| {
        if (t.shape.len != 2) {
            return OpsError.InvalidShape;
        }
        if (t.shape[0] != rows) {
            return OpsError.ShapeMismatch;
        }
        total_cols += t.shape[1];
    }

    var out_shape = [_]usize{ rows, total_cols };
    var result = try Tensor.init(allocator, &out_shape);
    errdefer result.deinit();

    for (0..rows) |row| {
        var col_offset: usize = 0;
        for (tensors) |t| {
            const t_cols = t.shape[1];
            const src_start = row * t_cols;
            const dst_start = row * total_cols + col_offset;
            @memcpy(
                result.data[dst_start..][0..t_cols],
                t.data[src_start..][0..t_cols],
            );
            col_offset += t_cols;
        }
    }

    return result;
}

/// Argmax along last dimension (rows)
/// Returns indices of maximum values for each row
pub fn argmax(allocator: std.mem.Allocator, tensor: *const Tensor) ![]usize {
    if (tensor.shape.len != 2) {
        return OpsError.InvalidShape;
    }

    const rows = tensor.shape[0];
    const cols = tensor.shape[1];

    var indices = try allocator.alloc(usize, rows);
    errdefer allocator.free(indices);

    for (0..rows) |row| {
        const row_start = row * cols;
        var max_idx: usize = 0;
        var max_val = tensor.data[row_start];

        for (1..cols) |col| {
            if (tensor.data[row_start + col] > max_val) {
                max_val = tensor.data[row_start + col];
                max_idx = col;
            }
        }
        indices[row] = max_idx;
    }

    return indices;
}

// ============================================================================
// Causal Masking (for Decoder Self-Attention)
// ============================================================================

/// Apply causal mask to attention scores in-place.
/// Masks positions where column > row with -inf (future positions cannot be attended to).
/// scores: [seq_len, seq_len] attention scores tensor
pub fn applyCausalMask(scores: *Tensor) void {
    std.debug.assert(scores.shape.len == 2);
    std.debug.assert(scores.shape[0] == scores.shape[1]);

    const seq_len = scores.shape[0];
    const neg_inf = -std.math.inf(f32);

    for (0..seq_len) |row| {
        // Mask all columns after the current row (future positions)
        for ((row + 1)..seq_len) |col| {
            scores.data[row * seq_len + col] = neg_inf;
        }
    }
}

/// Create a causal attention mask tensor.
/// Returns: [seq_len, seq_len] with 0.0 for valid positions, -inf for masked positions.
/// Can be added to attention scores before softmax.
pub fn createCausalMask(allocator: std.mem.Allocator, seq_len: usize) !Tensor {
    var out_shape = [_]usize{ seq_len, seq_len };
    var mask = try Tensor.init(allocator, &out_shape);
    errdefer mask.deinit();

    const neg_inf = -std.math.inf(f32);

    for (0..seq_len) |row| {
        for (0..seq_len) |col| {
            if (col <= row) {
                // Current and past positions are valid (0 added to scores)
                mask.data[row * seq_len + col] = 0.0;
            } else {
                // Future positions are masked (-inf added to scores -> 0 after softmax)
                mask.data[row * seq_len + col] = neg_inf;
            }
        }
    }

    return mask;
}

/// Apply an additive attention mask to scores in-place.
/// mask: [query_len, key_len] with 0 for valid, -inf for masked positions
/// scores: [query_len, key_len] attention scores
/// Used for both causal masking and padding masks.
pub fn applyAttentionMask(scores: *Tensor, mask: *const Tensor) OpsError!void {
    if (!shapesMatch(scores.shape, mask.shape)) {
        return OpsError.ShapeMismatch;
    }

    for (scores.data, mask.data) |*s, m| {
        s.* += m;
    }
}

// ============================================================================
// Tests: Transformer Primitives
// ============================================================================

test "embedding lookup" {
    const allocator = std.testing.allocator;

    // Embedding table: 3 tokens, 2 dims each
    // [[1,1], [2,2], [3,3]]
    var weight_shape = [_]usize{ 3, 2 };
    var weight = try Tensor.init(allocator, &weight_shape);
    defer weight.deinit();
    weight.data[0] = 1;
    weight.data[1] = 1; // token 0
    weight.data[2] = 2;
    weight.data[3] = 2; // token 1
    weight.data[4] = 3;
    weight.data[5] = 3; // token 2

    // Look up tokens [2, 0]
    const ids = [_]u32{ 2, 0 };
    var result = try embedding(allocator, &ids, &weight);
    defer result.deinit();

    // Expect [[3,3], [1,1]]
    try std.testing.expectEqual(@as(usize, 2), result.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), result.shape[1]);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), result.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), result.data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result.data[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result.data[3], 0.001);
}

test "embedding invalid token" {
    const allocator = std.testing.allocator;

    var weight_shape = [_]usize{ 3, 2 };
    var weight = try Tensor.init(allocator, &weight_shape);
    defer weight.deinit();

    // Token ID 5 is out of bounds for vocab size 3
    const ids = [_]u32{5};
    const result = embedding(allocator, &ids, &weight);
    try std.testing.expectError(OpsError.OutOfBounds, result);
}

test "layer normalization" {
    const allocator = std.testing.allocator;

    // Input: [1, 2, 3]
    var input_shape = [_]usize{ 1, 3 };
    var input = try Tensor.init(allocator, &input_shape);
    defer input.deinit();
    input.data[0] = 1.0;
    input.data[1] = 2.0;
    input.data[2] = 3.0;

    // gamma = [1, 1, 1], beta = [0, 0, 0] (no scaling/shifting)
    var param_shape = [_]usize{3};
    var gamma_tensor = try Tensor.init(allocator, &param_shape);
    defer gamma_tensor.deinit();
    gamma_tensor.fill(1.0);

    var beta_tensor = try Tensor.init(allocator, &param_shape);
    defer beta_tensor.deinit();
    beta_tensor.fill(0.0);

    var result = try layerNorm(allocator, &input, &gamma_tensor, &beta_tensor, 1e-5);
    defer result.deinit();

    // After normalization: mean ≈ 0, std ≈ 1
    var sum_val: f32 = 0.0;
    for (result.data) |v| sum_val += v;
    const mean_val = sum_val / 3.0;
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mean_val, 0.001);

    // Values should be approximately [-1.22, 0, 1.22]
    try std.testing.expectApproxEqAbs(@as(f32, -1.2247), result.data[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result.data[1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 1.2247), result.data[2], 0.01);
}

test "gelu activation" {
    const allocator = std.testing.allocator;

    var shape = [_]usize{5};
    var tensor = try Tensor.init(allocator, &shape);
    defer tensor.deinit();

    tensor.data[0] = -2.0;
    tensor.data[1] = -1.0;
    tensor.data[2] = 0.0;
    tensor.data[3] = 1.0;
    tensor.data[4] = 2.0;

    gelu(&tensor);

    // GELU(-2) ≈ -0.0454
    // GELU(-1) ≈ -0.1588
    // GELU(0) = 0
    // GELU(1) ≈ 0.8412
    // GELU(2) ≈ 1.9545
    try std.testing.expectApproxEqAbs(@as(f32, -0.0454), tensor.data[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, -0.1588), tensor.data[1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), tensor.data[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8412), tensor.data[3], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 1.9545), tensor.data[4], 0.01);
}

test "softmax" {
    const allocator = std.testing.allocator;

    var shape = [_]usize{ 1, 2 };
    var tensor = try Tensor.init(allocator, &shape);
    defer tensor.deinit();

    tensor.data[0] = 0.0;
    tensor.data[1] = 1.0;

    softmax(&tensor);

    // softmax([0, 1]) ≈ [0.2689, 0.7311]
    try std.testing.expectApproxEqAbs(@as(f32, 0.2689), tensor.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7311), tensor.data[1], 0.001);

    // Sum should be 1.0
    const sum_val = tensor.data[0] + tensor.data[1];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum_val, 0.0001);
}

test "softmax numerical stability" {
    const allocator = std.testing.allocator;

    var shape = [_]usize{ 1, 3 };
    var tensor = try Tensor.init(allocator, &shape);
    defer tensor.deinit();

    // Large values that would overflow without max subtraction
    tensor.data[0] = 1000.0;
    tensor.data[1] = 1001.0;
    tensor.data[2] = 1002.0;

    softmax(&tensor);

    // Should still sum to 1.0 and not produce inf/nan
    var sum_val: f32 = 0.0;
    for (tensor.data) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
        sum_val += v;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum_val, 0.0001);
}

test "slice columns" {
    const allocator = std.testing.allocator;

    // 2x4 tensor
    var shape = [_]usize{ 2, 4 };
    var t = try Tensor.init(allocator, &shape);
    defer t.deinit();
    // Row 0: [0, 1, 2, 3]
    // Row 1: [4, 5, 6, 7]
    for (t.data, 0..) |*v, i| {
        v.* = @floatFromInt(i);
    }

    // Slice columns [1:3]
    var sliced = try sliceColumns(allocator, &t, 1, 3);
    defer sliced.deinit();

    try std.testing.expectEqual(@as(usize, 2), sliced.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), sliced.shape[1]);
    // Row 0: [1, 2]
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sliced.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), sliced.data[1], 0.001);
    // Row 1: [5, 6]
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), sliced.data[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), sliced.data[3], 0.001);
}

test "concat columns" {
    const allocator = std.testing.allocator;

    // Tensor 1: 2x2
    var shape1 = [_]usize{ 2, 2 };
    var t1 = try Tensor.init(allocator, &shape1);
    defer t1.deinit();
    t1.data[0] = 1.0;
    t1.data[1] = 2.0;
    t1.data[2] = 5.0;
    t1.data[3] = 6.0;

    // Tensor 2: 2x3
    var shape2 = [_]usize{ 2, 3 };
    var t2 = try Tensor.init(allocator, &shape2);
    defer t2.deinit();
    t2.data[0] = 3.0;
    t2.data[1] = 4.0;
    t2.data[2] = 0.0;
    t2.data[3] = 7.0;
    t2.data[4] = 8.0;
    t2.data[5] = 0.0;

    const tensors = [_]Tensor{ t1, t2 };
    var concat = try concatColumns(allocator, &tensors);
    defer concat.deinit();

    // Result: 2x5
    try std.testing.expectEqual(@as(usize, 2), concat.shape[0]);
    try std.testing.expectEqual(@as(usize, 5), concat.shape[1]);

    // Row 0: [1, 2, 3, 4, 0]
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), concat.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), concat.data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), concat.data[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), concat.data[3], 0.001);
    // Row 1: [5, 6, 7, 8, 0]
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), concat.data[5], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), concat.data[6], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), concat.data[7], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), concat.data[8], 0.001);
}

test "argmax" {
    const allocator = std.testing.allocator;

    var shape = [_]usize{ 3, 4 };
    var t = try Tensor.init(allocator, &shape);
    defer t.deinit();

    // Row 0: max at index 2
    t.data[0] = 0.1;
    t.data[1] = 0.2;
    t.data[2] = 0.9;
    t.data[3] = 0.3;
    // Row 1: max at index 0
    t.data[4] = 0.8;
    t.data[5] = 0.2;
    t.data[6] = 0.1;
    t.data[7] = 0.3;
    // Row 2: max at index 3
    t.data[8] = 0.1;
    t.data[9] = 0.2;
    t.data[10] = 0.3;
    t.data[11] = 0.7;

    const indices = try argmax(allocator, &t);
    defer allocator.free(indices);

    try std.testing.expectEqual(@as(usize, 2), indices[0]);
    try std.testing.expectEqual(@as(usize, 0), indices[1]);
    try std.testing.expectEqual(@as(usize, 3), indices[2]);
}

test "add bias in place" {
    const allocator = std.testing.allocator;

    var input_shape = [_]usize{ 2, 3 };
    var input = try Tensor.init(allocator, &input_shape);
    defer input.deinit();
    input.fill(1.0);

    var bias_shape = [_]usize{3};
    var bias = try Tensor.init(allocator, &bias_shape);
    defer bias.deinit();
    bias.data[0] = 0.1;
    bias.data[1] = 0.2;
    bias.data[2] = 0.3;

    try addBiasInPlace(&input, &bias);

    // Row 0: [1.1, 1.2, 1.3]
    try std.testing.expectApproxEqAbs(@as(f32, 1.1), input.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.2), input.data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.3), input.data[2], 0.001);
    // Row 1: [1.1, 1.2, 1.3]
    try std.testing.expectApproxEqAbs(@as(f32, 1.1), input.data[3], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.2), input.data[4], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.3), input.data[5], 0.001);
}

// ============================================================================
// Tests: Causal Masking
// ============================================================================

test "create causal mask 4x4" {
    const allocator = std.testing.allocator;

    var mask = try createCausalMask(allocator, 4);
    defer mask.deinit();

    // Expected mask (0 = valid, -inf = masked):
    // [  0,  -∞,  -∞,  -∞ ]  row 0: can only see position 0
    // [  0,   0,  -∞,  -∞ ]  row 1: can see positions 0, 1
    // [  0,   0,   0,  -∞ ]  row 2: can see positions 0, 1, 2
    // [  0,   0,   0,   0 ]  row 3: can see all positions

    try std.testing.expectEqual(@as(usize, 2), mask.shape.len);
    try std.testing.expectEqual(@as(usize, 4), mask.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), mask.shape[1]);

    // Row 0: can only see position 0
    try std.testing.expectEqual(@as(f32, 0.0), mask.data[0]); // [0,0]
    try std.testing.expect(std.math.isNegativeInf(mask.data[1])); // [0,1]
    try std.testing.expect(std.math.isNegativeInf(mask.data[2])); // [0,2]
    try std.testing.expect(std.math.isNegativeInf(mask.data[3])); // [0,3]

    // Row 2: can see positions 0, 1, 2
    try std.testing.expectEqual(@as(f32, 0.0), mask.data[8]); // [2,0]
    try std.testing.expectEqual(@as(f32, 0.0), mask.data[9]); // [2,1]
    try std.testing.expectEqual(@as(f32, 0.0), mask.data[10]); // [2,2]
    try std.testing.expect(std.math.isNegativeInf(mask.data[11])); // [2,3]

    // Row 3 (last row): can see all positions
    try std.testing.expectEqual(@as(f32, 0.0), mask.data[12]); // [3,0]
    try std.testing.expectEqual(@as(f32, 0.0), mask.data[13]); // [3,1]
    try std.testing.expectEqual(@as(f32, 0.0), mask.data[14]); // [3,2]
    try std.testing.expectEqual(@as(f32, 0.0), mask.data[15]); // [3,3]
}

test "apply causal mask in place" {
    const allocator = std.testing.allocator;

    // Create 3x3 attention scores (all ones)
    var scores_shape = [_]usize{ 3, 3 };
    var scores = try Tensor.init(allocator, &scores_shape);
    defer scores.deinit();
    scores.fill(1.0);

    applyCausalMask(&scores);

    // Expected after masking:
    // [ 1,  -∞,  -∞ ]
    // [ 1,   1,  -∞ ]
    // [ 1,   1,   1 ]

    // Row 0
    try std.testing.expectEqual(@as(f32, 1.0), scores.data[0]);
    try std.testing.expect(std.math.isNegativeInf(scores.data[1]));
    try std.testing.expect(std.math.isNegativeInf(scores.data[2]));

    // Row 1
    try std.testing.expectEqual(@as(f32, 1.0), scores.data[3]);
    try std.testing.expectEqual(@as(f32, 1.0), scores.data[4]);
    try std.testing.expect(std.math.isNegativeInf(scores.data[5]));

    // Row 2 (no masking)
    try std.testing.expectEqual(@as(f32, 1.0), scores.data[6]);
    try std.testing.expectEqual(@as(f32, 1.0), scores.data[7]);
    try std.testing.expectEqual(@as(f32, 1.0), scores.data[8]);
}

test "apply attention mask additive" {
    const allocator = std.testing.allocator;

    // Scores: all 2.0
    var shape = [_]usize{ 2, 3 };
    var scores = try Tensor.init(allocator, &shape);
    defer scores.deinit();
    scores.fill(2.0);

    // Mask: 0 for valid, -inf for position [0,2] and [1,2]
    var mask = try Tensor.init(allocator, &shape);
    defer mask.deinit();
    mask.fill(0.0);
    mask.data[2] = -std.math.inf(f32); // [0,2]
    mask.data[5] = -std.math.inf(f32); // [1,2]

    try applyAttentionMask(&scores, &mask);

    // Expected: [2, 2, -inf], [2, 2, -inf]
    try std.testing.expectEqual(@as(f32, 2.0), scores.data[0]);
    try std.testing.expectEqual(@as(f32, 2.0), scores.data[1]);
    try std.testing.expect(std.math.isNegativeInf(scores.data[2]));
    try std.testing.expectEqual(@as(f32, 2.0), scores.data[3]);
    try std.testing.expectEqual(@as(f32, 2.0), scores.data[4]);
    try std.testing.expect(std.math.isNegativeInf(scores.data[5]));
}

const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const TensorError = @import("tensor.zig").TensorError;

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
/// Uses naive triple-loop implementation O(n^3).
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

    // Result shape: [M, N]
    var result_shape = [_]usize{ m, n };
    var result = try Tensor.init(allocator, &result_shape);
    errdefer result.deinit();

    // Triple loop: C[i,j] = sum_k(A[i,k] * B[k,j])
    for (0..m) |i| {
        for (0..n) |j| {
            var dot: f32 = 0.0;
            for (0..k) |kk| {
                // A[i, kk] is at index i * k + kk
                // B[kk, j] is at index kk * n + j
                const a_val = a.data[i * k + kk];
                const b_val = b.data[kk * n + j];
                dot += a_val * b_val;
            }
            // C[i, j] is at index i * n + j
            result.data[i * n + j] = dot;
        }
    }

    return result;
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
// Phase 2 Tests: Matrix Multiplication & Activations
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
// Phase 4: LSTM Operations
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
// Phase 4: Conv1D Operations
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
// Phase 4 Tests: LSTM and Conv1D
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

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

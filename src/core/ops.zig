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

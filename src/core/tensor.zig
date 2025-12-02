const std = @import("std");

/// Error types for Tensor operations
pub const TensorError = error{
    ShapeMismatch,
    OutOfBounds,
    InvalidShape,
};

/// Tensor - The fundamental data container for neural network operations.
/// Stores multi-dimensional float32 data with explicit memory management.
pub const Tensor = struct {
    /// Raw float32 data stored in row-major order
    data: []f32,
    /// Dimensions of the tensor (e.g., [batch, channels, height, width])
    shape: []usize,
    /// Allocator used for memory management
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Initialize a new tensor with given shape.
    /// All values are initialized to zero.
    /// Caller owns the returned tensor and must call deinit().
    pub fn init(allocator: std.mem.Allocator, shape: []const usize) !Self {
        if (shape.len == 0) {
            return TensorError.InvalidShape;
        }

        // Calculate total number of elements
        var total_size: usize = 1;
        for (shape) |dim| {
            if (dim == 0) {
                return TensorError.InvalidShape;
            }
            total_size *= dim;
        }

        // Allocate data buffer
        const data = try allocator.alloc(f32, total_size);
        @memset(data, 0.0);

        // Copy shape so we own it
        const owned_shape = try allocator.alloc(usize, shape.len);
        @memcpy(owned_shape, shape);

        return Self{
            .data = data,
            .shape = owned_shape,
            .allocator = allocator,
        };
    }

    /// Initialize a tensor from existing data (copies the data).
    /// Shape must match the data length.
    pub fn initWithData(allocator: std.mem.Allocator, shape: []const usize, src_data: []const f32) !Self {
        var tensor = try init(allocator, shape);

        if (src_data.len != tensor.data.len) {
            tensor.deinit();
            return TensorError.ShapeMismatch;
        }

        @memcpy(tensor.data, src_data);
        return tensor;
    }

    /// Free tensor memory.
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.data);
        self.allocator.free(self.shape);
        self.data = &[_]f32{};
        self.shape = &[_]usize{};
    }

    /// Get element at flattened index.
    pub fn get(self: *const Self, index: usize) TensorError!f32 {
        if (index >= self.data.len) {
            return TensorError.OutOfBounds;
        }
        return self.data[index];
    }

    /// Get element at flattened index (unchecked, for performance-critical code).
    pub fn getUnchecked(self: *const Self, index: usize) f32 {
        return self.data[index];
    }

    /// Set element at flattened index.
    pub fn set(self: *Self, index: usize, value: f32) TensorError!void {
        if (index >= self.data.len) {
            return TensorError.OutOfBounds;
        }
        self.data[index] = value;
    }

    /// Set element at flattened index (unchecked, for performance-critical code).
    pub fn setUnchecked(self: *Self, index: usize, value: f32) void {
        self.data[index] = value;
    }

    /// Get total number of elements in the tensor.
    pub fn size(self: *const Self) usize {
        return self.data.len;
    }

    /// Get number of dimensions (rank) of the tensor.
    pub fn rank(self: *const Self) usize {
        return self.shape.len;
    }

    /// Fill tensor with a constant value.
    pub fn fill(self: *Self, value: f32) void {
        @memset(self.data, value);
    }

    /// Copy data from another tensor (must have same size).
    pub fn copyFrom(self: *Self, other: *const Self) TensorError!void {
        if (self.data.len != other.data.len) {
            return TensorError.ShapeMismatch;
        }
        @memcpy(self.data, other.data);
    }

    /// Clone the tensor (creates a new tensor with copied data).
    pub fn clone(self: *const Self, allocator: std.mem.Allocator) !Self {
        return initWithData(allocator, self.shape, self.data);
    }

    /// Convert multi-dimensional indices to flat index.
    /// For a tensor with shape [d0, d1, d2], index [i, j, k] maps to:
    /// i * (d1 * d2) + j * d2 + k
    pub fn flatIndex(self: *const Self, indices: []const usize) TensorError!usize {
        if (indices.len != self.shape.len) {
            return TensorError.ShapeMismatch;
        }

        var flat: usize = 0;
        var stride: usize = 1;

        // Calculate from last dimension to first (row-major)
        var i: usize = self.shape.len;
        while (i > 0) {
            i -= 1;
            if (indices[i] >= self.shape[i]) {
                return TensorError.OutOfBounds;
            }
            flat += indices[i] * stride;
            stride *= self.shape[i];
        }

        return flat;
    }

    /// Get element using multi-dimensional indices.
    pub fn getAt(self: *const Self, indices: []const usize) TensorError!f32 {
        const flat = try self.flatIndex(indices);
        return self.data[flat];
    }

    /// Set element using multi-dimensional indices.
    pub fn setAt(self: *Self, indices: []const usize, value: f32) TensorError!void {
        const flat = try self.flatIndex(indices);
        self.data[flat] = value;
    }

    /// Check if two tensors have the same shape.
    pub fn sameShape(self: *const Self, other: *const Self) bool {
        if (self.shape.len != other.shape.len) {
            return false;
        }
        for (self.shape, other.shape) |a, b| {
            if (a != b) {
                return false;
            }
        }
        return true;
    }

    /// Print tensor info (for debugging).
    pub fn debugPrint(self: *const Self) void {
        std.debug.print("Tensor(shape=[", .{});
        for (self.shape, 0..) |dim, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{}", .{dim});
        }
        std.debug.print("], size={})\n", .{self.size()});

        // Print first few elements
        const max_print: usize = @min(10, self.data.len);
        std.debug.print("  data[0..{}] = [", .{max_print});
        for (self.data[0..max_print], 0..) |val, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{d:.4}", .{val});
        }
        if (self.data.len > max_print) {
            std.debug.print(", ...", .{});
        }
        std.debug.print("]\n", .{});
    }
};

// ============================================================================
// Tests
// ============================================================================

test "tensor creation and basic access" {
    const allocator = std.testing.allocator;

    // Create 2x2 tensor
    var shape = [_]usize{ 2, 2 };
    var tensor = try Tensor.init(allocator, &shape);
    defer tensor.deinit();

    // Verify initial state
    try std.testing.expectEqual(@as(usize, 4), tensor.size());
    try std.testing.expectEqual(@as(usize, 2), tensor.rank());
    try std.testing.expectEqual(@as(f32, 0.0), try tensor.get(0));

    // Set and get values
    try tensor.set(0, 1.0);
    try tensor.set(1, 2.0);
    try tensor.set(2, 3.0);
    try tensor.set(3, 4.0);

    try std.testing.expectEqual(@as(f32, 1.0), try tensor.get(0));
    try std.testing.expectEqual(@as(f32, 2.0), try tensor.get(1));
    try std.testing.expectEqual(@as(f32, 3.0), try tensor.get(2));
    try std.testing.expectEqual(@as(f32, 4.0), try tensor.get(3));
}

test "tensor fill" {
    const allocator = std.testing.allocator;

    var shape = [_]usize{ 3, 3 };
    var tensor = try Tensor.init(allocator, &shape);
    defer tensor.deinit();

    tensor.fill(5.0);

    for (tensor.data) |val| {
        try std.testing.expectEqual(@as(f32, 5.0), val);
    }
}

test "tensor multi-dimensional indexing" {
    const allocator = std.testing.allocator;

    // Create 2x3 tensor
    var shape = [_]usize{ 2, 3 };
    var tensor = try Tensor.init(allocator, &shape);
    defer tensor.deinit();

    // Fill with sequential values
    for (tensor.data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }

    // Test multi-dimensional access
    // Row 0: [0, 1, 2]
    // Row 1: [3, 4, 5]
    var idx = [_]usize{ 0, 0 };
    try std.testing.expectEqual(@as(f32, 0.0), try tensor.getAt(&idx));

    idx = [_]usize{ 0, 2 };
    try std.testing.expectEqual(@as(f32, 2.0), try tensor.getAt(&idx));

    idx = [_]usize{ 1, 0 };
    try std.testing.expectEqual(@as(f32, 3.0), try tensor.getAt(&idx));

    idx = [_]usize{ 1, 2 };
    try std.testing.expectEqual(@as(f32, 5.0), try tensor.getAt(&idx));
}

test "tensor clone" {
    const allocator = std.testing.allocator;

    var shape = [_]usize{ 2, 2 };
    var original = try Tensor.init(allocator, &shape);
    defer original.deinit();

    original.fill(7.0);

    var cloned = try original.clone(allocator);
    defer cloned.deinit();

    // Verify clone has same data
    try std.testing.expectEqual(@as(f32, 7.0), try cloned.get(0));
    try std.testing.expect(original.sameShape(&cloned));

    // Modify original, clone should be unaffected
    try original.set(0, 99.0);
    try std.testing.expectEqual(@as(f32, 7.0), try cloned.get(0));
}

test "tensor bounds checking" {
    const allocator = std.testing.allocator;

    var shape = [_]usize{ 2, 2 };
    var tensor = try Tensor.init(allocator, &shape);
    defer tensor.deinit();

    // Valid access
    _ = try tensor.get(3);

    // Invalid access should return error
    const result = tensor.get(4);
    try std.testing.expectError(TensorError.OutOfBounds, result);
}

test "tensor invalid shape" {
    const allocator = std.testing.allocator;

    // Empty shape should fail
    var empty_shape = [_]usize{};
    const result1 = Tensor.init(allocator, &empty_shape);
    try std.testing.expectError(TensorError.InvalidShape, result1);

    // Zero dimension should fail
    var zero_shape = [_]usize{ 2, 0, 3 };
    const result2 = Tensor.init(allocator, &zero_shape);
    try std.testing.expectError(TensorError.InvalidShape, result2);
}

test "tensor initWithData" {
    const allocator = std.testing.allocator;

    var shape = [_]usize{ 2, 3 };
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };

    var tensor = try Tensor.initWithData(allocator, &shape, &data);
    defer tensor.deinit();

    try std.testing.expectEqual(@as(f32, 1.0), try tensor.get(0));
    try std.testing.expectEqual(@as(f32, 6.0), try tensor.get(5));
}

const std = @import("std");
const tensor_import = @import("tensor.zig");
const Tensor = tensor_import.Tensor;

/// Quantization error types
pub const QuantError = error{
    InvalidShape,
    ShapeMismatch,
    OutOfMemory,
    InvalidFormat,
};

// ============================================================================
// Q8_0: Simple 8-bit Symmetric Quantization (Weight-Only)
// ============================================================================

/// Q8_0: Simple symmetric 8-bit quantization
/// Storage: 4-byte scale + n bytes of int8 data
/// Used for weight-only quantization (inputs stay float32)
pub const QuantizedTensorQ8 = struct {
    scale: f32,
    data: []i8,
    shape: []usize,
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Initialize with given shape (data is uninitialized)
    pub fn init(allocator: std.mem.Allocator, shape: []const usize) !Self {
        var total: usize = 1;
        for (shape) |dim| total *= dim;

        const shape_copy = try allocator.dupe(usize, shape);
        errdefer allocator.free(shape_copy);

        const data = try allocator.alloc(i8, total);
        errdefer allocator.free(data);

        return .{
            .scale = 0.0,
            .data = data,
            .shape = shape_copy,
            .allocator = allocator,
        };
    }

    /// Free memory
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.data);
        self.allocator.free(self.shape);
    }

    /// Get total number of elements
    pub fn numel(self: *const Self) usize {
        var total: usize = 1;
        for (self.shape) |dim| total *= dim;
        return total;
    }

    /// Memory size in bytes (scale + data)
    pub fn sizeBytes(self: *const Self) usize {
        return 4 + self.data.len; // 4 bytes for scale + 1 byte per element
    }

    /// Get compression ratio vs float32
    pub fn compressionRatio(self: *const Self) f32 {
        const f32_size: f32 = @floatFromInt(self.numel() * 4);
        const q8_size: f32 = @floatFromInt(self.sizeBytes());
        return f32_size / q8_size;
    }
};

// ============================================================================
// Quantization Functions
// ============================================================================

/// Quantize float32 tensor to Q8_0 format (symmetric, per-tensor scale)
/// Used for static weight quantization (offline)
pub fn quantizeQ8(allocator: std.mem.Allocator, tensor: *const Tensor) !QuantizedTensorQ8 {
    var result = try QuantizedTensorQ8.init(allocator, tensor.shape);
    errdefer result.deinit();

    // Find max absolute value
    var max_abs: f32 = 0.0;
    for (tensor.data) |v| {
        const abs_v = @abs(v);
        if (abs_v > max_abs) max_abs = abs_v;
    }

    // Avoid division by zero
    if (max_abs == 0.0) {
        result.scale = 1.0;
        @memset(result.data, 0);
        return result;
    }

    // Calculate scale (symmetric quantization)
    result.scale = max_abs / 127.0;
    const inv_scale = 127.0 / max_abs;

    // Quantize each element
    for (tensor.data, 0..) |v, i| {
        const scaled = v * inv_scale;
        const rounded = @round(scaled);
        const clamped = std.math.clamp(rounded, -127.0, 127.0);
        result.data[i] = @intFromFloat(clamped);
    }

    return result;
}

/// Dequantize Q8_0 tensor back to float32
/// Used for verification/debugging
pub fn dequantizeQ8(allocator: std.mem.Allocator, qtensor: *const QuantizedTensorQ8) !Tensor {
    var result = try Tensor.init(allocator, qtensor.shape);
    errdefer result.deinit();

    for (qtensor.data, 0..) |q, i| {
        result.data[i] = @as(f32, @floatFromInt(q)) * qtensor.scale;
    }

    return result;
}

// ============================================================================
// Weight-Only Quantized Matrix Multiplication
// ============================================================================

/// Weight-Only Quantized Matrix Multiplication (the correct approach)
/// a: [M, K] float32 input (activations - dynamic, keep precision)
/// b: [K, N] quantized weights (static, stored as int8)
/// Returns: [M, N] float32 result
///
/// This is the CORRECT way to do quantization for inference:
/// - Weights are static → quantize offline (easy)
/// - Inputs are dynamic → keep as float32 (preserves precision)
/// - Dequantize weight to float32 inside the loop
/// - Still saves memory bandwidth (fetching 1 byte instead of 4)
pub fn matmulF32Q8(
    allocator: std.mem.Allocator,
    a: *const Tensor,
    b: *const QuantizedTensorQ8,
) !Tensor {
    // Validate shapes
    if (a.shape.len != 2 or b.shape.len != 2) {
        return QuantError.InvalidShape;
    }

    const m = a.shape[0];
    const k_a = a.shape[1];
    const k_b = b.shape[0];
    const n = b.shape[1];

    if (k_a != k_b) {
        return QuantError.ShapeMismatch;
    }

    const k = k_a;

    var result_shape = [_]usize{ m, n };
    var result = try Tensor.init(allocator, &result_shape);
    errdefer result.deinit();

    const scale = b.scale;

    // Optimized i,k,j loop order for cache efficiency
    // Initialize result to zero
    @memset(result.data, 0.0);

    for (0..m) |i| {
        for (0..k) |kk| {
            const a_val = a.data[i * k + kk];

            for (0..n) |j| {
                // Dequantize weight on-the-fly
                const w_int = b.data[kk * n + j];
                const w_f32 = @as(f32, @floatFromInt(w_int)) * scale;
                result.data[i * n + j] += a_val * w_f32;
            }
        }
    }

    return result;
}

/// SIMD-optimized Weight-Only Matmul (32-wide for Q8)
/// Uses vector int→float conversion which is very fast on modern CPUs
pub fn matmulF32Q8Simd(
    allocator: std.mem.Allocator,
    a: *const Tensor,
    b: *const QuantizedTensorQ8,
) !Tensor {
    // Validate shapes
    if (a.shape.len != 2 or b.shape.len != 2) {
        return QuantError.InvalidShape;
    }

    const m = a.shape[0];
    const k_a = a.shape[1];
    const k_b = b.shape[0];
    const n = b.shape[1];

    if (k_a != k_b) {
        return QuantError.ShapeMismatch;
    }

    const k = k_a;

    var result_shape = [_]usize{ m, n };
    var result = try Tensor.init(allocator, &result_shape);
    errdefer result.deinit();

    const scale = b.scale;

    // Initialize result to zero
    @memset(result.data, 0.0);

    // SIMD vector width
    const VEC_WIDTH = 8;
    const Vec8i8 = @Vector(VEC_WIDTH, i8);
    const Vec8i32 = @Vector(VEC_WIDTH, i32);
    const Vec8f32 = @Vector(VEC_WIDTH, f32);

    // i, k, j loop order with SIMD on j dimension
    for (0..m) |i| {
        for (0..k) |kk| {
            const a_val = a.data[i * k + kk];
            const a_vec: Vec8f32 = @splat(a_val);
            const scale_vec: Vec8f32 = @splat(scale);

            var j: usize = 0;

            // SIMD vectorized loop
            while (j + VEC_WIDTH <= n) : (j += VEC_WIDTH) {
                // Load 8 int8 weights
                const w_ptr = b.data[kk * n + j ..];
                const w_i8: Vec8i8 = w_ptr[0..VEC_WIDTH].*;

                // Convert to i32 first (to avoid overflow in wider intermediates)
                const w_i32: Vec8i32 = w_i8;

                // Convert to float32
                const w_f32: Vec8f32 = @floatFromInt(w_i32);

                // Scale to dequantize
                const w_scaled = w_f32 * scale_vec;

                // Load current result
                const result_ptr = result.data[i * n + j ..];
                var result_vec: Vec8f32 = result_ptr[0..VEC_WIDTH].*;

                // Multiply and accumulate
                result_vec += a_vec * w_scaled;

                // Store back
                result_ptr[0..VEC_WIDTH].* = result_vec;
            }

            // Handle remainder (tail loop)
            while (j < n) : (j += 1) {
                const w_int = b.data[kk * n + j];
                const w_f32 = @as(f32, @floatFromInt(w_int)) * scale;
                result.data[i * n + j] += a_val * w_f32;
            }
        }
    }

    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "Q8_0 quantization roundtrip" {
    const allocator = std.testing.allocator;

    // Create test tensor
    var shape = [_]usize{ 2, 3 };
    var tensor = try Tensor.init(allocator, &shape);
    defer tensor.deinit();

    tensor.data[0] = 2.5;
    tensor.data[1] = -1.0;
    tensor.data[2] = 0.5;
    tensor.data[3] = -2.0;
    tensor.data[4] = 0.0;
    tensor.data[5] = 1.5;

    // Quantize
    var qtensor = try quantizeQ8(allocator, &tensor);
    defer qtensor.deinit();

    // Check compression ratio
    // For small tensors (6 elements), ratio = 24/(4+6) = 2.4
    // For large tensors, ratio approaches 4x (since 4 byte overhead is negligible)
    const ratio = qtensor.compressionRatio();
    try std.testing.expect(ratio > 2.0);

    // Dequantize
    var restored = try dequantizeQ8(allocator, &qtensor);
    defer restored.deinit();

    // Check accuracy (should be within 1% of max value)
    const tolerance = 2.5 * 0.01; // 1% of max
    for (tensor.data, 0..) |original, i| {
        try std.testing.expectApproxEqAbs(original, restored.data[i], tolerance);
    }
}

test "Q8_0 size reduction" {
    const allocator = std.testing.allocator;

    // 1000 element tensor
    var shape = [_]usize{1000};
    var tensor = try Tensor.init(allocator, &shape);
    defer tensor.deinit();

    for (tensor.data, 0..) |*v, i| {
        v.* = @sin(@as(f32, @floatFromInt(i)) * 0.1);
    }

    var qtensor = try quantizeQ8(allocator, &tensor);
    defer qtensor.deinit();

    // Float32: 1000 * 4 = 4000 bytes
    // Q8_0:    4 + 1000  = 1004 bytes (4x reduction)
    const f32_size = tensor.data.len * 4;
    const q8_size = qtensor.sizeBytes();

    try std.testing.expect(q8_size < f32_size / 3); // At least 3x smaller
}

test "Q8_0 zero tensor" {
    const allocator = std.testing.allocator;

    var shape = [_]usize{10};
    var tensor = try Tensor.init(allocator, &shape);
    defer tensor.deinit();
    tensor.fill(0.0);

    var qtensor = try quantizeQ8(allocator, &tensor);
    defer qtensor.deinit();

    // Should handle zero tensor gracefully
    try std.testing.expectEqual(@as(f32, 1.0), qtensor.scale);
    for (qtensor.data) |q| {
        try std.testing.expectEqual(@as(i8, 0), q);
    }
}

test "weight-only quantized matmul accuracy" {
    const allocator = std.testing.allocator;

    // Float32 input (simulating activations)
    var a_shape = [_]usize{ 2, 3 };
    var a = try Tensor.init(allocator, &a_shape);
    defer a.deinit();
    a.data[0] = 1.0;
    a.data[1] = 2.0;
    a.data[2] = 3.0;
    a.data[3] = 4.0;
    a.data[4] = 5.0;
    a.data[5] = 6.0;

    // Float32 weights (will be quantized)
    var b_shape = [_]usize{ 3, 2 };
    var b = try Tensor.init(allocator, &b_shape);
    defer b.deinit();
    b.data[0] = 1.0;
    b.data[1] = 2.0;
    b.data[2] = 3.0;
    b.data[3] = 4.0;
    b.data[4] = 5.0;
    b.data[5] = 6.0;

    // Float32 reference using standard ops
    const ops = @import("ops.zig");
    var ref = try ops.matmul(allocator, &a, &b);
    defer ref.deinit();

    // Quantize weights only (not inputs!)
    var qb = try quantizeQ8(allocator, &b);
    defer qb.deinit();

    // Weight-only quantized path
    var qresult = try matmulF32Q8(allocator, &a, &qb);
    defer qresult.deinit();

    // Compare (allow 5% error due to weight quantization)
    for (ref.data, 0..) |expected, i| {
        const tolerance = @abs(expected) * 0.05 + 0.1;
        try std.testing.expectApproxEqAbs(expected, qresult.data[i], tolerance);
    }
}

test "weight-only quantized matmul SIMD" {
    const allocator = std.testing.allocator;

    // Larger test to exercise SIMD path
    var a_shape = [_]usize{ 4, 32 };
    var a = try Tensor.init(allocator, &a_shape);
    defer a.deinit();
    for (a.data, 0..) |*v, i| {
        v.* = @sin(@as(f32, @floatFromInt(i)) * 0.1);
    }

    var b_shape = [_]usize{ 32, 16 };
    var b = try Tensor.init(allocator, &b_shape);
    defer b.deinit();
    for (b.data, 0..) |*v, i| {
        v.* = @cos(@as(f32, @floatFromInt(i)) * 0.1);
    }

    // Scalar reference
    var qb = try quantizeQ8(allocator, &b);
    defer qb.deinit();

    var scalar_result = try matmulF32Q8(allocator, &a, &qb);
    defer scalar_result.deinit();

    // SIMD path
    var simd_result = try matmulF32Q8Simd(allocator, &a, &qb);
    defer simd_result.deinit();

    // Both should produce the same result
    for (scalar_result.data, 0..) |expected, i| {
        try std.testing.expectApproxEqAbs(expected, simd_result.data[i], 0.0001);
    }
}

test "quantization preserves extreme values" {
    const allocator = std.testing.allocator;

    var shape = [_]usize{5};
    var tensor = try Tensor.init(allocator, &shape);
    defer tensor.deinit();

    // Test with extreme values
    tensor.data[0] = 100.0;
    tensor.data[1] = -100.0;
    tensor.data[2] = 0.0;
    tensor.data[3] = 50.0;
    tensor.data[4] = -50.0;

    var qtensor = try quantizeQ8(allocator, &tensor);
    defer qtensor.deinit();

    // Max and min should map to ±127
    try std.testing.expectEqual(@as(i8, 127), qtensor.data[0]);
    try std.testing.expectEqual(@as(i8, -127), qtensor.data[1]);
    try std.testing.expectEqual(@as(i8, 0), qtensor.data[2]);

    // Check scale
    try std.testing.expectApproxEqAbs(@as(f32, 100.0 / 127.0), qtensor.scale, 0.0001);
}

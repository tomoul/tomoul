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
// Q8_K: Block-wise 8-bit Symmetric Quantization (Weight-Only)
// ============================================================================

/// Block size for block-wise quantization
/// 32 is optimal for SIMD (AVX2 = 32 bytes)
pub const BLOCK_SIZE: usize = 32;

/// Q8_K: Block-wise symmetric 8-bit quantization
/// Storage: (n_blocks * 4) bytes for f32 scales + n bytes of int8 data
/// Each block of 32 elements gets its own scale for better accuracy
pub const QuantizedTensorQ8K = struct {
    block_size: usize,
    num_blocks: usize,
    scales: []f32, // One scale per block (f32 for precision)
    data: []i8,
    shape: []usize,
    element_count: usize,
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Initialize with given shape
    pub fn init(allocator: std.mem.Allocator, shape: []const usize) !Self {
        var total: usize = 1;
        for (shape) |dim| total *= dim;

        const num_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

        const shape_copy = try allocator.dupe(usize, shape);
        errdefer allocator.free(shape_copy);

        const scales = try allocator.alloc(f32, num_blocks);
        errdefer allocator.free(scales);

        const data = try allocator.alloc(i8, total);
        errdefer allocator.free(data);

        return .{
            .block_size = BLOCK_SIZE,
            .num_blocks = num_blocks,
            .scales = scales,
            .data = data,
            .shape = shape_copy,
            .element_count = total,
            .allocator = allocator,
        };
    }

    /// Free memory
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.scales);
        self.allocator.free(self.data);
        self.allocator.free(self.shape);
    }

    /// Get total number of elements
    pub fn numel(self: *const Self) usize {
        return self.element_count;
    }

    /// Memory size in bytes (scales + data)
    pub fn sizeBytes(self: *const Self) usize {
        return self.scales.len * 4 + self.data.len;
    }

    /// Get compression ratio vs float32
    pub fn compressionRatio(self: *const Self) f32 {
        const f32_size: f32 = @floatFromInt(self.numel() * 4);
        const q8k_size: f32 = @floatFromInt(self.sizeBytes());
        return f32_size / q8k_size;
    }

    /// Get block index and offset within block for a given element index
    pub fn getBlockInfo(self: *const Self, idx: usize) struct { block: usize, offset: usize } {
        return .{
            .block = idx / self.block_size,
            .offset = idx % self.block_size,
        };
    }

    /// Get dequantized value at index
    pub fn getDequantized(self: *const Self, idx: usize) f32 {
        const block_idx = idx / self.block_size;
        const scale = self.scales[block_idx];
        return @as(f32, @floatFromInt(self.data[idx])) * scale;
    }
};

/// Quantize float32 tensor to Q8_K format (block-wise, per-block scale)
pub fn quantizeQ8K(allocator: std.mem.Allocator, tensor: *const Tensor) !QuantizedTensorQ8K {
    var result = try QuantizedTensorQ8K.init(allocator, tensor.shape);
    errdefer result.deinit();

    const total = tensor.data.len;
    var block_idx: usize = 0;

    while (block_idx * BLOCK_SIZE < total) : (block_idx += 1) {
        const start = block_idx * BLOCK_SIZE;
        const end = @min(start + BLOCK_SIZE, total);
        const block = tensor.data[start..end];

        // Find max absolute value in this block
        var max_abs: f32 = 0.0;
        for (block) |v| {
            const abs_v = @abs(v);
            if (abs_v > max_abs) max_abs = abs_v;
        }

        // Calculate scale for this block
        if (max_abs == 0.0) {
            result.scales[block_idx] = 1.0;
            for (start..end) |i| {
                result.data[i] = 0;
            }
        } else {
            result.scales[block_idx] = max_abs / 127.0;
            const inv_scale = 127.0 / max_abs;

            // Quantize block elements
            for (block, start..) |v, i| {
                const scaled = v * inv_scale;
                const rounded = @round(scaled);
                const clamped = std.math.clamp(rounded, -127.0, 127.0);
                result.data[i] = @intFromFloat(clamped);
            }
        }
    }

    return result;
}

/// Dequantize Q8_K tensor back to float32
pub fn dequantizeQ8K(allocator: std.mem.Allocator, qtensor: *const QuantizedTensorQ8K) !Tensor {
    var result = try Tensor.init(allocator, qtensor.shape);
    errdefer result.deinit();

    const total = qtensor.element_count;
    var block_idx: usize = 0;

    while (block_idx * qtensor.block_size < total) : (block_idx += 1) {
        const start = block_idx * qtensor.block_size;
        const end = @min(start + qtensor.block_size, total);
        const scale = qtensor.scales[block_idx];

        for (start..end) |i| {
            result.data[i] = @as(f32, @floatFromInt(qtensor.data[i])) * scale;
        }
    }

    return result;
}

// ============================================================================
// Q8_K Weight-Only Quantized Matrix Multiplication
// ============================================================================

/// Weight-Only Quantized Matrix Multiplication for Q8_K (block-wise)
/// a: [M, K] float32 input (activations)
/// b: [K, N] quantized weights (block-wise Q8)
/// Returns: [M, N] float32 result
pub fn matmulF32Q8K(
    allocator: std.mem.Allocator,
    a: *const Tensor,
    b: *const QuantizedTensorQ8K,
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

    // Initialize result to zero
    @memset(result.data, 0.0);

    // i, k, j loop order
    for (0..m) |i| {
        for (0..k) |kk| {
            const a_val = a.data[i * k + kk];

            for (0..n) |j| {
                // Get block-wise scale for this weight
                const w_idx = kk * n + j;
                const block_idx = w_idx / b.block_size;
                const scale = b.scales[block_idx];

                // Dequantize weight on-the-fly
                const w_int = b.data[w_idx];
                const w_f32 = @as(f32, @floatFromInt(w_int)) * scale;
                result.data[i * n + j] += a_val * w_f32;
            }
        }
    }

    return result;
}

/// SIMD-optimized Weight-Only Matmul for Q8_K
pub fn matmulF32Q8KSimd(
    allocator: std.mem.Allocator,
    a: *const Tensor,
    b: *const QuantizedTensorQ8K,
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

    // Initialize result to zero
    @memset(result.data, 0.0);

    // SIMD vector width
    const VEC_WIDTH = 8;
    const Vec8i8 = @Vector(VEC_WIDTH, i8);
    const Vec8i32 = @Vector(VEC_WIDTH, i32);
    const Vec8f32 = @Vector(VEC_WIDTH, f32);

    for (0..m) |i| {
        for (0..k) |kk| {
            const a_val = a.data[i * k + kk];
            const a_vec: Vec8f32 = @splat(a_val);

            var j: usize = 0;

            // SIMD loop
            while (j + VEC_WIDTH <= n) : (j += VEC_WIDTH) {
                // Load 8 int8 weights
                const w_ptr = b.data[kk * n + j ..];
                const w_i8: Vec8i8 = w_ptr[0..VEC_WIDTH].*;

                // Get scales for each weight (may span multiple blocks)
                var scale_vec: Vec8f32 = undefined;
                inline for (0..VEC_WIDTH) |vi| {
                    const w_idx = kk * n + j + vi;
                    const block_idx = w_idx / b.block_size;
                    scale_vec[vi] = b.scales[block_idx];
                }

                // Convert to i32 then float32
                const w_i32: Vec8i32 = w_i8;
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

            // Handle remainder
            while (j < n) : (j += 1) {
                const w_idx = kk * n + j;
                const block_idx = w_idx / b.block_size;
                const scale = b.scales[block_idx];
                const w_int = b.data[w_idx];
                const w_f32 = @as(f32, @floatFromInt(w_int)) * scale;
                result.data[i * n + j] += a_val * w_f32;
            }
        }
    }

    return result;
}

// ============================================================================
// Q4_0: Simple 4-bit Symmetric Quantization (Weight-Only)
// ============================================================================

/// Q4_0: Simple symmetric 4-bit quantization
/// Storage: 4-byte scale + ceil(n/2) bytes of packed 4-bit data
/// Two 4-bit values packed per byte (low nibble = even index, high nibble = odd)
/// Used for aggressive weight compression (8x vs float32)
pub const QuantizedTensorQ4 = struct {
    scale: f32,
    data: []u8, // Packed 4-bit values (2 per byte)
    shape: []usize,
    element_count: usize,
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Initialize with given shape (data is uninitialized)
    pub fn init(allocator: std.mem.Allocator, shape: []const usize) !Self {
        var total: usize = 1;
        for (shape) |dim| total *= dim;

        const shape_copy = try allocator.dupe(usize, shape);
        errdefer allocator.free(shape_copy);

        // Packed bytes: ceil(total / 2)
        const packed_size = (total + 1) / 2;
        const data = try allocator.alloc(u8, packed_size);
        errdefer allocator.free(data);

        return .{
            .scale = 0.0,
            .data = data,
            .shape = shape_copy,
            .element_count = total,
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
        return self.element_count;
    }

    /// Memory size in bytes (scale + packed data)
    pub fn sizeBytes(self: *const Self) usize {
        return 4 + self.data.len; // 4 bytes for scale + packed bytes
    }

    /// Get compression ratio vs float32
    pub fn compressionRatio(self: *const Self) f32 {
        const f32_size: f32 = @floatFromInt(self.numel() * 4);
        const q4_size: f32 = @floatFromInt(self.sizeBytes());
        return f32_size / q4_size;
    }

    /// Get quantized value at index (unpacks from byte)
    pub fn get(self: *const Self, idx: usize) i8 {
        const byte_idx = idx / 2;
        const byte_val = self.data[byte_idx];

        if (idx % 2 == 0) {
            // Low nibble (sign-extend from 4-bit)
            const nibble: u4 = @truncate(byte_val & 0x0F);
            return signExtend4(nibble);
        } else {
            // High nibble (sign-extend from 4-bit)
            const nibble: u4 = @truncate(byte_val >> 4);
            return signExtend4(nibble);
        }
    }

    /// Set quantized value at index (packs into byte)
    pub fn set(self: *Self, idx: usize, value: i8) void {
        const byte_idx = idx / 2;
        const clamped = std.math.clamp(value, -7, 7);
        const nibble: u4 = @bitCast(@as(i4, @intCast(clamped)));

        if (idx % 2 == 0) {
            // Low nibble
            self.data[byte_idx] = (self.data[byte_idx] & 0xF0) | nibble;
        } else {
            // High nibble
            self.data[byte_idx] = (self.data[byte_idx] & 0x0F) | (@as(u8, nibble) << 4);
        }
    }
};

/// Sign-extend 4-bit value to i8
fn signExtend4(nibble: u4) i8 {
    const as_i4: i4 = @bitCast(nibble);
    return as_i4;
}

/// Quantize float32 tensor to Q4_0 format (symmetric, per-tensor scale)
/// 4-bit range: [-7, 7] (using symmetric quantization)
pub fn quantizeQ4(allocator: std.mem.Allocator, tensor: *const Tensor) !QuantizedTensorQ4 {
    var result = try QuantizedTensorQ4.init(allocator, tensor.shape);
    errdefer result.deinit();

    // Initialize packed data to zero
    @memset(result.data, 0);

    // Find max absolute value
    var max_abs: f32 = 0.0;
    for (tensor.data) |v| {
        const abs_v = @abs(v);
        if (abs_v > max_abs) max_abs = abs_v;
    }

    // Avoid division by zero
    if (max_abs == 0.0) {
        result.scale = 1.0;
        return result;
    }

    // Calculate scale (symmetric quantization)
    result.scale = max_abs / 7.0;
    const inv_scale = 7.0 / max_abs;

    // Quantize each element
    for (tensor.data, 0..) |v, i| {
        const scaled = v * inv_scale;
        const rounded = @round(scaled);
        const clamped = std.math.clamp(rounded, -7.0, 7.0);
        result.set(i, @intFromFloat(clamped));
    }

    return result;
}

/// Dequantize Q4_0 tensor back to float32
/// Used for verification/debugging
pub fn dequantizeQ4(allocator: std.mem.Allocator, qtensor: *const QuantizedTensorQ4) !Tensor {
    var result = try Tensor.init(allocator, qtensor.shape);
    errdefer result.deinit();

    for (0..qtensor.element_count) |i| {
        const q = qtensor.get(i);
        result.data[i] = @as(f32, @floatFromInt(q)) * qtensor.scale;
    }

    return result;
}

// ============================================================================
// Q4_0 Weight-Only Quantized Matrix Multiplication
// ============================================================================

/// Weight-Only Quantized Matrix Multiplication for Q4_0
/// a: [M, K] float32 input (activations)
/// b: [K, N] quantized weights (4-bit packed)
/// Returns: [M, N] float32 result
pub fn matmulF32Q4(
    allocator: std.mem.Allocator,
    a: *const Tensor,
    b: *const QuantizedTensorQ4,
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

    // Simple i,k,j loop (unpacking 4-bit values on the fly)
    for (0..m) |i| {
        for (0..k) |kk| {
            const a_val = a.data[i * k + kk];

            for (0..n) |j| {
                // Get 4-bit quantized weight (unpacks from packed byte)
                const w_idx = kk * n + j;
                const w_int = b.get(w_idx);
                const w_f32 = @as(f32, @floatFromInt(w_int)) * scale;
                result.data[i * n + j] += a_val * w_f32;
            }
        }
    }

    return result;
}

/// SIMD-optimized Weight-Only Matmul for Q4_0
/// Unpacks 8 values at a time for vectorized processing
pub fn matmulF32Q4Simd(
    allocator: std.mem.Allocator,
    a: *const Tensor,
    b: *const QuantizedTensorQ4,
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

    // SIMD vector width (8 floats)
    const VEC_WIDTH = 8;
    const Vec8f32 = @Vector(VEC_WIDTH, f32);
    const Vec8i32 = @Vector(VEC_WIDTH, i32);

    for (0..m) |i| {
        for (0..k) |kk| {
            const a_val = a.data[i * k + kk];
            const a_vec: Vec8f32 = @splat(a_val);
            const scale_vec: Vec8f32 = @splat(scale);

            var j: usize = 0;

            // SIMD loop (8 elements at a time)
            while (j + VEC_WIDTH <= n) : (j += VEC_WIDTH) {
                // Unpack 8 x 4-bit values (4 bytes) into i32 vector
                const base_idx = kk * n + j;
                var w_i32: Vec8i32 = undefined;

                // Unpack each 4-bit value
                inline for (0..VEC_WIDTH) |vi| {
                    w_i32[vi] = b.get(base_idx + vi);
                }

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
                const w_idx = kk * n + j;
                const w_int = b.get(w_idx);
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

// ============================================================================
// Q4_0 Tests
// ============================================================================

test "Q4_0 quantization roundtrip" {
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
    var qtensor = try quantizeQ4(allocator, &tensor);
    defer qtensor.deinit();

    // Check compression ratio (4-bit = 8x theoretical, minus scale overhead)
    const ratio = qtensor.compressionRatio();
    try std.testing.expect(ratio > 3.0); // Should be close to 8x for large tensors

    // Dequantize
    var restored = try dequantizeQ4(allocator, &qtensor);
    defer restored.deinit();

    // Check accuracy (4-bit has higher error than 8-bit)
    // With only 15 levels, error can be up to ~7% of max value
    const tolerance = 2.5 * 0.15; // 15% of max
    for (tensor.data, 0..) |original, i| {
        try std.testing.expectApproxEqAbs(original, restored.data[i], tolerance);
    }
}

test "Q4_0 packing correctness" {
    const allocator = std.testing.allocator;

    var shape = [_]usize{8};
    var qtensor = try QuantizedTensorQ4.init(allocator, &shape);
    defer qtensor.deinit();

    // Set values: [-7, -3, 0, 3, 7, -1, 1, -2]
    qtensor.set(0, -7);
    qtensor.set(1, -3);
    qtensor.set(2, 0);
    qtensor.set(3, 3);
    qtensor.set(4, 7);
    qtensor.set(5, -1);
    qtensor.set(6, 1);
    qtensor.set(7, -2);

    // Verify get returns correct values
    try std.testing.expectEqual(@as(i8, -7), qtensor.get(0));
    try std.testing.expectEqual(@as(i8, -3), qtensor.get(1));
    try std.testing.expectEqual(@as(i8, 0), qtensor.get(2));
    try std.testing.expectEqual(@as(i8, 3), qtensor.get(3));
    try std.testing.expectEqual(@as(i8, 7), qtensor.get(4));
    try std.testing.expectEqual(@as(i8, -1), qtensor.get(5));
    try std.testing.expectEqual(@as(i8, 1), qtensor.get(6));
    try std.testing.expectEqual(@as(i8, -2), qtensor.get(7));

    // Check packed size (8 elements = 4 bytes)
    try std.testing.expectEqual(@as(usize, 4), qtensor.data.len);
}

test "Q4_0 size reduction" {
    const allocator = std.testing.allocator;

    // 1000 element tensor
    var shape = [_]usize{1000};
    var tensor = try Tensor.init(allocator, &shape);
    defer tensor.deinit();

    for (tensor.data, 0..) |*v, i| {
        v.* = @sin(@as(f32, @floatFromInt(i)) * 0.1);
    }

    var qtensor = try quantizeQ4(allocator, &tensor);
    defer qtensor.deinit();

    // Float32: 1000 * 4 = 4000 bytes
    // Q4_0:    4 + 500   = 504 bytes (8x reduction)
    const f32_size = tensor.data.len * 4;
    const q4_size = qtensor.sizeBytes();

    try std.testing.expect(q4_size < f32_size / 6); // At least 6x smaller
}

test "Q4_0 zero tensor" {
    const allocator = std.testing.allocator;

    var shape = [_]usize{10};
    var tensor = try Tensor.init(allocator, &shape);
    defer tensor.deinit();
    tensor.fill(0.0);

    var qtensor = try quantizeQ4(allocator, &tensor);
    defer qtensor.deinit();

    // Should handle zero tensor gracefully
    try std.testing.expectEqual(@as(f32, 1.0), qtensor.scale);
    for (0..qtensor.element_count) |i| {
        try std.testing.expectEqual(@as(i8, 0), qtensor.get(i));
    }
}

test "Q4_0 weight-only matmul accuracy" {
    const allocator = std.testing.allocator;

    // Float32 input (activations)
    var a_shape = [_]usize{ 2, 3 };
    var a = try Tensor.init(allocator, &a_shape);
    defer a.deinit();
    a.data[0] = 1.0;
    a.data[1] = 2.0;
    a.data[2] = 3.0;
    a.data[3] = 4.0;
    a.data[4] = 5.0;
    a.data[5] = 6.0;

    // Float32 weights (will be quantized to Q4)
    var b_shape = [_]usize{ 3, 2 };
    var b = try Tensor.init(allocator, &b_shape);
    defer b.deinit();
    b.data[0] = 1.0;
    b.data[1] = 2.0;
    b.data[2] = 3.0;
    b.data[3] = 4.0;
    b.data[4] = 5.0;
    b.data[5] = 6.0;

    // Float32 reference
    const ops = @import("ops.zig");
    var ref = try ops.matmul(allocator, &a, &b);
    defer ref.deinit();

    // Quantize weights to Q4_0
    var qb = try quantizeQ4(allocator, &b);
    defer qb.deinit();

    // Q4 weight-only matmul
    var qresult = try matmulF32Q4(allocator, &a, &qb);
    defer qresult.deinit();

    // Compare (allow 15% error due to 4-bit quantization)
    for (ref.data, 0..) |expected, i| {
        const tolerance = @abs(expected) * 0.15 + 0.5;
        try std.testing.expectApproxEqAbs(expected, qresult.data[i], tolerance);
    }
}

test "Q4_0 weight-only matmul SIMD" {
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
    var qb = try quantizeQ4(allocator, &b);
    defer qb.deinit();

    var scalar_result = try matmulF32Q4(allocator, &a, &qb);
    defer scalar_result.deinit();

    // SIMD path
    var simd_result = try matmulF32Q4Simd(allocator, &a, &qb);
    defer simd_result.deinit();

    // Both should produce the same result
    for (scalar_result.data, 0..) |expected, i| {
        try std.testing.expectApproxEqAbs(expected, simd_result.data[i], 0.0001);
    }
}

test "Q4_0 extreme values" {
    const allocator = std.testing.allocator;

    var shape = [_]usize{5};
    var tensor = try Tensor.init(allocator, &shape);
    defer tensor.deinit();

    tensor.data[0] = 100.0;
    tensor.data[1] = -100.0;
    tensor.data[2] = 0.0;
    tensor.data[3] = 50.0;
    tensor.data[4] = -50.0;

    var qtensor = try quantizeQ4(allocator, &tensor);
    defer qtensor.deinit();

    // Max and min should map to ±7
    try std.testing.expectEqual(@as(i8, 7), qtensor.get(0));
    try std.testing.expectEqual(@as(i8, -7), qtensor.get(1));
    try std.testing.expectEqual(@as(i8, 0), qtensor.get(2));

    // Check scale
    try std.testing.expectApproxEqAbs(@as(f32, 100.0 / 7.0), qtensor.scale, 0.0001);
}

// ============================================================================
// Q8_K Block-wise Tests
// ============================================================================

test "Q8_K block-wise quantization roundtrip" {
    const allocator = std.testing.allocator;

    // Create test tensor with 100 elements (spans multiple blocks)
    var shape = [_]usize{100};
    var tensor = try Tensor.init(allocator, &shape);
    defer tensor.deinit();

    // Fill with values that have varying ranges across blocks
    for (tensor.data, 0..) |*v, i| {
        // Block 0 (0-31): small values
        // Block 1 (32-63): medium values
        // Block 2 (64-95): large values
        // Block 3 (96-99): mixed
        const block = i / 32;
        const scale_factor: f32 = @as(f32, @floatFromInt(block + 1)) * 10.0;
        v.* = @sin(@as(f32, @floatFromInt(i)) * 0.1) * scale_factor;
    }

    // Quantize with block-wise
    var qtensor = try quantizeQ8K(allocator, &tensor);
    defer qtensor.deinit();

    // Check we have multiple blocks
    try std.testing.expect(qtensor.num_blocks > 1);

    // Check compression (slightly less than Q8_0 due to per-block scales)
    const ratio = qtensor.compressionRatio();
    try std.testing.expect(ratio > 2.5);

    // Dequantize
    var restored = try dequantizeQ8K(allocator, &qtensor);
    defer restored.deinit();

    // Check accuracy - should be better than per-tensor Q8_0 for varying ranges
    for (tensor.data, 0..) |original, i| {
        const tolerance = @abs(original) * 0.02 + 0.1; // 2% tolerance
        try std.testing.expectApproxEqAbs(original, restored.data[i], tolerance);
    }
}

test "Q8_K accuracy vs Q8_0 with outliers" {
    const allocator = std.testing.allocator;

    // Create tensor with an outlier that hurts per-tensor quantization
    var shape = [_]usize{64};
    var tensor = try Tensor.init(allocator, &shape);
    defer tensor.deinit();

    // First 32 elements: small values around 0.1
    for (0..32) |i| {
        tensor.data[i] = 0.1 + @sin(@as(f32, @floatFromInt(i)) * 0.1) * 0.05;
    }
    // Element 32 is an outlier
    tensor.data[32] = 100.0;
    // Rest are medium values
    for (33..64) |i| {
        tensor.data[i] = 1.0 + @cos(@as(f32, @floatFromInt(i)) * 0.1) * 0.5;
    }

    // Quantize with per-tensor Q8_0
    var q8_tensor = try quantizeQ8(allocator, &tensor);
    defer q8_tensor.deinit();

    // Quantize with block-wise Q8_K
    var q8k_tensor = try quantizeQ8K(allocator, &tensor);
    defer q8k_tensor.deinit();

    // Dequantize both
    var restored_q8 = try dequantizeQ8(allocator, &q8_tensor);
    defer restored_q8.deinit();

    var restored_q8k = try dequantizeQ8K(allocator, &q8k_tensor);
    defer restored_q8k.deinit();

    // Calculate error for small values (first 32 elements)
    var q8_error: f32 = 0.0;
    var q8k_error: f32 = 0.0;

    for (0..32) |i| {
        q8_error += @abs(tensor.data[i] - restored_q8.data[i]);
        q8k_error += @abs(tensor.data[i] - restored_q8k.data[i]);
    }

    // Q8_K should have lower error for small values when outliers exist
    // Because it uses separate scale for block 0 vs block 1 (with outlier)
    try std.testing.expect(q8k_error <= q8_error);
}

test "Q8_K weight-only matmul accuracy" {
    const allocator = std.testing.allocator;

    // Float32 input
    var a_shape = [_]usize{ 2, 64 };
    var a = try Tensor.init(allocator, &a_shape);
    defer a.deinit();
    for (a.data, 0..) |*v, i| {
        v.* = @sin(@as(f32, @floatFromInt(i)) * 0.1);
    }

    // Float32 weights (will be quantized)
    var b_shape = [_]usize{ 64, 32 };
    var b = try Tensor.init(allocator, &b_shape);
    defer b.deinit();
    for (b.data, 0..) |*v, i| {
        v.* = @cos(@as(f32, @floatFromInt(i)) * 0.1);
    }

    // Float32 reference
    const ops = @import("ops.zig");
    var ref = try ops.matmul(allocator, &a, &b);
    defer ref.deinit();

    // Quantize weights to Q8_K
    var qb = try quantizeQ8K(allocator, &b);
    defer qb.deinit();

    // Q8_K weight-only matmul
    var qresult = try matmulF32Q8K(allocator, &a, &qb);
    defer qresult.deinit();

    // Compare (allow 5% error)
    for (ref.data, 0..) |expected, i| {
        const tolerance = @abs(expected) * 0.05 + 0.1;
        try std.testing.expectApproxEqAbs(expected, qresult.data[i], tolerance);
    }
}

test "Q8_K weight-only matmul SIMD" {
    const allocator = std.testing.allocator;

    // Test SIMD path
    var a_shape = [_]usize{ 4, 64 };
    var a = try Tensor.init(allocator, &a_shape);
    defer a.deinit();
    for (a.data, 0..) |*v, i| {
        v.* = @sin(@as(f32, @floatFromInt(i)) * 0.1);
    }

    var b_shape = [_]usize{ 64, 32 };
    var b = try Tensor.init(allocator, &b_shape);
    defer b.deinit();
    for (b.data, 0..) |*v, i| {
        v.* = @cos(@as(f32, @floatFromInt(i)) * 0.1);
    }

    var qb = try quantizeQ8K(allocator, &b);
    defer qb.deinit();

    var scalar_result = try matmulF32Q8K(allocator, &a, &qb);
    defer scalar_result.deinit();

    var simd_result = try matmulF32Q8KSimd(allocator, &a, &qb);
    defer simd_result.deinit();

    // Both should produce the same result
    for (scalar_result.data, 0..) |expected, i| {
        try std.testing.expectApproxEqAbs(expected, simd_result.data[i], 0.0001);
    }
}

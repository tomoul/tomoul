// src/core/attention_generic.zig
// Generic Multi-Head Attention implementation supporting multiple weight formats
//
// Supports:
// - F32: Standard float32 weights (Tensor)
// - Q8: Simple 8-bit symmetric quantization (QuantizedTensorQ8)
// - Q4: Simple 4-bit symmetric quantization (QuantizedTensorQ4)
// - Q8_K: Block-wise 8-bit quantization (QuantizedTensorQ8K)
//
// Weight-only quantization: inputs remain float32, weights are quantized.
// Uses compile-time generics for zero-cost abstraction.

const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const ops = @import("ops.zig");
const quant = @import("quantization.zig");

const QuantizedTensorQ8 = quant.QuantizedTensorQ8;
const QuantizedTensorQ4 = quant.QuantizedTensorQ4;
const QuantizedTensorQ8K = quant.QuantizedTensorQ8K;

/// Weight format enumeration for runtime checks and debugging
pub const WeightFormat = enum {
    f32,
    q8,
    q4,
    q8_k,

    pub fn name(self: WeightFormat) []const u8 {
        return switch (self) {
            .f32 => "F32",
            .q8 => "Q8_0",
            .q4 => "Q4_0",
            .q8_k => "Q8_K",
        };
    }

    pub fn bitsPerWeight(self: WeightFormat) u8 {
        return switch (self) {
            .f32 => 32,
            .q8 => 8,
            .q4 => 4,
            .q8_k => 8,
        };
    }
};

/// Generic attention weights for a single attention layer
/// WeightType can be: Tensor (f32), QuantizedTensorQ8, QuantizedTensorQ4, QuantizedTensorQ8K
pub fn AttentionWeights(comptime WeightType: type) type {
    return struct {
        // Projection weights [hidden, hidden] - pre-transposed
        q_weight: WeightType,
        k_weight: WeightType,
        v_weight: WeightType,
        o_weight: WeightType,

        // Biases remain float32 (small size, sensitive to quantization)
        q_bias: Tensor,
        k_bias: Tensor,
        v_bias: Tensor,
        o_bias: Tensor,

        const Self = @This();

        /// Get the weight format enum for this type
        pub fn format() WeightFormat {
            return comptime getWeightFormat(WeightType);
        }

        pub fn deinit(self: *Self) void {
            self.q_weight.deinit();
            self.k_weight.deinit();
            self.v_weight.deinit();
            self.o_weight.deinit();
            self.q_bias.deinit();
            self.k_bias.deinit();
            self.v_bias.deinit();
            self.o_bias.deinit();
        }
    };
}

/// Convenience type aliases for common weight formats
pub const AttentionWeightsF32 = AttentionWeights(Tensor);
pub const AttentionWeightsQ8 = AttentionWeights(QuantizedTensorQ8);
pub const AttentionWeightsQ4 = AttentionWeights(QuantizedTensorQ4);
pub const AttentionWeightsQ8K = AttentionWeights(QuantizedTensorQ8K);

/// Configuration for attention mechanism
pub const AttentionConfig = struct {
    num_heads: usize, // 12 for base, 16 for large
    hidden_dim: usize, // 768 for base, 1024 for large
    head_dim: usize, // hidden_dim / num_heads
};

/// Get the weight format enum for a given weight type
fn getWeightFormat(comptime T: type) WeightFormat {
    if (T == Tensor) return .f32;
    if (T == QuantizedTensorQ8) return .q8;
    if (T == QuantizedTensorQ4) return .q4;
    if (T == QuantizedTensorQ8K) return .q8_k;
    @compileError("Unsupported weight type: " ++ @typeName(T));
}

/// Generic matrix multiplication: float32 input @ weight -> float32 output
/// Dispatches to appropriate implementation based on weight type
fn matmulWithWeight(
    comptime WeightType: type,
    allocator: std.mem.Allocator,
    input: *const Tensor,
    weight: *const WeightType,
) !Tensor {
    const format = comptime getWeightFormat(WeightType);

    return switch (format) {
        .f32 => ops.matmul(allocator, input, weight),
        .q8 => quant.matmulF32Q8Simd(allocator, input, weight),
        .q4 => quant.matmulF32Q4Simd(allocator, input, weight),
        .q8_k => quant.matmulF32Q8KSimd(allocator, input, weight),
    };
}

/// Scaled dot-product attention (float32 only, used after projection)
/// Q, K, V: [seq_len, head_dim]
/// Returns: [seq_len, head_dim]
///
/// Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
fn scaledDotProductAttention(
    allocator: std.mem.Allocator,
    q: *const Tensor,
    k: *const Tensor,
    v: *const Tensor,
) !Tensor {
    const head_dim = q.shape[1];
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    // scores = Q @ K^T
    var k_t = try ops.transpose(allocator, k);
    defer k_t.deinit();

    var scores = try ops.matmul(allocator, q, &k_t);
    defer scores.deinit();

    // Scale
    ops.scaleInPlace(&scores, scale);

    // Softmax
    ops.softmax(&scores);

    // Output = scores @ V
    return ops.matmul(allocator, &scores, v);
}

/// Generic multi-head attention
/// input: [seq_len, hidden_dim]
/// Returns: [seq_len, hidden_dim]
///
/// MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_o
/// where head_i = Attention(Q @ W_q_i, K @ W_k_i, V @ W_v_i)
///
/// NOTE: Weights are PRE-TRANSPOSED at load time for optimal SIMD matmul performance.
pub fn multiHeadAttention(
    comptime WeightType: type,
    allocator: std.mem.Allocator,
    input: *const Tensor,
    weights: *const AttentionWeights(WeightType),
    config: AttentionConfig,
) !Tensor {
    const num_heads = config.num_heads;
    const head_dim = config.head_dim;

    // Project Q, K, V using appropriate matmul for weight type
    var q = try matmulWithWeight(WeightType, allocator, input, &weights.q_weight);
    defer q.deinit();
    var k = try matmulWithWeight(WeightType, allocator, input, &weights.k_weight);
    defer k.deinit();
    var v = try matmulWithWeight(WeightType, allocator, input, &weights.v_weight);
    defer v.deinit();

    try ops.addBiasInPlace(&q, &weights.q_bias);
    try ops.addBiasInPlace(&k, &weights.k_bias);
    try ops.addBiasInPlace(&v, &weights.v_bias);

    // Split into heads and compute attention
    var head_outputs = try allocator.alloc(Tensor, num_heads);
    var head_count: usize = 0;
    errdefer {
        for (head_outputs[0..head_count]) |*h| h.deinit();
        allocator.free(head_outputs);
    }

    for (0..num_heads) |h| {
        const start = h * head_dim;
        const end = start + head_dim;

        // Extract head slices
        var q_head = try ops.sliceColumns(allocator, &q, start, end);
        defer q_head.deinit();
        var k_head = try ops.sliceColumns(allocator, &k, start, end);
        defer k_head.deinit();
        var v_head = try ops.sliceColumns(allocator, &v, start, end);
        defer v_head.deinit();

        head_outputs[h] = try scaledDotProductAttention(
            allocator,
            &q_head,
            &k_head,
            &v_head,
        );
        head_count += 1;
    }
    defer {
        for (head_outputs) |*h| h.deinit();
        allocator.free(head_outputs);
    }

    // Concatenate heads
    var concat = try ops.concatColumns(allocator, head_outputs);
    defer concat.deinit();

    // Final projection using appropriate matmul for weight type
    var output = try matmulWithWeight(WeightType, allocator, &concat, &weights.o_weight);
    try ops.addBiasInPlace(&output, &weights.o_bias);

    return output;
}

/// Self-attention helper: uses same input for Q, K, V
pub fn selfAttention(
    comptime WeightType: type,
    allocator: std.mem.Allocator,
    input: *const Tensor,
    weights: *const AttentionWeights(WeightType),
    config: AttentionConfig,
) !Tensor {
    return multiHeadAttention(WeightType, allocator, input, weights, config);
}

// ============================================================================
// Convenience Functions (Non-Generic API)
// ============================================================================

/// Multi-head attention with F32 weights
pub fn multiHeadAttentionF32(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    weights: *const AttentionWeightsF32,
    config: AttentionConfig,
) !Tensor {
    return multiHeadAttention(Tensor, allocator, input, weights, config);
}

/// Multi-head attention with Q8 weights
pub fn multiHeadAttentionQ8(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    weights: *const AttentionWeightsQ8,
    config: AttentionConfig,
) !Tensor {
    return multiHeadAttention(QuantizedTensorQ8, allocator, input, weights, config);
}

/// Multi-head attention with Q4 weights
pub fn multiHeadAttentionQ4(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    weights: *const AttentionWeightsQ4,
    config: AttentionConfig,
) !Tensor {
    return multiHeadAttention(QuantizedTensorQ4, allocator, input, weights, config);
}

/// Multi-head attention with Q8_K weights
pub fn multiHeadAttentionQ8K(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    weights: *const AttentionWeightsQ8K,
    config: AttentionConfig,
) !Tensor {
    return multiHeadAttention(QuantizedTensorQ8K, allocator, input, weights, config);
}

// ============================================================================
// Tests
// ============================================================================

test "generic attention with F32 weights" {
    const allocator = std.testing.allocator;

    // Small config: 2 heads, 4 hidden dim, 2 head dim
    const config = AttentionConfig{
        .num_heads = 2,
        .hidden_dim = 4,
        .head_dim = 2,
    };

    // Input: [2, 4] (2 tokens, 4 hidden dim)
    var input_shape = [_]usize{ 2, 4 };
    var input = try Tensor.init(allocator, &input_shape);
    defer input.deinit();
    for (input.data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i)) * 0.1;
    }

    // Create identity-like weights
    var weight_shape = [_]usize{ 4, 4 };
    var bias_shape = [_]usize{4};

    var q_weight = try Tensor.init(allocator, &weight_shape);
    errdefer q_weight.deinit();
    q_weight.data[0] = 1.0;
    q_weight.data[5] = 1.0;
    q_weight.data[10] = 1.0;
    q_weight.data[15] = 1.0;

    var k_weight = try Tensor.init(allocator, &weight_shape);
    errdefer k_weight.deinit();
    @memcpy(k_weight.data, q_weight.data);

    var v_weight = try Tensor.init(allocator, &weight_shape);
    errdefer v_weight.deinit();
    @memcpy(v_weight.data, q_weight.data);

    var o_weight = try Tensor.init(allocator, &weight_shape);
    errdefer o_weight.deinit();
    @memcpy(o_weight.data, q_weight.data);

    var q_bias = try Tensor.init(allocator, &bias_shape);
    errdefer q_bias.deinit();
    var k_bias = try Tensor.init(allocator, &bias_shape);
    errdefer k_bias.deinit();
    var v_bias = try Tensor.init(allocator, &bias_shape);
    errdefer v_bias.deinit();
    var o_bias = try Tensor.init(allocator, &bias_shape);
    errdefer o_bias.deinit();

    var weights = AttentionWeightsF32{
        .q_weight = q_weight,
        .k_weight = k_weight,
        .v_weight = v_weight,
        .o_weight = o_weight,
        .q_bias = q_bias,
        .k_bias = k_bias,
        .v_bias = v_bias,
        .o_bias = o_bias,
    };
    defer weights.deinit();

    // Test using generic function
    var output = try multiHeadAttention(Tensor, allocator, &input, &weights, config);
    defer output.deinit();

    // Output should have same shape as input
    try std.testing.expectEqual(@as(usize, 2), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), output.shape[1]);

    // Values should be valid
    for (output.data) |val| {
        try std.testing.expect(!std.math.isNan(val));
        try std.testing.expect(!std.math.isInf(val));
    }

    // Test format detection
    try std.testing.expectEqual(WeightFormat.f32, AttentionWeightsF32.format());
}

test "generic attention with Q8 weights" {
    const allocator = std.testing.allocator;

    const config = AttentionConfig{
        .num_heads = 2,
        .hidden_dim = 4,
        .head_dim = 2,
    };

    var input_shape = [_]usize{ 2, 4 };
    var input = try Tensor.init(allocator, &input_shape);
    defer input.deinit();
    for (input.data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i)) * 0.1;
    }

    // Create float weights then quantize
    var weight_shape = [_]usize{ 4, 4 };
    var bias_shape = [_]usize{4};

    var weight_f32 = try Tensor.init(allocator, &weight_shape);
    defer weight_f32.deinit();
    weight_f32.data[0] = 1.0;
    weight_f32.data[5] = 1.0;
    weight_f32.data[10] = 1.0;
    weight_f32.data[15] = 1.0;

    // Quantize to Q8
    var q_weight = try quant.quantizeQ8(allocator, &weight_f32);
    errdefer q_weight.deinit();
    var k_weight = try quant.quantizeQ8(allocator, &weight_f32);
    errdefer k_weight.deinit();
    var v_weight = try quant.quantizeQ8(allocator, &weight_f32);
    errdefer v_weight.deinit();
    var o_weight = try quant.quantizeQ8(allocator, &weight_f32);
    errdefer o_weight.deinit();

    var q_bias = try Tensor.init(allocator, &bias_shape);
    errdefer q_bias.deinit();
    var k_bias = try Tensor.init(allocator, &bias_shape);
    errdefer k_bias.deinit();
    var v_bias = try Tensor.init(allocator, &bias_shape);
    errdefer v_bias.deinit();
    var o_bias = try Tensor.init(allocator, &bias_shape);
    errdefer o_bias.deinit();

    var weights = AttentionWeightsQ8{
        .q_weight = q_weight,
        .k_weight = k_weight,
        .v_weight = v_weight,
        .o_weight = o_weight,
        .q_bias = q_bias,
        .k_bias = k_bias,
        .v_bias = v_bias,
        .o_bias = o_bias,
    };
    defer weights.deinit();

    var output = try multiHeadAttention(QuantizedTensorQ8, allocator, &input, &weights, config);
    defer output.deinit();

    try std.testing.expectEqual(@as(usize, 2), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), output.shape[1]);

    for (output.data) |val| {
        try std.testing.expect(!std.math.isNan(val));
        try std.testing.expect(!std.math.isInf(val));
    }

    try std.testing.expectEqual(WeightFormat.q8, AttentionWeightsQ8.format());
}

test "generic attention with Q4 weights" {
    const allocator = std.testing.allocator;

    const config = AttentionConfig{
        .num_heads = 2,
        .hidden_dim = 4,
        .head_dim = 2,
    };

    var input_shape = [_]usize{ 2, 4 };
    var input = try Tensor.init(allocator, &input_shape);
    defer input.deinit();
    for (input.data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i)) * 0.1;
    }

    var weight_shape = [_]usize{ 4, 4 };
    var bias_shape = [_]usize{4};

    var weight_f32 = try Tensor.init(allocator, &weight_shape);
    defer weight_f32.deinit();
    weight_f32.data[0] = 1.0;
    weight_f32.data[5] = 1.0;
    weight_f32.data[10] = 1.0;
    weight_f32.data[15] = 1.0;

    // Quantize to Q4
    var q_weight = try quant.quantizeQ4(allocator, &weight_f32);
    errdefer q_weight.deinit();
    var k_weight = try quant.quantizeQ4(allocator, &weight_f32);
    errdefer k_weight.deinit();
    var v_weight = try quant.quantizeQ4(allocator, &weight_f32);
    errdefer v_weight.deinit();
    var o_weight = try quant.quantizeQ4(allocator, &weight_f32);
    errdefer o_weight.deinit();

    var q_bias = try Tensor.init(allocator, &bias_shape);
    errdefer q_bias.deinit();
    var k_bias = try Tensor.init(allocator, &bias_shape);
    errdefer k_bias.deinit();
    var v_bias = try Tensor.init(allocator, &bias_shape);
    errdefer v_bias.deinit();
    var o_bias = try Tensor.init(allocator, &bias_shape);
    errdefer o_bias.deinit();

    var weights = AttentionWeightsQ4{
        .q_weight = q_weight,
        .k_weight = k_weight,
        .v_weight = v_weight,
        .o_weight = o_weight,
        .q_bias = q_bias,
        .k_bias = k_bias,
        .v_bias = v_bias,
        .o_bias = o_bias,
    };
    defer weights.deinit();

    var output = try multiHeadAttention(QuantizedTensorQ4, allocator, &input, &weights, config);
    defer output.deinit();

    try std.testing.expectEqual(@as(usize, 2), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), output.shape[1]);

    for (output.data) |val| {
        try std.testing.expect(!std.math.isNan(val));
        try std.testing.expect(!std.math.isInf(val));
    }

    try std.testing.expectEqual(WeightFormat.q4, AttentionWeightsQ4.format());
}

test "generic attention with Q8_K weights" {
    const allocator = std.testing.allocator;

    const config = AttentionConfig{
        .num_heads = 2,
        .hidden_dim = 4,
        .head_dim = 2,
    };

    var input_shape = [_]usize{ 2, 4 };
    var input = try Tensor.init(allocator, &input_shape);
    defer input.deinit();
    for (input.data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i)) * 0.1;
    }

    var weight_shape = [_]usize{ 4, 4 };
    var bias_shape = [_]usize{4};

    var weight_f32 = try Tensor.init(allocator, &weight_shape);
    defer weight_f32.deinit();
    weight_f32.data[0] = 1.0;
    weight_f32.data[5] = 1.0;
    weight_f32.data[10] = 1.0;
    weight_f32.data[15] = 1.0;

    // Quantize to Q8_K
    var q_weight = try quant.quantizeQ8K(allocator, &weight_f32);
    errdefer q_weight.deinit();
    var k_weight = try quant.quantizeQ8K(allocator, &weight_f32);
    errdefer k_weight.deinit();
    var v_weight = try quant.quantizeQ8K(allocator, &weight_f32);
    errdefer v_weight.deinit();
    var o_weight = try quant.quantizeQ8K(allocator, &weight_f32);
    errdefer o_weight.deinit();

    var q_bias = try Tensor.init(allocator, &bias_shape);
    errdefer q_bias.deinit();
    var k_bias = try Tensor.init(allocator, &bias_shape);
    errdefer k_bias.deinit();
    var v_bias = try Tensor.init(allocator, &bias_shape);
    errdefer v_bias.deinit();
    var o_bias = try Tensor.init(allocator, &bias_shape);
    errdefer o_bias.deinit();

    var weights = AttentionWeightsQ8K{
        .q_weight = q_weight,
        .k_weight = k_weight,
        .v_weight = v_weight,
        .o_weight = o_weight,
        .q_bias = q_bias,
        .k_bias = k_bias,
        .v_bias = v_bias,
        .o_bias = o_bias,
    };
    defer weights.deinit();

    var output = try multiHeadAttention(QuantizedTensorQ8K, allocator, &input, &weights, config);
    defer output.deinit();

    try std.testing.expectEqual(@as(usize, 2), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), output.shape[1]);

    for (output.data) |val| {
        try std.testing.expect(!std.math.isNan(val));
        try std.testing.expect(!std.math.isInf(val));
    }

    try std.testing.expectEqual(WeightFormat.q8_k, AttentionWeightsQ8K.format());
}

test "convenience functions match generic API" {
    const allocator = std.testing.allocator;

    const config = AttentionConfig{
        .num_heads = 2,
        .hidden_dim = 4,
        .head_dim = 2,
    };

    var input_shape = [_]usize{ 2, 4 };
    var input = try Tensor.init(allocator, &input_shape);
    defer input.deinit();
    for (input.data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i)) * 0.1;
    }

    var weight_shape = [_]usize{ 4, 4 };
    var bias_shape = [_]usize{4};

    var weight_f32 = try Tensor.init(allocator, &weight_shape);
    defer weight_f32.deinit();
    weight_f32.data[0] = 1.0;
    weight_f32.data[5] = 1.0;
    weight_f32.data[10] = 1.0;
    weight_f32.data[15] = 1.0;

    // Create F32 weights
    var q_weight = try Tensor.init(allocator, &weight_shape);
    errdefer q_weight.deinit();
    @memcpy(q_weight.data, weight_f32.data);
    var k_weight = try Tensor.init(allocator, &weight_shape);
    errdefer k_weight.deinit();
    @memcpy(k_weight.data, weight_f32.data);
    var v_weight = try Tensor.init(allocator, &weight_shape);
    errdefer v_weight.deinit();
    @memcpy(v_weight.data, weight_f32.data);
    var o_weight = try Tensor.init(allocator, &weight_shape);
    errdefer o_weight.deinit();
    @memcpy(o_weight.data, weight_f32.data);

    var q_bias = try Tensor.init(allocator, &bias_shape);
    errdefer q_bias.deinit();
    var k_bias = try Tensor.init(allocator, &bias_shape);
    errdefer k_bias.deinit();
    var v_bias = try Tensor.init(allocator, &bias_shape);
    errdefer v_bias.deinit();
    var o_bias = try Tensor.init(allocator, &bias_shape);
    errdefer o_bias.deinit();

    var weights = AttentionWeightsF32{
        .q_weight = q_weight,
        .k_weight = k_weight,
        .v_weight = v_weight,
        .o_weight = o_weight,
        .q_bias = q_bias,
        .k_bias = k_bias,
        .v_bias = v_bias,
        .o_bias = o_bias,
    };
    defer weights.deinit();

    // Test generic API
    var output_generic = try multiHeadAttention(Tensor, allocator, &input, &weights, config);
    defer output_generic.deinit();

    // Test convenience function
    var output_convenience = try multiHeadAttentionF32(allocator, &input, &weights, config);
    defer output_convenience.deinit();

    // Results should be identical
    for (output_generic.data, output_convenience.data) |gen, conv| {
        try std.testing.expectApproxEqAbs(gen, conv, 0.0001);
    }
}

test "format detection" {
    try std.testing.expectEqual(WeightFormat.f32, getWeightFormat(Tensor));
    try std.testing.expectEqual(WeightFormat.q8, getWeightFormat(QuantizedTensorQ8));
    try std.testing.expectEqual(WeightFormat.q4, getWeightFormat(QuantizedTensorQ4));
    try std.testing.expectEqual(WeightFormat.q8_k, getWeightFormat(QuantizedTensorQ8K));

    try std.testing.expectEqualStrings("F32", WeightFormat.f32.name());
    try std.testing.expectEqualStrings("Q8_0", WeightFormat.q8.name());
    try std.testing.expectEqualStrings("Q4_0", WeightFormat.q4.name());
    try std.testing.expectEqualStrings("Q8_K", WeightFormat.q8_k.name());

    try std.testing.expectEqual(@as(u8, 32), WeightFormat.f32.bitsPerWeight());
    try std.testing.expectEqual(@as(u8, 8), WeightFormat.q8.bitsPerWeight());
    try std.testing.expectEqual(@as(u8, 4), WeightFormat.q4.bitsPerWeight());
    try std.testing.expectEqual(@as(u8, 8), WeightFormat.q8_k.bitsPerWeight());
}

test "F32 vs quantized accuracy comparison" {
    const allocator = std.testing.allocator;

    const config = AttentionConfig{
        .num_heads = 2,
        .hidden_dim = 8,
        .head_dim = 4,
    };

    // Larger input for meaningful comparison
    var input_shape = [_]usize{ 4, 8 };
    var input = try Tensor.init(allocator, &input_shape);
    defer input.deinit();
    for (input.data, 0..) |*val, i| {
        val.* = @sin(@as(f32, @floatFromInt(i)) * 0.1);
    }

    // Create realistic weights
    var weight_shape = [_]usize{ 8, 8 };
    var bias_shape = [_]usize{8};

    var weight_f32 = try Tensor.init(allocator, &weight_shape);
    defer weight_f32.deinit();
    for (weight_f32.data, 0..) |*val, i| {
        val.* = @cos(@as(f32, @floatFromInt(i)) * 0.1) * 0.5;
    }

    // F32 weights
    var q_weight_f32 = try Tensor.init(allocator, &weight_shape);
    errdefer q_weight_f32.deinit();
    @memcpy(q_weight_f32.data, weight_f32.data);
    var k_weight_f32 = try Tensor.init(allocator, &weight_shape);
    errdefer k_weight_f32.deinit();
    @memcpy(k_weight_f32.data, weight_f32.data);
    var v_weight_f32 = try Tensor.init(allocator, &weight_shape);
    errdefer v_weight_f32.deinit();
    @memcpy(v_weight_f32.data, weight_f32.data);
    var o_weight_f32 = try Tensor.init(allocator, &weight_shape);
    errdefer o_weight_f32.deinit();
    @memcpy(o_weight_f32.data, weight_f32.data);

    var q_bias_f32 = try Tensor.init(allocator, &bias_shape);
    errdefer q_bias_f32.deinit();
    var k_bias_f32 = try Tensor.init(allocator, &bias_shape);
    errdefer k_bias_f32.deinit();
    var v_bias_f32 = try Tensor.init(allocator, &bias_shape);
    errdefer v_bias_f32.deinit();
    var o_bias_f32 = try Tensor.init(allocator, &bias_shape);
    errdefer o_bias_f32.deinit();

    var weights_f32 = AttentionWeightsF32{
        .q_weight = q_weight_f32,
        .k_weight = k_weight_f32,
        .v_weight = v_weight_f32,
        .o_weight = o_weight_f32,
        .q_bias = q_bias_f32,
        .k_bias = k_bias_f32,
        .v_bias = v_bias_f32,
        .o_bias = o_bias_f32,
    };
    defer weights_f32.deinit();

    // Q8 weights
    var q_weight_q8 = try quant.quantizeQ8(allocator, &weight_f32);
    errdefer q_weight_q8.deinit();
    var k_weight_q8 = try quant.quantizeQ8(allocator, &weight_f32);
    errdefer k_weight_q8.deinit();
    var v_weight_q8 = try quant.quantizeQ8(allocator, &weight_f32);
    errdefer v_weight_q8.deinit();
    var o_weight_q8 = try quant.quantizeQ8(allocator, &weight_f32);
    errdefer o_weight_q8.deinit();

    var q_bias_q8 = try Tensor.init(allocator, &bias_shape);
    errdefer q_bias_q8.deinit();
    var k_bias_q8 = try Tensor.init(allocator, &bias_shape);
    errdefer k_bias_q8.deinit();
    var v_bias_q8 = try Tensor.init(allocator, &bias_shape);
    errdefer v_bias_q8.deinit();
    var o_bias_q8 = try Tensor.init(allocator, &bias_shape);
    errdefer o_bias_q8.deinit();

    var weights_q8 = AttentionWeightsQ8{
        .q_weight = q_weight_q8,
        .k_weight = k_weight_q8,
        .v_weight = v_weight_q8,
        .o_weight = o_weight_q8,
        .q_bias = q_bias_q8,
        .k_bias = k_bias_q8,
        .v_bias = v_bias_q8,
        .o_bias = o_bias_q8,
    };
    defer weights_q8.deinit();

    // Compute outputs
    var output_f32 = try multiHeadAttention(Tensor, allocator, &input, &weights_f32, config);
    defer output_f32.deinit();

    var output_q8 = try multiHeadAttention(QuantizedTensorQ8, allocator, &input, &weights_q8, config);
    defer output_q8.deinit();

    // Q8 should be close to F32 (within 5% relative error)
    var max_diff: f32 = 0.0;
    for (output_f32.data, output_q8.data) |f32_val, q8_val| {
        const diff = @abs(f32_val - q8_val);
        max_diff = @max(max_diff, diff);
    }

    // Allow reasonable quantization error
    try std.testing.expect(max_diff < 0.1);
}

// src/core/attention.zig
// Multi-Head Attention implementation for Transformer models (Phase 6)

const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const TensorError = @import("tensor.zig").TensorError;
const ops = @import("ops.zig");
const OpsError = ops.OpsError;

/// Attention weights for a single attention layer
pub const AttentionWeights = struct {
    q_weight: Tensor, // [hidden, hidden]
    k_weight: Tensor, // [hidden, hidden]
    v_weight: Tensor, // [hidden, hidden]
    o_weight: Tensor, // [hidden, hidden]
    q_bias: Tensor, // [hidden]
    k_bias: Tensor, // [hidden]
    v_bias: Tensor, // [hidden]
    o_bias: Tensor, // [hidden]

    const Self = @This();

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

/// Configuration for attention mechanism
pub const AttentionConfig = struct {
    num_heads: usize, // 12 for DistilBERT
    hidden_dim: usize, // 768 for DistilBERT
    head_dim: usize, // 64 (768 / 12)
};

/// Scaled dot-product attention
/// Q, K, V: [seq_len, head_dim]
/// Returns: [seq_len, head_dim]
///
/// Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
pub fn scaledDotProductAttention(
    allocator: std.mem.Allocator,
    q: *const Tensor,
    k: *const Tensor,
    v: *const Tensor,
    mask: ?*const Tensor,
) !Tensor {
    if (q.shape.len != 2 or k.shape.len != 2 or v.shape.len != 2) {
        return OpsError.InvalidShape;
    }

    const head_dim = q.shape[1];
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    // scores = Q @ K^T
    var k_t = try ops.transpose(allocator, k);
    defer k_t.deinit();

    var scores = try ops.matmul(allocator, q, &k_t);
    defer scores.deinit();

    // Scale
    ops.scaleInPlace(&scores, scale);

    // Apply mask if provided (for causal attention)
    if (mask) |m| {
        for (scores.data, 0..) |*val, i| {
            if (m.data[i] == 0) {
                val.* = -std.math.inf(f32);
            }
        }
    }

    // Softmax
    ops.softmax(&scores);

    // Output = scores @ V
    return ops.matmul(allocator, &scores, v);
}

/// Multi-head attention
/// input: [seq_len, hidden_dim]
/// Returns: [seq_len, hidden_dim]
///
/// MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_o
/// where head_i = Attention(Q @ W_q_i, K @ W_k_i, V @ W_v_i)
///
/// NOTE: Weights are PRE-TRANSPOSED at load time for optimal SIMD matmul performance.
pub fn multiHeadAttention(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    weights: *const AttentionWeights,
    config: AttentionConfig,
) !Tensor {
    const num_heads = config.num_heads;
    const head_dim = config.head_dim;

    // Project Q, K, V (weights are pre-transposed for optimal cache access)
    var q = try ops.matmul(allocator, input, &weights.q_weight);
    defer q.deinit();
    var k = try ops.matmul(allocator, input, &weights.k_weight);
    defer k.deinit();
    var v = try ops.matmul(allocator, input, &weights.v_weight);
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
            null,
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

    // Final projection: concat @ W_o (weight pre-transposed)
    var output = try ops.matmul(allocator, &concat, &weights.o_weight);
    try ops.addBiasInPlace(&output, &weights.o_bias);

    return output;
}

/// Self-attention helper: uses same input for Q, K, V
pub fn selfAttention(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    weights: *const AttentionWeights,
    config: AttentionConfig,
) !Tensor {
    return multiHeadAttention(allocator, input, weights, config);
}

// ============================================================================
// Tests
// ============================================================================

test "scaled dot product attention" {
    const allocator = std.testing.allocator;

    // Small test: 2 tokens, 4 head_dim
    var q_shape = [_]usize{ 2, 4 };
    var q = try Tensor.init(allocator, &q_shape);
    defer q.deinit();
    // Q = [[1, 0, 0, 0], [0, 1, 0, 0]]
    q.data[0] = 1.0;
    q.data[5] = 1.0;

    var k = try Tensor.init(allocator, &q_shape);
    defer k.deinit();
    // K = same as Q for identity-like behavior
    k.data[0] = 1.0;
    k.data[5] = 1.0;

    var v = try Tensor.init(allocator, &q_shape);
    defer v.deinit();
    // V = [[1, 2, 3, 4], [5, 6, 7, 8]]
    v.data[0] = 1.0;
    v.data[1] = 2.0;
    v.data[2] = 3.0;
    v.data[3] = 4.0;
    v.data[4] = 5.0;
    v.data[5] = 6.0;
    v.data[6] = 7.0;
    v.data[7] = 8.0;

    var output = try scaledDotProductAttention(allocator, &q, &k, &v, null);
    defer output.deinit();

    // Output shape should be [2, 4]
    try std.testing.expectEqual(@as(usize, 2), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), output.shape[1]);

    // Values should be weighted combination of V rows
    // Due to softmax, each output should be a valid weighted average
    for (output.data) |val| {
        try std.testing.expect(!std.math.isNan(val));
        try std.testing.expect(!std.math.isInf(val));
    }
}

test "multi head attention small" {
    const allocator = std.testing.allocator;

    // Very small config: 2 heads, 4 hidden dim, 2 head dim
    const config = AttentionConfig{
        .num_heads = 2,
        .hidden_dim = 4,
        .head_dim = 2,
    };

    // Input: [2, 4] (2 tokens, 4 hidden dim)
    var input_shape = [_]usize{ 2, 4 };
    var input = try Tensor.init(allocator, &input_shape);
    defer input.deinit();
    for (input.data, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) * 0.1;
    }

    // Create minimal weights
    var weight_shape = [_]usize{ 4, 4 };
    var bias_shape = [_]usize{4};

    var q_weight = try Tensor.init(allocator, &weight_shape);
    defer q_weight.deinit();
    // Initialize as identity-like
    q_weight.data[0] = 1.0;
    q_weight.data[5] = 1.0;
    q_weight.data[10] = 1.0;
    q_weight.data[15] = 1.0;

    var k_weight = try Tensor.init(allocator, &weight_shape);
    defer k_weight.deinit();
    k_weight.data[0] = 1.0;
    k_weight.data[5] = 1.0;
    k_weight.data[10] = 1.0;
    k_weight.data[15] = 1.0;

    var v_weight = try Tensor.init(allocator, &weight_shape);
    defer v_weight.deinit();
    v_weight.data[0] = 1.0;
    v_weight.data[5] = 1.0;
    v_weight.data[10] = 1.0;
    v_weight.data[15] = 1.0;

    var o_weight = try Tensor.init(allocator, &weight_shape);
    defer o_weight.deinit();
    o_weight.data[0] = 1.0;
    o_weight.data[5] = 1.0;
    o_weight.data[10] = 1.0;
    o_weight.data[15] = 1.0;

    var q_bias = try Tensor.init(allocator, &bias_shape);
    defer q_bias.deinit();
    var k_bias = try Tensor.init(allocator, &bias_shape);
    defer k_bias.deinit();
    var v_bias = try Tensor.init(allocator, &bias_shape);
    defer v_bias.deinit();
    var o_bias = try Tensor.init(allocator, &bias_shape);
    defer o_bias.deinit();

    const weights = AttentionWeights{
        .q_weight = q_weight,
        .k_weight = k_weight,
        .v_weight = v_weight,
        .o_weight = o_weight,
        .q_bias = q_bias,
        .k_bias = k_bias,
        .v_bias = v_bias,
        .o_bias = o_bias,
    };

    var output = try multiHeadAttention(allocator, &input, &weights, config);
    defer output.deinit();

    // Output should have same shape as input
    try std.testing.expectEqual(@as(usize, 2), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), output.shape[1]);

    // Values should be valid (not NaN or Inf)
    for (output.data) |val| {
        try std.testing.expect(!std.math.isNan(val));
        try std.testing.expect(!std.math.isInf(val));
    }
}

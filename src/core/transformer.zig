// src/core/transformer.zig
// Transformer Block implementation for DistilBERT (Phase 6)

const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const TensorError = @import("tensor.zig").TensorError;
const ops = @import("ops.zig");
const OpsError = ops.OpsError;
const attention = @import("attention.zig");
const AttentionWeights = attention.AttentionWeights;
const AttentionConfig = attention.AttentionConfig;

/// Weights for a complete Transformer block
pub const TransformerBlockWeights = struct {
    // Attention
    attention: AttentionWeights,
    attn_ln_gamma: Tensor,
    attn_ln_beta: Tensor,

    // Feed Forward
    ff_linear1_weight: Tensor, // [hidden, intermediate]
    ff_linear1_bias: Tensor,
    ff_linear2_weight: Tensor, // [intermediate, hidden]
    ff_linear2_bias: Tensor,
    ff_ln_gamma: Tensor,
    ff_ln_beta: Tensor,

    const Self = @This();

    pub fn deinit(self: *Self) void {
        self.attention.deinit();
        self.attn_ln_gamma.deinit();
        self.attn_ln_beta.deinit();
        self.ff_linear1_weight.deinit();
        self.ff_linear1_bias.deinit();
        self.ff_linear2_weight.deinit();
        self.ff_linear2_bias.deinit();
        self.ff_ln_gamma.deinit();
        self.ff_ln_beta.deinit();
    }
};

/// Configuration for Transformer
pub const TransformerConfig = struct {
    hidden_dim: usize, // 768 for DistilBERT
    intermediate_dim: usize, // 3072 (4x hidden)
    num_heads: usize, // 12
    layer_norm_eps: f32, // 1e-12

    pub fn getAttentionConfig(self: TransformerConfig) AttentionConfig {
        return AttentionConfig{
            .num_heads = self.num_heads,
            .hidden_dim = self.hidden_dim,
            .head_dim = self.hidden_dim / self.num_heads,
        };
    }
};

/// DistilBERT default configuration
pub const DISTILBERT_CONFIG = TransformerConfig{
    .hidden_dim = 768,
    .intermediate_dim = 3072,
    .num_heads = 12,
    .layer_norm_eps = 1e-12,
};

/// Forward pass through a single Transformer block
/// Architecture (DistilBERT style):
///   Input
///     |
///     +---> Multi-Head Attention
///     |        + Residual connection
///     |        + LayerNorm
///     |
///     +---> FeedForward (Linear->GELU->Linear)
///              + Residual connection
///              + LayerNorm
///     |
///   Output
pub fn transformerBlock(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    weights: *const TransformerBlockWeights,
    config: TransformerConfig,
) !Tensor {
    // 1. Multi-head attention with residual
    var attn_output = try attention.multiHeadAttention(
        allocator,
        input,
        &weights.attention,
        config.getAttentionConfig(),
    );
    defer attn_output.deinit();

    // Residual connection
    try ops.addInPlace(&attn_output, input);

    // Layer norm
    var normed1 = try ops.layerNorm(
        allocator,
        &attn_output,
        &weights.attn_ln_gamma,
        &weights.attn_ln_beta,
        config.layer_norm_eps,
    );
    defer normed1.deinit();

    // 2. Feed forward with residual
    // FFN(x) = Linear(GELU(Linear(x)))
    // NOTE: Weights are PRE-TRANSPOSED at load time for optimal SIMD matmul performance.

    // First linear: hidden -> intermediate (weight pre-transposed)
    var ff_hidden = try ops.matmul(allocator, &normed1, &weights.ff_linear1_weight);
    defer ff_hidden.deinit();
    try ops.addBiasInPlace(&ff_hidden, &weights.ff_linear1_bias);

    // GELU activation
    ops.gelu(&ff_hidden);

    // Second linear: intermediate -> hidden (weight pre-transposed)
    var ff_output = try ops.matmul(allocator, &ff_hidden, &weights.ff_linear2_weight);
    defer ff_output.deinit();
    try ops.addBiasInPlace(&ff_output, &weights.ff_linear2_bias);

    // Residual connection
    try ops.addInPlace(&ff_output, &normed1);

    // Layer norm
    return ops.layerNorm(
        allocator,
        &ff_output,
        &weights.ff_ln_gamma,
        &weights.ff_ln_beta,
        config.layer_norm_eps,
    );
}

/// Forward pass through multiple Transformer blocks
pub fn transformerStack(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    blocks: []const TransformerBlockWeights,
    config: TransformerConfig,
) !Tensor {
    if (blocks.len == 0) {
        return input.clone(allocator);
    }

    // Process first block
    var hidden = try transformerBlock(allocator, input, &blocks[0], config);

    // Process remaining blocks
    for (blocks[1..]) |*block| {
        const new_hidden = try transformerBlock(allocator, &hidden, block, config);
        hidden.deinit();
        hidden = new_hidden;
    }

    return hidden;
}

// ============================================================================
// Tests
// ============================================================================

test "transformer block small" {
    const allocator = std.testing.allocator;

    // Very small config for testing: 2 heads, 4 hidden, 8 intermediate
    const config = TransformerConfig{
        .hidden_dim = 4,
        .intermediate_dim = 8,
        .num_heads = 2,
        .layer_norm_eps = 1e-5,
    };

    // Input: [2, 4] (2 tokens, 4 hidden dim)
    var input_shape = [_]usize{ 2, 4 };
    var input = try Tensor.init(allocator, &input_shape);
    defer input.deinit();
    for (input.data, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) * 0.1 + 0.1;
    }

    // Create weights
    var attn_weight_shape = [_]usize{ 4, 4 };
    var attn_bias_shape = [_]usize{4};
    var ln_shape = [_]usize{4};
    var ff1_weight_shape = [_]usize{ 8, 4 }; // [intermediate, hidden]
    var ff1_bias_shape = [_]usize{8};
    var ff2_weight_shape = [_]usize{ 4, 8 }; // [hidden, intermediate]
    var ff2_bias_shape = [_]usize{4};

    // Attention weights (identity-like)
    var q_weight = try Tensor.init(allocator, &attn_weight_shape);
    defer q_weight.deinit();
    q_weight.data[0] = 1.0;
    q_weight.data[5] = 1.0;
    q_weight.data[10] = 1.0;
    q_weight.data[15] = 1.0;

    var k_weight = try Tensor.init(allocator, &attn_weight_shape);
    defer k_weight.deinit();
    k_weight.data[0] = 1.0;
    k_weight.data[5] = 1.0;
    k_weight.data[10] = 1.0;
    k_weight.data[15] = 1.0;

    var v_weight = try Tensor.init(allocator, &attn_weight_shape);
    defer v_weight.deinit();
    v_weight.data[0] = 1.0;
    v_weight.data[5] = 1.0;
    v_weight.data[10] = 1.0;
    v_weight.data[15] = 1.0;

    var o_weight = try Tensor.init(allocator, &attn_weight_shape);
    defer o_weight.deinit();
    o_weight.data[0] = 1.0;
    o_weight.data[5] = 1.0;
    o_weight.data[10] = 1.0;
    o_weight.data[15] = 1.0;

    var q_bias = try Tensor.init(allocator, &attn_bias_shape);
    defer q_bias.deinit();
    var k_bias = try Tensor.init(allocator, &attn_bias_shape);
    defer k_bias.deinit();
    var v_bias = try Tensor.init(allocator, &attn_bias_shape);
    defer v_bias.deinit();
    var o_bias = try Tensor.init(allocator, &attn_bias_shape);
    defer o_bias.deinit();

    // Layer norm weights
    var attn_ln_gamma = try Tensor.init(allocator, &ln_shape);
    defer attn_ln_gamma.deinit();
    attn_ln_gamma.fill(1.0);

    var attn_ln_beta = try Tensor.init(allocator, &ln_shape);
    defer attn_ln_beta.deinit();

    var ff_ln_gamma = try Tensor.init(allocator, &ln_shape);
    defer ff_ln_gamma.deinit();
    ff_ln_gamma.fill(1.0);

    var ff_ln_beta = try Tensor.init(allocator, &ln_shape);
    defer ff_ln_beta.deinit();

    // Feed forward weights (small random-ish values)
    var ff1_weight = try Tensor.init(allocator, &ff1_weight_shape);
    defer ff1_weight.deinit();
    for (ff1_weight.data, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 4)) * 0.1;
    }

    var ff1_bias = try Tensor.init(allocator, &ff1_bias_shape);
    defer ff1_bias.deinit();

    var ff2_weight = try Tensor.init(allocator, &ff2_weight_shape);
    defer ff2_weight.deinit();
    for (ff2_weight.data, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 4)) * 0.1;
    }

    var ff2_bias = try Tensor.init(allocator, &ff2_bias_shape);
    defer ff2_bias.deinit();

    const weights = TransformerBlockWeights{
        .attention = AttentionWeights{
            .q_weight = q_weight,
            .k_weight = k_weight,
            .v_weight = v_weight,
            .o_weight = o_weight,
            .q_bias = q_bias,
            .k_bias = k_bias,
            .v_bias = v_bias,
            .o_bias = o_bias,
        },
        .attn_ln_gamma = attn_ln_gamma,
        .attn_ln_beta = attn_ln_beta,
        .ff_linear1_weight = ff1_weight,
        .ff_linear1_bias = ff1_bias,
        .ff_linear2_weight = ff2_weight,
        .ff_linear2_bias = ff2_bias,
        .ff_ln_gamma = ff_ln_gamma,
        .ff_ln_beta = ff_ln_beta,
    };

    var output = try transformerBlock(allocator, &input, &weights, config);
    defer output.deinit();

    // Output should have same shape as input
    try std.testing.expectEqual(@as(usize, 2), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), output.shape[1]);

    // Values should be valid (not NaN or Inf)
    for (output.data) |val| {
        try std.testing.expect(!std.math.isNan(val));
        try std.testing.expect(!std.math.isInf(val));
    }

    // Layer norm output should have mean ≈ 0 per row (with gamma=1, beta=0)
    var row_sum: f32 = 0.0;
    for (output.data[0..4]) |v| {
        row_sum += v;
    }
    const row_mean = row_sum / 4.0;
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), row_mean, 0.01);
}

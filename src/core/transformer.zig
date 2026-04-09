// src/core/transformer_generic.zig
// Generic Transformer Block implementation supporting multiple weight formats
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
const attention_generic = @import("attention.zig");

const QuantizedTensorQ8 = quant.QuantizedTensorQ8;
const QuantizedTensorQ4 = quant.QuantizedTensorQ4;
const QuantizedTensorQ8K = quant.QuantizedTensorQ8K;
const F16Tensor = quant.F16Tensor;

// Re-export from attention_generic for convenience
pub const WeightFormat = attention_generic.WeightFormat;
pub const AttentionWeights = attention_generic.AttentionWeights;
pub const AttentionConfig = attention_generic.AttentionConfig;

/// Generic Transformer block weights
/// WeightType can be: Tensor (f32), QuantizedTensorQ8, QuantizedTensorQ4, QuantizedTensorQ8K
pub fn TransformerBlockWeights(comptime WeightType: type) type {
    return struct {
        // Attention weights (using generic attention)
        attention: AttentionWeights(WeightType),

        // Layer norm params (always float32 - small, sensitive to quantization)
        attn_ln_gamma: Tensor,
        attn_ln_beta: Tensor,

        // Feed Forward weights [hidden, intermediate] and [intermediate, hidden]
        // Pre-transposed for optimal matmul performance
        ff_linear1_weight: WeightType,
        ff_linear1_bias: Tensor,
        ff_linear2_weight: WeightType,
        ff_linear2_bias: Tensor,
        ff_ln_gamma: Tensor,
        ff_ln_beta: Tensor,

        const Self = @This();

        /// Get the weight format enum for this type
        pub fn format() WeightFormat {
            return comptime getWeightFormat(WeightType);
        }

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
}

/// Convenience type aliases for common weight formats
pub const TransformerBlockWeightsF32 = TransformerBlockWeights(Tensor);
pub const TransformerBlockWeightsQ8 = TransformerBlockWeights(QuantizedTensorQ8);
pub const TransformerBlockWeightsQ4 = TransformerBlockWeights(QuantizedTensorQ4);
pub const TransformerBlockWeightsQ8K = TransformerBlockWeights(QuantizedTensorQ8K);
pub const TransformerBlockWeightsF16 = TransformerBlockWeights(F16Tensor);

/// Configuration for Transformer
pub const TransformerConfig = struct {
    hidden_dim: usize, // 768 for base, 1024 for large
    intermediate_dim: usize, // 4x hidden typically
    num_heads: usize, // 12 for base, 16 for large
    layer_norm_eps: f32, // 1e-12 for BERT, 1e-5 for RoBERTa

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

/// RoBERTa base default configuration
pub const ROBERTA_BASE_CONFIG = TransformerConfig{
    .hidden_dim = 768,
    .intermediate_dim = 3072,
    .num_heads = 12,
    .layer_norm_eps = 1e-5,
};

/// RoBERTa large default configuration
pub const ROBERTA_LARGE_CONFIG = TransformerConfig{
    .hidden_dim = 1024,
    .intermediate_dim = 4096,
    .num_heads = 16,
    .layer_norm_eps = 1e-5,
};

/// Get the weight format enum for a given weight type
fn getWeightFormat(comptime T: type) WeightFormat {
    if (T == Tensor) return .f32;
    if (T == QuantizedTensorQ8) return .q8;
    if (T == QuantizedTensorQ4) return .q4;
    if (T == QuantizedTensorQ8K) return .q8_k;
    if (T == F16Tensor) return .f16;
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
        .f16 => quant.matmulF32F16(allocator, input, weight),
    };
}

/// Fused matrix multiplication + bias: C = input @ weight + bias
/// Uses fused sgemmBias for f32, separate matmul+bias for quantized.
fn matmulWithWeightBias(
    comptime WeightType: type,
    allocator: std.mem.Allocator,
    input: *const Tensor,
    weight: *const WeightType,
    bias: *const Tensor,
) !Tensor {
    const format = comptime getWeightFormat(WeightType);

    if (format == .f32) {
        return ops.matmulBias(allocator, input, weight, bias);
    }

    // Quantized: separate matmul then bias
    var result = try matmulWithWeight(WeightType, allocator, input, weight);
    errdefer result.deinit();
    try ops.addBiasInPlace(&result, bias);
    return result;
}

/// Generic forward pass through a single Transformer block
/// Architecture (post-LN style):
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
///
/// NOTE: Weights are PRE-TRANSPOSED at load time for optimal SIMD matmul performance.
pub fn transformerBlock(
    comptime WeightType: type,
    allocator: std.mem.Allocator,
    input: *const Tensor,
    weights: *const TransformerBlockWeights(WeightType),
    config: TransformerConfig,
) !Tensor {
    // 1. Multi-head attention with generic weights (fused — no per-head copies)
    var attn_output = try attention_generic.multiHeadAttentionFused(
        WeightType,
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

    // First linear: hidden -> intermediate (weight pre-transposed)
    var ff_hidden = try matmulWithWeightBias(WeightType, allocator, &normed1, &weights.ff_linear1_weight, &weights.ff_linear1_bias);
    defer ff_hidden.deinit();

    // GELU activation
    ops.gelu(&ff_hidden);

    // Second linear: intermediate -> hidden (weight pre-transposed)
    var ff_output = try matmulWithWeightBias(WeightType, allocator, &ff_hidden, &weights.ff_linear2_weight, &weights.ff_linear2_bias);
    errdefer ff_output.deinit();

    // Residual connection
    try ops.addInPlace(&ff_output, &normed1);

    // Layer norm (in-place — ff_output IS the return value, no clone needed)
    try ops.layerNormInPlace(
        &ff_output,
        &weights.ff_ln_gamma,
        &weights.ff_ln_beta,
        config.layer_norm_eps,
    );

    return ff_output;
}

/// Generic forward pass through multiple Transformer blocks
pub fn transformerStack(
    comptime WeightType: type,
    allocator: std.mem.Allocator,
    input: *const Tensor,
    blocks: []const TransformerBlockWeights(WeightType),
    config: TransformerConfig,
) !Tensor {
    if (blocks.len == 0) {
        return input.clone(allocator);
    }

    // Process first block
    var hidden = try transformerBlock(WeightType, allocator, input, &blocks[0], config);

    // Process remaining blocks
    for (blocks[1..]) |*block| {
        const new_hidden = try transformerBlock(WeightType, allocator, &hidden, block, config);
        hidden.deinit();
        hidden = new_hidden;
    }

    return hidden;
}

// ============================================================================
// Convenience Functions (Non-Generic API)
// ============================================================================

/// Transformer block with F32 weights
pub fn transformerBlockF32(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    weights: *const TransformerBlockWeightsF32,
    config: TransformerConfig,
) !Tensor {
    return transformerBlock(Tensor, allocator, input, weights, config);
}

/// Transformer block with Q8 weights
pub fn transformerBlockQ8(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    weights: *const TransformerBlockWeightsQ8,
    config: TransformerConfig,
) !Tensor {
    return transformerBlock(QuantizedTensorQ8, allocator, input, weights, config);
}

/// Transformer block with Q4 weights
pub fn transformerBlockQ4(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    weights: *const TransformerBlockWeightsQ4,
    config: TransformerConfig,
) !Tensor {
    return transformerBlock(QuantizedTensorQ4, allocator, input, weights, config);
}

/// Transformer block with Q8_K weights
pub fn transformerBlockQ8K(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    weights: *const TransformerBlockWeightsQ8K,
    config: TransformerConfig,
) !Tensor {
    return transformerBlock(QuantizedTensorQ8K, allocator, input, weights, config);
}

/// Transformer stack with F32 weights
pub fn transformerStackF32(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    blocks: []const TransformerBlockWeightsF32,
    config: TransformerConfig,
) !Tensor {
    return transformerStack(Tensor, allocator, input, blocks, config);
}

/// Transformer stack with Q8 weights
pub fn transformerStackQ8(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    blocks: []const TransformerBlockWeightsQ8,
    config: TransformerConfig,
) !Tensor {
    return transformerStack(QuantizedTensorQ8, allocator, input, blocks, config);
}

/// Transformer stack with Q4 weights
pub fn transformerStackQ4(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    blocks: []const TransformerBlockWeightsQ4,
    config: TransformerConfig,
) !Tensor {
    return transformerStack(QuantizedTensorQ4, allocator, input, blocks, config);
}

/// Transformer stack with Q8_K weights
pub fn transformerStackQ8K(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    blocks: []const TransformerBlockWeightsQ8K,
    config: TransformerConfig,
) !Tensor {
    return transformerStack(QuantizedTensorQ8K, allocator, input, blocks, config);
}

/// Transformer block with F16 weights
pub fn transformerBlockF16(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    weights: *const TransformerBlockWeightsF16,
    config: TransformerConfig,
) !Tensor {
    return transformerBlock(F16Tensor, allocator, input, weights, config);
}

/// Transformer stack with F16 weights
pub fn transformerStackF16(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    blocks: []const TransformerBlockWeightsF16,
    config: TransformerConfig,
) !Tensor {
    return transformerStack(F16Tensor, allocator, input, blocks, config);
}

// ============================================================================
// Helper Functions for Creating Test Weights
// ============================================================================

/// Create identity-like weight tensor for testing
fn createIdentityWeight(allocator: std.mem.Allocator, shape: []const usize) !Tensor {
    var weight = try Tensor.init(allocator, shape);
    errdefer weight.deinit();

    // Set diagonal to 1.0 (identity-like for square matrices)
    const rows = shape[0];
    const cols = shape[1];
    const min_dim = @min(rows, cols);

    for (0..min_dim) |i| {
        weight.data[i * cols + i] = 1.0;
    }

    return weight;
}

/// Create small random-like weight tensor for testing
fn createSmallWeight(allocator: std.mem.Allocator, shape: []const usize) !Tensor {
    var weight = try Tensor.init(allocator, shape);
    errdefer weight.deinit();

    for (weight.data, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 4)) * 0.1;
    }

    return weight;
}

// ============================================================================
// Tests
// ============================================================================

test "generic transformer block with F32 weights" {
    const allocator = std.testing.allocator;

    // Small config: 2 heads, 4 hidden, 8 intermediate
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
    var bias_shape = [_]usize{4};
    var ff1_weight_shape = [_]usize{ 4, 8 };
    var ff1_bias_shape = [_]usize{8};
    var ff2_weight_shape = [_]usize{ 8, 4 };

    // Attention weights
    var q_weight = try createIdentityWeight(allocator, &attn_weight_shape);
    errdefer q_weight.deinit();
    var k_weight = try createIdentityWeight(allocator, &attn_weight_shape);
    errdefer k_weight.deinit();
    var v_weight = try createIdentityWeight(allocator, &attn_weight_shape);
    errdefer v_weight.deinit();
    var o_weight = try createIdentityWeight(allocator, &attn_weight_shape);
    errdefer o_weight.deinit();

    var q_bias = try Tensor.init(allocator, &bias_shape);
    errdefer q_bias.deinit();
    var k_bias = try Tensor.init(allocator, &bias_shape);
    errdefer k_bias.deinit();
    var v_bias = try Tensor.init(allocator, &bias_shape);
    errdefer v_bias.deinit();
    var o_bias = try Tensor.init(allocator, &bias_shape);
    errdefer o_bias.deinit();

    // Layer norm weights
    var attn_ln_gamma = try Tensor.init(allocator, &bias_shape);
    errdefer attn_ln_gamma.deinit();
    attn_ln_gamma.fill(1.0);
    var attn_ln_beta = try Tensor.init(allocator, &bias_shape);
    errdefer attn_ln_beta.deinit();

    var ff_ln_gamma = try Tensor.init(allocator, &bias_shape);
    errdefer ff_ln_gamma.deinit();
    ff_ln_gamma.fill(1.0);
    var ff_ln_beta = try Tensor.init(allocator, &bias_shape);
    errdefer ff_ln_beta.deinit();

    // FFN weights
    var ff1_weight = try createSmallWeight(allocator, &ff1_weight_shape);
    errdefer ff1_weight.deinit();
    var ff1_bias = try Tensor.init(allocator, &ff1_bias_shape);
    errdefer ff1_bias.deinit();
    var ff2_weight = try createSmallWeight(allocator, &ff2_weight_shape);
    errdefer ff2_weight.deinit();
    var ff2_bias = try Tensor.init(allocator, &bias_shape);
    errdefer ff2_bias.deinit();

    var weights = TransformerBlockWeightsF32{
        .attention = attention_generic.AttentionWeightsF32{
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
    defer weights.deinit();

    var output = try transformerBlock(Tensor, allocator, &input, &weights, config);
    defer output.deinit();

    // Output should have same shape as input
    try std.testing.expectEqual(@as(usize, 2), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), output.shape[1]);

    // Values should be valid
    for (output.data) |val| {
        try std.testing.expect(!std.math.isNan(val));
        try std.testing.expect(!std.math.isInf(val));
    }

    // Layer norm output should have mean near 0 per row
    var row_sum: f32 = 0.0;
    for (output.data[0..4]) |v| {
        row_sum += v;
    }
    const row_mean = row_sum / 4.0;
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), row_mean, 0.01);

    // Test format detection
    try std.testing.expectEqual(WeightFormat.f32, TransformerBlockWeightsF32.format());
}

test "generic transformer block with Q8 weights" {
    const allocator = std.testing.allocator;

    const config = TransformerConfig{
        .hidden_dim = 4,
        .intermediate_dim = 8,
        .num_heads = 2,
        .layer_norm_eps = 1e-5,
    };

    var input_shape = [_]usize{ 2, 4 };
    var input = try Tensor.init(allocator, &input_shape);
    defer input.deinit();
    for (input.data, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) * 0.1 + 0.1;
    }

    // Create float weights then quantize
    var attn_weight_shape = [_]usize{ 4, 4 };
    var bias_shape = [_]usize{4};
    var ff1_weight_shape = [_]usize{ 4, 8 };
    var ff1_bias_shape = [_]usize{8};
    var ff2_weight_shape = [_]usize{ 8, 4 };

    var attn_weight_f32 = try createIdentityWeight(allocator, &attn_weight_shape);
    defer attn_weight_f32.deinit();
    var ff1_weight_f32 = try createSmallWeight(allocator, &ff1_weight_shape);
    defer ff1_weight_f32.deinit();
    var ff2_weight_f32 = try createSmallWeight(allocator, &ff2_weight_shape);
    defer ff2_weight_f32.deinit();

    // Quantize attention weights
    var q_weight = try quant.quantizeQ8(allocator, &attn_weight_f32);
    errdefer q_weight.deinit();
    var k_weight = try quant.quantizeQ8(allocator, &attn_weight_f32);
    errdefer k_weight.deinit();
    var v_weight = try quant.quantizeQ8(allocator, &attn_weight_f32);
    errdefer v_weight.deinit();
    var o_weight = try quant.quantizeQ8(allocator, &attn_weight_f32);
    errdefer o_weight.deinit();

    var q_bias = try Tensor.init(allocator, &bias_shape);
    errdefer q_bias.deinit();
    var k_bias = try Tensor.init(allocator, &bias_shape);
    errdefer k_bias.deinit();
    var v_bias = try Tensor.init(allocator, &bias_shape);
    errdefer v_bias.deinit();
    var o_bias = try Tensor.init(allocator, &bias_shape);
    errdefer o_bias.deinit();

    var attn_ln_gamma = try Tensor.init(allocator, &bias_shape);
    errdefer attn_ln_gamma.deinit();
    attn_ln_gamma.fill(1.0);
    var attn_ln_beta = try Tensor.init(allocator, &bias_shape);
    errdefer attn_ln_beta.deinit();

    var ff_ln_gamma = try Tensor.init(allocator, &bias_shape);
    errdefer ff_ln_gamma.deinit();
    ff_ln_gamma.fill(1.0);
    var ff_ln_beta = try Tensor.init(allocator, &bias_shape);
    errdefer ff_ln_beta.deinit();

    // Quantize FFN weights
    var ff1_weight = try quant.quantizeQ8(allocator, &ff1_weight_f32);
    errdefer ff1_weight.deinit();
    var ff1_bias = try Tensor.init(allocator, &ff1_bias_shape);
    errdefer ff1_bias.deinit();
    var ff2_weight = try quant.quantizeQ8(allocator, &ff2_weight_f32);
    errdefer ff2_weight.deinit();
    var ff2_bias = try Tensor.init(allocator, &bias_shape);
    errdefer ff2_bias.deinit();

    var weights = TransformerBlockWeightsQ8{
        .attention = attention_generic.AttentionWeightsQ8{
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
    defer weights.deinit();

    var output = try transformerBlock(QuantizedTensorQ8, allocator, &input, &weights, config);
    defer output.deinit();

    try std.testing.expectEqual(@as(usize, 2), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), output.shape[1]);

    for (output.data) |val| {
        try std.testing.expect(!std.math.isNan(val));
        try std.testing.expect(!std.math.isInf(val));
    }

    try std.testing.expectEqual(WeightFormat.q8, TransformerBlockWeightsQ8.format());
}

test "generic transformer block with Q4 weights" {
    const allocator = std.testing.allocator;

    const config = TransformerConfig{
        .hidden_dim = 4,
        .intermediate_dim = 8,
        .num_heads = 2,
        .layer_norm_eps = 1e-5,
    };

    var input_shape = [_]usize{ 2, 4 };
    var input = try Tensor.init(allocator, &input_shape);
    defer input.deinit();
    for (input.data, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) * 0.1 + 0.1;
    }

    var attn_weight_shape = [_]usize{ 4, 4 };
    var bias_shape = [_]usize{4};
    var ff1_weight_shape = [_]usize{ 4, 8 };
    var ff1_bias_shape = [_]usize{8};
    var ff2_weight_shape = [_]usize{ 8, 4 };

    var attn_weight_f32 = try createIdentityWeight(allocator, &attn_weight_shape);
    defer attn_weight_f32.deinit();
    var ff1_weight_f32 = try createSmallWeight(allocator, &ff1_weight_shape);
    defer ff1_weight_f32.deinit();
    var ff2_weight_f32 = try createSmallWeight(allocator, &ff2_weight_shape);
    defer ff2_weight_f32.deinit();

    // Quantize to Q4
    var q_weight = try quant.quantizeQ4(allocator, &attn_weight_f32);
    errdefer q_weight.deinit();
    var k_weight = try quant.quantizeQ4(allocator, &attn_weight_f32);
    errdefer k_weight.deinit();
    var v_weight = try quant.quantizeQ4(allocator, &attn_weight_f32);
    errdefer v_weight.deinit();
    var o_weight = try quant.quantizeQ4(allocator, &attn_weight_f32);
    errdefer o_weight.deinit();

    var q_bias = try Tensor.init(allocator, &bias_shape);
    errdefer q_bias.deinit();
    var k_bias = try Tensor.init(allocator, &bias_shape);
    errdefer k_bias.deinit();
    var v_bias = try Tensor.init(allocator, &bias_shape);
    errdefer v_bias.deinit();
    var o_bias = try Tensor.init(allocator, &bias_shape);
    errdefer o_bias.deinit();

    var attn_ln_gamma = try Tensor.init(allocator, &bias_shape);
    errdefer attn_ln_gamma.deinit();
    attn_ln_gamma.fill(1.0);
    var attn_ln_beta = try Tensor.init(allocator, &bias_shape);
    errdefer attn_ln_beta.deinit();

    var ff_ln_gamma = try Tensor.init(allocator, &bias_shape);
    errdefer ff_ln_gamma.deinit();
    ff_ln_gamma.fill(1.0);
    var ff_ln_beta = try Tensor.init(allocator, &bias_shape);
    errdefer ff_ln_beta.deinit();

    var ff1_weight = try quant.quantizeQ4(allocator, &ff1_weight_f32);
    errdefer ff1_weight.deinit();
    var ff1_bias = try Tensor.init(allocator, &ff1_bias_shape);
    errdefer ff1_bias.deinit();
    var ff2_weight = try quant.quantizeQ4(allocator, &ff2_weight_f32);
    errdefer ff2_weight.deinit();
    var ff2_bias = try Tensor.init(allocator, &bias_shape);
    errdefer ff2_bias.deinit();

    var weights = TransformerBlockWeightsQ4{
        .attention = attention_generic.AttentionWeightsQ4{
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
    defer weights.deinit();

    var output = try transformerBlock(QuantizedTensorQ4, allocator, &input, &weights, config);
    defer output.deinit();

    try std.testing.expectEqual(@as(usize, 2), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), output.shape[1]);

    for (output.data) |val| {
        try std.testing.expect(!std.math.isNan(val));
        try std.testing.expect(!std.math.isInf(val));
    }

    try std.testing.expectEqual(WeightFormat.q4, TransformerBlockWeightsQ4.format());
}

test "generic transformer block with Q8_K weights" {
    const allocator = std.testing.allocator;

    const config = TransformerConfig{
        .hidden_dim = 4,
        .intermediate_dim = 8,
        .num_heads = 2,
        .layer_norm_eps = 1e-5,
    };

    var input_shape = [_]usize{ 2, 4 };
    var input = try Tensor.init(allocator, &input_shape);
    defer input.deinit();
    for (input.data, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) * 0.1 + 0.1;
    }

    var attn_weight_shape = [_]usize{ 4, 4 };
    var bias_shape = [_]usize{4};
    var ff1_weight_shape = [_]usize{ 4, 8 };
    var ff1_bias_shape = [_]usize{8};
    var ff2_weight_shape = [_]usize{ 8, 4 };

    var attn_weight_f32 = try createIdentityWeight(allocator, &attn_weight_shape);
    defer attn_weight_f32.deinit();
    var ff1_weight_f32 = try createSmallWeight(allocator, &ff1_weight_shape);
    defer ff1_weight_f32.deinit();
    var ff2_weight_f32 = try createSmallWeight(allocator, &ff2_weight_shape);
    defer ff2_weight_f32.deinit();

    // Quantize to Q8_K
    var q_weight = try quant.quantizeQ8K(allocator, &attn_weight_f32);
    errdefer q_weight.deinit();
    var k_weight = try quant.quantizeQ8K(allocator, &attn_weight_f32);
    errdefer k_weight.deinit();
    var v_weight = try quant.quantizeQ8K(allocator, &attn_weight_f32);
    errdefer v_weight.deinit();
    var o_weight = try quant.quantizeQ8K(allocator, &attn_weight_f32);
    errdefer o_weight.deinit();

    var q_bias = try Tensor.init(allocator, &bias_shape);
    errdefer q_bias.deinit();
    var k_bias = try Tensor.init(allocator, &bias_shape);
    errdefer k_bias.deinit();
    var v_bias = try Tensor.init(allocator, &bias_shape);
    errdefer v_bias.deinit();
    var o_bias = try Tensor.init(allocator, &bias_shape);
    errdefer o_bias.deinit();

    var attn_ln_gamma = try Tensor.init(allocator, &bias_shape);
    errdefer attn_ln_gamma.deinit();
    attn_ln_gamma.fill(1.0);
    var attn_ln_beta = try Tensor.init(allocator, &bias_shape);
    errdefer attn_ln_beta.deinit();

    var ff_ln_gamma = try Tensor.init(allocator, &bias_shape);
    errdefer ff_ln_gamma.deinit();
    ff_ln_gamma.fill(1.0);
    var ff_ln_beta = try Tensor.init(allocator, &bias_shape);
    errdefer ff_ln_beta.deinit();

    var ff1_weight = try quant.quantizeQ8K(allocator, &ff1_weight_f32);
    errdefer ff1_weight.deinit();
    var ff1_bias = try Tensor.init(allocator, &ff1_bias_shape);
    errdefer ff1_bias.deinit();
    var ff2_weight = try quant.quantizeQ8K(allocator, &ff2_weight_f32);
    errdefer ff2_weight.deinit();
    var ff2_bias = try Tensor.init(allocator, &bias_shape);
    errdefer ff2_bias.deinit();

    var weights = TransformerBlockWeightsQ8K{
        .attention = attention_generic.AttentionWeightsQ8K{
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
    defer weights.deinit();

    var output = try transformerBlock(QuantizedTensorQ8K, allocator, &input, &weights, config);
    defer output.deinit();

    try std.testing.expectEqual(@as(usize, 2), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), output.shape[1]);

    for (output.data) |val| {
        try std.testing.expect(!std.math.isNan(val));
        try std.testing.expect(!std.math.isInf(val));
    }

    try std.testing.expectEqual(WeightFormat.q8_k, TransformerBlockWeightsQ8K.format());
}

test "convenience functions match generic API" {
    const allocator = std.testing.allocator;

    const config = TransformerConfig{
        .hidden_dim = 4,
        .intermediate_dim = 8,
        .num_heads = 2,
        .layer_norm_eps = 1e-5,
    };

    var input_shape = [_]usize{ 2, 4 };
    var input = try Tensor.init(allocator, &input_shape);
    defer input.deinit();
    for (input.data, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) * 0.1 + 0.1;
    }

    var attn_weight_shape = [_]usize{ 4, 4 };
    var bias_shape = [_]usize{4};
    var ff1_weight_shape = [_]usize{ 4, 8 };
    var ff1_bias_shape = [_]usize{8};
    var ff2_weight_shape = [_]usize{ 8, 4 };

    var q_weight = try createIdentityWeight(allocator, &attn_weight_shape);
    errdefer q_weight.deinit();
    var k_weight = try createIdentityWeight(allocator, &attn_weight_shape);
    errdefer k_weight.deinit();
    var v_weight = try createIdentityWeight(allocator, &attn_weight_shape);
    errdefer v_weight.deinit();
    var o_weight = try createIdentityWeight(allocator, &attn_weight_shape);
    errdefer o_weight.deinit();

    var q_bias = try Tensor.init(allocator, &bias_shape);
    errdefer q_bias.deinit();
    var k_bias = try Tensor.init(allocator, &bias_shape);
    errdefer k_bias.deinit();
    var v_bias = try Tensor.init(allocator, &bias_shape);
    errdefer v_bias.deinit();
    var o_bias = try Tensor.init(allocator, &bias_shape);
    errdefer o_bias.deinit();

    var attn_ln_gamma = try Tensor.init(allocator, &bias_shape);
    errdefer attn_ln_gamma.deinit();
    attn_ln_gamma.fill(1.0);
    var attn_ln_beta = try Tensor.init(allocator, &bias_shape);
    errdefer attn_ln_beta.deinit();

    var ff_ln_gamma = try Tensor.init(allocator, &bias_shape);
    errdefer ff_ln_gamma.deinit();
    ff_ln_gamma.fill(1.0);
    var ff_ln_beta = try Tensor.init(allocator, &bias_shape);
    errdefer ff_ln_beta.deinit();

    var ff1_weight = try createSmallWeight(allocator, &ff1_weight_shape);
    errdefer ff1_weight.deinit();
    var ff1_bias = try Tensor.init(allocator, &ff1_bias_shape);
    errdefer ff1_bias.deinit();
    var ff2_weight = try createSmallWeight(allocator, &ff2_weight_shape);
    errdefer ff2_weight.deinit();
    var ff2_bias = try Tensor.init(allocator, &bias_shape);
    errdefer ff2_bias.deinit();

    var weights = TransformerBlockWeightsF32{
        .attention = attention_generic.AttentionWeightsF32{
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
    defer weights.deinit();

    // Test generic API
    var output_generic = try transformerBlock(Tensor, allocator, &input, &weights, config);
    defer output_generic.deinit();

    // Test convenience function
    var output_convenience = try transformerBlockF32(allocator, &input, &weights, config);
    defer output_convenience.deinit();

    // Results should be identical
    for (output_generic.data, output_convenience.data) |gen, conv| {
        try std.testing.expectApproxEqAbs(gen, conv, 0.0001);
    }
}

test "transformer stack with multiple blocks" {
    const allocator = std.testing.allocator;

    const config = TransformerConfig{
        .hidden_dim = 4,
        .intermediate_dim = 8,
        .num_heads = 2,
        .layer_norm_eps = 1e-5,
    };

    var input_shape = [_]usize{ 2, 4 };
    var input = try Tensor.init(allocator, &input_shape);
    defer input.deinit();
    for (input.data, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) * 0.1 + 0.1;
    }

    // Create weights for 2 blocks
    var attn_weight_shape = [_]usize{ 4, 4 };
    var bias_shape = [_]usize{4};
    var ff1_weight_shape = [_]usize{ 4, 8 };
    var ff1_bias_shape = [_]usize{8};
    var ff2_weight_shape = [_]usize{ 8, 4 };

    // Block 1
    var q_weight1 = try createIdentityWeight(allocator, &attn_weight_shape);
    errdefer q_weight1.deinit();
    var k_weight1 = try createIdentityWeight(allocator, &attn_weight_shape);
    errdefer k_weight1.deinit();
    var v_weight1 = try createIdentityWeight(allocator, &attn_weight_shape);
    errdefer v_weight1.deinit();
    var o_weight1 = try createIdentityWeight(allocator, &attn_weight_shape);
    errdefer o_weight1.deinit();
    var q_bias1 = try Tensor.init(allocator, &bias_shape);
    errdefer q_bias1.deinit();
    var k_bias1 = try Tensor.init(allocator, &bias_shape);
    errdefer k_bias1.deinit();
    var v_bias1 = try Tensor.init(allocator, &bias_shape);
    errdefer v_bias1.deinit();
    var o_bias1 = try Tensor.init(allocator, &bias_shape);
    errdefer o_bias1.deinit();
    var attn_ln_gamma1 = try Tensor.init(allocator, &bias_shape);
    errdefer attn_ln_gamma1.deinit();
    attn_ln_gamma1.fill(1.0);
    var attn_ln_beta1 = try Tensor.init(allocator, &bias_shape);
    errdefer attn_ln_beta1.deinit();
    var ff_ln_gamma1 = try Tensor.init(allocator, &bias_shape);
    errdefer ff_ln_gamma1.deinit();
    ff_ln_gamma1.fill(1.0);
    var ff_ln_beta1 = try Tensor.init(allocator, &bias_shape);
    errdefer ff_ln_beta1.deinit();
    var ff1_weight1 = try createSmallWeight(allocator, &ff1_weight_shape);
    errdefer ff1_weight1.deinit();
    var ff1_bias1 = try Tensor.init(allocator, &ff1_bias_shape);
    errdefer ff1_bias1.deinit();
    var ff2_weight1 = try createSmallWeight(allocator, &ff2_weight_shape);
    errdefer ff2_weight1.deinit();
    var ff2_bias1 = try Tensor.init(allocator, &bias_shape);
    errdefer ff2_bias1.deinit();

    // Block 2
    var q_weight2 = try createIdentityWeight(allocator, &attn_weight_shape);
    errdefer q_weight2.deinit();
    var k_weight2 = try createIdentityWeight(allocator, &attn_weight_shape);
    errdefer k_weight2.deinit();
    var v_weight2 = try createIdentityWeight(allocator, &attn_weight_shape);
    errdefer v_weight2.deinit();
    var o_weight2 = try createIdentityWeight(allocator, &attn_weight_shape);
    errdefer o_weight2.deinit();
    var q_bias2 = try Tensor.init(allocator, &bias_shape);
    errdefer q_bias2.deinit();
    var k_bias2 = try Tensor.init(allocator, &bias_shape);
    errdefer k_bias2.deinit();
    var v_bias2 = try Tensor.init(allocator, &bias_shape);
    errdefer v_bias2.deinit();
    var o_bias2 = try Tensor.init(allocator, &bias_shape);
    errdefer o_bias2.deinit();
    var attn_ln_gamma2 = try Tensor.init(allocator, &bias_shape);
    errdefer attn_ln_gamma2.deinit();
    attn_ln_gamma2.fill(1.0);
    var attn_ln_beta2 = try Tensor.init(allocator, &bias_shape);
    errdefer attn_ln_beta2.deinit();
    var ff_ln_gamma2 = try Tensor.init(allocator, &bias_shape);
    errdefer ff_ln_gamma2.deinit();
    ff_ln_gamma2.fill(1.0);
    var ff_ln_beta2 = try Tensor.init(allocator, &bias_shape);
    errdefer ff_ln_beta2.deinit();
    var ff1_weight2 = try createSmallWeight(allocator, &ff1_weight_shape);
    errdefer ff1_weight2.deinit();
    var ff1_bias2 = try Tensor.init(allocator, &ff1_bias_shape);
    errdefer ff1_bias2.deinit();
    var ff2_weight2 = try createSmallWeight(allocator, &ff2_weight_shape);
    errdefer ff2_weight2.deinit();
    var ff2_bias2 = try Tensor.init(allocator, &bias_shape);
    errdefer ff2_bias2.deinit();

    var blocks = [_]TransformerBlockWeightsF32{
        .{
            .attention = attention_generic.AttentionWeightsF32{
                .q_weight = q_weight1,
                .k_weight = k_weight1,
                .v_weight = v_weight1,
                .o_weight = o_weight1,
                .q_bias = q_bias1,
                .k_bias = k_bias1,
                .v_bias = v_bias1,
                .o_bias = o_bias1,
            },
            .attn_ln_gamma = attn_ln_gamma1,
            .attn_ln_beta = attn_ln_beta1,
            .ff_linear1_weight = ff1_weight1,
            .ff_linear1_bias = ff1_bias1,
            .ff_linear2_weight = ff2_weight1,
            .ff_linear2_bias = ff2_bias1,
            .ff_ln_gamma = ff_ln_gamma1,
            .ff_ln_beta = ff_ln_beta1,
        },
        .{
            .attention = attention_generic.AttentionWeightsF32{
                .q_weight = q_weight2,
                .k_weight = k_weight2,
                .v_weight = v_weight2,
                .o_weight = o_weight2,
                .q_bias = q_bias2,
                .k_bias = k_bias2,
                .v_bias = v_bias2,
                .o_bias = o_bias2,
            },
            .attn_ln_gamma = attn_ln_gamma2,
            .attn_ln_beta = attn_ln_beta2,
            .ff_linear1_weight = ff1_weight2,
            .ff_linear1_bias = ff1_bias2,
            .ff_linear2_weight = ff2_weight2,
            .ff_linear2_bias = ff2_bias2,
            .ff_ln_gamma = ff_ln_gamma2,
            .ff_ln_beta = ff_ln_beta2,
        },
    };
    defer {
        for (&blocks) |*block| {
            block.deinit();
        }
    }

    var output = try transformerStack(Tensor, allocator, &input, &blocks, config);
    defer output.deinit();

    // Output should have same shape as input
    try std.testing.expectEqual(@as(usize, 2), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), output.shape[1]);

    // Values should be valid
    for (output.data) |val| {
        try std.testing.expect(!std.math.isNan(val));
        try std.testing.expect(!std.math.isInf(val));
    }
}

test "format detection" {
    try std.testing.expectEqual(WeightFormat.f32, getWeightFormat(Tensor));
    try std.testing.expectEqual(WeightFormat.q8, getWeightFormat(QuantizedTensorQ8));
    try std.testing.expectEqual(WeightFormat.q4, getWeightFormat(QuantizedTensorQ4));
    try std.testing.expectEqual(WeightFormat.q8_k, getWeightFormat(QuantizedTensorQ8K));
}

test "predefined configs" {
    // DistilBERT config
    try std.testing.expectEqual(@as(usize, 768), DISTILBERT_CONFIG.hidden_dim);
    try std.testing.expectEqual(@as(usize, 3072), DISTILBERT_CONFIG.intermediate_dim);
    try std.testing.expectEqual(@as(usize, 12), DISTILBERT_CONFIG.num_heads);
    try std.testing.expectEqual(@as(usize, 64), DISTILBERT_CONFIG.getAttentionConfig().head_dim);

    // RoBERTa base config
    try std.testing.expectEqual(@as(usize, 768), ROBERTA_BASE_CONFIG.hidden_dim);
    try std.testing.expectEqual(@as(usize, 64), ROBERTA_BASE_CONFIG.getAttentionConfig().head_dim);

    // RoBERTa large config
    try std.testing.expectEqual(@as(usize, 1024), ROBERTA_LARGE_CONFIG.hidden_dim);
    try std.testing.expectEqual(@as(usize, 16), ROBERTA_LARGE_CONFIG.num_heads);
    try std.testing.expectEqual(@as(usize, 64), ROBERTA_LARGE_CONFIG.getAttentionConfig().head_dim);
}

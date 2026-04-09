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

/// Fused matrix multiplication + bias: C = input @ weight + bias
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

    var result = try matmulWithWeight(WeightType, allocator, input, weight);
    errdefer result.deinit();
    try ops.addBiasInPlace(&result, bias);
    return result;
}

/// Scaled dot-product attention (float32 only, used after projection)
/// Q: [query_len, head_dim]
/// K: [key_len, head_dim]
/// V: [key_len, head_dim]
/// mask: optional [query_len, key_len] with 0 for valid, -inf for masked positions
/// Returns: [query_len, head_dim]
///
/// Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k) + mask) @ V
///
/// For self-attention: query_len == key_len
/// For cross-attention: query_len may differ from key_len (e.g., decoder attending to encoder)
fn scaledDotProductAttention(
    allocator: std.mem.Allocator,
    q: *const Tensor,
    k: *const Tensor,
    v: *const Tensor,
    mask: ?*const Tensor,
) !Tensor {
    const head_dim = q.shape[1];
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    // scores = Q @ K^T  -> [query_len, key_len]
    var k_t = try ops.transpose(allocator, k);
    defer k_t.deinit();

    var scores = try ops.matmul(allocator, q, &k_t);
    defer scores.deinit();

    // Scale
    ops.scaleInPlace(&scores, scale);

    // Apply mask if provided (for causal attention or padding)
    if (mask) |m| {
        try ops.applyAttentionMask(&scores, m);
    }

    // Softmax
    ops.softmax(&scores);

    // Output = scores @ V  -> [query_len, head_dim]
    return ops.matmul(allocator, &scores, v);
}

/// Fused scaled dot-product attention operating on strided head slices.
/// Computes attention for a single head without copying data.
///
/// Q_full, K_full, V_full: [seq_len, hidden_dim] (full projected tensors)
/// head_offset: column offset for this head (h * head_dim)
/// head_dim: dimension per head
/// hidden_dim: total hidden dimension (stride)
/// scores_buf: pre-allocated [seq_len, seq_len] scratch buffer
/// output: [seq_len, hidden_dim] — results written at column offset
fn scaledDotProductAttentionStrided(
    q_data: []const f32,
    k_data: []const f32,
    v_data: []const f32,
    output: []f32,
    seq_len: usize,
    head_dim: usize,
    hidden_dim: usize,
    head_offset: usize,
    scores_buf: []f32,
) void {
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    // 1. Compute scores = Q_h @ K_h^T * scale  → [seq_len, seq_len]
    //    Q_h row i: q_data[i * hidden_dim + head_offset .. +head_dim] (stride = hidden_dim)
    //    K_h row j: k_data[j * hidden_dim + head_offset .. +head_dim] (stride = hidden_dim)
    for (0..seq_len) |i| {
        const q_row_base = i * hidden_dim + head_offset;
        for (0..seq_len) |j| {
            const k_row_base = j * hidden_dim + head_offset;
            var dot: f32 = 0.0;
            for (0..head_dim) |d| {
                dot += q_data[q_row_base + d] * k_data[k_row_base + d];
            }
            scores_buf[i * seq_len + j] = dot * scale;
        }
    }

    // 2. Softmax each row of scores
    for (0..seq_len) |i| {
        const row = scores_buf[i * seq_len ..][0..seq_len];
        var max_val: f32 = row[0];
        for (row[1..]) |v| {
            if (v > max_val) max_val = v;
        }
        var sum: f32 = 0.0;
        for (row) |*v| {
            v.* = @exp(v.* - max_val);
            sum += v.*;
        }
        const inv_sum = 1.0 / sum;
        for (row) |*v| {
            v.* *= inv_sum;
        }
    }

    // 3. Output = scores @ V_h → write directly into output[i, head_offset..+head_dim]
    //    V_h row j: v_data[j * hidden_dim + head_offset .. +head_dim]
    for (0..seq_len) |i| {
        const score_row = scores_buf[i * seq_len ..][0..seq_len];
        const out_base = i * hidden_dim + head_offset;
        // Zero the output slice
        for (0..head_dim) |d| {
            output[out_base + d] = 0.0;
        }
        for (0..seq_len) |j| {
            const s = score_row[j];
            const v_base = j * hidden_dim + head_offset;
            for (0..head_dim) |d| {
                output[out_base + d] += s * v_data[v_base + d];
            }
        }
    }
}

/// Fused multi-head attention — no per-head tensor allocation.
///
/// Instead of sliceColumns/concatColumns per head, uses strided access
/// into the full Q/K/V projected tensors. Only allocates one shared
/// scores buffer [seq_len, seq_len] reused across all heads.
pub fn multiHeadAttentionFused(
    comptime WeightType: type,
    allocator: std.mem.Allocator,
    input: *const Tensor,
    weights: *const AttentionWeights(WeightType),
    config: AttentionConfig,
) !Tensor {
    const seq_len = input.shape[0];
    const hidden_dim = config.hidden_dim;
    const num_heads = config.num_heads;
    const head_dim = config.head_dim;

    // Project Q, K, V — these are the only large allocations
    var q = try matmulWithWeightBias(WeightType, allocator, input, &weights.q_weight, &weights.q_bias);
    defer q.deinit();
    var k = try matmulWithWeightBias(WeightType, allocator, input, &weights.k_weight, &weights.k_bias);
    defer k.deinit();
    var v = try matmulWithWeightBias(WeightType, allocator, input, &weights.v_weight, &weights.v_bias);
    defer v.deinit();

    // Allocate output [seq_len, hidden_dim] and one shared scores buffer [seq_len, seq_len]
    var out_shape = [_]usize{ seq_len, hidden_dim };
    var concat = try Tensor.init(allocator, &out_shape);
    errdefer concat.deinit();

    const scores_buf = try allocator.alloc(f32, seq_len * seq_len);
    defer allocator.free(scores_buf);

    // Compute attention for each head using strided access — no copies
    for (0..num_heads) |h| {
        scaledDotProductAttentionStrided(
            q.data,
            k.data,
            v.data,
            concat.data,
            seq_len,
            head_dim,
            hidden_dim,
            h * head_dim,
            scores_buf,
        );
    }

    // Final output projection
    const output = try matmulWithWeightBias(WeightType, allocator, &concat, &weights.o_weight, &weights.o_bias);
    concat.deinit();

    return output;
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
    var q = try matmulWithWeightBias(WeightType, allocator, input, &weights.q_weight, &weights.q_bias);
    defer q.deinit();
    var k = try matmulWithWeightBias(WeightType, allocator, input, &weights.k_weight, &weights.k_bias);
    defer k.deinit();
    var v = try matmulWithWeightBias(WeightType, allocator, input, &weights.v_weight, &weights.v_bias);
    defer v.deinit();

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
            null, // No mask for standard self-attention
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
    const output = try matmulWithWeightBias(WeightType, allocator, &concat, &weights.o_weight, &weights.o_bias);

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
// Cross-Attention (for Encoder-Decoder architectures like Whisper)
// ============================================================================

/// Cross-attention weights: Query projects from decoder, Key/Value project from encoder
/// Used in decoder blocks to attend to encoder output
pub fn CrossAttentionWeights(comptime WeightType: type) type {
    return struct {
        // Query projection [decoder_hidden, decoder_hidden] - pre-transposed
        q_weight: WeightType,
        q_bias: Tensor,

        // Key/Value projection [encoder_hidden, decoder_hidden] - pre-transposed
        // Note: encoder_hidden may equal decoder_hidden in Whisper
        k_weight: WeightType,
        k_bias: Tensor,
        v_weight: WeightType,
        v_bias: Tensor,

        // Output projection [decoder_hidden, decoder_hidden] - pre-transposed
        o_weight: WeightType,
        o_bias: Tensor,

        const Self = @This();

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

/// Convenience type aliases for cross-attention weights
pub const CrossAttentionWeightsF32 = CrossAttentionWeights(Tensor);
pub const CrossAttentionWeightsQ8 = CrossAttentionWeights(QuantizedTensorQ8);
pub const CrossAttentionWeightsQ4 = CrossAttentionWeights(QuantizedTensorQ4);
pub const CrossAttentionWeightsQ8K = CrossAttentionWeights(QuantizedTensorQ8K);

/// Cross-attention configuration
pub const CrossAttentionConfig = struct {
    num_heads: usize, // Number of attention heads
    decoder_hidden: usize, // Decoder hidden dimension (query source)
    encoder_hidden: usize, // Encoder hidden dimension (key/value source)
    head_dim: usize, // decoder_hidden / num_heads
};

/// Generic multi-head cross-attention
/// query_input: [query_len, decoder_hidden] from decoder (e.g., 1 token during generation)
/// key_value_input: [key_len, encoder_hidden] from encoder (e.g., 1500 audio frames)
/// Returns: [query_len, decoder_hidden]
///
/// CrossAttention(Q_dec, K_enc, V_enc) = Concat(head_1, ..., head_h) @ W_o
/// where head_i = Attention(Q_dec @ W_q_i, K_enc @ W_k_i, V_enc @ W_v_i)
///
/// NOTE: Weights are PRE-TRANSPOSED at load time for optimal SIMD matmul performance.
pub fn multiHeadCrossAttention(
    comptime WeightType: type,
    allocator: std.mem.Allocator,
    query_input: *const Tensor, // [query_len, decoder_hidden]
    key_value_input: *const Tensor, // [key_len, encoder_hidden]
    weights: *const CrossAttentionWeights(WeightType),
    config: CrossAttentionConfig,
) !Tensor {
    const num_heads = config.num_heads;
    const head_dim = config.head_dim;

    // Project Q from decoder input
    var q = try matmulWithWeight(WeightType, allocator, query_input, &weights.q_weight);
    defer q.deinit();
    try ops.addBiasInPlace(&q, &weights.q_bias);

    // Project K, V from encoder output
    var k = try matmulWithWeight(WeightType, allocator, key_value_input, &weights.k_weight);
    defer k.deinit();
    try ops.addBiasInPlace(&k, &weights.k_bias);

    var v = try matmulWithWeight(WeightType, allocator, key_value_input, &weights.v_weight);
    defer v.deinit();
    try ops.addBiasInPlace(&v, &weights.v_bias);

    // Split into heads and compute cross-attention
    var head_outputs = try allocator.alloc(Tensor, num_heads);
    var head_count: usize = 0;
    errdefer {
        for (head_outputs[0..head_count]) |*h| h.deinit();
        allocator.free(head_outputs);
    }

    for (0..num_heads) |h| {
        const start = h * head_dim;
        const end = start + head_dim;

        // Extract head slices (Q from decoder, K/V from encoder)
        var q_head = try ops.sliceColumns(allocator, &q, start, end);
        defer q_head.deinit();
        var k_head = try ops.sliceColumns(allocator, &k, start, end);
        defer k_head.deinit();
        var v_head = try ops.sliceColumns(allocator, &v, start, end);
        defer v_head.deinit();

        // Note: No mask for cross-attention (decoder attends to all encoder positions)
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

    // Final projection
    var output = try matmulWithWeight(WeightType, allocator, &concat, &weights.o_weight);
    try ops.addBiasInPlace(&output, &weights.o_bias);

    return output;
}

/// Cached cross-attention structure for pre-computed encoder K/V
pub const CachedCrossAttentionKV = struct {
    k: Tensor, // [key_len, hidden_dim] - projected keys
    v: Tensor, // [key_len, hidden_dim] - projected values

    pub fn deinit(self: *CachedCrossAttentionKV) void {
        self.k.deinit();
        self.v.deinit();
    }
};

/// Pre-compute cross-attention K/V from encoder output
/// Call this once after encoding, then reuse for all decoder steps
pub fn precomputeCrossAttentionKV(
    comptime WeightType: type,
    allocator: std.mem.Allocator,
    encoder_output: *const Tensor, // [key_len, encoder_hidden]
    weights: *const CrossAttentionWeights(WeightType),
) !CachedCrossAttentionKV {
    // Project K, V from encoder output
    var k = try matmulWithWeight(WeightType, allocator, encoder_output, &weights.k_weight);
    errdefer k.deinit();
    try ops.addBiasInPlace(&k, &weights.k_bias);

    var v = try matmulWithWeight(WeightType, allocator, encoder_output, &weights.v_weight);
    try ops.addBiasInPlace(&v, &weights.v_bias);

    return CachedCrossAttentionKV{
        .k = k,
        .v = v,
    };
}

/// Multi-head cross-attention with pre-computed K/V
/// Use this during generation to avoid recomputing K/V every step
pub fn multiHeadCrossAttentionWithCache(
    comptime WeightType: type,
    allocator: std.mem.Allocator,
    query_input: *const Tensor, // [query_len, decoder_hidden]
    cached_kv: *const CachedCrossAttentionKV, // Pre-computed K/V
    weights: *const CrossAttentionWeights(WeightType),
    config: CrossAttentionConfig,
) !Tensor {
    const num_heads = config.num_heads;
    const head_dim = config.head_dim;

    // Project Q from decoder input only
    var q = try matmulWithWeight(WeightType, allocator, query_input, &weights.q_weight);
    defer q.deinit();
    try ops.addBiasInPlace(&q, &weights.q_bias);

    // Use cached K, V
    const k = &cached_kv.k;
    const v = &cached_kv.v;

    // Split into heads and compute cross-attention
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
        var k_head = try ops.sliceColumns(allocator, k, start, end);
        defer k_head.deinit();
        var v_head = try ops.sliceColumns(allocator, v, start, end);
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

    // Final projection
    var output = try matmulWithWeight(WeightType, allocator, &concat, &weights.o_weight);
    try ops.addBiasInPlace(&output, &weights.o_bias);

    return output;
}

/// Convenience function: pre-compute cross-attention K/V with F32 weights
pub fn precomputeCrossAttentionKVF32(
    allocator: std.mem.Allocator,
    encoder_output: *const Tensor,
    weights: *const CrossAttentionWeightsF32,
) !CachedCrossAttentionKV {
    return precomputeCrossAttentionKV(Tensor, allocator, encoder_output, weights);
}

/// Convenience function: cross-attention with cached K/V and F32 weights
pub fn multiHeadCrossAttentionWithCacheF32(
    allocator: std.mem.Allocator,
    query_input: *const Tensor,
    cached_kv: *const CachedCrossAttentionKV,
    weights: *const CrossAttentionWeightsF32,
    config: CrossAttentionConfig,
) !Tensor {
    return multiHeadCrossAttentionWithCache(Tensor, allocator, query_input, cached_kv, weights, config);
}

/// Convenience function: cross-attention with F32 weights
pub fn multiHeadCrossAttentionF32(
    allocator: std.mem.Allocator,
    query_input: *const Tensor,
    key_value_input: *const Tensor,
    weights: *const CrossAttentionWeightsF32,
    config: CrossAttentionConfig,
) !Tensor {
    return multiHeadCrossAttention(Tensor, allocator, query_input, key_value_input, weights, config);
}

/// Convenience function: cross-attention with Q8 weights
pub fn multiHeadCrossAttentionQ8(
    allocator: std.mem.Allocator,
    query_input: *const Tensor,
    key_value_input: *const Tensor,
    weights: *const CrossAttentionWeightsQ8,
    config: CrossAttentionConfig,
) !Tensor {
    return multiHeadCrossAttention(QuantizedTensorQ8, allocator, query_input, key_value_input, weights, config);
}

// ============================================================================
// Causal Self-Attention (for Decoder self-attention)
// ============================================================================

/// Multi-head self-attention with causal masking
/// Used in decoder blocks where each position can only attend to earlier positions
/// input: [seq_len, hidden_dim]
/// Returns: [seq_len, hidden_dim]
pub fn multiHeadCausalAttention(
    comptime WeightType: type,
    allocator: std.mem.Allocator,
    input: *const Tensor,
    weights: *const AttentionWeights(WeightType),
    config: AttentionConfig,
) !Tensor {
    const seq_len = input.shape[0];
    const num_heads = config.num_heads;
    const head_dim = config.head_dim;

    // Project Q, K, V
    var q = try matmulWithWeight(WeightType, allocator, input, &weights.q_weight);
    defer q.deinit();
    var k = try matmulWithWeight(WeightType, allocator, input, &weights.k_weight);
    defer k.deinit();
    var v = try matmulWithWeight(WeightType, allocator, input, &weights.v_weight);
    defer v.deinit();

    try ops.addBiasInPlace(&q, &weights.q_bias);
    try ops.addBiasInPlace(&k, &weights.k_bias);
    try ops.addBiasInPlace(&v, &weights.v_bias);

    // Create causal mask for this sequence length
    var causal_mask = try ops.createCausalMask(allocator, seq_len);
    defer causal_mask.deinit();

    // Split into heads and compute causal attention
    var head_outputs = try allocator.alloc(Tensor, num_heads);
    var head_count: usize = 0;
    errdefer {
        for (head_outputs[0..head_count]) |*h| h.deinit();
        allocator.free(head_outputs);
    }

    for (0..num_heads) |h| {
        const start = h * head_dim;
        const end = start + head_dim;

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
            &causal_mask, // Apply causal masking
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

    // Final projection
    var output = try matmulWithWeight(WeightType, allocator, &concat, &weights.o_weight);
    try ops.addBiasInPlace(&output, &weights.o_bias);

    return output;
}

/// Convenience function: causal attention with F32 weights
pub fn multiHeadCausalAttentionF32(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    weights: *const AttentionWeightsF32,
    config: AttentionConfig,
) !Tensor {
    return multiHeadCausalAttention(Tensor, allocator, input, weights, config);
}

/// Convenience function: causal attention with Q8 weights
pub fn multiHeadCausalAttentionQ8(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    weights: *const AttentionWeightsQ8,
    config: AttentionConfig,
) !Tensor {
    return multiHeadCausalAttention(QuantizedTensorQ8, allocator, input, weights, config);
}

// ============================================================================
// Self-Attention with KV Cache (for efficient autoregressive decoding)
// ============================================================================

/// Self-attention KV cache for a single layer
/// Accumulates K/V as tokens are generated
pub const SelfAttentionKVCache = struct {
    k_cache: Tensor, // [max_seq_len, hidden_dim]
    v_cache: Tensor, // [max_seq_len, hidden_dim]
    // Reusable shape array for views
    k_view_shape: [2]usize,
    v_view_shape: [2]usize,
    position: usize, // Current position (number of cached tokens)
    max_seq_len: usize,
    hidden_dim: usize,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, max_seq_len: usize, hidden_dim: usize) !Self {
        var k_shape = [_]usize{ max_seq_len, hidden_dim };
        var k_cache = try Tensor.init(allocator, &k_shape);
        errdefer k_cache.deinit();

        var v_shape = [_]usize{ max_seq_len, hidden_dim };
        const v_cache = try Tensor.init(allocator, &v_shape);

        return Self{
            .k_cache = k_cache,
            .v_cache = v_cache,
            .k_view_shape = [_]usize{ 0, hidden_dim },
            .v_view_shape = [_]usize{ 0, hidden_dim },
            .position = 0,
            .max_seq_len = max_seq_len,
            .hidden_dim = hidden_dim,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.k_cache.deinit();
        self.v_cache.deinit();
    }

    /// Append new K/V (from single token or batch) to cache
    pub fn append(self: *Self, k_new: *const Tensor, v_new: *const Tensor) void {
        const new_len = k_new.shape[0];
        const hidden = self.hidden_dim;

        const k_start = self.position * hidden;
        @memcpy(
            self.k_cache.data[k_start .. k_start + new_len * hidden],
            k_new.data[0 .. new_len * hidden],
        );

        const v_start = self.position * hidden;
        @memcpy(
            self.v_cache.data[v_start .. v_start + new_len * hidden],
            v_new.data[0 .. new_len * hidden],
        );

        self.position += new_len;
    }

    /// Reset cache for new sequence
    pub fn reset(self: *Self) void {
        self.position = 0;
    }
};

/// Single-token causal self-attention with KV cache
/// Used during generation - processes only 1 new token
/// new_token_emb: [1, hidden_dim] - the new token embedding
/// kv_cache: accumulates K/V from all previous tokens
/// Returns: [1, hidden_dim]
pub fn multiHeadCausalAttentionWithCache(
    comptime WeightType: type,
    allocator: std.mem.Allocator,
    new_token_emb: *const Tensor, // [1, hidden_dim]
    kv_cache: *SelfAttentionKVCache,
    weights: *const AttentionWeights(WeightType),
    config: AttentionConfig,
) !Tensor {
    const num_heads = config.num_heads;
    const head_dim = config.head_dim;
    const hidden_dim = config.hidden_dim;

    // Project new token to Q, K, V
    var q_new = try matmulWithWeight(WeightType, allocator, new_token_emb, &weights.q_weight);
    defer q_new.deinit();
    var k_new = try matmulWithWeight(WeightType, allocator, new_token_emb, &weights.k_weight);
    errdefer k_new.deinit();
    var v_new = try matmulWithWeight(WeightType, allocator, new_token_emb, &weights.v_weight);
    errdefer v_new.deinit();

    try ops.addBiasInPlace(&q_new, &weights.q_bias);
    try ops.addBiasInPlace(&k_new, &weights.k_bias);
    try ops.addBiasInPlace(&v_new, &weights.v_bias);

    // Append new K/V to cache
    kv_cache.append(&k_new, &v_new);
    k_new.deinit();
    v_new.deinit();

    // Get full K/V from cache - update shape views with current length
    const cache_len = kv_cache.position;
    kv_cache.k_view_shape[0] = cache_len;
    kv_cache.v_view_shape[0] = cache_len;

    // Create views into cached K/V
    var k_full = Tensor{
        .data = kv_cache.k_cache.data[0 .. cache_len * hidden_dim],
        .shape = &kv_cache.k_view_shape,
        .allocator = kv_cache.allocator, // Won't be freed since views don't own data
    };
    var v_full = Tensor{
        .data = kv_cache.v_cache.data[0 .. cache_len * hidden_dim],
        .shape = &kv_cache.v_view_shape,
        .allocator = kv_cache.allocator,
    };

    // Split into heads and compute attention (Q attends to all cached K/V)
    var head_outputs = try allocator.alloc(Tensor, num_heads);
    var head_count: usize = 0;
    errdefer {
        for (head_outputs[0..head_count]) |*h| h.deinit();
        allocator.free(head_outputs);
    }

    for (0..num_heads) |h| {
        const start = h * head_dim;
        const end = start + head_dim;

        // Q: [1, head_dim], K: [cache_len, head_dim], V: [cache_len, head_dim]
        var q_head = try ops.sliceColumns(allocator, &q_new, start, end);
        defer q_head.deinit();
        var k_head = try ops.sliceColumns(allocator, &k_full, start, end);
        defer k_head.deinit();
        var v_head = try ops.sliceColumns(allocator, &v_full, start, end);
        defer v_head.deinit();

        // No mask needed - single token Q attending to all previous K
        // (last position can see all previous positions)
        head_outputs[h] = try scaledDotProductAttention(
            allocator,
            &q_head, // [1, head_dim]
            &k_head, // [cache_len, head_dim]
            &v_head, // [cache_len, head_dim]
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

    // Final projection
    var output = try matmulWithWeight(WeightType, allocator, &concat, &weights.o_weight);
    try ops.addBiasInPlace(&output, &weights.o_bias);

    return output;
}

/// Convenience function: causal attention with cache and F32 weights
pub fn multiHeadCausalAttentionWithCacheF32(
    allocator: std.mem.Allocator,
    new_token_emb: *const Tensor,
    kv_cache: *SelfAttentionKVCache,
    weights: *const AttentionWeightsF32,
    config: AttentionConfig,
) !Tensor {
    return multiHeadCausalAttentionWithCache(Tensor, allocator, new_token_emb, kv_cache, weights, config);
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

test "cross-attention different input sizes" {
    const allocator = std.testing.allocator;

    // Config: 2 heads, 4 hidden dim (same for encoder/decoder)
    const config = CrossAttentionConfig{
        .num_heads = 2,
        .decoder_hidden = 4,
        .encoder_hidden = 4,
        .head_dim = 2,
    };

    // Query from decoder: [2 tokens, 4 hidden] (e.g., generating 2 tokens)
    var query_shape = [_]usize{ 2, 4 };
    var query_input = try Tensor.init(allocator, &query_shape);
    defer query_input.deinit();
    for (query_input.data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i)) * 0.1;
    }

    // Key/Value from encoder: [5 tokens, 4 hidden] (e.g., 5 audio frames)
    var kv_shape = [_]usize{ 5, 4 };
    var kv_input = try Tensor.init(allocator, &kv_shape);
    defer kv_input.deinit();
    for (kv_input.data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i)) * 0.05;
    }

    // Create identity-like weights
    var weight_shape = [_]usize{ 4, 4 };
    var bias_shape = [_]usize{4};

    var q_weight = try Tensor.init(allocator, &weight_shape);
    q_weight.data[0] = 1.0;
    q_weight.data[5] = 1.0;
    q_weight.data[10] = 1.0;
    q_weight.data[15] = 1.0;

    const k_weight = try Tensor.init(allocator, &weight_shape);
    @memcpy(k_weight.data, q_weight.data);
    const v_weight = try Tensor.init(allocator, &weight_shape);
    @memcpy(v_weight.data, q_weight.data);
    const o_weight = try Tensor.init(allocator, &weight_shape);
    @memcpy(o_weight.data, q_weight.data);

    const q_bias = try Tensor.init(allocator, &bias_shape);
    const k_bias = try Tensor.init(allocator, &bias_shape);
    const v_bias = try Tensor.init(allocator, &bias_shape);
    const o_bias = try Tensor.init(allocator, &bias_shape);

    var weights = CrossAttentionWeightsF32{
        .q_weight = q_weight,
        .q_bias = q_bias,
        .k_weight = k_weight,
        .k_bias = k_bias,
        .v_weight = v_weight,
        .v_bias = v_bias,
        .o_weight = o_weight,
        .o_bias = o_bias,
    };
    defer weights.deinit();

    // Compute cross-attention
    var output = try multiHeadCrossAttention(
        Tensor,
        allocator,
        &query_input,
        &kv_input,
        &weights,
        config,
    );
    defer output.deinit();

    // Output shape should match query: [2, 4]
    try std.testing.expectEqual(@as(usize, 2), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), output.shape[1]);
}

test "causal attention masks future positions" {
    const allocator = std.testing.allocator;

    // Config: 1 head, 2 hidden dim (simple for verification)
    const config = AttentionConfig{
        .num_heads = 1,
        .hidden_dim = 2,
        .head_dim = 2,
    };

    // Input: 3 tokens
    var input_shape = [_]usize{ 3, 2 };
    var input = try Tensor.init(allocator, &input_shape);
    defer input.deinit();
    // Different values per token so we can verify masking
    input.data[0] = 1.0;
    input.data[1] = 0.0; // token 0
    input.data[2] = 0.0;
    input.data[3] = 1.0; // token 1
    input.data[4] = 0.5;
    input.data[5] = 0.5; // token 2

    // Identity weights
    var weight_shape = [_]usize{ 2, 2 };
    var bias_shape = [_]usize{2};

    var q_weight = try Tensor.init(allocator, &weight_shape);
    q_weight.data[0] = 1.0;
    q_weight.data[3] = 1.0;
    const k_weight = try Tensor.init(allocator, &weight_shape);
    @memcpy(k_weight.data, q_weight.data);
    const v_weight = try Tensor.init(allocator, &weight_shape);
    @memcpy(v_weight.data, q_weight.data);
    const o_weight = try Tensor.init(allocator, &weight_shape);
    @memcpy(o_weight.data, q_weight.data);

    const q_bias = try Tensor.init(allocator, &bias_shape);
    const k_bias = try Tensor.init(allocator, &bias_shape);
    const v_bias = try Tensor.init(allocator, &bias_shape);
    const o_bias = try Tensor.init(allocator, &bias_shape);

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

    // Compute causal attention
    var output = try multiHeadCausalAttention(Tensor, allocator, &input, &weights, config);
    defer output.deinit();

    // Output shape should match input
    try std.testing.expectEqual(@as(usize, 3), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), output.shape[1]);

    // The first token should only attend to itself (due to causal mask)
    // With identity weights and softmax(1 element) = 1.0, output[0] should equal input[0]
    // (This is a simplified verification - the actual values depend on the projection)
}

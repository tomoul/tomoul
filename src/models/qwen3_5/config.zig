// src/models/qwen3_5/config.zig
// Qwen3.5-0.8B model configuration
//
// Hybrid Gated DeltaNet + Attention architecture.
// 24 layers in [L L L A] × 6 pattern:
//   L = Gated DeltaNet (linear attention, 18 layers)
//   A = Full GQA Attention (6 layers)

const std = @import("std");

pub const LayerType = enum {
    linear_attention, // Gated DeltaNet
    full_attention, // Standard GQA
};

pub const Qwen3_5Config = struct {
    // Model architecture
    hidden_size: usize = 1024,
    num_hidden_layers: usize = 24,
    intermediate_size: usize = 3584,
    vocab_size: usize = 248320,

    // Full attention (GQA) parameters
    num_attention_heads: usize = 8,
    num_key_value_heads: usize = 2,
    head_dim: usize = 256,

    // DeltaNet parameters
    linear_num_key_heads: usize = 16,
    linear_num_value_heads: usize = 16,
    linear_key_head_dim: usize = 128,
    linear_value_head_dim: usize = 128,
    linear_conv_kernel_dim: usize = 4,

    // Normalization
    rms_norm_eps: f32 = 1e-6,

    // Position encoding
    rope_theta: f32 = 10_000_000.0,
    partial_rotary_factor: f32 = 0.25,
    max_position_embeddings: usize = 262144,

    // Layer layout: every Nth layer is full attention
    full_attention_interval: usize = 4,

    // Output gate for full attention layers
    attn_output_gate: bool = true,

    // Derived dimensions
    pub fn linearKeyDim(self: Qwen3_5Config) usize {
        return self.linear_num_key_heads * self.linear_key_head_dim;
    }

    pub fn linearValueDim(self: Qwen3_5Config) usize {
        return self.linear_num_value_heads * self.linear_value_head_dim;
    }

    /// Total QKV dimension for DeltaNet: key + key + value
    pub fn linearQkvDim(self: Qwen3_5Config) usize {
        return self.linearKeyDim() * 2 + self.linearValueDim();
    }

    pub fn fullAttentionQDim(self: Qwen3_5Config) usize {
        return self.num_attention_heads * self.head_dim;
    }

    /// Q projection output dimension (includes gate when attn_output_gate=true)
    pub fn fullAttentionQProjDim(self: Qwen3_5Config) usize {
        const q_dim = self.fullAttentionQDim();
        return if (self.attn_output_gate) q_dim * 2 else q_dim;
    }

    pub fn fullAttentionKvDim(self: Qwen3_5Config) usize {
        return self.num_key_value_heads * self.head_dim;
    }

    /// Number of RoPE dimensions (partial_rotary_factor * head_dim)
    pub fn ropePartialDim(self: Qwen3_5Config) usize {
        return @intFromFloat(@as(f32, @floatFromInt(self.head_dim)) * self.partial_rotary_factor);
    }

    /// GQA ratio: how many Q heads per KV head
    pub fn gqaRatio(self: Qwen3_5Config) usize {
        return self.num_attention_heads / self.num_key_value_heads;
    }

    pub fn getLayerType(self: Qwen3_5Config, layer_idx: usize) LayerType {
        // Pattern: [L L L A] × 6 → indices 3,7,11,15,19,23 are full attention
        if ((layer_idx + 1) % self.full_attention_interval == 0) {
            return .full_attention;
        }
        return .linear_attention;
    }

    pub fn numDeltaNetLayers(self: Qwen3_5Config) usize {
        var count: usize = 0;
        for (0..self.num_hidden_layers) |i| {
            if (self.getLayerType(i) == .linear_attention) count += 1;
        }
        return count;
    }

    pub fn numFullAttentionLayers(self: Qwen3_5Config) usize {
        var count: usize = 0;
        for (0..self.num_hidden_layers) |i| {
            if (self.getLayerType(i) == .full_attention) count += 1;
        }
        return count;
    }

    pub const default = Qwen3_5Config{};
};

pub const SpecialTokens = struct {
    pub const END_OF_TEXT: u32 = 248044;
    pub const IM_START: u32 = 248045;
    pub const IM_END: u32 = 248046;
    pub const IMAGE_PAD: u32 = 248056;
    pub const VIDEO_PAD: u32 = 248057;
    /// EOS token for generation stopping
    pub const EOS = IM_END;
};

// ============================================================================
// Tests
// ============================================================================

test "layer_types" {
    const cfg = Qwen3_5Config.default;

    // [L L L A] × 6 pattern
    try std.testing.expectEqual(LayerType.linear_attention, cfg.getLayerType(0));
    try std.testing.expectEqual(LayerType.linear_attention, cfg.getLayerType(1));
    try std.testing.expectEqual(LayerType.linear_attention, cfg.getLayerType(2));
    try std.testing.expectEqual(LayerType.full_attention, cfg.getLayerType(3));
    try std.testing.expectEqual(LayerType.linear_attention, cfg.getLayerType(4));
    try std.testing.expectEqual(LayerType.full_attention, cfg.getLayerType(7));
    try std.testing.expectEqual(LayerType.full_attention, cfg.getLayerType(23));

    try std.testing.expectEqual(@as(usize, 18), cfg.numDeltaNetLayers());
    try std.testing.expectEqual(@as(usize, 6), cfg.numFullAttentionLayers());
}

test "dimensions" {
    const cfg = Qwen3_5Config.default;

    try std.testing.expectEqual(@as(usize, 2048), cfg.linearKeyDim());
    try std.testing.expectEqual(@as(usize, 2048), cfg.linearValueDim());
    try std.testing.expectEqual(@as(usize, 6144), cfg.linearQkvDim());
    try std.testing.expectEqual(@as(usize, 2048), cfg.fullAttentionQDim());
    try std.testing.expectEqual(@as(usize, 512), cfg.fullAttentionKvDim());
    try std.testing.expectEqual(@as(usize, 64), cfg.ropePartialDim());
    try std.testing.expectEqual(@as(usize, 4), cfg.gqaRatio());
}

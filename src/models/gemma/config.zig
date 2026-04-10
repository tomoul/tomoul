// src/models/gemma/config.zig
// Gemma model configuration
//
// Supports Gemma 1 (2B, 7B) and Gemma 2 (9B, 27B) variants.
// Each variant uses GQA (grouped-query attention) with RMSNorm and GeGLU FFN.

const std = @import("std");
const attention = @import("attention.zig");
const GQAConfig = attention.GQAConfig;

/// Gemma model variant
pub const GemmaVariant = enum {
    gemma_2b, // Gemma 1 — 18 layers, MQA (1 KV head)
    gemma_7b, // Gemma 1 — 28 layers, MHA (16/16 heads)
    gemma2_9b, // Gemma 2 — 42 layers, GQA (16/8 heads)
    gemma2_27b, // Gemma 2 — 46 layers, GQA (32/16 heads)
};

/// Complete Gemma model configuration
pub const GemmaConfig = struct {
    variant: GemmaVariant,
    vocab_size: usize,
    num_layers: usize,
    hidden_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_dim: usize,
    max_position: usize,
    rms_norm_eps: f32,
    rope_base: f32,

    /// Whether this is a Gemma 2 variant (has post-attention/FFN norms)
    pub fn isGemma2(self: GemmaConfig) bool {
        return self.variant == .gemma2_9b or self.variant == .gemma2_27b;
    }

    /// Get GQA attention config for this model
    pub fn gqaConfig(self: GemmaConfig) GQAConfig {
        return .{
            .num_heads = self.num_heads,
            .num_kv_heads = self.num_kv_heads,
            .head_dim = self.head_dim,
            .hidden_dim = self.hidden_dim,
        };
    }

    /// KV dimension (num_kv_heads * head_dim) — used for cache sizing
    pub fn kvDim(self: GemmaConfig) usize {
        return self.num_kv_heads * self.head_dim;
    }

    /// Embedding scale factor: sqrt(hidden_dim)
    pub fn embeddingScale(self: GemmaConfig) f32 {
        return @sqrt(@as(f32, @floatFromInt(self.hidden_dim)));
    }

    /// Get config for a known variant
    pub fn forVariant(variant: GemmaVariant) GemmaConfig {
        return switch (variant) {
            .gemma_2b => .{
                .variant = .gemma_2b,
                .vocab_size = 256000,
                .num_layers = 18,
                .hidden_dim = 2048,
                .num_heads = 8,
                .num_kv_heads = 1,
                .head_dim = 256,
                .intermediate_dim = 16384,
                .max_position = 8192,
                .rms_norm_eps = 1e-6,
                .rope_base = 10000.0,
            },
            .gemma_7b => .{
                .variant = .gemma_7b,
                .vocab_size = 256000,
                .num_layers = 28,
                .hidden_dim = 3072,
                .num_heads = 16,
                .num_kv_heads = 16,
                .head_dim = 256,
                .intermediate_dim = 24576,
                .max_position = 8192,
                .rms_norm_eps = 1e-6,
                .rope_base = 10000.0,
            },
            .gemma2_9b => .{
                .variant = .gemma2_9b,
                .vocab_size = 256000,
                .num_layers = 42,
                .hidden_dim = 3584,
                .num_heads = 16,
                .num_kv_heads = 8,
                .head_dim = 256,
                .intermediate_dim = 14336,
                .max_position = 8192,
                .rms_norm_eps = 1e-6,
                .rope_base = 10000.0,
            },
            .gemma2_27b => .{
                .variant = .gemma2_27b,
                .vocab_size = 256000,
                .num_layers = 46,
                .hidden_dim = 4608,
                .num_heads = 32,
                .num_kv_heads = 16,
                .head_dim = 128,
                .intermediate_dim = 36864,
                .max_position = 8192,
                .rms_norm_eps = 1e-6,
                .rope_base = 10000.0,
            },
        };
    }
};

/// Special token IDs for Gemma (SentencePiece)
pub const GemmaTokens = struct {
    pub const PAD: u32 = 0;
    pub const EOS: u32 = 1;
    pub const BOS: u32 = 2;
    pub const UNK: u32 = 3;
};

/// Infer variant from weights file path
pub fn inferVariantFromPath(path: []const u8) GemmaVariant {
    if (std.mem.indexOf(u8, path, "27b") != null) return .gemma2_27b;
    if (std.mem.indexOf(u8, path, "9b") != null) return .gemma2_9b;
    if (std.mem.indexOf(u8, path, "7b") != null) return .gemma_7b;
    return .gemma_2b;
}

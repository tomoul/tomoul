// src/models/whisper/config.zig
// Whisper model configuration and constants
//
// Whisper comes in multiple sizes. This file defines configurations for each.

const std = @import("std");

/// Whisper model variant
pub const WhisperVariant = enum {
    tiny,
    base,
    small,
    medium,
    large,
    large_v3_turbo,
    distil_small_en,

    pub fn name(self: WhisperVariant) []const u8 {
        return switch (self) {
            .tiny => "tiny",
            .base => "base",
            .small => "small",
            .medium => "medium",
            .large => "large",
            .large_v3_turbo => "large-v3-turbo",
            .distil_small_en => "distil-small.en",
        };
    }

    pub fn fromString(s: []const u8) ?WhisperVariant {
        if (std.mem.eql(u8, s, "tiny")) return .tiny;
        if (std.mem.eql(u8, s, "base")) return .base;
        if (std.mem.eql(u8, s, "small")) return .small;
        if (std.mem.eql(u8, s, "medium")) return .medium;
        if (std.mem.eql(u8, s, "large")) return .large;
        if (std.mem.eql(u8, s, "large-v3-turbo")) return .large_v3_turbo;
        if (std.mem.eql(u8, s, "turbo")) return .large_v3_turbo;
        if (std.mem.eql(u8, s, "distil-small.en")) return .distil_small_en;
        if (std.mem.eql(u8, s, "distil-small-en")) return .distil_small_en;
        if (std.mem.eql(u8, s, "distil_small_en")) return .distil_small_en;
        return null;
    }
};

/// Audio encoder configuration
pub const EncoderConfig = struct {
    /// Number of Mel spectrogram bins (always 80 for Whisper)
    n_mels: usize = 80,

    /// Number of audio context frames after Conv1D (1500 for 30s audio)
    n_audio_ctx: usize = 1500,

    /// Hidden dimension / model state size
    n_audio_state: usize,

    /// Number of attention heads
    n_audio_head: usize,

    /// Number of transformer layers
    n_audio_layer: usize,

    /// Computed: head dimension
    pub fn headDim(self: EncoderConfig) usize {
        return self.n_audio_state / self.n_audio_head;
    }
};

/// Text decoder configuration
pub const DecoderConfig = struct {
    /// Vocabulary size (51865 for multilingual)
    n_vocab: usize = 51865,

    /// Maximum text context length (448 tokens)
    n_text_ctx: usize = 448,

    /// Hidden dimension / model state size
    n_text_state: usize,

    /// Number of attention heads
    n_text_head: usize,

    /// Number of transformer layers
    n_text_layer: usize,

    /// Computed: head dimension
    pub fn headDim(self: DecoderConfig) usize {
        return self.n_text_state / self.n_text_head;
    }
};

/// Complete Whisper model configuration
pub const WhisperConfig = struct {
    /// Model variant name
    variant: WhisperVariant,

    /// Encoder configuration
    encoder: EncoderConfig,

    /// Decoder configuration
    decoder: DecoderConfig,

    /// Get configuration for a specific variant
    pub fn forVariant(variant: WhisperVariant) WhisperConfig {
        return switch (variant) {
            .tiny => WhisperConfig{
                .variant = .tiny,
                .encoder = .{
                    .n_audio_state = 384,
                    .n_audio_head = 6,
                    .n_audio_layer = 4,
                },
                .decoder = .{
                    .n_text_state = 384,
                    .n_text_head = 6,
                    .n_text_layer = 4,
                },
            },
            .base => WhisperConfig{
                .variant = .base,
                .encoder = .{
                    .n_audio_state = 512,
                    .n_audio_head = 8,
                    .n_audio_layer = 6,
                },
                .decoder = .{
                    .n_text_state = 512,
                    .n_text_head = 8,
                    .n_text_layer = 6,
                },
            },
            .small => WhisperConfig{
                .variant = .small,
                .encoder = .{
                    .n_audio_state = 768,
                    .n_audio_head = 12,
                    .n_audio_layer = 12,
                },
                .decoder = .{
                    .n_text_state = 768,
                    .n_text_head = 12,
                    .n_text_layer = 12,
                },
            },
            .medium => WhisperConfig{
                .variant = .medium,
                .encoder = .{
                    .n_audio_state = 1024,
                    .n_audio_head = 16,
                    .n_audio_layer = 24,
                },
                .decoder = .{
                    .n_text_state = 1024,
                    .n_text_head = 16,
                    .n_text_layer = 24,
                },
            },
            .large => WhisperConfig{
                .variant = .large,
                .encoder = .{
                    .n_audio_state = 1280,
                    .n_audio_head = 20,
                    .n_audio_layer = 32,
                },
                .decoder = .{
                    .n_text_state = 1280,
                    .n_text_head = 20,
                    .n_text_layer = 32,
                },
            },
            .large_v3_turbo => WhisperConfig{
                .variant = .large_v3_turbo,
                .encoder = .{
                    .n_mels = 128, // large-v3-turbo uses 128 mels
                    .n_audio_state = 1280,
                    .n_audio_head = 20,
                    .n_audio_layer = 32,
                },
                .decoder = .{
                    .n_vocab = 51866, // one extra token
                    .n_text_state = 1280,
                    .n_text_head = 20,
                    .n_text_layer = 4, // turbo: only 4 decoder layers!
                },
            },
            .distil_small_en => WhisperConfig{
                .variant = .distil_small_en,
                .encoder = .{
                    // Same as small encoder
                    .n_audio_state = 768,
                    .n_audio_head = 12,
                    .n_audio_layer = 12,
                },
                .decoder = .{
                    .n_vocab = 51864, // slightly smaller vocab
                    .n_text_state = 768,
                    .n_text_head = 12,
                    .n_text_layer = 4, // distilled: only 4 decoder layers!
                },
            },
        };
    }

    /// Estimated model size in bytes (FP32)
    pub fn estimatedSizeBytes(self: WhisperConfig) usize {
        // Rough estimate: encoder + decoder params
        const enc = self.encoder;
        const dec = self.decoder;

        // Encoder: conv layers + transformer blocks + embeddings
        const enc_conv = (enc.n_mels * enc.n_audio_state * 3) + // conv1
            (enc.n_audio_state * enc.n_audio_state * 3); // conv2
        const enc_pos = enc.n_audio_ctx * enc.n_audio_state;
        const enc_block = (4 * enc.n_audio_state * enc.n_audio_state + // attention
            8 * enc.n_audio_state * enc.n_audio_state) * enc.n_audio_layer; // FFN

        // Decoder: embeddings + transformer blocks
        const dec_embed = dec.n_vocab * dec.n_text_state + dec.n_text_ctx * dec.n_text_state;
        const dec_block = (4 * dec.n_text_state * dec.n_text_state + // self-attn
            4 * dec.n_text_state * dec.n_text_state + // cross-attn
            8 * dec.n_text_state * dec.n_text_state) * dec.n_text_layer; // FFN

        return (enc_conv + enc_pos + enc_block + dec_embed + dec_block) * 4; // 4 bytes per float32
    }

    /// Estimated model size in MB (FP32)
    pub fn estimatedSizeMB(self: WhisperConfig) f32 {
        return @as(f32, @floatFromInt(self.estimatedSizeBytes())) / (1024.0 * 1024.0);
    }
};

/// Special token IDs for Whisper
pub const WhisperTokens = struct {
    /// <|endoftext|>
    pub const EOT: u32 = 50257;

    /// <|startoftranscript|>
    pub const SOT: u32 = 50258;

    /// Language tokens start here (50259 = <|en|>)
    pub const LANG_EN: u32 = 50259;
    pub const LANG_ZH: u32 = 50260;
    pub const LANG_DE: u32 = 50261;
    pub const LANG_ES: u32 = 50262;
    pub const LANG_RU: u32 = 50263;
    pub const LANG_KO: u32 = 50264;
    pub const LANG_FR: u32 = 50265;
    pub const LANG_JA: u32 = 50266;
    pub const LANG_PT: u32 = 50267;

    /// <|translate|>
    pub const TRANSLATE: u32 = 50358;

    /// <|transcribe|>
    pub const TRANSCRIBE: u32 = 50359;

    /// <|startoflm|>
    pub const SOL: u32 = 50360;

    /// <|startofprev|>
    pub const SOP: u32 = 50361;

    /// <|nospeech|>
    pub const NO_SPEECH: u32 = 50362;

    /// <|notimestamps|>
    pub const NO_TIMESTAMPS: u32 = 50363;

    /// Timestamp tokens start here (50364 = <|0.00|>)
    pub const TIMESTAMP_BEGIN: u32 = 50364;
};

/// Default prompt tokens for transcription (English, no timestamps)
pub fn defaultPromptTokens() [4]u32 {
    return .{
        WhisperTokens.SOT,
        WhisperTokens.LANG_EN,
        WhisperTokens.TRANSCRIBE,
        WhisperTokens.NO_TIMESTAMPS,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "config variants" {
    const tiny = WhisperConfig.forVariant(.tiny);
    try std.testing.expectEqual(@as(usize, 384), tiny.encoder.n_audio_state);
    try std.testing.expectEqual(@as(usize, 6), tiny.encoder.n_audio_head);
    try std.testing.expectEqual(@as(usize, 4), tiny.encoder.n_audio_layer);
    try std.testing.expectEqual(@as(usize, 64), tiny.encoder.headDim());

    const base = WhisperConfig.forVariant(.base);
    try std.testing.expectEqual(@as(usize, 512), base.encoder.n_audio_state);
    try std.testing.expectEqual(@as(usize, 8), base.encoder.n_audio_head);
    try std.testing.expectEqual(@as(usize, 6), base.encoder.n_audio_layer);
}

test "model size estimates" {
    const tiny = WhisperConfig.forVariant(.tiny);
    const base = WhisperConfig.forVariant(.base);

    // Tiny should be smaller than base
    try std.testing.expect(tiny.estimatedSizeBytes() < base.estimatedSizeBytes());

    // Rough sanity check: tiny is ~40MB
    try std.testing.expect(tiny.estimatedSizeMB() > 20.0);
    try std.testing.expect(tiny.estimatedSizeMB() < 100.0);
}

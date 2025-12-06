// src/models/whisper/model.zig
// Whisper Speech-to-Text Model
//
// Main entry point for Whisper inference.
// Combines encoder and decoder with generation loop.
//
// Usage:
//   const model = try Whisper.load(allocator, "whisper_tiny.tl");
//   defer model.deinit();
//
//   const tokens = try model.transcribe(&mel_spectrogram);
//   defer allocator.free(tokens);
//
// Note: Text decoding from tokens is deferred to the caller (Python/JS)
// to avoid implementing the complex BPE tokenizer in Zig for Phase 10.

const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const ops = @import("ops.zig");

pub const config = @import("config.zig");
pub const encoder = @import("encoder.zig");
pub const decoder = @import("decoder.zig");

const WhisperConfig = config.WhisperConfig;
const WhisperTokens = config.WhisperTokens;
const WhisperEncoder = encoder.WhisperEncoder;
const WhisperEncoderWeights = encoder.WhisperEncoderWeights;
const WhisperDecoder = decoder.WhisperDecoder;
const WhisperDecoderWeights = decoder.WhisperDecoderWeights;

/// Whisper transcription options
pub const TranscribeOptions = struct {
    /// Language token (default: English)
    language: u32 = WhisperTokens.LANG_EN,

    /// Task: transcribe or translate
    task: u32 = WhisperTokens.TRANSCRIBE,

    /// Include timestamps in output
    timestamps: bool = false,

    /// Maximum tokens to generate
    max_tokens: usize = 224,

    /// Temperature for sampling (0.0 = greedy)
    temperature: f32 = 0.0,
};

/// Complete Whisper model
pub const Whisper = struct {
    cfg: WhisperConfig,
    enc: WhisperEncoder,
    dec: WhisperDecoder,
    enc_weights: *WhisperEncoderWeights,
    dec_weights: *WhisperDecoderWeights,
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Initialize Whisper with pre-loaded weights
    pub fn init(
        allocator: std.mem.Allocator,
        cfg: WhisperConfig,
        enc_weights: *WhisperEncoderWeights,
        dec_weights: *WhisperDecoderWeights,
    ) Self {
        return .{
            .cfg = cfg,
            .enc = WhisperEncoder.init(allocator, cfg.encoder, enc_weights),
            .dec = WhisperDecoder.init(allocator, cfg.decoder, dec_weights),
            .enc_weights = enc_weights,
            .dec_weights = dec_weights,
            .allocator = allocator,
        };
    }

    /// Transcribe audio to token IDs using greedy decoding
    /// mel: [n_mels, n_frames] Mel spectrogram (e.g., [80, 3000])
    /// Returns: array of token IDs (caller must free)
    ///
    /// Note: The returned tokens need to be decoded to text using the
    /// Whisper BPE tokenizer. For Phase 10, this is done in Python/JS.
    pub fn transcribe(self: *const Self, mel: *const Tensor) ![]u32 {
        return self.transcribeWithOptions(mel, .{});
    }

    /// Transcribe with custom options
    pub fn transcribeWithOptions(
        self: *const Self,
        mel: *const Tensor,
        opts: TranscribeOptions,
    ) ![]u32 {
        const allocator = self.allocator;

        // 1. Encode audio once
        var audio_features = try self.enc.encode(mel);
        defer audio_features.deinit();

        // 2. Initialize prompt tokens
        var tokens = std.ArrayList(u32).init(allocator);
        errdefer tokens.deinit();

        // Add start tokens: <|startoftranscript|><|lang|><|task|><|notimestamps|>
        try tokens.append(WhisperTokens.SOT);
        try tokens.append(opts.language);
        try tokens.append(opts.task);
        if (!opts.timestamps) {
            try tokens.append(WhisperTokens.NO_TIMESTAMPS);
        }

        // 3. Process initial prompt (get logits for first predicted token)
        var logits = try self.dec.forward(tokens.items, &audio_features);
        defer logits.deinit();

        // Get the last row of logits (prediction for next token)
        const seq_len = tokens.items.len;
        const vocab_size = self.cfg.decoder.n_vocab;
        const last_logits_start = (seq_len - 1) * vocab_size;
        const last_logits = logits.data[last_logits_start..][0..vocab_size];

        // 4. Greedy decoding loop
        var position = seq_len;
        const max_position = @min(position + opts.max_tokens, self.cfg.decoder.n_text_ctx);

        while (position < max_position) {
            // Sample next token (greedy: argmax)
            const next_token = argmax(last_logits);

            // Check for end of text
            if (next_token == WhisperTokens.EOT) {
                break;
            }

            try tokens.append(next_token);

            // Decode next step
            var next_logits = try self.dec.decodeStep(next_token, position, &audio_features);
            defer next_logits.deinit();

            // Update last_logits for next iteration
            // Note: next_logits is [1, vocab_size], so row 0
            @memcpy(
                @as([*]f32, @ptrCast(last_logits.ptr))[0..vocab_size],
                next_logits.data[0..vocab_size],
            );

            position += 1;
        }

        return tokens.toOwnedSlice();
    }

    /// Clean up resources
    pub fn deinit(self: *Self) void {
        self.enc_weights.deinit();
        self.dec_weights.deinit();
        // The weights are owned by the model, free them
        self.allocator.destroy(self.enc_weights);
        self.allocator.destroy(self.dec_weights);
    }
};

/// Find the index of the maximum value (greedy sampling)
fn argmax(logits: []const f32) u32 {
    var max_idx: u32 = 0;
    var max_val = logits[0];

    for (logits[1..], 1..) |val, i| {
        if (val > max_val) {
            max_val = val;
            max_idx = @intCast(i);
        }
    }

    return max_idx;
}

/// Softmax with temperature (for sampling)
fn softmaxWithTemperature(logits: []f32, temperature: f32) void {
    if (temperature <= 0.0) return; // Use argmax for greedy

    // Scale by temperature
    for (logits) |*l| {
        l.* /= temperature;
    }

    // Find max for numerical stability
    var max_val = logits[0];
    for (logits[1..]) |l| {
        max_val = @max(max_val, l);
    }

    // Compute exp and sum
    var sum: f32 = 0.0;
    for (logits) |*l| {
        l.* = @exp(l.* - max_val);
        sum += l.*;
    }

    // Normalize
    for (logits) |*l| {
        l.* /= sum;
    }
}

// ============================================================================
// Tests
// ============================================================================

test "argmax" {
    const logits = [_]f32{ 0.1, 0.5, 0.2, 0.8, 0.3 };
    try std.testing.expectEqual(@as(u32, 3), argmax(&logits));
}

test "argmax first element" {
    const logits = [_]f32{ 1.0, 0.5, 0.2 };
    try std.testing.expectEqual(@as(u32, 0), argmax(&logits));
}

test "argmax last element" {
    const logits = [_]f32{ 0.1, 0.2, 0.9 };
    try std.testing.expectEqual(@as(u32, 2), argmax(&logits));
}

test "whisper tokens" {
    try std.testing.expectEqual(@as(u32, 50257), WhisperTokens.EOT);
    try std.testing.expectEqual(@as(u32, 50258), WhisperTokens.SOT);
    try std.testing.expectEqual(@as(u32, 50259), WhisperTokens.LANG_EN);
}

// src/models/whisper/encoder.zig
// Whisper Audio Encoder
//
// Architecture:
//   Mel Spectrogram [80, 3000]
//       │
//       ▼
//   Conv1D (kernel=3, stride=1, padding=1) + GELU
//       │   [n_audio_state, 3000]
//       ▼
//   Conv1D (kernel=3, stride=2, padding=1) + GELU
//       │   [n_audio_state, 1500]  (downsampled 2x)
//       ▼
//   Transpose → [1500, n_audio_state]
//       │
//       ▼
//   + Positional Embedding [1500, n_audio_state]
//       │
//       ▼
//   N × Encoder Blocks (Self-Attention + FFN)
//       │
//       ▼
//   Final LayerNorm
//       │
//       ▼
//   Audio Features [1500, n_audio_state]

const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const ops = @import("ops.zig");
const attention = @import("attention.zig");
const config = @import("config.zig");
const loader_mod = @import("loader.zig");

const ModelLoader = loader_mod.ModelLoader;
const LoadError = loader_mod.LoadError;
const EncoderConfig = config.EncoderConfig;
const AttentionConfig = attention.AttentionConfig;
const AttentionWeightsF32 = attention.AttentionWeightsF32;

/// Encoder block weights (self-attention + FFN)
pub const EncoderBlockWeights = struct {
    // Self-attention
    attn_ln_gamma: Tensor, // [n_audio_state]
    attn_ln_beta: Tensor, // [n_audio_state]
    attn: AttentionWeightsF32,

    // Feed-forward network
    ffn_ln_gamma: Tensor, // [n_audio_state]
    ffn_ln_beta: Tensor, // [n_audio_state]
    ffn_fc1_weight: Tensor, // [n_audio_state, 4 * n_audio_state] pre-transposed
    ffn_fc1_bias: Tensor, // [4 * n_audio_state]
    ffn_fc2_weight: Tensor, // [4 * n_audio_state, n_audio_state] pre-transposed
    ffn_fc2_bias: Tensor, // [n_audio_state]

    pub fn deinit(self: *EncoderBlockWeights) void {
        self.attn_ln_gamma.deinit();
        self.attn_ln_beta.deinit();
        self.attn.deinit();
        self.ffn_ln_gamma.deinit();
        self.ffn_ln_beta.deinit();
        self.ffn_fc1_weight.deinit();
        self.ffn_fc1_bias.deinit();
        self.ffn_fc2_weight.deinit();
        self.ffn_fc2_bias.deinit();
    }
};

/// Complete Whisper encoder weights
pub const WhisperEncoderWeights = struct {
    // Convolutional layers
    conv1_weight: Tensor, // [n_audio_state, n_mels, 3]
    conv1_bias: Tensor, // [n_audio_state]
    conv2_weight: Tensor, // [n_audio_state, n_audio_state, 3]
    conv2_bias: Tensor, // [n_audio_state]

    // Positional embedding
    positional_embedding: Tensor, // [n_audio_ctx, n_audio_state]

    // Encoder blocks
    blocks: []EncoderBlockWeights,

    // Final layer norm
    ln_post_gamma: Tensor, // [n_audio_state]
    ln_post_beta: Tensor, // [n_audio_state]

    allocator: std.mem.Allocator,

    /// Load encoder weights from a ModelLoader
    pub fn loadFromLoader(allocator: std.mem.Allocator, loader: *ModelLoader, cfg: EncoderConfig) !*WhisperEncoderWeights {
        var weights = try allocator.create(WhisperEncoderWeights);
        errdefer allocator.destroy(weights);
        weights.allocator = allocator;

        // Load convolutional layers
        weights.conv1_weight = try loader.getTensorDequantized("encoder.conv1.weight");
        errdefer weights.conv1_weight.deinit();
        weights.conv1_bias = try loader.getTensorDequantized("encoder.conv1.bias");
        errdefer weights.conv1_bias.deinit();
        weights.conv2_weight = try loader.getTensorDequantized("encoder.conv2.weight");
        errdefer weights.conv2_weight.deinit();
        weights.conv2_bias = try loader.getTensorDequantized("encoder.conv2.bias");
        errdefer weights.conv2_bias.deinit();

        // Load positional embedding
        weights.positional_embedding = try loader.getTensorDequantized("encoder.positional_embedding");
        errdefer weights.positional_embedding.deinit();

        // Load encoder blocks
        weights.blocks = try allocator.alloc(EncoderBlockWeights, cfg.n_audio_layer);
        errdefer allocator.free(weights.blocks);

        var loaded_blocks: usize = 0;
        errdefer {
            for (weights.blocks[0..loaded_blocks]) |*block| block.deinit();
        }

        for (0..cfg.n_audio_layer) |i| {
            var buf: [64]u8 = undefined;

            // Self-attention LayerNorm
            const attn_ln_w = try std.fmt.bufPrint(&buf, "encoder.blocks.{d}.attn_ln.weight", .{i});
            weights.blocks[i].attn_ln_gamma = try loader.getTensorDequantized(attn_ln_w);
            const attn_ln_b = try std.fmt.bufPrint(&buf, "encoder.blocks.{d}.attn_ln.bias", .{i});
            weights.blocks[i].attn_ln_beta = try loader.getTensorDequantized(attn_ln_b);

            // Self-attention Q, K, V, O projections
            const q_w = try std.fmt.bufPrint(&buf, "encoder.blocks.{d}.attn.q_weight", .{i});
            weights.blocks[i].attn.q_weight = try loader.getTensorDequantized(q_w);
            const q_b = try std.fmt.bufPrint(&buf, "encoder.blocks.{d}.attn.q_bias", .{i});
            weights.blocks[i].attn.q_bias = try loader.getTensorDequantized(q_b);
            const k_w = try std.fmt.bufPrint(&buf, "encoder.blocks.{d}.attn.k_weight", .{i});
            weights.blocks[i].attn.k_weight = try loader.getTensorDequantized(k_w);
            const k_b = try std.fmt.bufPrint(&buf, "encoder.blocks.{d}.attn.k_bias", .{i});
            weights.blocks[i].attn.k_bias = try loader.getTensorDequantized(k_b);
            const v_w = try std.fmt.bufPrint(&buf, "encoder.blocks.{d}.attn.v_weight", .{i});
            weights.blocks[i].attn.v_weight = try loader.getTensorDequantized(v_w);
            const v_b = try std.fmt.bufPrint(&buf, "encoder.blocks.{d}.attn.v_bias", .{i});
            weights.blocks[i].attn.v_bias = try loader.getTensorDequantized(v_b);
            const o_w = try std.fmt.bufPrint(&buf, "encoder.blocks.{d}.attn.o_weight", .{i});
            weights.blocks[i].attn.o_weight = try loader.getTensorDequantized(o_w);
            const o_b = try std.fmt.bufPrint(&buf, "encoder.blocks.{d}.attn.o_bias", .{i});
            weights.blocks[i].attn.o_bias = try loader.getTensorDequantized(o_b);

            // FFN LayerNorm
            const mlp_ln_w = try std.fmt.bufPrint(&buf, "encoder.blocks.{d}.mlp_ln.weight", .{i});
            weights.blocks[i].ffn_ln_gamma = try loader.getTensorDequantized(mlp_ln_w);
            const mlp_ln_b = try std.fmt.bufPrint(&buf, "encoder.blocks.{d}.mlp_ln.bias", .{i});
            weights.blocks[i].ffn_ln_beta = try loader.getTensorDequantized(mlp_ln_b);

            // FFN fc1, fc2
            const fc1_w = try std.fmt.bufPrint(&buf, "encoder.blocks.{d}.mlp.fc1.weight", .{i});
            weights.blocks[i].ffn_fc1_weight = try loader.getTensorDequantized(fc1_w);
            const fc1_b = try std.fmt.bufPrint(&buf, "encoder.blocks.{d}.mlp.fc1.bias", .{i});
            weights.blocks[i].ffn_fc1_bias = try loader.getTensorDequantized(fc1_b);
            const fc2_w = try std.fmt.bufPrint(&buf, "encoder.blocks.{d}.mlp.fc2.weight", .{i});
            weights.blocks[i].ffn_fc2_weight = try loader.getTensorDequantized(fc2_w);
            const fc2_b = try std.fmt.bufPrint(&buf, "encoder.blocks.{d}.mlp.fc2.bias", .{i});
            weights.blocks[i].ffn_fc2_bias = try loader.getTensorDequantized(fc2_b);

            loaded_blocks += 1;
        }

        // Load final LayerNorm
        weights.ln_post_gamma = try loader.getTensorDequantized("encoder.ln_post.weight");
        errdefer weights.ln_post_gamma.deinit();
        weights.ln_post_beta = try loader.getTensorDequantized("encoder.ln_post.bias");

        return weights;
    }

    pub fn deinit(self: *WhisperEncoderWeights) void {
        self.conv1_weight.deinit();
        self.conv1_bias.deinit();
        self.conv2_weight.deinit();
        self.conv2_bias.deinit();
        self.positional_embedding.deinit();
        for (self.blocks) |*block| block.deinit();
        self.allocator.free(self.blocks);
        self.ln_post_gamma.deinit();
        self.ln_post_beta.deinit();
        // Free the struct itself (allocated by loadFromLoader)
        self.allocator.destroy(self);
    }
};

/// Whisper audio encoder
pub const WhisperEncoder = struct {
    cfg: EncoderConfig,
    weights: *WhisperEncoderWeights,
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Create encoder with loaded weights
    pub fn init(
        allocator: std.mem.Allocator,
        cfg: EncoderConfig,
        weights: *WhisperEncoderWeights,
    ) Self {
        return .{
            .cfg = cfg,
            .weights = weights,
            .allocator = allocator,
        };
    }

    /// Forward pass through a single encoder block
    fn forwardBlock(
        self: *const Self,
        allocator: std.mem.Allocator,
        x: *const Tensor,
        block: *const EncoderBlockWeights,
    ) !Tensor {
        // Pre-norm self-attention
        var attn_ln = try ops.layerNorm(allocator, x, &block.attn_ln_gamma, &block.attn_ln_beta, 1e-5);
        defer attn_ln.deinit();

        const attn_config = AttentionConfig{
            .num_heads = self.cfg.n_audio_head,
            .hidden_dim = self.cfg.n_audio_state,
            .head_dim = self.cfg.headDim(),
        };

        var attn_out = try attention.multiHeadAttention(
            Tensor,
            allocator,
            &attn_ln,
            &block.attn,
            attn_config,
        );
        defer attn_out.deinit();

        // Residual connection
        var x1 = try ops.add(allocator, x, &attn_out);
        defer x1.deinit();

        // Pre-norm FFN
        var ffn_ln = try ops.layerNorm(allocator, &x1, &block.ffn_ln_gamma, &block.ffn_ln_beta, 1e-5);
        defer ffn_ln.deinit();

        // FFN: fc1 -> GELU (exact) -> fc2
        // Whisper uses exact GELU (approximate='none'), not the tanh approximation
        var fc1 = try ops.matmul(allocator, &ffn_ln, &block.ffn_fc1_weight);
        defer fc1.deinit();
        try ops.addBiasInPlace(&fc1, &block.ffn_fc1_bias);
        ops.geluExact(&fc1);

        var fc2 = try ops.matmul(allocator, &fc1, &block.ffn_fc2_weight);
        try ops.addBiasInPlace(&fc2, &block.ffn_fc2_bias);

        // Second residual connection
        const result = try ops.add(allocator, &x1, &fc2);
        fc2.deinit();

        return result;
    }

    /// Encode audio Mel spectrogram to features
    /// mel: [n_mels, n_frames] (e.g., [80, 3000] for 30s audio)
    /// Returns: [n_audio_ctx, n_audio_state] (e.g., [1500, 384] for Whisper Tiny)
    pub fn encode(self: *const Self, mel: *const Tensor) !Tensor {
        const allocator = self.allocator;
        const w = self.weights;

        // 1. Conv1D: [80, 3000] → [n_audio_state, 3000]
        // stride=1, padding=1
        // Whisper uses exact GELU (approximate='none')
        var conv1_out = try ops.conv1d(
            allocator,
            mel,
            &w.conv1_weight,
            &w.conv1_bias,
            1, // stride
            1, // padding
        );
        defer conv1_out.deinit();
        ops.geluExact(&conv1_out);

        // 2. Conv1D with stride 2: [n_audio_state, 3000] → [n_audio_state, 1500]
        // stride=2, padding=1
        var conv2_out = try ops.conv1d(
            allocator,
            &conv1_out,
            &w.conv2_weight,
            &w.conv2_bias,
            2, // stride
            1, // padding
        );
        defer conv2_out.deinit();
        ops.geluExact(&conv2_out);

        // 3. Transpose: [n_audio_state, 1500] → [1500, n_audio_state]
        var x = try ops.transpose(allocator, &conv2_out);
        defer x.deinit();

        // 4. Add positional embedding
        // Note: x and positional_embedding should have same shape [n_audio_ctx, n_audio_state]
        var x_pos = try ops.add(allocator, &x, &w.positional_embedding);

        // 5. Encoder blocks
        for (w.blocks) |*block| {
            const new_x = try self.forwardBlock(allocator, &x_pos, block);
            x_pos.deinit();
            x_pos = new_x;
        }

        // 6. Final LayerNorm
        const result = try ops.layerNorm(allocator, &x_pos, &w.ln_post_gamma, &w.ln_post_beta, 1e-5);
        x_pos.deinit();

        return result;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "encoder config" {
    const cfg = config.WhisperConfig.forVariant(.tiny);
    try std.testing.expectEqual(@as(usize, 384), cfg.encoder.n_audio_state);
    try std.testing.expectEqual(@as(usize, 6), cfg.encoder.n_audio_head);
    try std.testing.expectEqual(@as(usize, 64), cfg.encoder.headDim());
}

test "encoder output validation against PyTorch" {
    const allocator = std.testing.allocator;
    const cfg = config.WhisperConfig.forVariant(.tiny);

    // Load validation fixture (expected outputs from PyTorch)
    var fixture_loader = ModelLoader.init(allocator, "models/whisper_tiny_fixture.tl") catch |err| {
        if (err == LoadError.FileNotFound) {
            std.debug.print("\nSkipping encoder validation: fixture not found.\n", .{});
            std.debug.print("Run 'python3 tools/validate_whisper.py --create-fixture' first.\n", .{});
            return;
        }
        return err;
    };
    defer fixture_loader.deinit();

    // Load model weights
    var model_loader = ModelLoader.init(allocator, "models/whisper_tiny.tl") catch |err| {
        if (err == LoadError.FileNotFound) {
            std.debug.print("\nSkipping encoder validation: model not found.\n", .{});
            std.debug.print("Run 'python3 tools/export_whisper.py -m tiny' first.\n", .{});
            return;
        }
        return err;
    };
    defer model_loader.deinit();

    // Load Mel spectrogram input and expected encoder output
    var mel = try fixture_loader.getTensor("mel");
    defer mel.deinit();
    var expected_encoder_output = try fixture_loader.getTensor("encoder_output");
    defer expected_encoder_output.deinit();

    // Load encoder weights
    var enc_weights = try WhisperEncoderWeights.loadFromLoader(allocator, &model_loader, cfg.encoder);
    defer enc_weights.deinit();

    // Create encoder and run forward pass
    const encoder = WhisperEncoder.init(allocator, cfg.encoder, enc_weights);
    var encoder_output = try encoder.encode(&mel);
    defer encoder_output.deinit();

    // Verify output shape matches expected
    try std.testing.expectEqual(expected_encoder_output.shape.len, encoder_output.shape.len);
    for (expected_encoder_output.shape, encoder_output.shape) |expected_dim, actual_dim| {
        try std.testing.expectEqual(expected_dim, actual_dim);
    }

    // Compare outputs element-wise
    // Tolerance of 1e-3 for float differences accumulated through deep network
    const tolerance: f32 = 1e-3;
    var max_diff: f32 = 0.0;
    var mean_diff: f32 = 0.0;
    var num_errors: usize = 0;

    for (expected_encoder_output.data, encoder_output.data) |expected, actual| {
        const diff = @abs(expected - actual);
        max_diff = @max(max_diff, diff);
        mean_diff += diff;
        if (diff > tolerance) {
            num_errors += 1;
        }
    }
    mean_diff /= @as(f32, @floatFromInt(encoder_output.data.len));

    std.debug.print("\n=== Encoder Validation Results ===\n", .{});
    std.debug.print("Output shape: [{d}, {d}]\n", .{ encoder_output.shape[0], encoder_output.shape[1] });
    std.debug.print("Max diff: {d:.6}\n", .{max_diff});
    std.debug.print("Mean diff: {d:.6}\n", .{mean_diff});
    std.debug.print("Errors (> {d:.0e}): {d}/{d}\n", .{ tolerance, num_errors, encoder_output.data.len });

    // Test passes if max diff is within tolerance
    try std.testing.expect(max_diff < tolerance);
}

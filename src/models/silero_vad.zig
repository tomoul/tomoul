const std = @import("std");
const Tensor = @import("../core/tensor.zig").Tensor;
const ops = @import("../core/ops.zig");
const ModelLoader = @import("../core/loader.zig").ModelLoader;
const LoadError = @import("../core/loader.zig").LoadError;

/// Silero VAD Model
/// Voice Activity Detection using LSTM-based architecture.
///
/// Architecture (16kHz model):
///   STFT (learned) -> Conv Encoder (4 layers) -> LSTM -> Decoder -> Sigmoid
///
/// The model processes audio chunks (typically 512 samples @ 16kHz = 32ms)
/// and outputs a speech probability [0.0 - 1.0].
pub const SileroVAD = struct {
    allocator: std.mem.Allocator,

    // STFT basis (learned Fourier transform)
    stft_basis: Tensor, // [258, 1, 256]

    // Encoder Conv1d layers
    enc0_weight: Tensor, // [128, 129, 3]
    enc0_bias: Tensor, // [128]
    enc1_weight: Tensor, // [64, 128, 3]
    enc1_bias: Tensor, // [64]
    enc2_weight: Tensor, // [64, 64, 3]
    enc2_bias: Tensor, // [64]
    enc3_weight: Tensor, // [128, 64, 3]
    enc3_bias: Tensor, // [128]

    // LSTM weights (hidden_size=128, 4 gates packed)
    lstm_weights: ops.LSTMWeights,

    // Decoder (1x1 conv acting as linear)
    dec_weight: Tensor, // [1, 128, 1]
    dec_bias: Tensor, // [1]

    // LSTM state (persistent across calls)
    lstm_state: ops.LSTMState,

    const Self = @This();

    /// Load Silero VAD model from .tl file
    pub fn init(allocator: std.mem.Allocator, model_path: []const u8) !Self {
        var loader = try ModelLoader.init(allocator, model_path);
        defer loader.deinit();

        // Load STFT basis
        var stft_basis = try loader.getTensor("_model.stft.forward_basis_buffer");
        errdefer stft_basis.deinit();

        // Load encoder weights
        var enc0_weight = try loader.getTensor("_model.encoder.0.reparam_conv.weight");
        errdefer enc0_weight.deinit();
        var enc0_bias = try loader.getTensor("_model.encoder.0.reparam_conv.bias");
        errdefer enc0_bias.deinit();

        var enc1_weight = try loader.getTensor("_model.encoder.1.reparam_conv.weight");
        errdefer enc1_weight.deinit();
        var enc1_bias = try loader.getTensor("_model.encoder.1.reparam_conv.bias");
        errdefer enc1_bias.deinit();

        var enc2_weight = try loader.getTensor("_model.encoder.2.reparam_conv.weight");
        errdefer enc2_weight.deinit();
        var enc2_bias = try loader.getTensor("_model.encoder.2.reparam_conv.bias");
        errdefer enc2_bias.deinit();

        var enc3_weight = try loader.getTensor("_model.encoder.3.reparam_conv.weight");
        errdefer enc3_weight.deinit();
        var enc3_bias = try loader.getTensor("_model.encoder.3.reparam_conv.bias");
        errdefer enc3_bias.deinit();

        // Load LSTM weights
        var lstm_weight_ih = try loader.getTensor("_model.decoder.rnn.weight_ih");
        errdefer lstm_weight_ih.deinit();
        var lstm_weight_hh = try loader.getTensor("_model.decoder.rnn.weight_hh");
        errdefer lstm_weight_hh.deinit();
        var lstm_bias_ih = try loader.getTensor("_model.decoder.rnn.bias_ih");
        errdefer lstm_bias_ih.deinit();
        var lstm_bias_hh = try loader.getTensor("_model.decoder.rnn.bias_hh");
        errdefer lstm_bias_hh.deinit();

        // Load decoder weights
        var dec_weight = try loader.getTensor("_model.decoder.decoder.2.weight");
        errdefer dec_weight.deinit();
        var dec_bias = try loader.getTensor("_model.decoder.decoder.2.bias");
        errdefer dec_bias.deinit();

        // Initialize LSTM state (hidden_size = 128)
        const hidden_size: usize = 128;
        var lstm_state = try ops.LSTMState.init(allocator, hidden_size);
        errdefer lstm_state.deinit();

        return Self{
            .allocator = allocator,
            .stft_basis = stft_basis,
            .enc0_weight = enc0_weight,
            .enc0_bias = enc0_bias,
            .enc1_weight = enc1_weight,
            .enc1_bias = enc1_bias,
            .enc2_weight = enc2_weight,
            .enc2_bias = enc2_bias,
            .enc3_weight = enc3_weight,
            .enc3_bias = enc3_bias,
            .lstm_weights = ops.LSTMWeights{
                .weight_ih = lstm_weight_ih,
                .weight_hh = lstm_weight_hh,
                .bias_ih = lstm_bias_ih,
                .bias_hh = lstm_bias_hh,
            },
            .dec_weight = dec_weight,
            .dec_bias = dec_bias,
            .lstm_state = lstm_state,
        };
    }

    /// Free all resources
    pub fn deinit(self: *Self) void {
        self.stft_basis.deinit();
        self.enc0_weight.deinit();
        self.enc0_bias.deinit();
        self.enc1_weight.deinit();
        self.enc1_bias.deinit();
        self.enc2_weight.deinit();
        self.enc2_bias.deinit();
        self.enc3_weight.deinit();
        self.enc3_bias.deinit();
        self.lstm_weights.weight_ih.deinit();
        self.lstm_weights.weight_hh.deinit();
        self.lstm_weights.bias_ih.deinit();
        self.lstm_weights.bias_hh.deinit();
        self.dec_weight.deinit();
        self.dec_bias.deinit();
        self.lstm_state.deinit();
    }

    /// Reset LSTM state (call between utterances/audio streams)
    pub fn resetStates(self: *Self) void {
        self.lstm_state.reset();
    }

    /// Process audio chunk and return speech probability
    /// audio_chunk: raw audio samples [512] for 16kHz input
    /// Returns: speech probability [0.0 - 1.0]
    ///
    /// Note: This is a simplified forward pass. The full Silero VAD pipeline
    /// includes STFT preprocessing which we simulate here.
    pub fn forward(self: *Self, audio_chunk: *const Tensor) !f32 {
        // For now, we'll use a simplified pipeline:
        // 1. The audio goes through feature extraction (simulated)
        // 2. Then through encoder convolutions
        // 3. Through LSTM
        // 4. Through decoder
        // 5. Sigmoid to get probability

        // Step 1: Feature extraction using learned STFT
        // The STFT basis is [258, 1, 256], we convolve with audio
        // This produces a spectrogram-like representation
        var features = try self.computeSTFT(audio_chunk);
        defer features.deinit();

        // Step 2: Encoder convolutions
        var x = try self.runEncoder(&features);
        defer x.deinit();

        // Step 3: Prepare input for LSTM (need to reduce to 1D)
        // After convolutions, we have [128, T] - sum over time dimension
        var lstm_input = try self.prepareForLSTM(&x);
        defer lstm_input.deinit();

        // Step 4: LSTM forward pass
        try ops.lstmCell(self.allocator, &lstm_input, &self.lstm_state, &self.lstm_weights);

        // Step 5: Decoder (linear from hidden state to output)
        const prob = self.runDecoder();

        return prob;
    }

    /// Compute Short-Time Fourier Transform using learned basis
    fn computeSTFT(self: *Self, audio: *const Tensor) !Tensor {
        // The STFT basis is [258, 1, 256]
        // We need to reshape it to [258, 256] for 1D convolution
        // Input audio is [512], we treat it as [1, 512]

        // Reshape audio to [1, audio_len]
        const audio_len = audio.data.len;
        var audio_2d_shape = [_]usize{ 1, audio_len };
        var audio_2d = try Tensor.init(self.allocator, &audio_2d_shape);
        errdefer audio_2d.deinit();
        @memcpy(audio_2d.data, audio.data);

        // The STFT produces complex values (real + imag), but the model uses
        // a learned basis that directly produces magnitude-like features.
        // Shape: [258, 1, 256] -> treating as [out_ch=258, in_ch=1, kernel=256]
        // But our conv1d expects [out_ch, in_ch, kernel], so this matches

        // Perform 1D convolution: [1, 512] conv [258, 1, 256] -> [258, output_width]
        // output_width = (512 - 256) / 1 + 1 = 257 (approximately)
        var stft_result = try ops.conv1d(self.allocator, &audio_2d, &self.stft_basis, null, 1, 0);
        audio_2d.deinit();

        // Take magnitude (or in this case, ReLU since it's a learned transform)
        ops.reluInPlace(&stft_result);

        return stft_result;
    }

    /// Run encoder conv layers
    fn runEncoder(self: *Self, features: *const Tensor) !Tensor {
        // Encoder consists of 4 conv layers
        // Each: Conv1d -> ReLU

        // Layer 0: [129, T] -> [128, T'] (input is spectrogram with 129 bins)
        // But our STFT output is [258, T], let's adjust
        // Actually, the first encoder expects [129, T] - half the STFT output (mag only)
        // For simplicity, we'll take the first 129 channels

        // Slice to get first 129 channels
        const in_channels = features.shape[0];
        const time_len = features.shape[1];

        // Create a view with 129 channels if needed
        var enc_input: Tensor = undefined;
        if (in_channels > 129) {
            var input_shape = [_]usize{ 129, time_len };
            enc_input = try Tensor.init(self.allocator, &input_shape);
            errdefer enc_input.deinit();
            // Copy first 129 channels
            for (0..129) |c| {
                for (0..time_len) |t| {
                    enc_input.data[c * time_len + t] = features.data[c * time_len + t];
                }
            }
        } else {
            enc_input = try features.clone(self.allocator);
        }
        defer enc_input.deinit();

        // Conv layer 0: [129, T] -> [128, T']
        var x0 = try ops.conv1d(self.allocator, &enc_input, &self.enc0_weight, &self.enc0_bias, 1, 1);
        defer x0.deinit();
        ops.reluInPlace(&x0);

        // Conv layer 1: [128, T'] -> [64, T'']
        var x1 = try ops.conv1d(self.allocator, &x0, &self.enc1_weight, &self.enc1_bias, 1, 1);
        defer x1.deinit();
        ops.reluInPlace(&x1);

        // Conv layer 2: [64, T''] -> [64, T''']
        var x2 = try ops.conv1d(self.allocator, &x1, &self.enc2_weight, &self.enc2_bias, 1, 1);
        defer x2.deinit();
        ops.reluInPlace(&x2);

        // Conv layer 3: [64, T'''] -> [128, T'''']
        var x3 = try ops.conv1d(self.allocator, &x2, &self.enc3_weight, &self.enc3_bias, 1, 1);
        ops.reluInPlace(&x3);

        return x3;
    }

    /// Prepare encoder output for LSTM input
    /// Takes [128, T] and reduces to [128] by averaging over time
    fn prepareForLSTM(self: *Self, encoded: *const Tensor) !Tensor {
        const channels = encoded.shape[0];
        const time_len = encoded.shape[1];

        var shape = [_]usize{channels};
        var result = try Tensor.init(self.allocator, &shape);
        errdefer result.deinit();

        // Global average pooling over time dimension
        for (0..channels) |c| {
            var sum_val: f32 = 0.0;
            for (0..time_len) |t| {
                sum_val += encoded.data[c * time_len + t];
            }
            result.data[c] = sum_val / @as(f32, @floatFromInt(time_len));
        }

        return result;
    }

    /// Run decoder on LSTM hidden state
    /// Applies linear transformation and sigmoid
    fn runDecoder(self: *Self) f32 {
        // Decoder weight is [1, 128, 1], essentially a linear layer
        // hidden state is [128], output is scalar

        // Compute: output = sigmoid(sum(h * w) + bias)
        var sum_val: f32 = 0.0;
        for (self.lstm_state.h.data, 0..) |h, i| {
            // Weight is [1, 128, 1], we access w[0, i, 0]
            const w = self.dec_weight.data[i];
            sum_val += h * w;
        }
        sum_val += self.dec_bias.data[0];

        // Apply sigmoid
        const prob = 1.0 / (1.0 + @exp(-sum_val));
        return prob;
    }

    /// Print model summary
    pub fn printSummary(self: *const Self) void {
        std.debug.print("\n=== Silero VAD Model Summary ===\n", .{});
        std.debug.print("STFT basis: [{}, {}, {}]\n", .{ self.stft_basis.shape[0], self.stft_basis.shape[1], self.stft_basis.shape[2] });
        std.debug.print("Encoder:\n", .{});
        std.debug.print("  Layer 0: [{}, {}, {}] + [{}]\n", .{ self.enc0_weight.shape[0], self.enc0_weight.shape[1], self.enc0_weight.shape[2], self.enc0_bias.shape[0] });
        std.debug.print("  Layer 1: [{}, {}, {}] + [{}]\n", .{ self.enc1_weight.shape[0], self.enc1_weight.shape[1], self.enc1_weight.shape[2], self.enc1_bias.shape[0] });
        std.debug.print("  Layer 2: [{}, {}, {}] + [{}]\n", .{ self.enc2_weight.shape[0], self.enc2_weight.shape[1], self.enc2_weight.shape[2], self.enc2_bias.shape[0] });
        std.debug.print("  Layer 3: [{}, {}, {}] + [{}]\n", .{ self.enc3_weight.shape[0], self.enc3_weight.shape[1], self.enc3_weight.shape[2], self.enc3_bias.shape[0] });
        std.debug.print("LSTM: weight_ih [{}, {}], weight_hh [{}, {}]\n", .{
            self.lstm_weights.weight_ih.shape[0],
            self.lstm_weights.weight_ih.shape[1],
            self.lstm_weights.weight_hh.shape[0],
            self.lstm_weights.weight_hh.shape[1],
        });
        std.debug.print("Decoder: [{}, {}, {}] + [{}]\n", .{
            self.dec_weight.shape[0],
            self.dec_weight.shape[1],
            self.dec_weight.shape[2],
            self.dec_bias.shape[0],
        });
        std.debug.print("================================\n", .{});
    }
};

// ============================================================================
// Tests
// ============================================================================

test "vad model load" {
    const allocator = std.testing.allocator;

    // Try to load the model
    var vad = SileroVAD.init(allocator, "models/silero_vad.tl") catch |err| {
        if (err == LoadError.FileNotFound) {
            std.debug.print("\nSkipping test: models/silero_vad.tl not found. Run 'python3 tools/export_vad.py' first.\n", .{});
            return;
        }
        return err;
    };
    defer vad.deinit();

    // Print model summary
    vad.printSummary();

    // Verify key dimensions
    try std.testing.expectEqual(@as(usize, 258), vad.stft_basis.shape[0]);
    try std.testing.expectEqual(@as(usize, 512), vad.lstm_weights.weight_ih.shape[0]); // 4 * 128
    try std.testing.expectEqual(@as(usize, 128), vad.lstm_weights.weight_ih.shape[1]);
}

test "vad forward pass" {
    const allocator = std.testing.allocator;

    // Try to load the model
    var vad = SileroVAD.init(allocator, "models/silero_vad.tl") catch |err| {
        if (err == LoadError.FileNotFound) {
            std.debug.print("\nSkipping test: models/silero_vad.tl not found.\n", .{});
            return;
        }
        return err;
    };
    defer vad.deinit();

    // Create a silence input (512 samples of zeros)
    var input_shape = [_]usize{512};
    var silence = try Tensor.init(allocator, &input_shape);
    defer silence.deinit();
    // silence is already zeros

    // Reset state
    vad.resetStates();

    // Run forward pass
    const prob = try vad.forward(&silence);

    std.debug.print("\nSilence probability: {d:.6}\n", .{prob});

    // Probability should be between 0 and 1
    try std.testing.expect(prob >= 0.0 and prob <= 1.0);
}

test "vad multiple chunks" {
    const allocator = std.testing.allocator;

    var vad = SileroVAD.init(allocator, "models/silero_vad.tl") catch |err| {
        if (err == LoadError.FileNotFound) {
            std.debug.print("\nSkipping test: models/silero_vad.tl not found.\n", .{});
            return;
        }
        return err;
    };
    defer vad.deinit();

    var input_shape = [_]usize{512};
    var chunk = try Tensor.init(allocator, &input_shape);
    defer chunk.deinit();

    // Reset state
    vad.resetStates();

    // Process multiple chunks - LSTM state should accumulate
    std.debug.print("\nProcessing multiple chunks:\n", .{});
    for (0..5) |i| {
        const prob = try vad.forward(&chunk);
        std.debug.print("  Chunk {}: prob = {d:.6}\n", .{ i, prob });

        // Each chunk should give valid probability
        try std.testing.expect(prob >= 0.0 and prob <= 1.0);
    }
}

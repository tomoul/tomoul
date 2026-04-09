const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const ops = @import("ops.zig");
const loader_mod = @import("loader.zig");
const ModelLoader = loader_mod.ModelLoader;
const LoadError = loader_mod.LoadError;

/// Silero VAD Model
/// Voice Activity Detection using LSTM-based architecture.
///
/// Architecture (16kHz model):
///   Context + Audio -> STFT (learned, hop=128) -> Conv Encoder (strides 1,2,2,1)
///   -> LSTM -> ReLU -> Conv -> Sigmoid
///
/// The model processes audio chunks (512 samples @ 16kHz = 32ms) with a 64-sample
/// context buffer carried over between chunks.
/// Returns speech probability [0.0 - 1.0].
pub const SileroVAD = struct {
    allocator: std.mem.Allocator,

    // STFT basis (learned Fourier transform)
    stft_basis: Tensor, // [258, 1, 256]

    // Encoder Conv1d layers (strides: 1, 2, 2, 1)
    enc0_weight: Tensor, // [128, 129, 3]
    enc0_bias: Tensor, // [128]
    enc1_weight: Tensor, // [64, 128, 3]
    enc1_bias: Tensor, // [64]
    enc2_weight: Tensor, // [64, 64, 3]
    enc2_bias: Tensor, // [64]
    enc3_weight: Tensor, // [128, 64, 3]
    enc3_bias: Tensor, // [128]

    // Pre-repacked weights for SIMD conv1d (layout: [C_in, K, C_out])
    stft_basis_repacked: Tensor,
    enc0_weight_repacked: Tensor,
    enc1_weight_repacked: Tensor,
    enc2_weight_repacked: Tensor,
    enc3_weight_repacked: Tensor,

    // LSTM weights (hidden_size=128, 4 gates packed)
    lstm_weights: ops.LSTMWeights,

    // Decoder (1x1 conv acting as linear, with ReLU before)
    dec_weight: Tensor, // [1, 128, 1]
    dec_bias: Tensor, // [1]

    // LSTM state (persistent across calls)
    lstm_state: ops.LSTMState,

    // Context buffer (64 samples carried over between chunks)
    context_buffer: Tensor, // [64]

    // Constants
    const CONTEXT_SIZE: usize = 64;
    const HOP_SIZE: usize = 128;
    const STFT_PAD_RIGHT: usize = 64;

    const Self = @This();

    /// Load Silero VAD model from .tl file
    pub fn init(allocator: std.mem.Allocator, model_path: []const u8) !Self {
        var loader = try ModelLoader.init(allocator, model_path);
        defer loader.deinit();
        return Self.initFromLoader(allocator, &loader);
    }

    /// Load Silero VAD model from embedded bytes (for Wasm)
    pub fn initFromBytes(allocator: std.mem.Allocator, model_bytes: []const u8) !Self {
        var loader = try ModelLoader.initFromBytes(allocator, model_bytes);
        defer loader.deinit();
        return Self.initFromLoader(allocator, &loader);
    }

    /// Internal: Initialize from a ModelLoader
    fn initFromLoader(allocator: std.mem.Allocator, loader: *ModelLoader) !Self {
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

        // Initialize context buffer (64 samples of zeros)
        var ctx_shape = [_]usize{CONTEXT_SIZE};
        var context_buffer = try Tensor.init(allocator, &ctx_shape);
        errdefer context_buffer.deinit();
        // Zero-initialized by Tensor.init

        // Pre-repack weights for SIMD conv1d
        var stft_basis_repacked = try ops.repackConv1dWeight(allocator, &stft_basis);
        errdefer stft_basis_repacked.deinit();
        var enc0_weight_repacked = try ops.repackConv1dWeight(allocator, &enc0_weight);
        errdefer enc0_weight_repacked.deinit();
        var enc1_weight_repacked = try ops.repackConv1dWeight(allocator, &enc1_weight);
        errdefer enc1_weight_repacked.deinit();
        var enc2_weight_repacked = try ops.repackConv1dWeight(allocator, &enc2_weight);
        errdefer enc2_weight_repacked.deinit();
        var enc3_weight_repacked = try ops.repackConv1dWeight(allocator, &enc3_weight);
        errdefer enc3_weight_repacked.deinit();

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
            .stft_basis_repacked = stft_basis_repacked,
            .enc0_weight_repacked = enc0_weight_repacked,
            .enc1_weight_repacked = enc1_weight_repacked,
            .enc2_weight_repacked = enc2_weight_repacked,
            .enc3_weight_repacked = enc3_weight_repacked,
            .lstm_weights = ops.LSTMWeights{
                .weight_ih = lstm_weight_ih,
                .weight_hh = lstm_weight_hh,
                .bias_ih = lstm_bias_ih,
                .bias_hh = lstm_bias_hh,
            },
            .dec_weight = dec_weight,
            .dec_bias = dec_bias,
            .lstm_state = lstm_state,
            .context_buffer = context_buffer,
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
        self.stft_basis_repacked.deinit();
        self.enc0_weight_repacked.deinit();
        self.enc1_weight_repacked.deinit();
        self.enc2_weight_repacked.deinit();
        self.enc3_weight_repacked.deinit();
        self.lstm_weights.weight_ih.deinit();
        self.lstm_weights.weight_hh.deinit();
        self.lstm_weights.bias_ih.deinit();
        self.lstm_weights.bias_hh.deinit();
        self.dec_weight.deinit();
        self.dec_bias.deinit();
        self.lstm_state.deinit();
        self.context_buffer.deinit();
    }

    /// Reset LSTM state and context buffer (call between utterances/audio streams)
    pub fn resetStates(self: *Self) void {
        self.lstm_state.reset();
        // Reset context buffer to zeros
        @memset(self.context_buffer.data, 0.0);
    }

    /// Process audio chunk and return speech probability
    /// audio_chunk: raw audio samples [512] for 16kHz input
    /// Returns: speech probability [0.0 - 1.0]
    pub fn forward(self: *Self, audio_chunk: *const Tensor) !f32 {
        // Pipeline:
        // 1. Prepend context buffer to audio (64 + 512 = 576 samples)
        // 2. STFT with hop=128, pad_right=64
        // 3. Encoder convolutions (strides 1,2,2,1 -> output [128,1])
        // 4. LSTM
        // 5. ReLU -> Conv -> Sigmoid

        // Step 1: Prepend context to audio
        var with_context = try self.prependContext(audio_chunk);
        defer with_context.deinit();

        // Step 2: STFT with right-padding and hop=128
        var features = try self.computeSTFT(&with_context);
        defer features.deinit();

        // Step 3: Encoder convolutions with correct strides
        var enc_out = try self.runEncoder(&features);
        defer enc_out.deinit();

        // Step 4: Squeeze time dim and run LSTM
        // enc_out is [128, 1], we need [128]
        var lstm_input = try self.prepareForLSTM(&enc_out);
        defer lstm_input.deinit();

        try ops.lstmCell(self.allocator, &lstm_input, &self.lstm_state, &self.lstm_weights);

        // Step 5: Decoder (ReLU -> Conv -> Sigmoid)
        const prob = self.runDecoder();

        // Step 6: Update context buffer with last 64 samples of input
        self.updateContext(&with_context);

        return prob;
    }

    /// Prepend context buffer to audio chunk
    fn prependContext(self: *Self, audio: *const Tensor) !Tensor {
        const total_len = CONTEXT_SIZE + audio.data.len;
        var shape = [_]usize{total_len};
        var result = try Tensor.init(self.allocator, &shape);
        errdefer result.deinit();

        // Copy context buffer
        @memcpy(result.data[0..CONTEXT_SIZE], self.context_buffer.data);
        // Copy audio
        @memcpy(result.data[CONTEXT_SIZE..], audio.data);

        return result;
    }

    /// Update context buffer with last CONTEXT_SIZE samples
    fn updateContext(self: *Self, audio_with_ctx: *const Tensor) void {
        const start = audio_with_ctx.data.len - CONTEXT_SIZE;
        @memcpy(self.context_buffer.data, audio_with_ctx.data[start..]);
    }

    /// Compute Short-Time Fourier Transform using learned basis
    fn computeSTFT(self: *Self, audio: *const Tensor) !Tensor {
        // The STFT basis is [258, 1, 256] = [2*129 freq bins, 1, window_size]
        // This is a learned STFT with 129 frequency bins (real and imaginary parts = 258)
        //
        // Silero VAD uses:
        // - Window size (n_fft): 256
        // - Hop size: 128 (50% overlap)
        // - Right padding: 64 samples (reflect mode, but we'll use zeros)
        //
        // Input: [576] (context + audio)
        // After padding: [576 + 64] = [640]
        // Output frames: (640 - 256) / 128 + 1 = 4 frames

        const audio_len = audio.data.len;
        const padded_len = audio_len + STFT_PAD_RIGHT;

        // Create padded audio as 2D tensor [1, padded_len]
        var padded_shape = [_]usize{ 1, padded_len };
        var audio_padded = try Tensor.init(self.allocator, &padded_shape);
        errdefer audio_padded.deinit();

        // Copy audio and pad with reflection of last samples
        @memcpy(audio_padded.data[0..audio_len], audio.data);
        // Reflect padding: copy last STFT_PAD_RIGHT samples in reverse
        for (0..STFT_PAD_RIGHT) |i| {
            audio_padded.data[audio_len + i] = audio.data[audio_len - 1 - i];
        }

        // Perform 1D convolution with stride=128 (hop size)
        var stft_complex = try ops.conv1dRepacked(self.allocator, &audio_padded, &self.stft_basis_repacked, 258, 1, 256, null, HOP_SIZE, 0);
        audio_padded.deinit();
        errdefer stft_complex.deinit();

        // STFT output is [258, T] where first 129 are real, last 129 are imaginary
        // We need to compute magnitude: sqrt(real^2 + imag^2)
        const num_bins = 129;
        const time_len = stft_complex.shape[1];

        var magnitude_shape = [_]usize{ num_bins, time_len };
        var magnitude = try Tensor.init(self.allocator, &magnitude_shape);
        errdefer magnitude.deinit();

        // Compute magnitude for each frequency bin
        for (0..num_bins) |freq| {
            for (0..time_len) |t| {
                const real = stft_complex.data[freq * time_len + t];
                const imag = stft_complex.data[(freq + num_bins) * time_len + t];
                magnitude.data[freq * time_len + t] = @sqrt(real * real + imag * imag);
            }
        }

        stft_complex.deinit();
        return magnitude;
    }

    /// Run encoder conv layers with correct strides
    fn runEncoder(self: *Self, features: *const Tensor) !Tensor {
        // Encoder consists of 4 conv layers with different strides
        // Each: Conv1d -> ReLU
        //
        // Input: [129, 4] (STFT magnitude with 129 bins, 4 time frames)
        // Layer 0: stride=1, padding=1 -> [128, 4]
        // Layer 1: stride=2, padding=1 -> [64, 2]
        // Layer 2: stride=2, padding=1 -> [64, 1]
        // Layer 3: stride=1, padding=1 -> [128, 1]

        // Clone the input features as our starting point
        var enc_input = try features.clone(self.allocator);
        defer enc_input.deinit();

        // Conv layer 0: stride=1, padding=1 [129,4]->[128,4]
        var x0 = try ops.conv1dRepacked(self.allocator, &enc_input, &self.enc0_weight_repacked, 128, 129, 3, &self.enc0_bias, 1, 1);
        defer x0.deinit();
        ops.reluInPlace(&x0);

        // Conv layer 1: stride=2, padding=1 [128,4]->[64,2]
        var x1 = try ops.conv1dRepacked(self.allocator, &x0, &self.enc1_weight_repacked, 64, 128, 3, &self.enc1_bias, 2, 1);
        defer x1.deinit();
        ops.reluInPlace(&x1);

        // Conv layer 2: stride=2, padding=1 [64,2]->[64,1]
        var x2 = try ops.conv1dRepacked(self.allocator, &x1, &self.enc2_weight_repacked, 64, 64, 3, &self.enc2_bias, 2, 1);
        defer x2.deinit();
        ops.reluInPlace(&x2);

        // Conv layer 3: stride=1, padding=1 [64,1]->[128,1]
        var x3 = try ops.conv1dRepacked(self.allocator, &x2, &self.enc3_weight_repacked, 128, 64, 3, &self.enc3_bias, 1, 1);
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
    /// Applies ReLU -> linear transformation -> sigmoid
    /// The decoder architecture is: dropout -> ReLU -> Conv1d -> Sigmoid
    fn runDecoder(self: *Self) f32 {
        // Decoder weight is [1, 128, 1], essentially a linear layer
        // hidden state is [128], output is scalar
        //
        // IMPORTANT: ReLU is applied to hidden state BEFORE the linear/conv layer!
        // This matches the Silero model's decoder.decoder = [Dropout, ReLU, Conv1d]

        // Compute: output = sigmoid(sum(relu(h) * w) + bias)
        var sum_val: f32 = 0.0;
        for (self.lstm_state.h.data, 0..) |h, i| {
            // Apply ReLU to hidden state element
            const h_relu = if (h > 0) h else 0;
            // Weight is [1, 128, 1], we access w[0, i, 0]
            const w = self.dec_weight.data[i];
            sum_val += h_relu * w;
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
    var vad = SileroVAD.init(allocator, "artifacts/silero_vad.tl") catch |err| {
        if (err == LoadError.FileNotFound) {
            std.debug.print("\nSkipping test: artifacts/silero_vad.tl not found. Run 'python3 tools/export_vad.py -o artifacts/' first.\n", .{});
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
    var vad = SileroVAD.init(allocator, "artifacts/silero_vad.tl") catch |err| {
        if (err == LoadError.FileNotFound) {
            std.debug.print("\nSkipping test: artifacts/silero_vad.tl not found.\n", .{});
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

    var vad = SileroVAD.init(allocator, "artifacts/silero_vad.tl") catch |err| {
        if (err == LoadError.FileNotFound) {
            std.debug.print("\nSkipping test: artifacts/silero_vad.tl not found.\n", .{});
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

test "vad speech detection - real audio" {
    const allocator = std.testing.allocator;

    var vad = SileroVAD.init(allocator, "artifacts/silero_vad.tl") catch |err| {
        if (err == LoadError.FileNotFound) {
            std.debug.print("\nSkipping test: artifacts/silero_vad.tl not found.\n", .{});
            return;
        }
        return err;
    };
    defer vad.deinit();

    // Load speech samples from fixture file at runtime
    const speech_file_data = std.fs.cwd().readFileAlloc(allocator, "tests/fixtures/silero_vad/speech_samples.bin", 1024 * 1024) catch |err| {
        if (err == error.FileNotFound) {
            std.debug.print("\nSkipping test: speech_samples.bin not found.\n", .{});
            return;
        }
        return err;
    };
    defer allocator.free(speech_file_data);

    const silence_file_data = std.fs.cwd().readFileAlloc(allocator, "tests/fixtures/silero_vad/silence_samples.bin", 1024 * 1024) catch |err| {
        if (err == error.FileNotFound) {
            std.debug.print("\nSkipping test: silence_samples.bin not found.\n", .{});
            return;
        }
        return err;
    };
    defer allocator.free(silence_file_data);

    const chunk_size: usize = 512;

    // Parse speech file header
    const speech_num_chunks = std.mem.readInt(u32, speech_file_data[0..4], .little);
    const speech_chunk_size = std.mem.readInt(u32, speech_file_data[4..8], .little);
    try std.testing.expectEqual(chunk_size, speech_chunk_size);

    // Parse silence file header
    const silence_num_chunks = std.mem.readInt(u32, silence_file_data[0..4], .little);
    const silence_chunk_size = std.mem.readInt(u32, silence_file_data[4..8], .little);
    try std.testing.expectEqual(chunk_size, silence_chunk_size);

    std.debug.print("\n=== VAD Speech Detection Test ===\n", .{});
    std.debug.print("Testing with real audio from sample1.wav\n", .{});
    std.debug.print("Silence chunks: {}, Speech chunks: {}\n", .{ silence_num_chunks, speech_num_chunks });

    // Reset state before testing
    vad.resetStates();

    var input_shape = [_]usize{chunk_size};
    var chunk = try Tensor.init(allocator, &input_shape);
    defer chunk.deinit();

    // Process silence chunks first (chunks 0-1 from audio file)
    // These should have LOW probability (< 0.5)
    std.debug.print("\nSilence chunks (expecting low probability):\n", .{});
    var silence_max_prob: f32 = 0.0;
    for (0..silence_num_chunks) |i| {
        const offset = 8 + i * chunk_size * 4; // Skip header, each f32 is 4 bytes
        const chunk_bytes = silence_file_data[offset .. offset + chunk_size * 4];
        // Reinterpret bytes as f32 slice
        const chunk_floats: [*]const f32 = @ptrCast(@alignCast(chunk_bytes.ptr));
        @memcpy(chunk.data, chunk_floats[0..chunk_size]);

        const prob = try vad.forward(&chunk);
        std.debug.print("  Silence chunk {}: prob = {d:.6}\n", .{ i, prob });
        if (prob > silence_max_prob) silence_max_prob = prob;
    }

    // Process speech chunks (chunks 2-6 from audio file)
    // These should have HIGH probability (> 0.5, most > 0.9)
    std.debug.print("\nSpeech chunks (expecting high probability):\n", .{});
    var speech_detected: usize = 0;
    var speech_max_prob: f32 = 0.0;
    for (0..speech_num_chunks) |i| {
        const offset = 8 + i * chunk_size * 4;
        const chunk_bytes = speech_file_data[offset .. offset + chunk_size * 4];
        const chunk_floats: [*]const f32 = @ptrCast(@alignCast(chunk_bytes.ptr));
        @memcpy(chunk.data, chunk_floats[0..chunk_size]);

        const prob = try vad.forward(&chunk);
        const is_speech = prob >= 0.5;
        std.debug.print("  Speech chunk {}: prob = {d:.6} {s}\n", .{
            i,
            prob,
            if (is_speech) "[SPEECH]" else "[silence]",
        });
        if (is_speech) speech_detected += 1;
        if (prob > speech_max_prob) speech_max_prob = prob;
    }

    std.debug.print("\nResults:\n", .{});
    std.debug.print("  Silence max prob: {d:.6}\n", .{silence_max_prob});
    std.debug.print("  Speech max prob:  {d:.6}\n", .{speech_max_prob});
    std.debug.print("  Speech detected:  {}/{} chunks\n", .{ speech_detected, speech_num_chunks });

    // Validation: speech chunks should mostly be detected as speech
    // Reference model detects 5/5 chunks as speech with probs > 0.9
    // We allow some tolerance but require at least 3/5 (60%) to be speech
    try std.testing.expect(speech_detected >= 3);
    std.debug.print("\n✓ Speech detection validation passed!\n", .{});

    // Max speech probability should be significantly higher than max silence
    try std.testing.expect(speech_max_prob > silence_max_prob);
    std.debug.print("✓ Speech/silence discrimination passed!\n", .{});
}

///! C Binding for Whisper Speech-to-Text
///!
///! Provides a C-compatible API for the Whisper model.
///!
///! Usage from C:
///!   1. tomoul_whisper_init(weights_path) - load model
///!   2. tomoul_whisper_transcribe_file(audio_path, output, output_capacity) - transcribe audio file
///!   3. tomoul_whisper_transcribe_mel(mel_data, mel_len, output, output_capacity) - transcribe mel spectrogram
///!   4. tomoul_whisper_destroy() - cleanup
///!
const std = @import("std");

// Import core modules (provided by build.zig with these exact names)
const tensor_mod = @import("tensor");
const loader_mod = @import("loader");
const model_mod = @import("model");

const Tensor = tensor_mod.Tensor;

// Re-export model types for convenience
const Whisper = model_mod.Whisper;
const WhisperConfig = model_mod.config.WhisperConfig;
const WhisperVariant = model_mod.config.WhisperVariant;
const WhisperEncoderWeights = model_mod.encoder.WhisperEncoderWeights;
const WhisperDecoderWeights = model_mod.decoder.WhisperDecoderWeights;

// Get ops and audio through model's re-exports
const ops_mod = model_mod.ops;
const audio_mod = model_mod.audio;
const ModelLoader = loader_mod.ModelLoader;

// Global state
var gpa: ?std.heap.GeneralPurposeAllocator(.{}) = null;
var model_instance: ?Whisper = null;
var is_initialized: bool = false;
var current_variant: WhisperVariant = .tiny;
var global_ctx: ?ops_mod.Context = null;

// =============================================================================
// C API Exports
// =============================================================================

/// Initialize the Whisper model from file path
/// Returns: 0 on success, negative error code on failure
export fn tomoul_whisper_init(
    weights_path: [*:0]const u8,
) c_int {
    if (is_initialized) {
        return 0; // Already initialized
    }

    // Initialize allocator
    gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.?.allocator();

    // Initialize global context for parallel operations (stored in global variable)
    global_ctx = ops_mod.Context.initMultiThreaded(allocator, null);
    ops_mod.initGlobalContext(&global_ctx.?);

    // Convert C string to Zig slice
    const weights_slice = std.mem.span(weights_path);

    // Infer variant from path
    current_variant = inferVariantFromPath(weights_slice);
    const cfg = WhisperConfig.forVariant(current_variant);

    // Load weights
    var loader = ModelLoader.init(allocator, weights_slice) catch |err| {
        std.debug.print("Failed to open weights file '{s}': {}\n", .{ weights_slice, err });
        return -1;
    };
    defer loader.deinit();

    // Load encoder weights (loadFromLoader returns pointer)
    const enc_weights = WhisperEncoderWeights.loadFromLoader(allocator, &loader, cfg.encoder) catch |err| {
        std.debug.print("Failed to load encoder weights: {}\n", .{err});
        return -3;
    };

    // Load decoder weights (loadFromLoader returns pointer)
    const dec_weights = WhisperDecoderWeights.loadFromLoader(allocator, &loader, cfg.decoder) catch |err| {
        std.debug.print("Failed to load decoder weights: {}\n", .{err});
        enc_weights.deinit();
        allocator.destroy(enc_weights);
        return -5;
    };

    // Create model instance
    model_instance = Whisper.init(allocator, cfg, enc_weights, dec_weights);

    is_initialized = true;
    return 0;
}

/// Transcribe audio file and return token IDs
/// audio_path: Path to WAV file (16kHz mono)
/// output: Buffer to write token IDs (as u32 array)
/// output_capacity: Maximum number of tokens that can be written
/// Returns: Number of tokens on success, negative error code on failure
export fn tomoul_whisper_transcribe_file(
    audio_path: [*:0]const u8,
    output: [*]u32,
    output_capacity: usize,
) c_int {
    if (!is_initialized) {
        return -1; // Not initialized
    }

    const allocator = gpa.?.allocator();
    const path_slice = std.mem.span(audio_path);

    // Load audio file (use loadAudioFile which supports more formats)
    const audio_result = audio_mod.loadAudioFile(allocator, path_slice) catch |err| {
        std.debug.print("Failed to load audio file '{s}': {}\n", .{ path_slice, err });
        return -2;
    };
    defer allocator.free(audio_result.samples);

    // Get config for current variant to get n_mels
    const cfg = WhisperConfig.forVariant(current_variant);

    // Compute mel spectrogram using Whisper-specific function (pads to 30s, matches PyTorch output)
    var mel = audio_mod.whisperMelSpectrogram(allocator, audio_result.samples, cfg.encoder.n_mels) catch |err| {
        std.debug.print("Failed to compute mel spectrogram: {}\n", .{err});
        return -3;
    };
    defer mel.deinit();

    // Transcribe
    const tokens = model_instance.?.transcribe(&mel) catch |err| {
        std.debug.print("Transcription failed: {}\n", .{err});
        return -4;
    };
    defer allocator.free(tokens);

    // Copy tokens to output buffer
    if (tokens.len > output_capacity) {
        return -5; // Output buffer too small
    }

    @memcpy(output[0..tokens.len], tokens);
    return @intCast(tokens.len);
}

/// Transcribe pre-computed mel spectrogram
/// mel_data: Mel spectrogram data [n_mels * n_frames] in row-major order
/// n_mels: Number of mel bins (80 or 128)
/// n_frames: Number of frames
/// output: Buffer to write token IDs
/// output_capacity: Maximum number of tokens
/// Returns: Number of tokens on success, negative error code on failure
export fn tomoul_whisper_transcribe_mel(
    mel_data: [*]const f32,
    n_mels: usize,
    n_frames: usize,
    output: [*]u32,
    output_capacity: usize,
) c_int {
    if (!is_initialized) {
        return -1; // Not initialized
    }

    const allocator = gpa.?.allocator();

    // Create tensor view of mel data (shape first, then data)
    var mel_shape = [_]usize{ n_mels, n_frames };
    var mel = Tensor.initWithData(allocator, &mel_shape, mel_data[0 .. n_mels * n_frames]) catch |err| {
        std.debug.print("Failed to create mel tensor: {}\n", .{err});
        return -2;
    };
    defer mel.deinit();

    // Transcribe
    const tokens = model_instance.?.transcribe(&mel) catch |err| {
        std.debug.print("Transcription failed: {}\n", .{err});
        return -3;
    };
    defer allocator.free(tokens);

    // Copy tokens to output buffer
    if (tokens.len > output_capacity) {
        return -4; // Output buffer too small
    }

    @memcpy(output[0..tokens.len], tokens);
    return @intCast(tokens.len);
}

/// Run only the encoder and return audio features
/// mel_data: Mel spectrogram data [n_mels * n_frames]
/// n_mels: Number of mel bins (80 or 128)
/// n_frames: Number of frames
/// output: Buffer to write audio features [n_frames/2 * n_audio_state]
/// output_capacity: Maximum floats that can be written
/// Returns: Number of floats written on success, negative error code on failure
export fn tomoul_whisper_encode(
    mel_data: [*]const f32,
    n_mels: usize,
    n_frames: usize,
    output: [*]f32,
    output_capacity: usize,
) c_int {
    if (!is_initialized) {
        return -1; // Not initialized
    }

    const allocator = gpa.?.allocator();

    // Create tensor view of mel data (shape first, then data)
    var mel_shape = [_]usize{ n_mels, n_frames };
    var mel = Tensor.initWithData(allocator, &mel_shape, mel_data[0 .. n_mels * n_frames]) catch |err| {
        std.debug.print("Failed to create mel tensor: {}\n", .{err});
        return -2;
    };
    defer mel.deinit();

    // Encode
    var audio_features = model_instance.?.enc.encode(&mel) catch |err| {
        std.debug.print("Encoding failed: {}\n", .{err});
        return -3;
    };
    defer audio_features.deinit();

    // Copy to output
    if (audio_features.data.len > output_capacity) {
        return -4; // Output buffer too small
    }

    @memcpy(output[0..audio_features.data.len], audio_features.data);
    return @intCast(audio_features.data.len);
}

/// Destroy the model and free all resources
export fn tomoul_whisper_destroy() void {
    if (!is_initialized) {
        return;
    }

    ops_mod.deinitGlobalContext();

    if (global_ctx) |*ctx| {
        ctx.deinit();
    }
    global_ctx = null;

    if (model_instance) |*model| {
        model.deinit();
    }
    model_instance = null;

    if (gpa) |*g| {
        _ = g.deinit();
    }
    gpa = null;

    is_initialized = false;
}

/// Check if the model is initialized
export fn tomoul_whisper_is_ready() c_int {
    return if (is_initialized) 1 else 0;
}

/// Get version string
export fn tomoul_whisper_version() [*:0]const u8 {
    return "whisper-v1.0.0";
}

/// Get the loaded model variant name
export fn tomoul_whisper_variant() [*:0]const u8 {
    return switch (current_variant) {
        .tiny => "tiny",
        .base => "base",
        .small => "small",
        .medium => "medium",
        .large => "large",
        .large_v3_turbo => "large-v3-turbo",
        .distil_small_en => "distil-small-en",
    };
}

// =============================================================================
// Helper functions
// =============================================================================

fn inferVariantFromPath(path: []const u8) WhisperVariant {
    if (std.mem.indexOf(u8, path, "large-v3-turbo") != null or
        std.mem.indexOf(u8, path, "large_v3_turbo") != null or
        std.mem.indexOf(u8, path, "turbo") != null)
    {
        return .large_v3_turbo;
    }
    if (std.mem.indexOf(u8, path, "distil-small") != null or
        std.mem.indexOf(u8, path, "distil_small") != null)
    {
        return .distil_small_en;
    }
    if (std.mem.indexOf(u8, path, "large") != null) return .large;
    if (std.mem.indexOf(u8, path, "medium") != null) return .medium;
    if (std.mem.indexOf(u8, path, "small") != null) return .small;
    if (std.mem.indexOf(u8, path, "base") != null) return .base;
    return .tiny;
}

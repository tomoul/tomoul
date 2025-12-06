// src/models/whisper/cli.zig
// CLI for Whisper Speech-to-Text model
//
// Usage:
//   zig build -Dmodel=whisper-tiny run -- transcribe <audio.wav>  # Transcribe audio
//   zig build -Dmodel=whisper-tiny run -- transcribe-mel <mel.tl>  # Transcribe pre-computed mel
//   zig build -Dmodel=whisper-tiny run -- validate-encoder  # Validate encoder
//

const std = @import("std");

// Use module imports from build.zig
const Tensor = @import("tensor.zig").Tensor;
const loader_mod = @import("loader.zig");
const ModelLoader = loader_mod.ModelLoader;
const LoadError = loader_mod.LoadError;
const audio = @import("audio.zig");

// Import whisper modules via model.zig to ensure single module
const model = @import("model.zig");
const Whisper = model.Whisper;
const WhisperConfig = model.config.WhisperConfig;
const WhisperTokens = model.config.WhisperTokens;
const defaultPromptTokens = model.config.defaultPromptTokens;
const WhisperEncoder = model.encoder.WhisperEncoder;
const WhisperEncoderWeights = model.encoder.WhisperEncoderWeights;
const WhisperDecoder = model.decoder.WhisperDecoder;
const WhisperDecoderWeights = model.decoder.WhisperDecoderWeights;

const WhisperVariant = model.config.WhisperVariant;

const VERSION = "0.1.0";
const DEFAULT_WEIGHTS_PATH = "models/whisper_tiny.tl";
const DEFAULT_VARIANT: WhisperVariant = .tiny;

/// Infer the model variant from the weights file path
fn inferVariantFromPath(path: []const u8) WhisperVariant {
    // Check for known patterns in the filename (order matters - check specific patterns first)
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
    // Default to tiny
    return .tiny;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    // Parse command
    if (args.len < 2) {
        printUsage();
        return;
    }

    const command = args[1];

    // Parse options from remaining args
    var weights_path: []const u8 = DEFAULT_WEIGHTS_PATH;
    var explicit_variant: ?WhisperVariant = null;
    var positional_args = std.ArrayListUnmanaged([]const u8){};
    defer positional_args.deinit(allocator);

    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--weights") or std.mem.eql(u8, args[i], "-w")) {
            if (i + 1 < args.len) {
                weights_path = args[i + 1];
                i += 1;
            } else {
                std.debug.print("Error: --weights requires a path argument\n", .{});
                return;
            }
        } else if (std.mem.eql(u8, args[i], "--model") or std.mem.eql(u8, args[i], "-m")) {
            if (i + 1 < args.len) {
                if (WhisperVariant.fromString(args[i + 1])) |v| {
                    explicit_variant = v;
                } else {
                    std.debug.print("Error: Unknown model variant: {s}\n", .{args[i + 1]});
                    std.debug.print("Valid variants: tiny, base, small, medium, large, large-v3-turbo (or turbo)\n", .{});
                    return;
                }
                i += 1;
            } else {
                std.debug.print("Error: --model requires a variant name\n", .{});
                return;
            }
        } else {
            try positional_args.append(allocator, args[i]);
        }
    }

    // Use explicit variant if provided, otherwise infer from weights path
    const variant = explicit_variant orelse inferVariantFromPath(weights_path);

    if (std.mem.eql(u8, command, "validate-encoder")) {
        try validateEncoderWithWeightsAndVariant(allocator, weights_path, variant);
    } else if (std.mem.eql(u8, command, "validate-decoder")) {
        try validateDecoderWithWeightsAndVariant(allocator, weights_path, variant);
    } else if (std.mem.eql(u8, command, "transcribe")) {
        if (positional_args.items.len < 1) {
            std.debug.print("Error: transcribe requires an audio file\n", .{});
            std.debug.print("Usage: whisper transcribe <audio.wav> [--weights path] [--model variant]\n", .{});
            return;
        }
        try transcribeAudioWithWeightsAndVariant(allocator, positional_args.items[0], weights_path, variant);
    } else if (std.mem.eql(u8, command, "transcribe-mel")) {
        if (positional_args.items.len < 1) {
            std.debug.print("Error: transcribe-mel requires a mel spectrogram file\n", .{});
            std.debug.print("Usage: whisper transcribe-mel <mel.tl> [--weights path] [--model variant]\n", .{});
            return;
        }
        try transcribeMelWithWeightsAndVariant(allocator, positional_args.items[0], weights_path, variant);
    } else if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "-h") or std.mem.eql(u8, command, "--help")) {
        printUsage();
    } else if (std.mem.eql(u8, command, "version") or std.mem.eql(u8, command, "-v") or std.mem.eql(u8, command, "--version")) {
        std.debug.print("Whisper CLI v{s}\n", .{VERSION});
    } else {
        std.debug.print("Unknown command: {s}\n", .{command});
        printUsage();
    }
}

// Legacy wrappers for backwards compatibility
fn validateEncoder(allocator: std.mem.Allocator) !void {
    return validateEncoderWithWeightsAndVariant(allocator, DEFAULT_WEIGHTS_PATH, DEFAULT_VARIANT);
}

fn validateDecoder(allocator: std.mem.Allocator) !void {
    return validateDecoderWithWeightsAndVariant(allocator, DEFAULT_WEIGHTS_PATH, DEFAULT_VARIANT);
}

fn transcribeAudio(allocator: std.mem.Allocator, audio_path: []const u8) !void {
    return transcribeAudioWithWeightsAndVariant(allocator, audio_path, DEFAULT_WEIGHTS_PATH, DEFAULT_VARIANT);
}

fn transcribeMel(allocator: std.mem.Allocator, mel_path: []const u8) !void {
    return transcribeMelWithWeightsAndVariant(allocator, mel_path, DEFAULT_WEIGHTS_PATH, DEFAULT_VARIANT);
}

fn validateEncoderWithWeights(allocator: std.mem.Allocator, weights_path: []const u8) !void {
    return validateEncoderWithWeightsAndVariant(allocator, weights_path, DEFAULT_VARIANT);
}

fn validateDecoderWithWeights(allocator: std.mem.Allocator, weights_path: []const u8) !void {
    return validateDecoderWithWeightsAndVariant(allocator, weights_path, DEFAULT_VARIANT);
}

fn transcribeAudioWithWeights(allocator: std.mem.Allocator, audio_path: []const u8, weights_path: []const u8) !void {
    return transcribeAudioWithWeightsAndVariant(allocator, audio_path, weights_path, DEFAULT_VARIANT);
}

fn transcribeMelWithWeights(allocator: std.mem.Allocator, mel_path: []const u8, weights_path: []const u8) !void {
    return transcribeMelWithWeightsAndVariant(allocator, mel_path, weights_path, DEFAULT_VARIANT);
}

fn printUsage() void {
    std.debug.print(
        \\Whisper Speech-to-Text CLI
        \\
        \\Usage:
        \\  whisper <command> [options]
        \\
        \\Commands:
        \\  transcribe <audio.wav>  Transcribe audio file (WAV format)
        \\  transcribe-mel <mel.tl> Transcribe from pre-computed mel spectrogram
        \\  validate-encoder        Validate encoder output against PyTorch reference
        \\  validate-decoder        Validate decoder output against PyTorch reference
        \\  help                    Show this help message
        \\  version                 Show version
        \\
        \\Options:
        \\  --weights, -w <path>    Path to model weights (variant auto-detected from filename)
        \\  --model, -m <variant>   Override variant (tiny, base, small, medium, large, turbo, distil-small-en)
        \\
        \\Examples:
        \\  zig build -Dmodel=whisper-tiny run -- transcribe audio.wav
        \\  zig build -Dmodel=whisper-tiny run -- transcribe audio.wav --weights models/whisper_tiny_q8_0.tl
        \\  zig build -Dmodel=whisper-tiny run -- transcribe audio.wav --weights models/whisper_large-v3-turbo_q8_0.tl
        \\  zig build -Dmodel=whisper-tiny run -- validate-encoder
        \\
    , .{});
}

fn validateEncoderWithWeightsAndVariant(allocator: std.mem.Allocator, weights_path: []const u8, variant: WhisperVariant) !void {
    std.debug.print("\n=== Whisper Encoder Validation ({s}) ===\n\n", .{variant.name()});

    const cfg = WhisperConfig.forVariant(variant);

    // Load validation fixture
    std.debug.print("Loading validation fixture...\n", .{});
    var fixture_loader = ModelLoader.init(allocator, "models/whisper_tiny_fixture.tl") catch |err| {
        if (err == LoadError.FileNotFound) {
            std.debug.print("Error: Fixture not found at models/whisper_tiny_fixture.tl\n", .{});
            std.debug.print("Run: python3 tools/validate_whisper.py --create-fixture\n", .{});
            return;
        }
        return err;
    };
    defer fixture_loader.deinit();

    // Load model weights
    std.debug.print("Loading model weights from {s}...\n", .{weights_path});
    var model_loader = ModelLoader.init(allocator, weights_path) catch |err| {
        if (err == LoadError.FileNotFound) {
            std.debug.print("Error: Model not found at {s}\n", .{weights_path});
            std.debug.print("Run: python3 tools/export_whisper.py -m tiny\n", .{});
            return;
        }
        return err;
    };
    defer model_loader.deinit();

    // Load Mel spectrogram input
    std.debug.print("Loading Mel spectrogram...\n", .{});
    var mel = try fixture_loader.getTensor("mel");
    defer mel.deinit();
    std.debug.print("  Mel shape: [{d}, {d}]\n", .{ mel.shape[0], mel.shape[1] });

    // Load expected encoder output
    std.debug.print("Loading expected encoder output...\n", .{});
    var expected_output = try fixture_loader.getTensor("encoder_output");
    defer expected_output.deinit();
    std.debug.print("  Expected shape: [{d}, {d}]\n", .{ expected_output.shape[0], expected_output.shape[1] });

    // Load encoder weights
    std.debug.print("Loading encoder weights ({d} layers)...\n", .{cfg.encoder.n_audio_layer});
    var enc_weights = try WhisperEncoderWeights.loadFromLoader(allocator, &model_loader, cfg.encoder);
    defer enc_weights.deinit();

    // Create encoder and run forward pass
    std.debug.print("Running encoder forward pass...\n", .{});
    const encoder = WhisperEncoder.init(allocator, cfg.encoder, enc_weights);
    var encoder_output = try encoder.encode(&mel);
    defer encoder_output.deinit();

    std.debug.print("  Output shape: [{d}, {d}]\n", .{ encoder_output.shape[0], encoder_output.shape[1] });

    // Verify shapes match
    if (expected_output.shape.len != encoder_output.shape.len) {
        std.debug.print("\nFAIL: Shape dimension mismatch!\n", .{});
        return;
    }
    for (expected_output.shape, encoder_output.shape) |exp, act| {
        if (exp != act) {
            std.debug.print("\nFAIL: Shape mismatch - expected {d}, got {d}\n", .{ exp, act });
            return;
        }
    }

    // Compare outputs element-wise
    // Use 0.025 tolerance to account for floating-point differences between
    // Zig and PyTorch (order of operations, erf approximation, etc.)
    const tolerance: f32 = 0.025;
    var max_diff: f32 = 0.0;
    var mean_diff: f32 = 0.0;
    var num_errors: usize = 0;

    for (expected_output.data, encoder_output.data) |expected, actual| {
        const diff = @abs(expected - actual);
        max_diff = @max(max_diff, diff);
        mean_diff += diff;
        if (diff > tolerance) {
            num_errors += 1;
        }
    }
    mean_diff /= @as(f32, @floatFromInt(encoder_output.data.len));

    std.debug.print("\n=== Validation Results ===\n", .{});
    std.debug.print("Total elements: {d}\n", .{encoder_output.data.len});
    std.debug.print("Max diff:       {d:.6}\n", .{max_diff});
    std.debug.print("Mean diff:      {d:.6}\n", .{mean_diff});
    std.debug.print("Tolerance:      {e}\n", .{tolerance});
    std.debug.print("Errors:         {d} ({d:.2}%)\n", .{
        num_errors,
        @as(f32, @floatFromInt(num_errors)) / @as(f32, @floatFromInt(encoder_output.data.len)) * 100.0,
    });

    if (max_diff < tolerance) {
        std.debug.print("\nPASS: Encoder output matches PyTorch reference!\n", .{});
    } else {
        std.debug.print("\nFAIL: Encoder output differs from PyTorch reference.\n", .{});
    }
}

fn validateDecoderWithWeightsAndVariant(allocator: std.mem.Allocator, weights_path: []const u8, variant: WhisperVariant) !void {
    std.debug.print("\n=== Whisper Decoder Validation ({s}) ===\n\n", .{variant.name()});

    const cfg = WhisperConfig.forVariant(variant);

    // Load validation fixture
    std.debug.print("Loading validation fixture...\n", .{});
    var fixture_loader = ModelLoader.init(allocator, "models/whisper_tiny_fixture.tl") catch |err| {
        if (err == LoadError.FileNotFound) {
            std.debug.print("Error: Fixture not found at models/whisper_tiny_fixture.tl\n", .{});
            std.debug.print("Run: python3 tools/validate_whisper.py --create-fixture\n", .{});
            return;
        }
        return err;
    };
    defer fixture_loader.deinit();

    // Load model weights
    std.debug.print("Loading model weights from {s}...\n", .{weights_path});
    var model_loader = ModelLoader.init(allocator, weights_path) catch |err| {
        if (err == LoadError.FileNotFound) {
            std.debug.print("Error: Model not found at {s}\n", .{weights_path});
            std.debug.print("Run: python3 tools/export_whisper.py -m tiny\n", .{});
            return;
        }
        return err;
    };
    defer model_loader.deinit();

    // Load encoder output (input to decoder)
    std.debug.print("Loading encoder output...\n", .{});
    var encoder_output = try fixture_loader.getTensor("encoder_output");
    defer encoder_output.deinit();
    std.debug.print("  Encoder output shape: [{d}, {d}]\n", .{ encoder_output.shape[0], encoder_output.shape[1] });

    // Load expected decoder logits
    std.debug.print("Loading expected decoder logits...\n", .{});
    var expected_logits = try fixture_loader.getTensor("decoder_logits");
    defer expected_logits.deinit();
    std.debug.print("  Expected logits shape: [{d}, {d}]\n", .{ expected_logits.shape[0], expected_logits.shape[1] });

    // Load decoder weights
    std.debug.print("Loading decoder weights ({d} layers)...\n", .{cfg.decoder.n_text_layer});
    var dec_weights = try WhisperDecoderWeights.loadFromLoader(allocator, &model_loader, cfg.decoder);
    defer dec_weights.deinit();

    // Create decoder and run forward pass with prompt tokens
    std.debug.print("Running decoder forward pass with prompt tokens...\n", .{});
    const decoder = WhisperDecoder.init(allocator, cfg.decoder, dec_weights);

    // Prompt tokens: [SOT, LANG_EN, TRANSCRIBE, NO_TIMESTAMPS]
    const prompt_tokens = [_]u32{
        WhisperTokens.SOT,
        WhisperTokens.LANG_EN,
        WhisperTokens.TRANSCRIBE,
        WhisperTokens.NO_TIMESTAMPS,
    };

    var decoder_logits = try decoder.forward(&prompt_tokens, &encoder_output);
    defer decoder_logits.deinit();

    std.debug.print("  Output logits shape: [{d}, {d}]\n", .{ decoder_logits.shape[0], decoder_logits.shape[1] });

    // Compare last row of logits (prediction for next token)
    const seq_len = prompt_tokens.len;
    const vocab_size = cfg.decoder.n_vocab;

    const expected_last = expected_logits.data[(seq_len - 1) * vocab_size ..][0..vocab_size];
    const actual_last = decoder_logits.data[(seq_len - 1) * vocab_size ..][0..vocab_size];

    // Compare argmax (predicted token)
    var expected_token: u32 = 0;
    var expected_max: f32 = expected_last[0];
    for (expected_last[1..], 1..) |v, i| {
        if (v > expected_max) {
            expected_max = v;
            expected_token = @intCast(i);
        }
    }

    var actual_token: u32 = 0;
    var actual_max: f32 = actual_last[0];
    for (actual_last[1..], 1..) |v, i| {
        if (v > actual_max) {
            actual_max = v;
            actual_token = @intCast(i);
        }
    }

    std.debug.print("\n=== Token Prediction ===\n", .{});
    std.debug.print("Expected token: {d}\n", .{expected_token});
    std.debug.print("Actual token:   {d}\n", .{actual_token});

    // Also compare logit values
    var max_diff: f32 = 0.0;
    for (expected_last, actual_last) |exp, act| {
        const diff = @abs(exp - act);
        max_diff = @max(max_diff, diff);
    }

    std.debug.print("Max logit diff: {d:.6}\n", .{max_diff});

    if (expected_token == actual_token) {
        std.debug.print("\nPASS: Decoder produces same token prediction!\n", .{});
    } else {
        std.debug.print("\nFAIL: Token mismatch.\n", .{});
    }
}

/// Transcribe audio file directly (loads WAV, computes mel spectrogram in Zig)
fn transcribeAudioWithWeightsAndVariant(allocator: std.mem.Allocator, audio_path: []const u8, weights_path: []const u8, variant: WhisperVariant) !void {
    std.debug.print("\n=== Whisper Audio Transcription ({s}) ===\n\n", .{variant.name()});

    const cfg = WhisperConfig.forVariant(variant);

    // Load audio file
    std.debug.print("Loading audio: {s}\n", .{audio_path});
    var timer = try std.time.Timer.start();
    const audio_result = audio.loadAudioFile(allocator, audio_path) catch |err| {
        std.debug.print("Error loading audio file: {}\n", .{err});
        std.debug.print("Supported formats: WAV (16/24/32-bit PCM, 32-bit float)\n", .{});
        return;
    };
    defer allocator.free(audio_result.samples);
    const load_time = @as(f64, @floatFromInt(timer.read())) / 1_000_000_000.0;
    std.debug.print("  Loaded {d} samples ({d:.2}s audio) in {d:.3}s\n", .{
        audio_result.samples.len,
        @as(f64, @floatFromInt(audio_result.samples.len)) / @as(f64, @floatFromInt(audio.SAMPLE_RATE)),
        load_time,
    });

    // Compute mel spectrogram
    std.debug.print("Computing mel spectrogram...\n", .{});
    timer.reset();
    var mel = try audio.whisperMelSpectrogram(allocator, audio_result.samples, cfg.encoder.n_mels);
    defer mel.deinit();
    const mel_time = @as(f64, @floatFromInt(timer.read())) / 1_000_000_000.0;
    std.debug.print("  Mel shape: [{d}, {d}] in {d:.3}s\n", .{ mel.shape[0], mel.shape[1], mel_time });

    // Debug: show mel statistics
    var mel_min: f32 = mel.data[0];
    var mel_max: f32 = mel.data[0];
    var mel_sum: f64 = 0.0;
    for (mel.data) |v| {
        mel_min = @min(mel_min, v);
        mel_max = @max(mel_max, v);
        mel_sum += v;
    }
    const mel_mean: f32 = @floatCast(mel_sum / @as(f64, @floatFromInt(mel.data.len)));
    std.debug.print("  Mel range: [{d:.4}, {d:.4}], mean: {d:.4}\n", .{ mel_min, mel_max, mel_mean });
    std.debug.print("  mel[0, :5]: [{d:.4}, {d:.4}, {d:.4}, {d:.4}, {d:.4}]\n", .{
        mel.data[0],
        mel.data[1],
        mel.data[2],
        mel.data[3],
        mel.data[4],
    });

    // Load model weights
    std.debug.print("Loading model weights from {s}...\n", .{weights_path});
    var model_loader = ModelLoader.init(allocator, weights_path) catch |err| {
        if (err == LoadError.FileNotFound) {
            std.debug.print("Error: Model not found at {s}\n", .{weights_path});
            std.debug.print("Run: python3 tools/export_whisper.py -m tiny\n", .{});
            return;
        }
        return err;
    };
    defer model_loader.deinit();

    // Load encoder
    std.debug.print("Loading encoder ({d} layers)...\n", .{cfg.encoder.n_audio_layer});
    var enc_weights = try WhisperEncoderWeights.loadFromLoader(allocator, &model_loader, cfg.encoder);
    defer enc_weights.deinit();
    const encoder = WhisperEncoder.init(allocator, cfg.encoder, enc_weights);

    // Load decoder
    std.debug.print("Loading decoder ({d} layers)...\n", .{cfg.decoder.n_text_layer});
    var dec_weights = try WhisperDecoderWeights.loadFromLoader(allocator, &model_loader, cfg.decoder);
    defer dec_weights.deinit();
    const decoder = WhisperDecoder.init(allocator, cfg.decoder, dec_weights);

    // Run encoder
    std.debug.print("\nRunning encoder...\n", .{});
    timer.reset();
    var encoder_output = try encoder.encode(&mel);
    defer encoder_output.deinit();
    const encode_time = @as(f64, @floatFromInt(timer.read())) / 1_000_000_000.0;
    std.debug.print("  Encoder time: {d:.3}s\n", .{encode_time});
    std.debug.print("  Output shape: [{d}, {d}]\n", .{ encoder_output.shape[0], encoder_output.shape[1] });

    // Run greedy decoding
    std.debug.print("\nRunning greedy decoding (with full KV cache)...\n", .{});
    timer.reset();
    const prompt = defaultPromptTokens();
    const tokens = try decoder.greedyDecodeFullCache(&encoder_output, &prompt, 224);
    defer allocator.free(tokens);
    const decode_time = @as(f64, @floatFromInt(timer.read())) / 1_000_000_000.0;

    std.debug.print("  Decode time: {d:.3}s\n", .{decode_time});
    std.debug.print("  Generated {d} tokens\n", .{tokens.len - prompt.len});

    // Print token IDs
    std.debug.print("\n=== Generated Token IDs ===\n", .{});
    std.debug.print("Prompt: ", .{});
    for (prompt) |t| {
        std.debug.print("{d} ", .{t});
    }
    std.debug.print("\nGenerated: ", .{});
    for (tokens[prompt.len..]) |t| {
        std.debug.print("{d} ", .{t});
    }
    std.debug.print("\n", .{});

    // Print timing summary
    std.debug.print("\n=== Timing Summary ===\n", .{});
    std.debug.print("Audio load:   {d:.3}s\n", .{load_time});
    std.debug.print("Mel compute:  {d:.3}s\n", .{mel_time});
    std.debug.print("Encoder:      {d:.3}s\n", .{encode_time});
    std.debug.print("Decoder:      {d:.3}s\n", .{decode_time});
    std.debug.print("Total:        {d:.3}s\n", .{load_time + mel_time + encode_time + decode_time});
}

/// Transcribe from pre-computed mel spectrogram file
fn transcribeMelWithWeightsAndVariant(allocator: std.mem.Allocator, mel_path: []const u8, weights_path: []const u8, variant: WhisperVariant) !void {
    std.debug.print("\n=== Whisper Transcription ({s}, from mel) ===\n\n", .{variant.name()});

    const cfg = WhisperConfig.forVariant(variant);

    // Load mel spectrogram
    std.debug.print("Loading mel spectrogram: {s}\n", .{mel_path});
    var mel_loader = ModelLoader.init(allocator, mel_path) catch |err| {
        if (err == LoadError.FileNotFound) {
            std.debug.print("Error: Mel file not found at {s}\n", .{mel_path});
            std.debug.print("Create with: python3 tools/export_mel.py <audio.wav> -o mel.tl\n", .{});
            return;
        }
        return err;
    };
    defer mel_loader.deinit();

    var mel = try mel_loader.getTensor("mel");
    defer mel.deinit();
    std.debug.print("  Mel shape: [{d}, {d}]\n", .{ mel.shape[0], mel.shape[1] });

    // Load model weights
    std.debug.print("Loading model weights from {s}...\n", .{weights_path});
    var model_loader = ModelLoader.init(allocator, weights_path) catch |err| {
        if (err == LoadError.FileNotFound) {
            std.debug.print("Error: Model not found at {s}\n", .{weights_path});
            std.debug.print("Run: python3 tools/export_whisper.py -m tiny\n", .{});
            return;
        }
        return err;
    };
    defer model_loader.deinit();

    // Load encoder
    std.debug.print("Loading encoder ({d} layers)...\n", .{cfg.encoder.n_audio_layer});
    var enc_weights = try WhisperEncoderWeights.loadFromLoader(allocator, &model_loader, cfg.encoder);
    defer enc_weights.deinit();
    const encoder = WhisperEncoder.init(allocator, cfg.encoder, enc_weights);

    // Load decoder
    std.debug.print("Loading decoder ({d} layers)...\n", .{cfg.decoder.n_text_layer});
    var dec_weights = try WhisperDecoderWeights.loadFromLoader(allocator, &model_loader, cfg.decoder);
    defer dec_weights.deinit();
    const decoder = WhisperDecoder.init(allocator, cfg.decoder, dec_weights);

    // Run encoder
    std.debug.print("\nRunning encoder...\n", .{});
    var timer = try std.time.Timer.start();
    var encoder_output = try encoder.encode(&mel);
    defer encoder_output.deinit();
    const encode_time = @as(f64, @floatFromInt(timer.read())) / 1_000_000_000.0;
    std.debug.print("  Encoder time: {d:.3}s\n", .{encode_time});
    std.debug.print("  Output shape: [{d}, {d}]\n", .{ encoder_output.shape[0], encoder_output.shape[1] });

    // Run greedy decoding with full KV cache (cross-attention + self-attention)
    std.debug.print("\nRunning greedy decoding (with full KV cache)...\n", .{});
    timer.reset();
    const prompt = defaultPromptTokens();
    const tokens = try decoder.greedyDecodeFullCache(&encoder_output, &prompt, 224);
    defer allocator.free(tokens);
    const decode_time = @as(f64, @floatFromInt(timer.read())) / 1_000_000_000.0;

    std.debug.print("  Decode time: {d:.3}s\n", .{decode_time});
    std.debug.print("  Generated {d} tokens\n", .{tokens.len - prompt.len});

    // Print token IDs (we don't have a tokenizer yet)
    std.debug.print("\n=== Generated Token IDs ===\n", .{});
    std.debug.print("Prompt: ", .{});
    for (prompt) |t| {
        std.debug.print("{d} ", .{t});
    }
    std.debug.print("\nGenerated: ", .{});
    for (tokens[prompt.len..]) |t| {
        std.debug.print("{d} ", .{t});
    }
    std.debug.print("\n", .{});

    // Print timing summary
    std.debug.print("\n=== Timing Summary ===\n", .{});
    std.debug.print("Encoder:  {d:.3}s\n", .{encode_time});
    std.debug.print("Decoder:  {d:.3}s\n", .{decode_time});
    std.debug.print("Total:    {d:.3}s\n", .{encode_time + decode_time});
}

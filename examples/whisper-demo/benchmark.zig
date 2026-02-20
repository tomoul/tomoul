// Whisper Benchmark - Pure Zig
//
// Loads the model once, then runs multiple inference calls to get accurate timing.
//
// Build and run:
//   zig build -Dmodel=whisper-tiny -Doptimize=ReleaseFast
//   ./zig-out/bin/tomoul_whisper-tiny benchmark models/english_man.wav
//
// Or use the CLI transcribe command which shows timing:
//   ./zig-out/bin/tomoul_whisper-tiny transcribe models/english_man.wav

const std = @import("std");

// Use module imports from build.zig
const Tensor = @import("tensor.zig").Tensor;
const ops = @import("ops.zig");
const loader_mod = @import("loader.zig");
const ModelLoader = loader_mod.ModelLoader;
const audio = @import("audio.zig");

// Import whisper modules via model.zig
const model = @import("model.zig");
const Whisper = model.Whisper;
const WhisperConfig = model.config.WhisperConfig;
const WhisperVariant = model.config.WhisperVariant;
const WhisperEncoderWeights = model.encoder.WhisperEncoderWeights;
const WhisperDecoderWeights = model.decoder.WhisperDecoderWeights;

const DEFAULT_WEIGHTS_PATH = "models/whisper_tiny.tl";
const DEFAULT_AUDIO_PATH = "models/english_man.wav";
const NUM_ITERATIONS: usize = 5;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize execution context for parallel matrix operations
    var ctx = ops.Context.initMultiThreaded(allocator, null);
    defer ctx.deinit();
    ops.initGlobalContext(&ctx);
    defer ops.deinitGlobalContext();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    // Parse arguments
    var weights_path: []const u8 = DEFAULT_WEIGHTS_PATH;
    var audio_path: []const u8 = DEFAULT_AUDIO_PATH;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--weights") or std.mem.eql(u8, args[i], "-w")) {
            if (i + 1 < args.len) {
                weights_path = args[i + 1];
                i += 1;
            }
        } else if (std.mem.eql(u8, args[i], "--help") or std.mem.eql(u8, args[i], "-h")) {
            printUsage();
            return;
        } else if (!std.mem.startsWith(u8, args[i], "-")) {
            audio_path = args[i];
        }
    }

    std.debug.print("\n", .{});
    std.debug.print("============================================================\n", .{});
    std.debug.print("Tomoul Whisper Benchmark (Pure Zig)\n", .{});
    std.debug.print("============================================================\n\n", .{});

    // Infer variant from weights path
    const variant = inferVariantFromPath(weights_path);
    const cfg = WhisperConfig.forVariant(variant);

    std.debug.print("Model variant: {s}\n", .{variant.name()});
    std.debug.print("Weights: {s}\n", .{weights_path});
    std.debug.print("Audio: {s}\n\n", .{audio_path});

    // Load model weights
    std.debug.print("Loading model...\n", .{});
    var load_timer = try std.time.Timer.start();

    var model_loader = ModelLoader.init(allocator, weights_path) catch |err| {
        std.debug.print("Error: Failed to load weights from {s}: {}\n", .{ weights_path, err });
        return;
    };
    defer model_loader.deinit();

    const enc_weights = try WhisperEncoderWeights.loadFromLoader(allocator, &model_loader, cfg.encoder);
    defer enc_weights.deinit();

    const dec_weights = try WhisperDecoderWeights.loadFromLoader(allocator, &model_loader, cfg.decoder);
    defer dec_weights.deinit();

    var whisper = Whisper.init(allocator, cfg, enc_weights, dec_weights);
    // Don't call whisper.deinit() - weights are deferred above

    const load_time = @as(f64, @floatFromInt(load_timer.read())) / 1_000_000_000.0;
    std.debug.print("  Model loaded in {d:.2}s\n\n", .{load_time});

    // Load audio once (reuse for all iterations)
    std.debug.print("Loading audio...\n", .{});
    var audio_timer = try std.time.Timer.start();

    const audio_result = audio.loadAudioFile(allocator, audio_path) catch |err| {
        std.debug.print("Error: Failed to load audio from {s}: {}\n", .{ audio_path, err });
        return;
    };
    defer allocator.free(audio_result.samples);

    const audio_load_time = @as(f64, @floatFromInt(audio_timer.read())) / 1_000_000_000.0;
    const audio_duration = @as(f64, @floatFromInt(audio_result.samples.len)) / @as(f64, @floatFromInt(audio.SAMPLE_RATE));
    std.debug.print("  Loaded {d} samples ({d:.2}s audio) in {d:.3}s\n\n", .{
        audio_result.samples.len,
        audio_duration,
        audio_load_time,
    });

    // Compute mel spectrogram once
    std.debug.print("Computing mel spectrogram...\n", .{});
    var mel_timer = try std.time.Timer.start();

    var mel = try audio.whisperMelSpectrogram(allocator, audio_result.samples, cfg.encoder.n_mels);
    defer mel.deinit();

    const mel_time = @as(f64, @floatFromInt(mel_timer.read())) / 1_000_000_000.0;
    std.debug.print("  Mel shape: [{d}, {d}] in {d:.3}s\n\n", .{ mel.shape[0], mel.shape[1], mel_time });

    // Warm-up run
    std.debug.print("Warm-up run...\n", .{});
    {
        const warmup_tokens = try whisper.transcribe(&mel);
        defer allocator.free(warmup_tokens);
        std.debug.print("  Generated {d} tokens\n\n", .{warmup_tokens.len});
    }

    // Benchmark runs
    std.debug.print("--- Transcription ({d} runs) ---\n", .{NUM_ITERATIONS});

    var times: [NUM_ITERATIONS]f64 = undefined;
    var first_tokens: ?[]u32 = null;

    for (0..NUM_ITERATIONS) |iter| {
        var iter_timer = try std.time.Timer.start();

        const tokens = try whisper.transcribe(&mel);

        const iter_time = @as(f64, @floatFromInt(iter_timer.read())) / 1_000_000_000.0;
        times[iter] = iter_time;

        if (iter == 0) {
            first_tokens = tokens;
            std.debug.print("  Run {d}: {d:.2}s - {d} tokens\n", .{ iter + 1, iter_time, tokens.len });

            // Show first 10 token IDs
            std.debug.print("  Token IDs: [", .{});
            const show_count = @min(tokens.len, 10);
            for (tokens[0..show_count], 0..) |tok, j| {
                if (j > 0) std.debug.print(", ", .{});
                std.debug.print("{d}", .{tok});
            }
            if (tokens.len > 10) std.debug.print("...", .{});
            std.debug.print("]\n", .{});
        } else {
            std.debug.print("  Run {d}: {d:.2}s - {d} tokens\n", .{ iter + 1, iter_time, tokens.len });
            allocator.free(tokens);
        }
    }

    // Compute statistics
    var total: f64 = 0.0;
    var min_time: f64 = times[0];
    var max_time: f64 = times[0];

    for (times) |t| {
        total += t;
        min_time = @min(min_time, t);
        max_time = @max(max_time, t);
    }

    const avg_time = total / @as(f64, NUM_ITERATIONS);

    std.debug.print("\nResults ({d} runs):\n", .{NUM_ITERATIONS});
    std.debug.print("  Average: {d:.2}s\n", .{avg_time});
    std.debug.print("  Min: {d:.2}s\n", .{min_time});
    std.debug.print("  Max: {d:.2}s\n", .{max_time});
    std.debug.print("  Real-time factor: {d:.2}x\n", .{avg_time / audio_duration});

    if (first_tokens) |tokens| {
        std.debug.print("  Tokens: {d}\n", .{tokens.len});
        allocator.free(tokens);
    }

    std.debug.print("\n============================================================\n", .{});
    std.debug.print("Benchmark Complete!\n", .{});
    std.debug.print("============================================================\n", .{});
}

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

fn printUsage() void {
    std.debug.print(
        \\Whisper Benchmark - Pure Zig
        \\
        \\Usage:
        \\  benchmark [audio.wav] [options]
        \\
        \\Options:
        \\  --weights, -w <path>  Path to model weights (default: models/whisper_tiny.tl)
        \\  --help, -h            Show this help message
        \\
        \\Examples:
        \\  ./benchmark models/english_man.wav
        \\  ./benchmark models/english_man.wav --weights models/whisper_tiny_q8_0.tl
        \\
    , .{});
}

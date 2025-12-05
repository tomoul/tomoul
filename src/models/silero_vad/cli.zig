const std = @import("std");
const SileroVAD = @import("model.zig").SileroVAD;
const Tensor = @import("tensor.zig").Tensor;
const build_options = @import("build_options");

const SAMPLE_RATE = 16000;
const CHUNK_SIZE = 512; // 32ms @ 16kHz
const BUNDLED = build_options.bundled;

const WavHeader = struct {
    riff: [4]u8,
    file_size: u32,
    wave: [4]u8,
    fmt: [4]u8,
    fmt_size: u32,
    audio_format: u16,
    num_channels: u16,
    sample_rate: u32,
    byte_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
};

fn parseWav(allocator: std.mem.Allocator, data: []const u8) ![]f32 {
    if (data.len < 44) return error.InvalidWavFile;

    // Check RIFF header
    if (!std.mem.eql(u8, data[0..4], "RIFF")) return error.InvalidWavFile;
    if (!std.mem.eql(u8, data[8..12], "WAVE")) return error.InvalidWavFile;

    // Parse format chunk
    if (!std.mem.eql(u8, data[12..16], "fmt ")) return error.InvalidWavFile;

    const audio_format = std.mem.readInt(u16, data[20..22], .little);
    const num_channels = std.mem.readInt(u16, data[22..24], .little);
    const sample_rate = std.mem.readInt(u32, data[24..28], .little);
    const bits_per_sample = std.mem.readInt(u16, data[34..36], .little);

    // Validate format
    if (num_channels != 1) {
        std.debug.print("Error: WAV must be mono (got {} channels)\n", .{num_channels});
        return error.InvalidWavFormat;
    }
    if (sample_rate != SAMPLE_RATE) {
        std.debug.print("Error: WAV must be 16kHz (got {}Hz)\n", .{sample_rate});
        return error.InvalidWavFormat;
    }
    if (audio_format != 1) {
        std.debug.print("Error: WAV must be PCM format (got format {})\n", .{audio_format});
        return error.InvalidWavFormat;
    }

    // Find data chunk
    var offset: usize = 12;
    while (offset + 8 < data.len) {
        const chunk_id = data[offset .. offset + 4];
        const chunk_size = std.mem.readInt(u32, data[offset + 4 ..][0..4], .little);

        if (std.mem.eql(u8, chunk_id, "data")) {
            // Found data chunk
            const audio_data = data[offset + 8 .. offset + 8 + chunk_size];
            const num_samples = chunk_size / (bits_per_sample / 8);

            // Allocate output buffer
            const samples = try allocator.alloc(f32, num_samples);
            errdefer allocator.free(samples);

            // Convert to f32
            if (bits_per_sample == 16) {
                for (0..num_samples) |i| {
                    const sample_i16 = std.mem.readInt(i16, audio_data[i * 2 ..][0..2], .little);
                    samples[i] = @as(f32, @floatFromInt(sample_i16)) / 32768.0;
                }
            } else if (bits_per_sample == 32) {
                @memcpy(samples, std.mem.bytesAsSlice(f32, audio_data));
            } else {
                std.debug.print("Error: Unsupported bit depth: {}\n", .{bits_per_sample});
                return error.UnsupportedBitDepth;
            }

            return samples;
        }

        offset += 8 + chunk_size;
    }

    return error.DataChunkNotFound;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (!BUNDLED and args.len < 3) {
        std.debug.print("Usage: {s} <audio_file> <model.tl>\n", .{args[0]});
        std.debug.print("\nDescription:\n", .{});
        std.debug.print("  Process audio file with Silero VAD model\n", .{});
        std.debug.print("  Supports:\n", .{});
        std.debug.print("    - WAV files (16kHz, mono, 16-bit PCM)\n", .{});
        std.debug.print("    - RAW files (16kHz, mono, 32-bit float, little-endian)\n", .{});
        std.debug.print("\nExamples:\n", .{});
        std.debug.print("  {s} audio.wav artifacts/silero_vad.tl\n", .{args[0]});
        std.debug.print("  {s} audio.raw artifacts/silero_vad.tl\n", .{args[0]});
        std.debug.print("\nOutput:\n", .{});
        std.debug.print("  Prints speech probability for each 32ms chunk (512 samples)\n", .{});
        std.debug.print("  Probability range: 0.0 (silence) to 1.0 (speech)\n", .{});
        std.process.exit(1);
    }

    if (BUNDLED and args.len < 2) {
        std.debug.print("Usage: {s} <audio_file>\n", .{args[0]});
        std.debug.print("\nDescription:\n", .{});
        std.debug.print("  Process audio file with bundled Silero VAD model\n", .{});
        std.debug.print("  Model weights are embedded in this executable.\n", .{});
        std.debug.print("\nSupports:\n", .{});
        std.debug.print("  - WAV files (16kHz, mono, 16-bit PCM)\n", .{});
        std.debug.print("  - RAW files (16kHz, mono, 32-bit float, little-endian)\n", .{});
        std.debug.print("\nExamples:\n", .{});
        std.debug.print("  {s} audio.wav\n", .{args[0]});
        std.debug.print("  {s} audio.raw\n", .{args[0]});
        std.process.exit(1);
    }

    const audio_path = args[1];

    // Load model (either from embedded weights or external file)
    var vad = if (BUNDLED) blk: {
        std.debug.print("Loading Silero VAD model from embedded weights...\n", .{});
        const model_weights_bytes = build_options.embedded_weights.?;
        break :blk SileroVAD.initFromBytes(allocator, model_weights_bytes) catch |err| {
            std.debug.print("Failed to load embedded model: {}\n", .{err});
            return err;
        };
    } else blk: {
        const model_path = args[2];
        std.debug.print("Loading Silero VAD model from {s}...\n", .{model_path});
        break :blk SileroVAD.init(allocator, model_path) catch |err| {
            std.debug.print("Failed to load model: {}\n", .{err});
            return err;
        };
    };
    defer vad.deinit();

    vad.printSummary();
    std.debug.print("\n", .{});

    // Read audio file
    std.debug.print("Loading audio from {s}...\n", .{audio_path});
    const file_data = std.fs.cwd().readFileAlloc(allocator, audio_path, 100 * 1024 * 1024) catch |err| {
        std.debug.print("Failed to read audio file: {}\n", .{err});
        std.debug.print("Make sure the file exists and is readable\n", .{});
        return err;
    };
    defer allocator.free(file_data);

    // Detect file type and parse audio
    const is_wav = file_data.len >= 4 and std.mem.eql(u8, file_data[0..4], "RIFF");

    var audio_samples: []f32 = undefined;
    var should_free_samples = false;
    defer if (should_free_samples) allocator.free(audio_samples);

    if (is_wav) {
        std.debug.print("Detected WAV file, parsing...\n", .{});
        audio_samples = try parseWav(allocator, file_data);
        should_free_samples = true;
    } else {
        // Assume raw f32
        std.debug.print("Detected RAW file (f32)...\n", .{});
        if (file_data.len % 4 != 0) {
            std.debug.print("Error: RAW file size must be multiple of 4 bytes (f32)\n", .{});
            std.debug.print("Got {d} bytes\n", .{file_data.len});
            return error.InvalidAudioFile;
        }
        const audio_floats: [*]const f32 = @ptrCast(@alignCast(file_data.ptr));
        audio_samples = @constCast(audio_floats[0 .. file_data.len / 4]);
    }

    const num_samples = audio_samples.len;
    const num_chunks = num_samples / CHUNK_SIZE;
    const duration_ms = (num_samples * 1000) / SAMPLE_RATE;

    std.debug.print("Audio info:\n", .{});
    std.debug.print("  Samples: {d}\n", .{num_samples});
    std.debug.print("  Chunks:  {d} ({d} samples each)\n", .{ num_chunks, CHUNK_SIZE });
    std.debug.print("  Duration: {d}ms ({d:.2}s)\n", .{ duration_ms, @as(f32, @floatFromInt(duration_ms)) / 1000.0 });
    std.debug.print("\n", .{});

    if (num_chunks == 0) {
        std.debug.print("Error: Audio file too short (need at least {d} samples / {d}ms)\n", .{ CHUNK_SIZE, (CHUNK_SIZE * 1000) / SAMPLE_RATE });
        return error.AudioTooShort;
    }

    // Create chunk tensor
    var chunk_shape = [_]usize{CHUNK_SIZE};
    var chunk = try Tensor.init(allocator, &chunk_shape);
    defer chunk.deinit();

    // Reset VAD state
    vad.resetStates();

    // Statistics tracking
    var speech_chunks: usize = 0;
    var total_speech_prob: f32 = 0.0;
    var max_prob: f32 = 0.0;
    var min_prob: f32 = 1.0;

    // Process audio in chunks
    std.debug.print("Processing audio (threshold 0.5 for speech detection):\n", .{});
    std.debug.print("─────────────────────────────────────────────────────────────\n", .{});

    for (0..num_chunks) |i| {
        // Copy chunk data
        const offset = i * CHUNK_SIZE;
        @memcpy(chunk.data, audio_samples[offset .. offset + CHUNK_SIZE]);

        // Run VAD
        const prob = try vad.forward(&chunk);
        const is_speech = prob >= 0.5;

        // Update statistics
        total_speech_prob += prob;
        if (prob > max_prob) max_prob = prob;
        if (prob < min_prob) min_prob = prob;
        if (is_speech) speech_chunks += 1;

        // Calculate timing
        const chunk_start_ms = (i * CHUNK_SIZE * 1000) / SAMPLE_RATE;
        const chunk_end_ms = ((i + 1) * CHUNK_SIZE * 1000) / SAMPLE_RATE;

        // Print result with visual indicator
        const indicator = if (is_speech) "█████" else "     ";
        std.debug.print("{d:4} | {d:6}ms - {d:6}ms | {d:.4} | {s} {s}\n", .{
            i,
            chunk_start_ms,
            chunk_end_ms,
            prob,
            indicator,
            if (is_speech) "SPEECH" else "silence",
        });
    }

    // Print summary statistics
    std.debug.print("─────────────────────────────────────────────────────────────\n", .{});
    std.debug.print("\n=== Summary ===\n", .{});
    std.debug.print("Total chunks:      {d}\n", .{num_chunks});
    std.debug.print("Speech chunks:     {d} ({d:.1}%)\n", .{
        speech_chunks,
        @as(f32, @floatFromInt(speech_chunks)) * 100.0 / @as(f32, @floatFromInt(num_chunks)),
    });
    std.debug.print("Average prob:      {d:.4}\n", .{total_speech_prob / @as(f32, @floatFromInt(num_chunks))});
    std.debug.print("Min prob:          {d:.4}\n", .{min_prob});
    std.debug.print("Max prob:          {d:.4}\n", .{max_prob});

    const speech_duration_ms = (speech_chunks * CHUNK_SIZE * 1000) / SAMPLE_RATE;
    const silence_duration_ms = duration_ms - speech_duration_ms;
    std.debug.print("Speech duration:   {d}ms ({d:.2}s)\n", .{
        speech_duration_ms,
        @as(f32, @floatFromInt(speech_duration_ms)) / 1000.0,
    });
    std.debug.print("Silence duration:  {d}ms ({d:.2}s)\n", .{
        silence_duration_ms,
        @as(f32, @floatFromInt(silence_duration_ms)) / 1000.0,
    });
    std.debug.print("\n", .{});
}

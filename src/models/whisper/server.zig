// Whisper HTTP Server - Pure Zig
//
// A persistent HTTP server that loads the model once and handles transcription
// requests without startup overhead.
//
// Build and run:
//   zig build -Dmodel=whisper-tiny -Doptimize=ReleaseFast
//   ./zig-out/bin/whisper-server --weights models/whisper_tiny.tl
//
// API Endpoints:
//   POST /transcribe - Transcribe audio file (JSON body: {"audio_path": "path/to/file.wav"})
//   GET  /status     - Check if model is ready
//   GET  /test       - Quick test with default audio file
//

const std = @import("std");
const net = std.net;
const posix = std.posix;

// Use module imports from build.zig
const Tensor = @import("tensor.zig").Tensor;
const ops = @import("ops.zig");
const loader_mod = @import("loader.zig");
const ModelLoader = loader_mod.ModelLoader;
const audio = @import("audio.zig");

// Import whisper modules via model.zig
const model = @import("model.zig");
const WhisperConfig = model.config.WhisperConfig;
const WhisperVariant = model.config.WhisperVariant;
const WhisperTokens = model.config.WhisperTokens;
const defaultPromptTokens = model.config.defaultPromptTokens;
const WhisperEncoder = model.encoder.WhisperEncoder;
const WhisperEncoderWeights = model.encoder.WhisperEncoderWeights;
const WhisperDecoder = model.decoder.WhisperDecoder;
const WhisperDecoderWeights = model.decoder.WhisperDecoderWeights;

// Tokenizer for text decoding
const tokenizer_mod = @import("tokenizer.zig");
const WhisperTokenizer = tokenizer_mod.WhisperTokenizer;

const DEFAULT_WEIGHTS_PATH = "models/whisper_tiny.tl";
const DEFAULT_AUDIO_PATH = "models/english_man.wav";
const DEFAULT_VOCAB_PATH = "models/whisper_vocab.bin";
const DEFAULT_PORT: u16 = 8080;
const MAX_REQUEST_SIZE: usize = 64 * 1024; // 64KB max request

// Global model state (loaded once at startup)
var global_encoder: ?WhisperEncoder = null;
var global_decoder: ?WhisperDecoder = null;
var global_cfg: ?WhisperConfig = null;
var global_tokenizer: ?WhisperTokenizer = null;
var global_allocator: std.mem.Allocator = undefined;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    global_allocator = allocator;

    // Initialize execution context for parallel matrix operations
    var ctx = ops.Context.initMultiThreaded(allocator, null);
    defer ctx.deinit();
    ops.initGlobalContext(&ctx);
    defer ops.deinitGlobalContext();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    // Parse arguments
    var weights_path: []const u8 = DEFAULT_WEIGHTS_PATH;
    var vocab_path: []const u8 = DEFAULT_VOCAB_PATH;
    var port: u16 = DEFAULT_PORT;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--weights") or std.mem.eql(u8, args[i], "-w")) {
            if (i + 1 < args.len) {
                weights_path = args[i + 1];
                i += 1;
            }
        } else if (std.mem.eql(u8, args[i], "--vocab") or std.mem.eql(u8, args[i], "-v")) {
            if (i + 1 < args.len) {
                vocab_path = args[i + 1];
                i += 1;
            }
        } else if (std.mem.eql(u8, args[i], "--port") or std.mem.eql(u8, args[i], "-p")) {
            if (i + 1 < args.len) {
                port = std.fmt.parseInt(u16, args[i + 1], 10) catch DEFAULT_PORT;
                i += 1;
            }
        } else if (std.mem.eql(u8, args[i], "--help") or std.mem.eql(u8, args[i], "-h")) {
            printUsage();
            return;
        }
    }

    std.debug.print("\n", .{});
    std.debug.print("============================================================\n", .{});
    std.debug.print("Tomoul Whisper HTTP Server (Pure Zig)\n", .{});
    std.debug.print("============================================================\n\n", .{});

    // Infer variant from weights path
    const variant = inferVariantFromPath(weights_path);
    global_cfg = WhisperConfig.forVariant(variant);
    const cfg = global_cfg.?;

    std.debug.print("Model variant: {s}\n", .{variant.name()});
    std.debug.print("Weights: {s}\n\n", .{weights_path});

    // Load model weights
    std.debug.print("Loading model...\n", .{});
    var load_timer = try std.time.Timer.start();

    var model_loader = ModelLoader.init(allocator, weights_path) catch |err| {
        std.debug.print("Error: Failed to load weights from {s}: {}\n", .{ weights_path, err });
        return;
    };
    defer model_loader.deinit();

    const enc_weights = try WhisperEncoderWeights.loadFromLoader(allocator, &model_loader, cfg.encoder);
    const dec_weights = try WhisperDecoderWeights.loadFromLoader(allocator, &model_loader, cfg.decoder);

    global_encoder = WhisperEncoder.init(allocator, cfg.encoder, enc_weights);
    global_decoder = WhisperDecoder.init(allocator, cfg.decoder, dec_weights);

    const load_time = @as(f64, @floatFromInt(load_timer.read())) / 1_000_000_000.0;
    std.debug.print("  Model loaded in {d:.2}s\n", .{load_time});

    // Load tokenizer for text decoding
    std.debug.print("Loading tokenizer from {s}...\n", .{vocab_path});
    if (WhisperTokenizer.loadFromFile(allocator, vocab_path)) |tok| {
        global_tokenizer = tok;
        std.debug.print("  Tokenizer loaded ({d} tokens)\n", .{tok.vocab_size});
    } else |err| {
        std.debug.print("  Warning: Failed to load tokenizer: {}\n", .{err});
        std.debug.print("  Text output will not be available (tokens only)\n", .{});
        global_tokenizer = null;
    }
    std.debug.print("\n", .{});

    // Start HTTP server
    std.debug.print("Starting HTTP server on port {d}...\n", .{port});
    std.debug.print("  POST /transcribe - Transcribe audio\n", .{});
    std.debug.print("  GET  /status     - Check model status\n", .{});
    std.debug.print("  GET  /test       - Test with default audio\n", .{});
    std.debug.print("\nServer ready! Press Ctrl+C to stop.\n", .{});
    std.debug.print("============================================================\n\n", .{});

    // Create TCP listener
    const address = net.Address.initIp4(.{ 0, 0, 0, 0 }, port);
    var server = try address.listen(.{
        .reuse_address = true,
    });
    defer server.deinit();

    // Handle connections
    while (true) {
        var connection = server.accept() catch |err| {
            std.debug.print("Accept error: {}\n", .{err});
            continue;
        };

        handleConnection(allocator, &connection) catch |err| {
            std.debug.print("Connection error: {}\n", .{err});
        };

        connection.stream.close();
    }
}

fn handleConnection(allocator: std.mem.Allocator, connection: *net.Server.Connection) !void {
    var buf: [MAX_REQUEST_SIZE]u8 = undefined;

    // Read the HTTP request
    const bytes_read = connection.stream.read(&buf) catch |err| {
        std.debug.print("Read error: {}\n", .{err});
        return;
    };

    if (bytes_read == 0) return;

    const request = buf[0..bytes_read];

    // Parse the request line
    const request_line_end = std.mem.indexOf(u8, request, "\r\n") orelse return;
    const request_line = request[0..request_line_end];

    // Parse method and path
    var parts = std.mem.splitScalar(u8, request_line, ' ');
    const method = parts.next() orelse return;
    const path = parts.next() orelse return;

    // Find body (after double CRLF)
    const body_start = std.mem.indexOf(u8, request, "\r\n\r\n");
    const body = if (body_start) |start| request[start + 4 ..] else "";

    // Route the request
    if (std.mem.eql(u8, path, "/status") and std.mem.eql(u8, method, "GET")) {
        try handleStatus(allocator, connection);
    } else if (std.mem.eql(u8, path, "/test") and std.mem.eql(u8, method, "GET")) {
        try handleTest(allocator, connection);
    } else if (std.mem.eql(u8, path, "/transcribe") and std.mem.eql(u8, method, "POST")) {
        try handleTranscribe(allocator, connection, body);
    } else {
        try sendResponse(connection, "404 Not Found", "{\"error\": \"Not found\"}");
    }
}

fn handleStatus(allocator: std.mem.Allocator, connection: *net.Server.Connection) !void {
    const ready = global_encoder != null and global_decoder != null;
    const variant_name = if (global_cfg) |cfg| cfg.variant.name() else "unknown";

    var buf: [512]u8 = undefined;
    const json = try std.fmt.bufPrint(&buf, "{{\"ready\": {s}, \"variant\": \"{s}\", \"version\": \"whisper-v1.0.0\"}}", .{
        if (ready) "true" else "false",
        variant_name,
    });

    _ = allocator;
    try sendResponse(connection, "200 OK", json);
}

fn handleTest(allocator: std.mem.Allocator, connection: *net.Server.Connection) !void {
    if (global_encoder == null or global_decoder == null) {
        try sendResponse(connection, "503 Service Unavailable", "{\"error\": \"Model not loaded\"}");
        return;
    }

    // Check if test audio exists
    std.fs.cwd().access(DEFAULT_AUDIO_PATH, .{}) catch {
        try sendResponse(connection, "400 Bad Request", "{\"error\": \"Test audio file not found\"}");
        return;
    };

    // Transcribe
    const result = transcribeAudioFile(allocator, DEFAULT_AUDIO_PATH) catch |err| {
        var buf: [256]u8 = undefined;
        const json = std.fmt.bufPrint(&buf, "{{\"error\": \"Transcription failed: {}\"}}", .{err}) catch
            "{\"error\": \"Transcription failed\"}";
        try sendResponse(connection, "500 Internal Server Error", json);
        return;
    };
    defer allocator.free(result.json);

    try sendResponse(connection, "200 OK", result.json);
}

fn handleTranscribe(allocator: std.mem.Allocator, connection: *net.Server.Connection, body: []const u8) !void {
    if (global_encoder == null or global_decoder == null) {
        try sendResponse(connection, "503 Service Unavailable", "{\"error\": \"Model not loaded\"}");
        return;
    }

    // Parse JSON to extract audio_path
    const audio_path = parseAudioPath(body) orelse {
        try sendResponse(connection, "400 Bad Request", "{\"error\": \"Missing or invalid audio_path in JSON body\"}");
        return;
    };

    // Check if audio file exists
    std.fs.cwd().access(audio_path, .{}) catch {
        try sendResponse(connection, "400 Bad Request", "{\"error\": \"Audio file not found\"}");
        return;
    };

    // Transcribe
    const result = transcribeAudioFile(allocator, audio_path) catch |err| {
        var buf: [256]u8 = undefined;
        const json = std.fmt.bufPrint(&buf, "{{\"error\": \"Transcription failed: {}\"}}", .{err}) catch
            "{\"error\": \"Transcription failed\"}";
        try sendResponse(connection, "500 Internal Server Error", json);
        return;
    };
    defer allocator.free(result.json);

    try sendResponse(connection, "200 OK", result.json);
}

const TranscriptionResult = struct {
    json: []u8,
    inference_time_ms: f64,
    text: ?[]u8 = null,
};

fn transcribeAudioFile(allocator: std.mem.Allocator, audio_path: []const u8) !TranscriptionResult {
    const encoder = &global_encoder.?;
    const decoder = &global_decoder.?;
    const cfg = global_cfg.?;

    var total_timer = try std.time.Timer.start();

    // Load audio
    const audio_result = try audio.loadAudioFile(allocator, audio_path);
    defer allocator.free(audio_result.samples);

    // Compute mel spectrogram
    var mel = try audio.whisperMelSpectrogram(allocator, audio_result.samples, cfg.encoder.n_mels);
    defer mel.deinit();

    // Run encoder
    var encoder_output = try encoder.encode(&mel);
    defer encoder_output.deinit();

    // Run greedy decoding with full KV cache (same as CLI)
    const prompt = defaultPromptTokens();
    const tokens = try decoder.greedyDecodeFullCache(&encoder_output, &prompt, 224);
    defer allocator.free(tokens);

    const total_time = @as(f64, @floatFromInt(total_timer.read())) / 1_000_000.0; // Convert to ms

    // Decode tokens to text if tokenizer is available
    var text: ?[]u8 = null;
    if (global_tokenizer) |*tokenizer| {
        text = tokenizer.decodeTokens(allocator, tokens, false) catch null;
    }
    defer if (text) |t| allocator.free(t);

    // Build JSON response
    var json_buf: std.ArrayListUnmanaged(u8) = .{};
    errdefer json_buf.deinit(allocator);

    // Add text field first if available
    if (text) |t| {
        try json_buf.appendSlice(allocator, "{\"text\": \"");
        // Escape special characters in text for JSON
        for (t) |c| {
            switch (c) {
                '"' => try json_buf.appendSlice(allocator, "\\\""),
                '\\' => try json_buf.appendSlice(allocator, "\\\\"),
                '\n' => try json_buf.appendSlice(allocator, "\\n"),
                '\r' => try json_buf.appendSlice(allocator, "\\r"),
                '\t' => try json_buf.appendSlice(allocator, "\\t"),
                else => {
                    if (c < 0x20) {
                        // Control character - skip or encode
                        var hex_buf: [6]u8 = undefined;
                        const hex = std.fmt.bufPrint(&hex_buf, "\\u{x:0>4}", .{c}) catch continue;
                        try json_buf.appendSlice(allocator, hex);
                    } else {
                        try json_buf.append(allocator, c);
                    }
                },
            }
        }
        try json_buf.appendSlice(allocator, "\", \"tokens\": [");
    } else {
        try json_buf.appendSlice(allocator, "{\"tokens\": [");
    }

    for (tokens, 0..) |tok, idx| {
        if (idx > 0) try json_buf.appendSlice(allocator, ", ");
        var num_buf: [16]u8 = undefined;
        const num_str = std.fmt.bufPrint(&num_buf, "{d}", .{tok}) catch continue;
        try json_buf.appendSlice(allocator, num_str);
    }

    var footer_buf: [128]u8 = undefined;
    const footer = try std.fmt.bufPrint(&footer_buf, "], \"num_tokens\": {d}, \"inference_time_ms\": {d:.2}}}", .{
        tokens.len,
        total_time,
    });
    try json_buf.appendSlice(allocator, footer);

    return TranscriptionResult{
        .json = try json_buf.toOwnedSlice(allocator),
        .inference_time_ms = total_time,
    };
}

fn parseAudioPath(json: []const u8) ?[]const u8 {
    // Simple JSON parser for {"audio_path": "..."}
    const key = "\"audio_path\"";
    const key_pos = std.mem.indexOf(u8, json, key) orelse return null;
    const after_key = json[key_pos + key.len ..];

    // Find the colon
    const colon_pos = std.mem.indexOf(u8, after_key, ":") orelse return null;
    const after_colon = after_key[colon_pos + 1 ..];

    // Find opening quote
    const quote_start = std.mem.indexOf(u8, after_colon, "\"") orelse return null;
    const value_start = after_colon[quote_start + 1 ..];

    // Find closing quote
    const quote_end = std.mem.indexOf(u8, value_start, "\"") orelse return null;

    return value_start[0..quote_end];
}

fn sendResponse(connection: *net.Server.Connection, status: []const u8, body: []const u8) !void {
    var header_buf: [512]u8 = undefined;
    const header = try std.fmt.bufPrint(&header_buf, "HTTP/1.1 {s}\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n", .{
        status,
        body.len,
    });

    _ = try connection.stream.write(header);
    _ = try connection.stream.write(body);
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
    const usage =
        \\Whisper HTTP Server - Pure Zig
        \\
        \\A persistent HTTP server that loads the Whisper model once and serves
        \\transcription requests without startup overhead.
        \\
        \\Usage:
        \\  whisper-server [options]
        \\
        \\Options:
        \\  --weights, -w <path>  Path to model weights (default: models/whisper_tiny.tl)
        \\  --vocab, -v <path>    Path to vocabulary file (default: models/whisper_vocab.bin)
        \\  --port, -p <port>     Port to listen on (default: 8080)
        \\  --help, -h            Show this help message
        \\
        \\API Endpoints:
        \\  POST /transcribe      Transcribe audio file
        \\                        Body: {"audio_path": "path/to/file.wav"}
        \\                        Returns: {"text": "...", "tokens": [...], "num_tokens": N, "inference_time_ms": X}
        \\
        \\  GET  /status          Check if model is ready
        \\                        Returns: {"ready": true, "variant": "tiny", "version": "..."}
        \\
        \\  GET  /test            Quick test with default audio file
        \\                        Returns: Same as /transcribe
        \\
        \\Examples:
        \\  ./whisper-server --weights models/whisper_tiny.tl --port 8080
        \\
        \\  curl -X POST http://localhost:8080/transcribe \
        \\       -H "Content-Type: application/json" \
        \\       -d '{"audio_path": "models/english_man.wav"}'
        \\
    ;
    std.debug.print("{s}", .{usage});
}

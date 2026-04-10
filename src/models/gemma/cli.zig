// src/models/gemma/cli.zig
// CLI for Gemma text generation
//
// Usage:
//   tomoul_gemma generate "Once upon a time" --model gemma_2b_q8.tl --tokenizer gemma_vocab.bin
//   tomoul_gemma chat --model gemma_2b_q8.tl --tokenizer gemma_vocab.bin

const std = @import("std");
const model_mod = @import("model.zig");
const Gemma = model_mod.Gemma;
const GemmaTokens = model_mod.config.GemmaTokens;
const Tokenizer = model_mod.tokenizer.Tokenizer;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    // Parse arguments
    var model_path: ?[]const u8 = null;
    var tokenizer_path: ?[]const u8 = null;
    var prompt_text: ?[]const u8 = null;
    var max_tokens: usize = 256;
    var temperature: f32 = 0.0;
    var mode: enum { generate, chat } = .generate;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--model") and i + 1 < args.len) {
            i += 1;
            model_path = args[i];
        } else if (std.mem.eql(u8, args[i], "--tokenizer") and i + 1 < args.len) {
            i += 1;
            tokenizer_path = args[i];
        } else if (std.mem.eql(u8, args[i], "--max-tokens") and i + 1 < args.len) {
            i += 1;
            max_tokens = std.fmt.parseInt(usize, args[i], 10) catch 256;
        } else if (std.mem.eql(u8, args[i], "--temperature") and i + 1 < args.len) {
            i += 1;
            temperature = std.fmt.parseFloat(f32, args[i]) catch 0.0;
        } else if (std.mem.eql(u8, args[i], "generate")) {
            mode = .generate;
            if (i + 1 < args.len and args[i + 1][0] != '-') {
                i += 1;
                prompt_text = args[i];
            }
        } else if (std.mem.eql(u8, args[i], "chat")) {
            mode = .chat;
        } else if (std.mem.eql(u8, args[i], "--help") or std.mem.eql(u8, args[i], "-h")) {
            printUsage(args[0]);
            return;
        }
    }

    if (model_path == null or tokenizer_path == null) {
        std.debug.print("Error: --model and --tokenizer are required\n\n", .{});
        printUsage(args[0]);
        std.process.exit(1);
    }

    // Load model
    std.debug.print("Loading tokenizer from {s}...\n", .{tokenizer_path.?});
    var tok = Tokenizer.init(allocator, tokenizer_path.?) catch |err| {
        std.debug.print("Failed to load tokenizer: {}\n", .{err});
        return err;
    };
    defer tok.deinit();

    std.debug.print("Loading model from {s}...\n", .{model_path.?});
    var model = Gemma.init(allocator, model_path.?) catch |err| {
        std.debug.print("Failed to load model: {}\n", .{err});
        return err;
    };
    defer model.deinit();
    std.debug.print("Model loaded.\n", .{});

    const stdout_file = std.fs.File.stdout();

    switch (mode) {
        .generate => {
            const text = prompt_text orelse {
                std.debug.print("Error: generate mode requires a prompt\n", .{});
                std.process.exit(1);
            };

            const input_ids = try tok.encode(text);
            defer allocator.free(input_ids);

            std.debug.print("Generating (prompt: {d} tokens, max: {d})...\n", .{ input_ids.len, max_tokens });

            var timer = try std.time.Timer.start();
            const tokens = try model.generate(input_ids, max_tokens, temperature);
            defer allocator.free(tokens);
            const elapsed_ns = timer.read();

            // Decode all tokens (including prompt echo)
            const output_text = try tok.decode(tokens);
            defer allocator.free(output_text);

            try stdout_file.writeAll(output_text);
            try stdout_file.writeAll("\n");

            const generated_count = tokens.len - input_ids.len;
            const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
            const tokens_per_sec = @as(f64, @floatFromInt(generated_count)) / (elapsed_ms / 1000.0);
            std.debug.print("\n--- {d} tokens in {d:.1}ms ({d:.1} tok/s) ---\n", .{ generated_count, elapsed_ms, tokens_per_sec });
        },
        .chat => {
            try stdout_file.writeAll("Gemma chat (type 'quit' to exit)\n\n");

            while (true) {
                try stdout_file.writeAll("> ");

                // Read a line from stdin
                var line_buf: [4096]u8 = undefined;
                var line_len: usize = 0;
                while (line_len < line_buf.len) {
                    const n = std.fs.File.stdin().read(line_buf[line_len..][0..1]) catch break;
                    if (n == 0) break;
                    if (line_buf[line_len] == '\n') break;
                    line_len += 1;
                }
                const line = std.mem.trim(u8, line_buf[0..line_len], " \t\r");
                if (line.len == 0) continue;
                if (std.mem.eql(u8, line, "quit")) break;

                const input_ids = tok.encode(line) catch |err| {
                    std.debug.print("Tokenization error: {}\n", .{err});
                    continue;
                };
                defer allocator.free(input_ids);

                model.resetCache();

                const tokens = model.generate(input_ids, max_tokens, temperature) catch |err| {
                    std.debug.print("Generation error: {}\n", .{err});
                    continue;
                };
                defer allocator.free(tokens);

                const generated = if (tokens.len > input_ids.len) tokens[input_ids.len..] else tokens[0..0];
                const output_text = tok.decode(generated) catch |err| {
                    std.debug.print("Decode error: {}\n", .{err});
                    continue;
                };
                defer allocator.free(output_text);

                try stdout_file.writeAll(output_text);
                try stdout_file.writeAll("\n\n");
            }
        },
    }
}

fn printUsage(prog: []const u8) void {
    std.debug.print(
        \\Usage: {s} <command> [options]
        \\
        \\Commands:
        \\  generate "prompt"   Generate text from a prompt
        \\  chat                Interactive chat mode
        \\
        \\Options:
        \\  --model <path>      Path to model weights (.tl file) [required]
        \\  --tokenizer <path>  Path to tokenizer file (.bin) [required]
        \\  --max-tokens <n>    Maximum tokens to generate (default: 256)
        \\  --temperature <f>   Sampling temperature, 0.0=greedy (default: 0.0)
        \\  --help, -h          Show this help
        \\
    , .{prog});
}

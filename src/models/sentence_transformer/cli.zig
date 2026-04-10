const std = @import("std");
const model_mod = @import("model.zig");
const SentenceTransformer = model_mod.SentenceTransformer;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    // Parse arguments
    var model_path: ?[]const u8 = null;
    var vocab_path: ?[]const u8 = null;
    var single_text: ?[]const u8 = null;
    var batch_path: ?[]const u8 = null;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--model") and i + 1 < args.len) {
            i += 1;
            model_path = args[i];
        } else if (std.mem.eql(u8, args[i], "--vocab") and i + 1 < args.len) {
            i += 1;
            vocab_path = args[i];
        } else if (std.mem.eql(u8, args[i], "--text") and i + 1 < args.len) {
            i += 1;
            single_text = args[i];
        } else if (std.mem.eql(u8, args[i], "--batch") and i + 1 < args.len) {
            i += 1;
            batch_path = args[i];
        } else if (std.mem.eql(u8, args[i], "--help") or std.mem.eql(u8, args[i], "-h")) {
            printUsage(args[0]);
            return;
        }
    }

    if (model_path == null or vocab_path == null) {
        std.debug.print("Error: --model and --vocab are required\n\n", .{});
        printUsage(args[0]);
        std.process.exit(1);
    }

    if (single_text == null and batch_path == null) {
        std.debug.print("Error: --text or --batch is required\n\n", .{});
        printUsage(args[0]);
        std.process.exit(1);
    }

    // Load model
    std.debug.print("Loading model from {s}...\n", .{model_path.?});
    var st = SentenceTransformer.init(allocator, model_path.?, vocab_path.?) catch |err| {
        std.debug.print("Failed to load model: {}\n", .{err});
        return err;
    };
    defer st.deinit();
    std.debug.print("Model loaded.\n", .{});

    const stdout_file = std.fs.File.stdout();

    if (single_text) |text| {
        // Single text mode
        const embedding = try st.embed(text);
        try writeJson(allocator, stdout_file, &[_][384]f32{embedding});
    } else if (batch_path) |path| {
        // Batch mode: read lines from file
        const file = std.fs.cwd().openFile(path, .{}) catch |err| {
            std.debug.print("Failed to open batch file '{s}': {}\n", .{ path, err });
            return err;
        };
        defer file.close();

        const content = try file.readToEndAlloc(allocator, 100 * 1024 * 1024);
        defer allocator.free(content);

        // Collect lines
        var lines: std.ArrayList([]const u8) = .{};
        defer lines.deinit(allocator);

        var line_iter = std.mem.splitScalar(u8, content, '\n');
        while (line_iter.next()) |line| {
            if (line.len > 0) {
                try lines.append(allocator, line);
            }
        }

        // Embed all
        const embeddings = try allocator.alloc([384]f32, lines.items.len);
        defer allocator.free(embeddings);

        for (lines.items, 0..) |line, idx| {
            embeddings[idx] = try st.embed(line);
        }

        try writeJson(allocator, stdout_file, embeddings);
    }
}

fn writeJson(allocator: std.mem.Allocator, file: std.fs.File, embeddings: []const [384]f32) !void {
    var buf: std.ArrayList(u8) = .{};
    defer buf.deinit(allocator);

    try buf.appendSlice(allocator, "{\"embeddings\": [");
    for (embeddings, 0..) |emb, i| {
        if (i > 0) try buf.appendSlice(allocator, ", ");
        try buf.appendSlice(allocator, "[");
        for (emb, 0..) |v, j| {
            if (j > 0) try buf.appendSlice(allocator, ", ");
            var num_buf: [32]u8 = undefined;
            const num_str = std.fmt.bufPrint(&num_buf, "{d:.6}", .{v}) catch unreachable;
            try buf.appendSlice(allocator, num_str);
        }
        try buf.appendSlice(allocator, "]");
    }
    var dim_buf: [64]u8 = undefined;
    const dim_str = std.fmt.bufPrint(&dim_buf, "], \"dimensions\": {d}}}\n", .{@as(usize, 384)}) catch unreachable;
    try buf.appendSlice(allocator, dim_str);

    try file.writeAll(buf.items);
}

fn printUsage(prog: []const u8) void {
    std.debug.print(
        \\Usage: {s} --model <path.tl> --vocab <path.txt> [--text "input"] [--batch <file>]
        \\
        \\Options:
        \\  --model <path>   Path to model weights (.tl file)
        \\  --vocab <path>   Path to vocabulary file (.txt)
        \\  --text  <text>   Single text to embed
        \\  --batch <file>   File with one sentence per line
        \\  --help, -h       Show this help
        \\
        \\Output: JSON with embeddings array and dimensions field
        \\
    , .{prog});
}

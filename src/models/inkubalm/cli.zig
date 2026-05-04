// src/models/inkubalm/cli.zig
//
// Token-id-driven CLI for InkubaLM-0.4B.
//
// Phase 1 scope: the SentencePiece tokenizer is not yet implemented in Zig.
// Until it is, this CLI accepts prompt token IDs as decimal arguments and
// emits generated token IDs to stdout. A Python harness can drive this end
// to end:
//
//   ids="$(python -c 'from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained("lelapa/InkubaLM-0.4B"); print(" ".join(map(str, t("Habari").input_ids)))')"
//   ./tomoul_inkubalm-0.4b artifacts/inkubalm.safetensors --max-new 20 -- $ids
//
// Once the SentencePiece reader lands, this CLI will accept raw text instead.

const std = @import("std");
const inkubalm = @import("model.zig");

const VERSION = "0.1.0";

const Args = struct {
    weights: []const u8,
    max_new: usize = 32,
    eos: ?u32 = null,
    prompt_ids: []const u32,
};

fn printUsage() void {
    std.debug.print(
        \\inkubalm-0.4b — Tomoul (token-id input)
        \\
        \\Usage:
        \\  tomoul_inkubalm-0.4b <weights.safetensors> [options] -- <id1> <id2> ...
        \\
        \\Options:
        \\  --max-new N    Max new tokens to generate (default 32)
        \\  --eos N        Stop when this token id is produced
        \\  -h, --help     Show this help
        \\
    , .{});
}

fn parseArgs(allocator: std.mem.Allocator, argv: []const []const u8) !Args {
    // Help anywhere short-circuits.
    for (argv[1..]) |a| {
        if (std.mem.eql(u8, a, "-h") or std.mem.eql(u8, a, "--help")) {
            return error.HelpRequested;
        }
    }

    if (argv.len < 2) return error.MissingWeightsPath;

    const weights = argv[1];
    var max_new: usize = 32;
    var eos: ?u32 = null;

    var i: usize = 2;
    var prompt_start: usize = argv.len;
    while (i < argv.len) : (i += 1) {
        const a = argv[i];
        if (std.mem.eql(u8, a, "--")) {
            prompt_start = i + 1;
            break;
        } else if (std.mem.eql(u8, a, "--max-new")) {
            i += 1;
            if (i >= argv.len) return error.MissingValue;
            max_new = try std.fmt.parseInt(usize, argv[i], 10);
        } else if (std.mem.eql(u8, a, "--eos")) {
            i += 1;
            if (i >= argv.len) return error.MissingValue;
            eos = try std.fmt.parseInt(u32, argv[i], 10);
        } else if (std.mem.eql(u8, a, "-h") or std.mem.eql(u8, a, "--help")) {
            return error.HelpRequested;
        } else {
            return error.UnknownFlag;
        }
    }

    if (prompt_start > argv.len) return error.MissingPrompt;

    const ids = try allocator.alloc(u32, argv.len - prompt_start);
    errdefer allocator.free(ids);
    for (argv[prompt_start..], ids) |s, *out| {
        out.* = try std.fmt.parseInt(u32, s, 10);
    }
    if (ids.len == 0) return error.MissingPrompt;

    return .{ .weights = weights, .max_new = max_new, .eos = eos, .prompt_ids = ids };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const argv = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, argv);

    const args = parseArgs(allocator, argv) catch |err| switch (err) {
        error.HelpRequested => {
            printUsage();
            return;
        },
        else => {
            std.debug.print("error: {s}\n\n", .{@errorName(err)});
            printUsage();
            std.process.exit(1);
        },
    };
    defer allocator.free(args.prompt_ids);

    std.debug.print("inkubalm-{s}: loading {s}\n", .{ VERSION, args.weights });

    var model = try inkubalm.InkubaLM.loadFromFile(allocator, args.weights);
    defer model.deinit();

    const generated = try model.generateGreedy(
        allocator,
        args.prompt_ids,
        args.max_new,
        args.eos,
    );
    defer allocator.free(generated);

    const stdout = std.fs.File.stdout();
    var buf: [32]u8 = undefined;
    for (generated, 0..) |tid, i| {
        if (i > 0) try stdout.writeAll(" ");
        const s = try std.fmt.bufPrint(&buf, "{d}", .{tid});
        try stdout.writeAll(s);
    }
    try stdout.writeAll("\n");
}

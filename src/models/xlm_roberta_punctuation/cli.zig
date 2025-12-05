const std = @import("std");
// Use module import from build.zig
const model_mod = @import("model.zig");
const PunctuationModel = model_mod.PunctuationModel;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 4) {
        std.debug.print("Usage: {s} <text> <weights.tl> <vocab.txt>\n", .{args[0]});
        std.debug.print("Example: {s} \"hello world\" fullstop.tl fullstop_vocab.txt\n", .{args[0]});
        std.process.exit(1);
    }

    const input_text = args[1];
    const weights_path = args[2];
    const vocab_path = args[3];

    // Load model (LITE mode - requires external weights)
    std.debug.print("Loading model from {s}...\n", .{weights_path});
    var model = PunctuationModel.init(allocator, weights_path, vocab_path) catch |err| {
        std.debug.print("Failed to load model: {}\n", .{err});
        return err;
    };
    defer model.deinit();

    std.debug.print("Processing text...\n", .{});

    // Process text
    const result = model.process(input_text) catch |err| {
        std.debug.print("Processing failed: {}\n", .{err});
        return err;
    };
    defer allocator.free(result);

    // Output result
    std.debug.print("{s}\n", .{result});
}

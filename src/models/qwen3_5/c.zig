///! C Binding for Qwen3.5-0.8B
///!
///! Usage from C:
///!   1. tomoul_qwen3_5_init(weights_path, tokenizer_path) — load model
///!   2. tomoul_qwen3_5_generate(prompt, len, max_tokens, temperature) — generate text
///!   3. tomoul_qwen3_5_get_output_buffer_ptr() / get_output_length() — read output
///!   4. tomoul_qwen3_5_destroy() — cleanup

const std = @import("std");
const tensor_mod = @import("tensor.zig");
const loader_mod = @import("loader.zig");
const model_mod = @import("model.zig");

const Tensor = tensor_mod.Tensor;
const ModelLoader = loader_mod.ModelLoader;
const Qwen3_5 = model_mod.Qwen3_5;
const Qwen3_5Config = model_mod.config.Qwen3_5Config;
const ops_mod = model_mod.ops;

const Tokenizer = model_mod.tokenizer_mod.Tokenizer;

// Global state
var gpa: ?std.heap.GeneralPurposeAllocator(.{}) = null;
var model_instance: ?Qwen3_5 = null;
var tokenizer_instance: ?Tokenizer = null;
var is_initialized: bool = false;
var global_ctx: ?ops_mod.Context = null;
var output_buffer: ?[]u8 = null;
var output_length: usize = 0;

// =============================================================================
// C API Exports
// =============================================================================

export fn tomoul_qwen3_5_init(
    weights_path: [*:0]const u8,
    tokenizer_path: [*:0]const u8,
) c_int {
    if (is_initialized) return 0;

    gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.?.allocator();

    global_ctx = ops_mod.Context.initMultiThreaded(allocator, null);
    ops_mod.initGlobalContext(&global_ctx.?);

    const weights_slice = std.mem.span(weights_path);
    const tokenizer_slice = std.mem.span(tokenizer_path);

    // Load tokenizer
    tokenizer_instance = Tokenizer.init(allocator, tokenizer_slice) catch return -1;

    // Load model
    var loader = ModelLoader.init(allocator, weights_slice) catch return -2;
    defer loader.deinit();

    model_instance = Qwen3_5.loadWithConfig(allocator, &loader, Qwen3_5Config.default, 4096) catch return -3;

    // Pre-allocate output buffer
    output_buffer = allocator.alloc(u8, 65536) catch return -4;
    output_length = 0;

    is_initialized = true;
    return 0;
}

export fn tomoul_qwen3_5_generate(
    prompt_ptr: [*]const u8,
    prompt_len: usize,
    max_tokens: usize,
    temperature: f32,
) c_int {
    if (!is_initialized) return -1;
    const allocator = gpa.?.allocator();
    const model = &model_instance.?;
    const tokenizer = &tokenizer_instance.?;
    const prompt = prompt_ptr[0..prompt_len];

    // Format + encode
    const formatted = Tokenizer.formatChatPrompt(allocator, null, prompt) catch return -2;
    defer allocator.free(formatted);

    const prompt_ids = tokenizer.encode(allocator, formatted) catch return -3;
    defer allocator.free(prompt_ids);

    // Generate
    model.reset();
    const output_ids = model.generate(prompt_ids, .{
        .max_tokens = max_tokens,
        .temperature = temperature,
    }) catch return -4;
    defer allocator.free(output_ids);

    // Decode
    const text = tokenizer.decode(allocator, output_ids) catch return -5;
    defer allocator.free(text);

    // Copy to output buffer
    const len = @min(text.len, output_buffer.?.len);
    @memcpy(output_buffer.?[0..len], text[0..len]);
    output_length = len;

    return @intCast(output_ids.len);
}

export fn tomoul_qwen3_5_get_output_buffer_ptr() ?[*]const u8 {
    if (output_buffer) |buf| return buf.ptr;
    return null;
}

export fn tomoul_qwen3_5_get_output_length() usize {
    return output_length;
}

export fn tomoul_qwen3_5_is_ready() c_int {
    return if (is_initialized) 1 else 0;
}

export fn tomoul_qwen3_5_destroy() void {
    if (!is_initialized) return;
    const allocator = gpa.?.allocator();

    if (output_buffer) |buf| allocator.free(buf);
    output_buffer = null;

    model_instance.?.deinit();
    model_instance = null;

    tokenizer_instance.?.deinit();
    tokenizer_instance = null;

    ops_mod.deinitGlobalContext();
    if (global_ctx) |*ctx| ctx.deinit();
    global_ctx = null;

    is_initialized = false;
    _ = gpa.?.deinit();
    gpa = null;
}

fn inferVariantFromPath(path: []const u8) void {
    _ = path;
}

///! C Binding for Gemma Text Generation
///!
///! Provides a C-compatible API for Gemma text generation.
///!
///! Usage from C/FFI:
///!   1. tomoul_gemma_init(weights_path, tokenizer_path) — load model + tokenizer
///!   2. tomoul_gemma_generate(prompt, prompt_len, max_tokens, temperature) — generate text
///!   3. tomoul_gemma_get_output_buffer_ptr() — get pointer to output text
///!   4. tomoul_gemma_get_output_length() — get length of generated text
///!   5. tomoul_gemma_destroy() — cleanup
///!
const std = @import("std");

const model_mod = @import("model");
const Gemma = model_mod.Gemma;
const GemmaTokens = model_mod.config.GemmaTokens;
const Tokenizer = model_mod.tokenizer.Tokenizer;

// Global state
var gpa: ?std.heap.GeneralPurposeAllocator(.{}) = null;
var model_instance: ?Gemma = null;
var tokenizer_instance: ?Tokenizer = null;
var is_initialized: bool = false;

// Output buffer for generated text
var output_buffer: ?[]u8 = null;
var output_length: usize = 0;

// =============================================================================
// C API Exports
// =============================================================================

/// Initialize the Gemma model from file paths.
/// Returns: 0 on success, negative error code on failure.
export fn tomoul_gemma_init(
    weights_path: [*:0]const u8,
    tokenizer_path: [*:0]const u8,
) c_int {
    if (is_initialized) return 0;

    gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.?.allocator();

    const weights_slice = std.mem.span(weights_path);
    const tokenizer_slice = std.mem.span(tokenizer_path);

    // Load tokenizer
    tokenizer_instance = Tokenizer.init(allocator, tokenizer_slice) catch |err| {
        std.debug.print("Failed to load tokenizer: {}\n", .{err});
        return -1;
    };

    // Load model
    model_instance = Gemma.init(allocator, weights_slice) catch |err| {
        std.debug.print("Failed to load model: {}\n", .{err});
        if (tokenizer_instance) |*t| t.deinit();
        tokenizer_instance = null;
        return -2;
    };

    is_initialized = true;
    return 0;
}

/// Generate text from a prompt.
/// prompt: Input UTF-8 text
/// prompt_len: Length of input text in bytes
/// max_tokens: Maximum tokens to generate
/// temperature: Sampling temperature (0.0 = greedy)
/// Returns: 0 on success, negative error code on failure.
/// Use get_output_buffer_ptr() and get_output_length() to retrieve the result.
export fn tomoul_gemma_generate(
    prompt: [*]const u8,
    prompt_len: usize,
    max_tokens: c_int,
    temperature: f32,
) c_int {
    if (!is_initialized) return -1;

    const allocator = gpa.?.allocator();
    const input_text = prompt[0..prompt_len];

    // Free previous output
    if (output_buffer) |buf| {
        allocator.free(buf);
        output_buffer = null;
        output_length = 0;
    }

    // Encode prompt
    const input_ids = tokenizer_instance.?.encode(input_text) catch return -2;
    defer allocator.free(input_ids);

    // Generate
    const max_tok: usize = if (max_tokens > 0) @intCast(max_tokens) else 128;
    const tokens = model_instance.?.generate(input_ids, max_tok, temperature) catch return -3;
    defer allocator.free(tokens);

    // Decode output tokens (skip prompt tokens)
    const generated = if (tokens.len > input_ids.len) tokens[input_ids.len..] else tokens[0..0];
    const text = tokenizer_instance.?.decode(generated) catch return -4;

    output_buffer = text;
    output_length = text.len;
    return 0;
}

/// Get pointer to the output text buffer.
export fn tomoul_gemma_get_output_buffer_ptr() [*]const u8 {
    if (output_buffer) |buf| return buf.ptr;
    return @as([*]const u8, @ptrCast(""));
}

/// Get length of the output text.
export fn tomoul_gemma_get_output_length() usize {
    return output_length;
}

/// Destroy the model and free all resources.
export fn tomoul_gemma_destroy() void {
    if (!is_initialized) return;

    const allocator = gpa.?.allocator();

    if (output_buffer) |buf| {
        allocator.free(buf);
        output_buffer = null;
        output_length = 0;
    }

    if (model_instance) |*m| m.deinit();
    model_instance = null;

    if (tokenizer_instance) |*t| t.deinit();
    tokenizer_instance = null;

    if (gpa) |*g| _ = g.deinit();
    gpa = null;

    is_initialized = false;
}

/// Check if the model is initialized.
export fn tomoul_gemma_is_ready() c_int {
    return if (is_initialized) 1 else 0;
}

/// Get version string.
export fn tomoul_gemma_get_version() [*:0]const u8 {
    return "0.1.0";
}

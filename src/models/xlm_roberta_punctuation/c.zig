///! C Binding for XLM-RoBERTa Punctuation Restoration
///!
///! Provides a C-compatible API for the punctuation model.
///!
///! Usage from C:
///!   1. tomoul_xlm_roberta_punctuation_init(weights_path, vocab_path) - load model
///!   2. tomoul_xlm_roberta_punctuation_process(input, input_len, output, output_capacity) - process text
///!   3. tomoul_xlm_roberta_punctuation_destroy() - cleanup
///!
const std = @import("std");

// Import core modules (provided by build.zig)
const tensor_mod = @import("tensor");
const model_mod = @import("model");

const Tensor = tensor_mod.Tensor;
const PunctuationModel = model_mod.PunctuationModel;

// Global state
var gpa: ?std.heap.GeneralPurposeAllocator(.{}) = null;
var model_instance: ?PunctuationModel = null;
var is_initialized: bool = false;

// =============================================================================
// C API Exports
// =============================================================================

/// Initialize the punctuation model from file paths
/// Returns: 0 on success, negative error code on failure
export fn tomoul_xlm_roberta_punctuation_init(
    weights_path: [*:0]const u8,
    vocab_path: [*:0]const u8,
) c_int {
    if (is_initialized) {
        return 0; // Already initialized
    }

    // Initialize allocator
    gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.?.allocator();

    // Convert C strings to Zig slices
    const weights_slice = std.mem.span(weights_path);
    const vocab_slice = std.mem.span(vocab_path);

    // Initialize the combined model (XLM-RoBERTa + Tokenizer)
    model_instance = PunctuationModel.init(allocator, weights_slice, vocab_slice) catch |err| {
        std.debug.print("Failed to load model from '{s}': {}\n", .{ weights_slice, err });
        return -1;
    };

    is_initialized = true;
    return 0;
}

/// Process input text and return punctuated text
/// input: Input UTF-8 text (null-terminated)
/// input_len: Length of input text in bytes
/// output: Buffer to write punctuated text
/// output_capacity: Maximum bytes that can be written to output
/// Returns: Length of output text on success, negative error code on failure
export fn tomoul_xlm_roberta_punctuation_process(
    input: [*]const u8,
    input_len: usize,
    output: [*]u8,
    output_capacity: usize,
) c_int {
    if (!is_initialized) {
        return -1; // Not initialized
    }

    if (input_len == 0) {
        return -2; // Empty input
    }

    const allocator = gpa.?.allocator();
    const input_text = input[0..input_len];

    // Process text through the combined model
    const result = model_instance.?.process(input_text) catch |err| {
        std.debug.print("Processing error: {}\n", .{err});
        return -3; // Processing failed
    };
    defer allocator.free(result);

    // Copy result to output buffer
    if (result.len > output_capacity) {
        return -4; // Output buffer too small
    }

    @memcpy(output[0..result.len], result);
    return @intCast(result.len);
}

/// Destroy the model and free all resources
export fn tomoul_xlm_roberta_punctuation_destroy() void {
    if (!is_initialized) {
        return;
    }

    if (model_instance) |*model| {
        model.deinit();
    }
    model_instance = null;

    if (gpa) |*g| {
        _ = g.deinit();
    }
    gpa = null;

    is_initialized = false;
}

/// Check if the model is initialized
export fn tomoul_xlm_roberta_punctuation_is_ready() c_int {
    return if (is_initialized) 1 else 0;
}

/// Get version string
export fn tomoul_xlm_roberta_punctuation_version() [*:0]const u8 {
    return "xlm_roberta_punctuation-v1.0.0";
}

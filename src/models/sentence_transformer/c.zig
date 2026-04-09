///! C Binding for Sentence Transformer (all-MiniLM-L6-v2)
///!
///! Provides a C-compatible API for generating 384-dim sentence embeddings.
///!
///! Usage from C:
///!   1. tomoul_sentence_transformer_init(weights_path, vocab_path) — load model
///!   2. tomoul_sentence_transformer_embed(text, text_len, output_384) — embed text
///!   3. tomoul_sentence_transformer_destroy() — cleanup
///!
const std = @import("std");

// Import core modules (provided by build.zig)
const model_mod = @import("model");
const SentenceTransformer = model_mod.SentenceTransformer;

// Global state
var gpa: ?std.heap.GeneralPurposeAllocator(.{}) = null;
var model_instance: ?SentenceTransformer = null;
var is_initialized: bool = false;

// =============================================================================
// C API Exports
// =============================================================================

/// Initialize the sentence transformer from file paths.
/// Returns: 0 on success, negative error code on failure.
export fn tomoul_sentence_transformer_init(
    weights_path: [*:0]const u8,
    vocab_path: [*:0]const u8,
) c_int {
    if (is_initialized) {
        return 0;
    }

    gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.?.allocator();

    const weights_slice = std.mem.span(weights_path);
    const vocab_slice = std.mem.span(vocab_path);

    model_instance = SentenceTransformer.init(allocator, weights_slice, vocab_slice) catch |err| {
        std.debug.print("Failed to load model: {}\n", .{err});
        return -1;
    };

    is_initialized = true;
    return 0;
}

/// Embed a text string into a 384-dimensional normalized vector.
/// text: Input UTF-8 text
/// text_len: Length of input text in bytes
/// output: Pointer to f32[384] buffer to write the embedding
/// Returns: 0 on success, negative error code on failure.
export fn tomoul_sentence_transformer_embed(
    text: [*]const u8,
    text_len: usize,
    output: [*]f32,
) c_int {
    if (!is_initialized) {
        return -1; // Not initialized
    }

    if (text_len == 0) {
        return -2; // Empty input
    }

    const input_text = text[0..text_len];

    const embedding = model_instance.?.embed(input_text) catch {
        return -3; // Embedding failed
    };

    @memcpy(output[0..384], &embedding);
    return 0;
}

/// Destroy the model and free all resources.
export fn tomoul_sentence_transformer_destroy() void {
    if (!is_initialized) {
        return;
    }

    if (model_instance) |*m| {
        m.deinit();
    }
    model_instance = null;

    if (gpa) |*g| {
        _ = g.deinit();
    }
    gpa = null;

    is_initialized = false;
}

/// Check if the model is initialized.
/// Returns: 1 if ready, 0 if not.
export fn tomoul_sentence_transformer_is_ready() c_int {
    return if (is_initialized) 1 else 0;
}

/// Get version string.
export fn tomoul_sentence_transformer_version() [*:0]const u8 {
    return "0.1.0";
}

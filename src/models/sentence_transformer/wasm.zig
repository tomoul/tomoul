///! WebAssembly Binding for Sentence Transformer (all-MiniLM-L6-v2)
///!
///! Exposes the sentence embedding model to JavaScript/WebAssembly.
///! Produces normalized 384-dimensional vectors for semantic similarity.
///!
///! Usage from JavaScript:
///!   1. Call init() to initialize the model
///!   2. Get input buffer with get_input_buffer_ptr()
///!   3. Write UTF-8 text bytes to the buffer
///!   4. Call embed(text_len) — writes 384 floats to output buffer
///!   5. Read result from get_output_buffer_ptr() (384 × f32 = 1536 bytes)
///!
const std = @import("std");

// Import core modules (provided by build.zig)
const model_mod = @import("model");
const SentenceTransformer = model_mod.SentenceTransformer;

// Model weights and vocab are embedded at compile time
const model_bytes = @embedFile("model_weights");
const vocab_bytes = @embedFile("vocab");

// =============================================================================
// Memory Management
// =============================================================================

// Static heap — model ~87MB (F32) or ~24MB (Q8K), inference ~2MB temp
var heap: [96 * 1024 * 1024]u8 = undefined;
var fba = std.heap.FixedBufferAllocator.init(&heap);
const allocator = fba.allocator();

// High-water mark after model initialization
var init_end_index: usize = 0;

// Text input buffer
const MAX_INPUT_BYTES: usize = 8 * 1024; // 8KB — 512 tokens is ~2KB typical
var input_buffer: [MAX_INPUT_BYTES]u8 = undefined;

// Embedding output buffer (384 × f32)
const EMBEDDING_DIM: usize = 384;
var output_buffer: [EMBEDDING_DIM]f32 = undefined;

// Global model instance
var model_instance: ?SentenceTransformer = null;
var is_initialized: bool = false;

// =============================================================================
// Exported Functions (callable from JavaScript)
// =============================================================================

/// Initialize the sentence transformer from embedded weights and vocab.
/// Returns: 1 on success, 0 on failure.
export fn init() u32 {
    if (is_initialized) {
        return 1;
    }

    model_instance = SentenceTransformer.initFromBytes(allocator, model_bytes, vocab_bytes) catch {
        return 0;
    };

    init_end_index = fba.end_index;
    is_initialized = true;
    return 1;
}

/// Get pointer to the input text buffer.
/// JavaScript should write UTF-8 text bytes here.
export fn get_input_buffer_ptr() [*]u8 {
    return &input_buffer;
}

/// Get the maximum number of bytes the input buffer can hold.
export fn get_max_input_bytes() usize {
    return MAX_INPUT_BYTES;
}

/// Get pointer to the output embedding buffer (384 × f32).
export fn get_output_buffer_ptr() [*]const f32 {
    return &output_buffer;
}

/// Embed the text in the input buffer.
/// text_len: Number of UTF-8 bytes written to the input buffer.
/// Returns: EMBEDDING_DIM (384) on success, 0 on error.
export fn embed(text_len: usize) u32 {
    if (!is_initialized) {
        return 0;
    }

    if (text_len == 0 or text_len > MAX_INPUT_BYTES) {
        return 0;
    }

    // Reset allocator to post-init state
    fba.end_index = init_end_index;

    const input_text = input_buffer[0..text_len];

    const embedding = model_instance.?.embed(input_text) catch {
        return 0;
    };

    @memcpy(&output_buffer, &embedding);
    return EMBEDDING_DIM;
}

/// Check if the model is initialized.
/// Returns: 1 if ready, 0 if not.
export fn is_ready() u32 {
    return if (is_initialized) 1 else 0;
}

/// Get version identifier.
export fn get_version() u32 {
    return 0x000100; // 0.1.0
}

/// Reset the allocator to reclaim memory (no persistent state to reset).
export fn reset() void {
    if (is_initialized) {
        fba.end_index = init_end_index;
    }
}

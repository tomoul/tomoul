///! WebAssembly Binding for XLM-RoBERTa Punctuation Restoration
///!
///! This exposes the punctuation model to JavaScript/WebAssembly.
///! The model takes unpunctuated text and returns properly punctuated text.
///!
///! Usage from JavaScript:
///!   1. Call init() to initialize the model
///!   2. Get input buffer with get_input_buffer_ptr()
///!   3. Write UTF-8 text bytes to that memory location
///!   4. Call process_text(text_len) to get punctuated text
///!   5. Read result from get_output_buffer_ptr() with length get_output_length()
///!
const std = @import("std");

// Import core modules (provided by build.zig)
const tensor_mod = @import("tensor");
const model_mod = @import("model");

const Tensor = tensor_mod.Tensor;
const PunctuationModel = model_mod.PunctuationModel;

// Model weights and vocab are provided via anonymous import from build.zig
const model_bytes = @embedFile("model_weights");
const vocab_bytes = @embedFile("vocab");

// =============================================================================
// Memory Management
// =============================================================================

// Static heap for all allocations
// Model weights use ~1GB, inference needs ~100MB temp allocations
// For WASM, we need a much smaller heap - the model may need streaming
var heap: [128 * 1024 * 1024]u8 = undefined; // 128MB for now
var fba = std.heap.FixedBufferAllocator.init(&heap);
const allocator = fba.allocator();

// High-water mark after model initialization
var init_end_index: usize = 0;

// Text input/output buffers
const MAX_INPUT_BYTES: usize = 16 * 1024; // 16KB max input text
const MAX_OUTPUT_BYTES: usize = 32 * 1024; // 32KB max output (punctuation may add chars)
var input_buffer: [MAX_INPUT_BYTES]u8 = undefined;
var output_buffer: [MAX_OUTPUT_BYTES]u8 = undefined;
var output_length: usize = 0;

// Global model instance
var model_instance: ?PunctuationModel = null;
var is_initialized: bool = false;

// =============================================================================
// Exported Functions (callable from JavaScript)
// =============================================================================

/// Initialize the punctuation model from embedded weights
/// Returns: 1 on success, 0 on failure
export fn init() u32 {
    if (is_initialized) {
        return 1; // Already initialized
    }

    // Initialize combined model (XLM-RoBERTa + Tokenizer)
    model_instance = PunctuationModel.initFromBytes(allocator, model_bytes, vocab_bytes) catch {
        return 0;
    };

    // Record high-water mark after init
    init_end_index = fba.end_index;

    is_initialized = true;
    return 1;
}

/// Get pointer to the input text buffer
/// JavaScript should write UTF-8 text bytes here
export fn get_input_buffer_ptr() [*]u8 {
    return &input_buffer;
}

/// Get the maximum number of bytes the input buffer can hold
export fn get_max_input_bytes() usize {
    return MAX_INPUT_BYTES;
}

/// Get pointer to the output text buffer
/// Read the punctuated text from here after process_text()
export fn get_output_buffer_ptr() [*]const u8 {
    return &output_buffer;
}

/// Get the length of the output text after process_text()
export fn get_output_length() usize {
    return output_length;
}

/// Process input text and return punctuated text
/// text_len: Number of UTF-8 bytes written to the input buffer
/// Returns: Length of punctuated text, or negative on error:
///   -1: Not initialized
///   -2: Invalid text length
///   -3: Processing failed
///   -4: Output too large
export fn process_text(text_len: usize) i32 {
    if (!is_initialized) {
        return -1;
    }

    if (text_len == 0 or text_len > MAX_INPUT_BYTES) {
        return -2;
    }

    // Reset allocator to post-init state
    fba.end_index = init_end_index;

    const input_text = input_buffer[0..text_len];

    // Process text through the combined model (tokenize + forward + apply labels)
    const result = model_instance.?.process(input_text) catch {
        return -3;
    };
    defer allocator.free(result);

    // Copy result to output buffer
    if (result.len > MAX_OUTPUT_BYTES) {
        return -4;
    }

    @memcpy(output_buffer[0..result.len], result);
    output_length = result.len;

    return @intCast(result.len);
}

/// Reset the model state (if any stateful components)
/// For punctuation, this is mostly a no-op but included for API consistency
export fn reset() void {
    output_length = 0;
}

/// Check if the model is initialized and ready
export fn is_ready() u32 {
    return if (is_initialized) 1 else 0;
}

/// Get version string (pointer to static memory)
export fn get_version() [*:0]const u8 {
    return "xlm_roberta_punctuation-v1.0.0";
}

/// Get quantization format of loaded model
/// Returns: 0 = not loaded, 1 = float32, 2 = Q8_0
export fn get_quant_format() u32 {
    if (!is_initialized) {
        return 0;
    }

    return switch (model_instance.?.model.quant_format) {
        .f32 => 1, // float32
        .q8_0 => 2, // Q8_0 quantized
        .q4_0 => 3, // Q4_0 quantized
        .q8_k => 4, // Q8_K block-wise quantized
    };
}

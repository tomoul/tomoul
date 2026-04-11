///! C Binding for Sentence Transformer (all-MiniLM-L6-v2)
///!
///! Provides a C-compatible API for generating 384-dim sentence embeddings.
///!
///! Usage from C:
///!   1. tomoul_sentence_transformer_init(weights_path, vocab_path) — load model
///!   2. tomoul_sentence_transformer_embed(text, text_len, output_384) — embed text
///!   3. tomoul_sentence_transformer_destroy() — cleanup
///!
///! GPU acceleration (optional):
///!   1. tomoul_sentence_transformer_init(weights_path, vocab_path) — load model (CPU, required first)
///!   2. tomoul_sentence_transformer_gpu_init() — initialize GPU backend (auto-detects Vulkan/Metal)
///!   3. tomoul_sentence_transformer_gpu_embed(text, text_len, output_384) — embed on GPU
///!   4. tomoul_sentence_transformer_gpu_destroy() — cleanup GPU resources
///!   5. tomoul_sentence_transformer_destroy() — cleanup CPU model
///!
const std = @import("std");

// Import core modules (provided by build.zig)
const model_mod = @import("model");
const SentenceTransformer = model_mod.SentenceTransformer;
const SentenceTransformerModel = model_mod.SentenceTransformerModel;
const gpu_model_mod = @import("gpu_model");
const GpuModel = gpu_model_mod.GpuModel;

// Global state
var gpa: ?std.heap.GeneralPurposeAllocator(.{}) = null;
var model_instance: ?SentenceTransformer = null;
var is_initialized: bool = false;

// GPU state
var gpu_instance: ?GpuModel = null;
var gpu_initialized: bool = false;

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

    // Clean up GPU first if active
    if (gpu_initialized) {
        tomoul_sentence_transformer_gpu_destroy();
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

/// Embed a batch of texts in a single forward pass.
/// texts: array of pointers to UTF-8 text strings
/// text_lengths: array of byte lengths per text
/// batch_size: number of texts
/// output: pointer to batch_size * 384 floats
/// Returns: 0 on success, negative error code on failure.
export fn tomoul_sentence_transformer_embed_batch(
    texts: [*]const [*]const u8,
    text_lengths: [*]const usize,
    batch_size: usize,
    output: [*]f32,
) c_int {
    if (!is_initialized) {
        return -1;
    }

    if (batch_size == 0) {
        return -2;
    }

    const allocator = gpa.?.allocator();

    // Build slice array
    const text_slices = allocator.alloc([]const u8, batch_size) catch return -4;
    defer allocator.free(text_slices);

    for (0..batch_size) |i| {
        if (text_lengths[i] == 0) return -2;
        text_slices[i] = texts[i][0..text_lengths[i]];
    }

    const embeddings = model_instance.?.embedBatch(text_slices) catch return -3;
    defer allocator.free(embeddings);

    // Copy results to output buffer
    for (0..batch_size) |i| {
        @memcpy(output[i * 384 ..][0..384], &embeddings[i]);
    }

    return 0;
}

/// Get version string.
export fn tomoul_sentence_transformer_version() [*:0]const u8 {
    return "0.1.0";
}

// =============================================================================
// GPU C API Exports
// =============================================================================

/// Initialize GPU backend for accelerated embedding.
/// Requires: tomoul_sentence_transformer_init() must be called first.
/// Auto-detects Vulkan (Linux/Windows) or Metal (macOS) backend.
/// Returns: 0 on success, -1 if CPU model not loaded, -5 if GPU init failed.
export fn tomoul_sentence_transformer_gpu_init() c_int {
    if (!is_initialized) {
        return -1; // CPU model must be loaded first
    }

    if (gpu_initialized) {
        return 0; // Already initialized
    }

    const allocator = gpa.?.allocator();

    gpu_instance = GpuModel.init(allocator, &model_instance.?.model) catch {
        return -5; // GPU initialization failed
    };

    gpu_initialized = true;
    return 0;
}

/// Check if GPU backend is active (not CPU fallback).
/// Returns: 1 if GPU active, 0 if not initialized or using CPU fallback.
export fn tomoul_sentence_transformer_gpu_is_active() c_int {
    if (!gpu_initialized) return 0;
    if (gpu_instance) |*g| {
        return if (g.isGpuActive()) @as(c_int, 1) else @as(c_int, 0);
    }
    return 0;
}

/// Get GPU device name (null-terminated).
/// Returns: device name string, or "none" if GPU not initialized.
export fn tomoul_sentence_transformer_gpu_device_name() [*:0]const u8 {
    if (!gpu_initialized) return "none";
    if (gpu_instance) |*g| {
        const name = g.getDeviceName();
        // Return pointer to the slice data (backed by HAL/Vulkan static or allocated memory)
        // The name is stable for the lifetime of the GPU instance.
        if (name.len > 0 and name[name.len - 1] == 0) {
            return @ptrCast(name.ptr);
        }
        // Fallback: not null-terminated, return a static string
        return "GPU";
    }
    return "none";
}

/// Embed a text string on GPU into a 384-dimensional normalized vector.
/// Returns: 0 on success, -1 if not initialized, -5 if GPU not ready, -6 if GPU embed failed.
export fn tomoul_sentence_transformer_gpu_embed(
    text: [*]const u8,
    text_len: usize,
    output: [*]f32,
) c_int {
    if (!is_initialized) return -1;
    if (!gpu_initialized) return -5;

    if (text_len == 0) return -2;

    const input_text = text[0..text_len];

    const embedding = gpu_instance.?.embed(&model_instance.?.tokenizer, input_text) catch {
        return -6;
    };

    @memcpy(output[0..384], &embedding);
    return 0;
}

/// Embed a batch of texts on GPU in a single dispatch.
/// Returns: 0 on success, negative error code on failure.
export fn tomoul_sentence_transformer_gpu_embed_batch(
    texts: [*]const [*]const u8,
    text_lengths: [*]const usize,
    batch_size: usize,
    output: [*]f32,
) c_int {
    if (!is_initialized) return -1;
    if (!gpu_initialized) return -5;

    if (batch_size == 0) return -2;

    const allocator = gpa.?.allocator();

    // Build text slices
    const text_slices = allocator.alloc([]const u8, batch_size) catch return -4;
    defer allocator.free(text_slices);

    for (0..batch_size) |i| {
        if (text_lengths[i] == 0) return -2;
        text_slices[i] = texts[i][0..text_lengths[i]];
    }

    // Allocate results
    const results = allocator.alloc([384]f32, batch_size) catch return -4;
    defer allocator.free(results);

    gpu_instance.?.embedBatch(&model_instance.?.tokenizer, text_slices, results) catch {
        return -6;
    };

    // Copy to flat output
    for (0..batch_size) |i| {
        @memcpy(output[i * 384 ..][0..384], &results[i]);
    }

    return 0;
}

/// Destroy GPU resources. CPU model remains loaded.
export fn tomoul_sentence_transformer_gpu_destroy() void {
    if (!gpu_initialized) return;

    if (gpu_instance) |*g| {
        g.deinit();
    }
    gpu_instance = null;
    gpu_initialized = false;
}

///! WebAssembly Binding for Silero VAD
///!
///! This is a "dumb wrapper" that exposes the SileroVAD model to JavaScript.
///! The actual model logic lives in src/models/silero_vad.zig.
///!
///! Usage from JavaScript:
///!   1. Call init() to initialize the model
///!   2. Get input buffer pointer with get_input_buffer_ptr()
///!   3. Write audio samples (f32) to that memory location
///!   4. Call process_audio(num_samples) to get speech probability
///!
const std = @import("std");

// Import core modules (provided by build.zig)
const tensor_mod = @import("tensor");
const vad_mod = @import("vad");

const Tensor = tensor_mod.Tensor;
const SileroVAD = vad_mod.SileroVAD;

// Model weights are provided via anonymous import from build.zig
const model_bytes = @embedFile("model_weights");

// =============================================================================
// Memory Management
// =============================================================================

// Static heap for all allocations (~4MB total)
// Model weights use ~1.8MB, forward pass needs ~200KB of temporary allocations
var heap: [4 * 1024 * 1024]u8 = undefined;
var fba = std.heap.FixedBufferAllocator.init(&heap);
const allocator = fba.allocator();

// High-water mark after model initialization
// We reset to this point after each forward pass to reclaim temporary memory
var init_end_index: usize = 0;

// Audio input buffer (1536 samples = 96ms @ 16kHz, supports various chunk sizes)
const MAX_INPUT_SAMPLES: usize = 1536;
var input_buffer: [MAX_INPUT_SAMPLES]f32 = undefined;

// Global VAD instance
var vad_instance: ?SileroVAD = null;
var is_initialized: bool = false;

// =============================================================================
// Exported Functions (callable from JavaScript)
// =============================================================================

/// Initialize the VAD model from embedded weights
/// Returns: 1 on success, 0 on failure
export fn init() u32 {
    if (is_initialized) {
        return 1; // Already initialized
    }

    vad_instance = SileroVAD.initFromBytes(allocator, model_bytes) catch {
        return 0; // Initialization failed
    };

    // Record high-water mark after init - we'll reset to here after each forward pass
    init_end_index = fba.end_index;

    is_initialized = true;
    return 1;
}

/// Get pointer to the input audio buffer
/// JavaScript should write f32 audio samples here
export fn get_input_buffer_ptr() [*]f32 {
    return &input_buffer;
}

/// Get the maximum number of samples the input buffer can hold
export fn get_max_input_samples() usize {
    return MAX_INPUT_SAMPLES;
}

/// Process audio samples and return speech probability
/// num_samples: Number of f32 samples written to the input buffer
/// Returns: Speech probability [0.0, 1.0], or negative on error:
///   -1.0: Not initialized
///   -2.0: Invalid sample count
///   -3.0: Failed to create tensor
///   -4.0: Forward pass failed
export fn process_audio(num_samples: usize) f32 {
    if (!is_initialized) {
        return -1.0; // Not initialized
    }

    if (num_samples == 0 or num_samples > MAX_INPUT_SAMPLES) {
        return -2.0; // Invalid sample count
    }

    // Reset allocator to post-init state to reclaim memory from previous forward passes
    // This is safe because forward() doesn't store any persistent allocations
    fba.end_index = init_end_index;

    // Create a tensor from the input buffer
    var input_shape = [_]usize{num_samples};
    var input_tensor = Tensor.init(allocator, &input_shape) catch {
        return -3.0; // Failed to create tensor
    };
    // No defer needed - we reset allocator at start of each call

    // Copy input buffer data to tensor
    @memcpy(input_tensor.data, input_buffer[0..num_samples]);

    // Run VAD forward pass
    const prob = vad_instance.?.forward(&input_tensor) catch {
        return -4.0; // Forward pass failed
    };

    return prob;
}

/// Reset the VAD state (call between different audio streams)
export fn reset_state() void {
    if (is_initialized) {
        vad_instance.?.resetStates();
    }
}

/// Check if the model is initialized
export fn is_ready() u32 {
    return if (is_initialized) 1 else 0;
}

/// Get the model version embedded in the binary
export fn get_version() u32 {
    return 1; // Version 1 of the Wasm API
}

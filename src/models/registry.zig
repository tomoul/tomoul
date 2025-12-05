// model_registry.zig
// Master list of all models in Tomoul
// Add new models here - build.zig will automatically generate targets for them
//
// Convention over configuration:
// Given a model name like "silero_vad", paths are derived as:
//   - wasm_binding  → src/models/silero_vad/wasm.zig
//   - c_binding     → src/models/silero_vad/c.zig
//   - model_module  → src/models/silero_vad/model.zig
//   - weights_path  → artifacts/silero_vad.tl
//   - example_dir   → examples/silero-vad (if has_example=true)
//   - hf_repo       → tomoul/silero-vad
//
// Override any path by setting it explicitly.
//
// Weight variants:
// For models with multiple quantization levels (float32, q8, q4), use weight_variants
// to define all variants. Each variant has a suffix and description.
// Example: { .suffix = "-q8", .path = "artifacts/model_q8.tl", .description = "Q8 quantized" }

const std = @import("std");

pub const ModelKind = enum { audio, text, vision };

pub const ReleaseMode = enum {
    small, // ReleaseSmall - smaller binary, slightly slower
    fast, // ReleaseFast - larger binary, faster execution
};

/// Weight variant for quantized models
pub const WeightVariant = struct {
    suffix: []const u8, // e.g., "-q8", "-q4" (empty for default/float32)
    path: []const u8, // e.g., "artifacts/model_q8.tl"
    description: []const u8 = "", // e.g., "Q8 quantized - 4x smaller"
};

pub const ModelConfig = struct {
    name: []const u8, // e.g., "silero_vad"
    kind: ModelKind, // Audio, Text, Vision
    description: []const u8 = "", // Human-readable description

    // Export symbols for the wasm module
    export_symbols: []const []const u8,

    // Optional path overrides (if not following convention)
    wasm_binding: ?[]const u8 = null,
    c_binding: ?[]const u8 = null,
    model_module: ?[]const u8 = null,
    cli_module: ?[]const u8 = null, // CLI executable source
    weights_path: ?[]const u8 = null, // Default weights (float32)
    example_dir: ?[]const u8 = null,
    hf_repo: ?[]const u8 = null,

    // Weight variants for quantized models (Q8, Q4, etc.)
    // Each variant creates a separate build target: {name}{suffix}
    weight_variants: []const WeightVariant = &.{},

    // Flags
    has_example: bool = false, // Whether to copy wasm to examples/{name-with-dashes}
    has_cli: bool = false, // Whether model has a CLI executable
    supports_bundled: bool = true,
    release_mode: ReleaseMode = .small, // Default to small for VAD-like models
};

// Helper to create model config with conventional paths
pub fn model(
    name: []const u8,
    kind: ModelKind,
    description: []const u8,
    export_symbols: []const []const u8,
) ModelConfig {
    return .{
        .name = name,
        .kind = kind,
        .description = description,
        .export_symbols = export_symbols,
        // Paths derived by build.zig from name
    };
}

pub const models = [_]ModelConfig{
    .{
        .name = "silero_vad",
        .kind = .audio,
        .description = "Voice Activity Detection",
        .export_symbols = &.{
            "init",
            "get_input_buffer_ptr",
            "get_max_input_samples",
            "process_audio",
            "reset_state",
            "is_ready",
            "get_version",
        },
        .has_example = true,
        .has_cli = true,
    },
    .{
        .name = "fullstop-punctuation-multilang-large",
        .kind = .text,
        .description = "Punctuation Restoration (XLM-RoBERTa Large)",
        .wasm_binding = "src/models/xlm_roberta_punctuation/wasm.zig",
        .c_binding = "src/models/xlm_roberta_punctuation/c.zig",
        .model_module = "src/models/xlm_roberta_punctuation/model.zig",
        .weights_path = "artifacts/fullstop_punctuation_multilang_large.tl",
        .hf_repo = "tomoul/fullstop-punctuation-multilang-large",
        .export_symbols = &.{
            "init",
            "get_input_buffer_ptr",
            "get_max_input_bytes",
            "get_output_buffer_ptr",
            "get_output_length",
            "process_text",
            "reset",
            "is_ready",
            "get_version",
        },
        .weight_variants = &.{
            .{ .suffix = "-q8", .path = "artifacts/fullstop_punctuation_multilang_large_q8.tl", .description = "Q8 quantized - 4x smaller, 1.9x faster" },
            .{ .suffix = "-q4", .path = "artifacts/fullstop_punctuation_multilang_large_q4.tl", .description = "Q4 quantized - 8x smaller" },
        },
        .has_example = false,
        .supports_bundled = false, // Weights loaded separately at runtime
        .release_mode = .fast, // Large model benefits from ReleaseFast (~15% faster)
    },
    // Add more models - paths are derived from name automatically:
    // .{
    //     .name = "whisper_tiny",
    //     .kind = .audio,
    //     .description = "Speech Recognition",
    //     .export_symbols = &.{ "init", "transcribe", ... },
    // },
};

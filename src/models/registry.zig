// model_registry.zig
// Master list of all models in Tomoul
// Add new models here - build.zig will automatically generate targets for them
//
// Convention over configuration:
// Given a model name like "silero_vad", paths are derived as:
//   - wasm_binding  → src/models/silero_vad/wasm.zig
//   - c_binding     → src/models/silero_vad/c.zig
//   - model_module  → src/models/silero_vad/model.zig
//   - weights_path  → models/silero_vad.tl
//   - example_dir   → examples/silero-vad (if has_example=true)
//   - hf_repo       → tomoul/silero-vad
//
// Override any path by setting it explicitly.

const std = @import("std");

pub const ModelKind = enum { audio, text, vision };

pub const ModelConfig = struct {
    name: []const u8, // e.g., "silero_vad"
    kind: ModelKind, // Audio, Text, Vision
    description: []const u8 = "", // Human-readable description

    // Export symbols for the wasm module
    export_symbols: []const []const u8,

    // Flags
    has_example: bool = false, // Whether to copy wasm to examples/{name-with-dashes}
    supports_bundled: bool = true,
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
    },
    // Add more models - paths are derived from name automatically:
    // .{
    //     .name = "whisper_tiny",
    //     .kind = .audio,
    //     .description = "Speech Recognition",
    //     .export_symbols = &.{ "init", "transcribe", ... },
    // },
};

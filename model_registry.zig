// model_registry.zig
// Master list of all models in Tomoul
// Add new models here - build.zig will automatically generate targets for them

pub const ModelConfig = struct {
    name: []const u8, // e.g., "silero_vad"
    kind: ModelKind, // Audio, Text, Vision
    wasm_binding: []const u8, // Path to wasm_*.zig
    model_module: []const u8, // Path to the model implementation
    weights_path: []const u8, // Path to .tl file (for bundled builds)
    example_dir: ?[]const u8 = null, // Optional example directory to copy wasm to
    supports_bundled: bool = true,
    // Export symbols for the wasm module
    export_symbols: []const []const u8,
    // Hugging Face registry info
    hf_repo: []const u8 = "", // e.g., "tomoul/silero-vad"
    description: []const u8 = "", // Human-readable description
};

pub const ModelKind = enum { audio, text, vision };

// THIS IS YOUR MASTER LIST
pub const models = [_]ModelConfig{
    .{
        .name = "silero_vad",
        .kind = .audio,
        .wasm_binding = "src/bindings/wasm_vad.zig",
        .model_module = "src/models/silero_vad.zig",
        .weights_path = "models/silero_vad.tl",
        .example_dir = "examples/silero-vad",
        .supports_bundled = true,
        .export_symbols = &.{
            "init",
            "get_input_buffer_ptr",
            "get_max_input_samples",
            "process_audio",
            "reset_state",
            "is_ready",
            "get_version",
        },
        .hf_repo = "tomoul/silero-vad",
        .description = "Voice Activity Detection",
    },
    // Add more models here:
    // .{
    //     .name = "bert_punct",
    //     .kind = .text,
    //     .wasm_binding = "src/bindings/wasm_punct.zig",
    //     .model_module = "src/models/bert_punct.zig",
    //     .weights_path = "models/bert_punct.tl",
    //     .supports_bundled = true,
    //     .export_symbols = &.{ "init", "punctuate", ... },
    // },
};

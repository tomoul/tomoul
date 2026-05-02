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
    vocab_path: ?[]const u8 = null, // Vocab file for text models (embedded alongside weights)
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
            .{ .suffix = "-q8k", .path = "artifacts/fullstop_punctuation_multilang_large_q8k.tl", .description = "Q8_K block-wise - 3.5x smaller, better accuracy" },
            .{ .suffix = "-q4", .path = "artifacts/fullstop_punctuation_multilang_large_q4.tl", .description = "Q4 quantized - 8x smaller" },
        },
        .has_example = false,
        .supports_bundled = false, // Weights loaded separately at runtime
        .release_mode = .fast, // Large model benefits from ReleaseFast (~15% faster)
    },
    .{
        .name = "fullstop-punctuation-multilingual-sonar-base",
        .kind = .text,
        .description = "Punctuation Restoration (XLM-RoBERTa Base - 3.5x faster, 3.7x smaller than Large)",
        .wasm_binding = "src/models/xlm_roberta_punctuation/wasm.zig",
        .c_binding = "src/models/xlm_roberta_punctuation/c.zig",
        .model_module = "src/models/xlm_roberta_punctuation/model.zig",
        .weights_path = "artifacts/fullstop_punctuation_multilingual_sonar_base.tl",
        .hf_repo = "tomoul/fullstop-punctuation-multilingual-sonar-base",
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
            .{ .suffix = "-q8", .path = "artifacts/fullstop_punctuation_multilingual_sonar_base_q8.tl", .description = "Q8 quantized - 4x smaller" },
        },
        .has_example = false,
        .supports_bundled = false, // Weights loaded separately at runtime
        .release_mode = .fast,
    },
    // Sentence Transformer (all-MiniLM-L6-v2) — embedding model
    .{
        .name = "sentence_transformer",
        .kind = .text,
        .description = "Sentence Embeddings (all-MiniLM-L6-v2, 384-dim)",
        .model_module = "src/models/sentence_transformer/model.zig",
        .cli_module = "src/models/sentence_transformer/cli.zig",
        .wasm_binding = "src/models/sentence_transformer/wasm.zig",
        .c_binding = "src/models/sentence_transformer/c.zig",
        .weights_path = "artifacts/all_minilm_l6_v2.tl",
        .vocab_path = "artifacts/all_minilm_l6_v2_vocab.txt",
        .export_symbols = &.{
            "init",
            "embed",
            "get_input_buffer_ptr",
            "get_max_input_bytes",
            "get_output_buffer_ptr",
            "is_ready",
            "get_version",
            "reset",
            "get_heap_used",
            "get_heap_size",
            "gpu_init",
            "gpu_is_active",
            "gpu_embed",
            "gpu_destroy",
        },
        .weight_variants = &.{
            .{ .suffix = "-q8k", .path = "artifacts/all_minilm_l6_v2_q8k.tl", .description = "Q8_K block-wise - 3.6x smaller, 0.9997+ accuracy" },
        },
        .has_example = false,
        .has_cli = true,
        .supports_bundled = true,
        .release_mode = .fast,
    },
    // Whisper Speech-to-Text models (Phase 10)
    .{
        .name = "whisper-tiny",
        .kind = .audio,
        .description = "Speech Recognition (Whisper Tiny - 39M params)",
        .wasm_binding = "src/models/whisper/wasm.zig",
        .c_binding = "src/models/whisper/c.zig",
        .model_module = "src/models/whisper/model.zig",
        .weights_path = "artifacts/whisper_tiny.tl",
        .hf_repo = "tomoul/whisper-tiny",
        .export_symbols = &.{
            "init",
            "get_mel_buffer_ptr",
            "get_max_mel_frames",
            "get_output_buffer_ptr",
            "get_output_length",
            "transcribe",
            "reset",
            "is_ready",
            "get_version",
        },
        .weight_variants = &.{
            .{ .suffix = "-q8", .path = "artifacts/whisper_tiny_q8.tl", .description = "Q8 quantized - 4x smaller" },
        },
        .has_example = false,
        .supports_bundled = false, // Weights loaded separately at runtime
        .release_mode = .fast,
    },
    // Qwen3.5-0.8B — Hybrid DeltaNet + Attention Language Model
    .{
        .name = "qwen3_5-0.8b",
        .kind = .text,
        .description = "Text Generation (Qwen3.5-0.8B — Hybrid DeltaNet + GQA, 0.8B params)",
        .model_module = "src/models/qwen3_5/model.zig",
        .cli_module = "src/models/qwen3_5/cli.zig",
        .c_binding = "src/models/qwen3_5/c.zig",
        .weights_path = "artifacts/qwen3_5_0.8b.tl",
        .vocab_path = "artifacts/qwen3_5_vocab.bin",
        .export_symbols = &.{
            "tomoul_qwen3_5_init",
            "tomoul_qwen3_5_generate",
            "tomoul_qwen3_5_get_output_buffer_ptr",
            "tomoul_qwen3_5_get_output_length",
            "tomoul_qwen3_5_is_ready",
            "tomoul_qwen3_5_destroy",
        },
        .weight_variants = &.{
            .{ .suffix = "-q8k", .path = "artifacts/qwen3_5_0.8b_q8k.tl", .description = "Q8_K block-wise - ~0.8 GB" },
            .{ .suffix = "-q4", .path = "artifacts/qwen3_5_0.8b_q4.tl", .description = "Q4 quantized - ~0.4 GB" },
        },
        .has_example = false,
        .has_cli = true,
        .supports_bundled = false,
        .release_mode = .fast,
    },
    .{
        .name = "whisper-base",
        .kind = .audio,
        .description = "Speech Recognition (Whisper Base - 74M params)",
        .wasm_binding = "src/models/whisper/wasm.zig",
        .c_binding = "src/models/whisper/c.zig",
        .model_module = "src/models/whisper/model.zig",
        .weights_path = "artifacts/whisper_base.tl",
        .hf_repo = "tomoul/whisper-base",
        .export_symbols = &.{
            "init",
            "get_mel_buffer_ptr",
            "get_max_mel_frames",
            "get_output_buffer_ptr",
            "get_output_length",
            "transcribe",
            "reset",
            "is_ready",
            "get_version",
        },
        .weight_variants = &.{
            .{ .suffix = "-q8", .path = "artifacts/whisper_base_q8.tl", .description = "Q8 quantized - 4x smaller" },
        },
        .has_example = false,
        .supports_bundled = false, // Weights loaded separately at runtime
        .release_mode = .fast,
    },
};

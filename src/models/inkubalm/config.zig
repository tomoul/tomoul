// src/models/inkubalm/config.zig
//
// Compile-time configuration for InkubaLM-0.4B.
// Source: lelapa/InkubaLM-0.4B (HuggingFace)
//
// Architecture is plain Llama-2 / Llama-3 style: dense decoder, MHA (no GQA),
// SwiGLU FFN, RoPE, RMSNorm, tied embeddings. So this config is consumed
// directly by src/arch/llama.zig — no model-specific architecture code.
//
// Mirrors HF config.json:
//   { "hidden_size": 2048, "intermediate_size": 5632,
//     "num_attention_heads": 32, "num_hidden_layers": 8,
//     "num_key_value_heads": 32, "vocab_size": 61788,
//     "max_position_embeddings": 2048, "rms_norm_eps": 1e-5,
//     "rope_theta": 10000.0, "tie_word_embeddings": true }

const llama = @import("llama.zig");

pub const inkubalm_0_4b: llama.LlamaConfig = .{
    .hidden_size = 2048,
    .num_layers = 8,
    .num_heads = 32,
    .num_kv_heads = 32, // MHA — no grouped-query attention
    .head_dim = 64, // 2048 / 32
    .intermediate_size = 5632,
    .vocab_size = 61788,
    .max_seq_len = 2048,
    .rms_norm_eps = 1e-5,
    .rope_theta = 10000.0,
    .tie_word_embeddings = true,
};

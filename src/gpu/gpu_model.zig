// src/gpu/gpu_model.zig
//
// GPU-accelerated Sentence Transformer wrapper.
// Wraps the existing SentenceTransformerModel and provides embedGpu()
// that runs the full forward pass on GPU via Vulkan compute shaders.
//
// Usage:
//   var model = try SentenceTransformerModel.init(alloc, path);
//   var gpu = try GpuModel.init(alloc, &model);
//   defer gpu.deinit();
//   const embedding = try gpu.embed(&tokenizer, "hello world");

const std = @import("std");
const gpu_fwd = @import("gpu_forward");
const gpu = @import("vulkan");

// These types come from the model module
const model_mod = @import("model");
const SentenceTransformerModel = model_mod.SentenceTransformerModel;
const SentenceTransformerConfig = model_mod.SentenceTransformerConfig;
const SentenceTransformer = model_mod.SentenceTransformer;
const Tokenizer = @import("tokenizer").Tokenizer;
const transformer_mod = @import("transformer");
const TransformerBlockWeightsF32 = transformer_mod.TransformerBlockWeightsF32;
const TransformerBlockWeightsQ8K = transformer_mod.TransformerBlockWeightsQ8K;
const TransformerBlockWeightsF16 = transformer_mod.TransformerBlockWeightsF16;

pub const GpuModel = struct {
    allocator: std.mem.Allocator,
    fwd: gpu_fwd.GpuForward,
    hidden_dim: u32,

    const Self = @This();

    /// Initialize GPU model from an already-loaded SentenceTransformerModel.
    /// Uploads all F32 weights to GPU. For Q8K/F16 models, weights are
    /// dequantized to F32 for GPU upload (GPU Q8K kernels are future work).
    pub fn init(allocator: std.mem.Allocator, model: *const SentenceTransformerModel) !Self {
        const config = model.config;
        const hidden: u32 = @intCast(config.hidden_dim);
        const num_heads: u32 = @intCast(config.num_heads);

        const gpu_config = gpu_fwd.GpuConfig{
            .hidden_dim = hidden,
            .num_heads = num_heads,
            .head_dim = hidden / num_heads,
            .ffn_dim = @intCast(config.intermediate_dim),
            .num_layers = @intCast(config.num_layers),
            .vocab_size = @intCast(config.vocab_size),
            .max_seq_len = @intCast(config.max_seq_len),
            .max_batch_tokens = @intCast(config.max_seq_len * 32), // support batch up to 32 sentences
        };

        const embeddings = gpu_fwd.EmbeddingData{
            .word_emb = model.word_embeddings.data,
            .pos_emb = model.position_embeddings.data,
            .type_emb = model.token_type_embeddings.data,
            .ln_gamma = model.embed_ln_gamma.data,
            .ln_beta = model.embed_ln_beta.data,
        };

        // Extract per-layer weight data (dequantize Q8K/F16 → F32 for GPU upload)
        const num_layers = config.num_layers;
        const layer_data = try allocator.alloc(gpu_fwd.LayerData, num_layers);
        defer allocator.free(layer_data);

        // Arena for temporary dequantized weight buffers (freed after GpuForward.init uploads them)
        var dequant_arena = std.heap.ArenaAllocator.init(allocator);
        defer dequant_arena.deinit();
        const da = dequant_arena.allocator();

        switch (model.weights) {
            .f32 => |f| {
                for (f.blocks, 0..) |*blk, i| {
                    layer_data[i] = extractLayerF32(blk);
                }
            },
            .q8k => |q| {
                for (q.blocks, 0..) |*blk, i| {
                    layer_data[i] = try extractLayerQ8K(da, blk);
                }
            },
            .f16 => |h| {
                for (h.blocks, 0..) |*blk, i| {
                    layer_data[i] = try extractLayerF16(da, blk);
                }
            },
        }

        const fwd = try gpu_fwd.GpuForward.init(allocator, gpu_config, embeddings, layer_data);

        return Self{
            .allocator = allocator,
            .fwd = fwd,
            .hidden_dim = hidden,
        };
    }

    /// Embed a single text on GPU → normalized 384-dim vector
    pub fn embed(self: *Self, tokenizer: *Tokenizer, text: []const u8) ![384]f32 {
        var enc = try tokenizer.encode(text);
        defer enc.deinit(self.allocator);

        var output: [384]f32 = undefined;
        try self.fwd.forward(enc.input_ids, &output);
        return output;
    }

    /// Embed a batch of texts on GPU → array of normalized 384-dim vectors
    /// All sentences processed together in a single GPU submission.
    pub fn embedBatch(self: *Self, tokenizer: *Tokenizer, texts: []const []const u8, results: [][384]f32) !void {
        // Tokenize all sentences
        const encs = try self.allocator.alloc(@import("tokenizer").TokenizerOutput, texts.len);
        defer self.allocator.free(encs);
        var initialized: usize = 0;
        defer for (encs[0..initialized]) |*e| e.deinit(self.allocator);

        const batch_ids = try self.allocator.alloc([]const u32, texts.len);
        defer self.allocator.free(batch_ids);

        for (texts, 0..) |text, i| {
            encs[i] = try tokenizer.encode(text);
            initialized = i + 1;
            batch_ids[i] = encs[i].input_ids;
        }

        // Flat output buffer for GPU
        const hidden = self.hidden_dim;
        const flat = try self.allocator.alloc(f32, texts.len * hidden);
        defer self.allocator.free(flat);

        try self.fwd.forwardBatch(batch_ids, flat);

        // Copy to structured output
        for (0..texts.len) |i| {
            @memcpy(&results[i], flat[i * hidden .. (i + 1) * hidden]);
        }
    }

    pub fn deinit(self: *Self) void {
        self.fwd.deinit();
    }

    // =========================================================================
    // Weight extraction helpers
    // =========================================================================

    fn extractLayerF32(blk: *const TransformerBlockWeightsF32) gpu_fwd.LayerData {
        return gpu_fwd.LayerData{
            .q_weight = blk.attention.q_weight.data,
            .q_bias = blk.attention.q_bias.data,
            .k_weight = blk.attention.k_weight.data,
            .k_bias = blk.attention.k_bias.data,
            .v_weight = blk.attention.v_weight.data,
            .v_bias = blk.attention.v_bias.data,
            .o_weight = blk.attention.o_weight.data,
            .o_bias = blk.attention.o_bias.data,
            .ff1_weight = blk.ff_linear1_weight.data,
            .ff1_bias = blk.ff_linear1_bias.data,
            .ff2_weight = blk.ff_linear2_weight.data,
            .ff2_bias = blk.ff_linear2_bias.data,
            .attn_ln_gamma = blk.attn_ln_gamma.data,
            .attn_ln_beta = blk.attn_ln_beta.data,
            .ff_ln_gamma = blk.ff_ln_gamma.data,
            .ff_ln_beta = blk.ff_ln_beta.data,
        };
    }

    fn extractLayerQ8K(arena: std.mem.Allocator, blk: *const TransformerBlockWeightsQ8K) !gpu_fwd.LayerData {
        return gpu_fwd.LayerData{
            .q_weight = try dequantQ8K(arena, &blk.attention.q_weight),
            .q_bias = blk.attention.q_bias.data,
            .k_weight = try dequantQ8K(arena, &blk.attention.k_weight),
            .k_bias = blk.attention.k_bias.data,
            .v_weight = try dequantQ8K(arena, &blk.attention.v_weight),
            .v_bias = blk.attention.v_bias.data,
            .o_weight = try dequantQ8K(arena, &blk.attention.o_weight),
            .o_bias = blk.attention.o_bias.data,
            .ff1_weight = try dequantQ8K(arena, &blk.ff_linear1_weight),
            .ff1_bias = blk.ff_linear1_bias.data,
            .ff2_weight = try dequantQ8K(arena, &blk.ff_linear2_weight),
            .ff2_bias = blk.ff_linear2_bias.data,
            .attn_ln_gamma = blk.attn_ln_gamma.data,
            .attn_ln_beta = blk.attn_ln_beta.data,
            .ff_ln_gamma = blk.ff_ln_gamma.data,
            .ff_ln_beta = blk.ff_ln_beta.data,
        };
    }

    fn extractLayerF16(arena: std.mem.Allocator, blk: *const TransformerBlockWeightsF16) !gpu_fwd.LayerData {
        return gpu_fwd.LayerData{
            .q_weight = try dequantF16(arena, &blk.attention.q_weight),
            .q_bias = blk.attention.q_bias.data,
            .k_weight = try dequantF16(arena, &blk.attention.k_weight),
            .k_bias = blk.attention.k_bias.data,
            .v_weight = try dequantF16(arena, &blk.attention.v_weight),
            .v_bias = blk.attention.v_bias.data,
            .o_weight = try dequantF16(arena, &blk.attention.o_weight),
            .o_bias = blk.attention.o_bias.data,
            .ff1_weight = try dequantF16(arena, &blk.ff_linear1_weight),
            .ff1_bias = blk.ff_linear1_bias.data,
            .ff2_weight = try dequantF16(arena, &blk.ff_linear2_weight),
            .ff2_bias = blk.ff_linear2_bias.data,
            .attn_ln_gamma = blk.attn_ln_gamma.data,
            .attn_ln_beta = blk.attn_ln_beta.data,
            .ff_ln_gamma = blk.ff_ln_gamma.data,
            .ff_ln_beta = blk.ff_ln_beta.data,
        };
    }

    // =========================================================================
    // Dequantization helpers (Q8K/F16 → F32 slice, arena-allocated)
    // =========================================================================

    /// Dequantize Q8K tensor: value = int8_data[i] * scale[i / block_size]
    fn dequantQ8K(arena: std.mem.Allocator, qt: anytype) ![]const f32 {
        const n = qt.element_count;
        const result = try arena.alloc(f32, n);
        const block_size = qt.block_size;
        for (0..n) |i| {
            const block_idx = i / block_size;
            result[i] = @as(f32, @floatFromInt(qt.data[i])) * qt.scales[block_idx];
        }
        return result;
    }

    /// Convert F16 tensor to F32 slice
    fn dequantF16(arena: std.mem.Allocator, ht: anytype) ![]const f32 {
        const n = ht.data.len;
        const result = try arena.alloc(f32, n);
        for (0..n) |i| {
            result[i] = @floatCast(ht.data[i]);
        }
        return result;
    }
};

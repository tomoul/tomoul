// src/models/sentence_transformer/model.zig
// Sentence Transformer Model (all-MiniLM-L6-v2)
//
// Architecture: BERT encoder (6 layers, 384 hidden, 12 heads)
//   WordPiece tokenizer → BERT embeddings → 6 transformer blocks → mean pooling → L2 normalize → 384-dim vector
//
// Supports float32 and Q8_K quantized weights (auto-detected from .tl file).

const std = @import("std");
const tensor_mod = @import("tensor.zig");
const Tensor = tensor_mod.Tensor;
const ops = @import("ops.zig");
const loader_mod = @import("loader.zig");
const ModelLoader = loader_mod.ModelLoader;
const QuantFormat = loader_mod.QuantFormat;

const transformer = @import("transformer.zig");
const attention = @import("attention.zig");
const TransformerBlockWeights = transformer.TransformerBlockWeightsF32;
const TransformerBlockWeightsQ8K = transformer.TransformerBlockWeightsQ8K;
const TransformerConfig = transformer.TransformerConfig;
const AttentionWeights = attention.AttentionWeightsF32;
const AttentionWeightsQ8K = attention.AttentionWeightsQ8K;

const quant = @import("quantization.zig");
const QuantizedTensorQ8K = quant.QuantizedTensorQ8K;

const tokenizer_mod = @import("tokenizer.zig");
const Tokenizer = tokenizer_mod.Tokenizer;
const TokenizerOutput = tokenizer_mod.TokenizerOutput;

// =============================================================================
// Configuration
// =============================================================================

pub const SentenceTransformerConfig = struct {
    vocab_size: usize = 30522,
    hidden_dim: usize = 384,
    intermediate_dim: usize = 1536,
    num_layers: usize = 6,
    num_heads: usize = 12,
    max_seq_len: usize = 512,
    layer_norm_eps: f32 = 1e-12, // BERT uses 1e-12

    pub fn getTransformerConfig(self: SentenceTransformerConfig) TransformerConfig {
        return TransformerConfig{
            .hidden_dim = self.hidden_dim,
            .intermediate_dim = self.intermediate_dim,
            .num_heads = self.num_heads,
            .layer_norm_eps = self.layer_norm_eps,
        };
    }
};

pub const DEFAULT_CONFIG = SentenceTransformerConfig{};

// =============================================================================
// Weight Storage
// =============================================================================

pub const WeightStorage = union(enum) {
    f32: struct {
        blocks: []TransformerBlockWeights,
    },
    q8k: struct {
        blocks: []TransformerBlockWeightsQ8K,
    },

    pub fn deinit(self: *WeightStorage, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .f32 => |*f| {
                for (f.blocks) |*block| block.deinit();
                allocator.free(f.blocks);
            },
            .q8k => |*q| {
                for (q.blocks) |*block| block.deinit();
                allocator.free(q.blocks);
            },
        }
    }
};

// =============================================================================
// Sentence Transformer Model
// =============================================================================

pub const SentenceTransformerModel = struct {
    allocator: std.mem.Allocator,
    config: SentenceTransformerConfig,
    quant_format: QuantFormat,

    // Embeddings (always float32)
    word_embeddings: Tensor, // [vocab_size, hidden_dim]
    position_embeddings: Tensor, // [max_seq_len, hidden_dim]
    token_type_embeddings: Tensor, // [2, hidden_dim]
    embed_ln_gamma: Tensor, // [hidden_dim]
    embed_ln_beta: Tensor, // [hidden_dim]

    // Transformer blocks
    weights: WeightStorage,

    const Self = @This();

    /// Load model from .tl file path
    pub fn init(allocator: std.mem.Allocator, model_path: []const u8) !Self {
        var loader = try ModelLoader.init(allocator, model_path);
        defer loader.deinit();
        return Self.initFromLoader(allocator, &loader);
    }

    /// Load model from embedded bytes (for WASM)
    pub fn initFromBytes(allocator: std.mem.Allocator, model_bytes: []const u8) !Self {
        var loader = try ModelLoader.initFromBytes(allocator, model_bytes);
        defer loader.deinit();
        return Self.initFromLoader(allocator, &loader);
    }

    fn initFromLoader(allocator: std.mem.Allocator, loader: *ModelLoader) !Self {
        const config = try detectConfig(loader);
        const quant_format = loader.quant_format;
        const is_quantized = quant_format == .q8_k;

        // Load embeddings (always float32)
        var word_embeddings = if (is_quantized)
            try loader.getTensorDequantized("embeddings.word_embeddings.weight")
        else
            try loader.getTensor("embeddings.word_embeddings.weight");
        errdefer word_embeddings.deinit();

        var position_embeddings = if (is_quantized)
            try loader.getTensorDequantized("embeddings.position_embeddings.weight")
        else
            try loader.getTensor("embeddings.position_embeddings.weight");
        errdefer position_embeddings.deinit();

        var token_type_embeddings = if (is_quantized)
            try loader.getTensorDequantized("embeddings.token_type_embeddings.weight")
        else
            try loader.getTensor("embeddings.token_type_embeddings.weight");
        errdefer token_type_embeddings.deinit();

        var embed_ln_gamma = if (is_quantized)
            try loader.getTensorDequantized("embeddings.LayerNorm.weight")
        else
            try loader.getTensor("embeddings.LayerNorm.weight");
        errdefer embed_ln_gamma.deinit();

        var embed_ln_beta = if (is_quantized)
            try loader.getTensorDequantized("embeddings.LayerNorm.bias")
        else
            try loader.getTensor("embeddings.LayerNorm.bias");
        errdefer embed_ln_beta.deinit();

        // Load transformer blocks
        var weights: WeightStorage = undefined;
        if (quant_format == .q8_k) {
            var blocks = try allocator.alloc(TransformerBlockWeightsQ8K, config.num_layers);
            var loaded: usize = 0;
            errdefer {
                for (blocks[0..loaded]) |*b| b.deinit();
                allocator.free(blocks);
            }
            for (0..config.num_layers) |i| {
                blocks[i] = try loadTransformerBlockQ8K(allocator, loader, i);
                loaded += 1;
            }
            weights = .{ .q8k = .{ .blocks = blocks } };
        } else {
            var blocks = try allocator.alloc(TransformerBlockWeights, config.num_layers);
            var loaded: usize = 0;
            errdefer {
                for (blocks[0..loaded]) |*b| b.deinit();
                allocator.free(blocks);
            }
            for (0..config.num_layers) |i| {
                blocks[i] = try loadTransformerBlock(loader, i);
                loaded += 1;
            }
            weights = .{ .f32 = .{ .blocks = blocks } };
        }

        return Self{
            .allocator = allocator,
            .config = config,
            .quant_format = quant_format,
            .word_embeddings = word_embeddings,
            .position_embeddings = position_embeddings,
            .token_type_embeddings = token_type_embeddings,
            .embed_ln_gamma = embed_ln_gamma,
            .embed_ln_beta = embed_ln_beta,
            .weights = weights,
        };
    }

    /// Auto-detect config from model weights
    fn detectConfig(loader: *ModelLoader) !SentenceTransformerConfig {
        const is_quantized = loader.quant_format == .q8_k;
        var word_emb = if (is_quantized)
            try loader.getTensorDequantized("embeddings.word_embeddings.weight")
        else
            try loader.getTensor("embeddings.word_embeddings.weight");
        defer word_emb.deinit();

        const hidden_dim = word_emb.shape[1];

        // Count layers
        var num_layers: usize = 0;
        var buf: [128]u8 = undefined;
        while (num_layers < 48) : (num_layers += 1) {
            const name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.attention.self.query.weight", .{num_layers}) catch unreachable;
            if (is_quantized) {
                var t = loader.getTensorDequantized(name) catch break;
                t.deinit();
            } else {
                var t = loader.getTensor(name) catch break;
                t.deinit();
            }
        }

        return SentenceTransformerConfig{
            .hidden_dim = hidden_dim,
            .intermediate_dim = hidden_dim * 4,
            .num_layers = num_layers,
            .num_heads = if (hidden_dim == 384) 12 else if (hidden_dim == 768) 12 else 16,
        };
    }

    /// Run forward pass: token IDs → hidden states (before pooling)
    /// Uses a scratch arena allocator for all temporaries to minimize allocation overhead.
    pub fn forward(self: *Self, scratch: std.mem.Allocator, input_ids: []const u32, attention_mask: []const u32, token_type_ids: []const u32) !Tensor {
        const seq_len = input_ids.len;
        const config = self.config;
        const transformer_config = config.getTransformerConfig();

        // 1. Embeddings: word + position + token_type
        var word_emb = try ops.embedding(scratch, input_ids, &self.word_embeddings);
        defer word_emb.deinit();

        // BERT position IDs start at 0 (no offset, unlike RoBERTa)
        const position_ids = try scratch.alloc(u32, seq_len);
        defer scratch.free(position_ids);
        for (position_ids, 0..) |*p, i| {
            p.* = @intCast(i);
        }

        var pos_emb = try ops.embedding(scratch, position_ids, &self.position_embeddings);
        defer pos_emb.deinit();

        var type_emb = try ops.embedding(scratch, token_type_ids, &self.token_type_embeddings);
        defer type_emb.deinit();

        // Combine embeddings
        try ops.addInPlace(&word_emb, &pos_emb);
        try ops.addInPlace(&word_emb, &type_emb);

        // Embedding layer norm
        var hidden = try ops.layerNorm(
            scratch,
            &word_emb,
            &self.embed_ln_gamma,
            &self.embed_ln_beta,
            config.layer_norm_eps,
        );
        defer hidden.deinit();

        // 2. Run through transformer blocks
        var output: Tensor = undefined;
        switch (self.weights) {
            .f32 => |f| {
                output = try transformer.transformerStackF32(
                    scratch,
                    &hidden,
                    f.blocks,
                    transformer_config,
                );
            },
            .q8k => |q| {
                output = try transformer.transformerStackQ8K(
                    scratch,
                    &hidden,
                    q.blocks,
                    transformer_config,
                );
            },
        }

        // output is now [seq_len, hidden_dim] — caller owns it
        _ = attention_mask; // used in mean pooling, not in transformer
        return output;
    }

    /// Compute embedding for a single text
    /// Returns a normalized 384-dim vector
    /// Uses an arena allocator for all inference temporaries — bulk-freed after each call.
    pub fn embed(self: *Self, tokenizer: *Tokenizer, text: []const u8) ![384]f32 {
        var enc = try tokenizer.encode(text);
        defer enc.deinit(self.allocator);

        // Use arena for forward pass scratch memory — all temporaries bulk-freed at end
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const scratch = arena.allocator();

        var hidden = try self.forward(scratch, enc.input_ids, enc.attention_mask, enc.token_type_ids);
        // No need to defer hidden.deinit() — arena handles it

        // Mean pooling over non-padding tokens
        var pooled = meanPool(&hidden, enc.attention_mask, self.config.hidden_dim);

        // L2 normalize
        l2Normalize(&pooled);

        return pooled;
    }

    pub fn deinit(self: *Self) void {
        self.word_embeddings.deinit();
        self.position_embeddings.deinit();
        self.token_type_embeddings.deinit();
        self.embed_ln_gamma.deinit();
        self.embed_ln_beta.deinit();
        self.weights.deinit(self.allocator);
    }

    // =========================================================================
    // Weight loading helpers
    // =========================================================================

    fn loadTransformerBlock(loader: *ModelLoader, layer_idx: usize) !TransformerBlockWeights {
        var buf: [128]u8 = undefined;

        const q_w_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.attention.self.query.weight", .{layer_idx}) catch unreachable;
        var q_w_pt = try loader.getTensor(q_w_name);
        var q_w = try ops.transpose(loader.allocator, &q_w_pt);
        q_w_pt.deinit();
        errdefer q_w.deinit();

        const k_w_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.attention.self.key.weight", .{layer_idx}) catch unreachable;
        var k_w_pt = try loader.getTensor(k_w_name);
        var k_w = try ops.transpose(loader.allocator, &k_w_pt);
        k_w_pt.deinit();
        errdefer k_w.deinit();

        const v_w_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.attention.self.value.weight", .{layer_idx}) catch unreachable;
        var v_w_pt = try loader.getTensor(v_w_name);
        var v_w = try ops.transpose(loader.allocator, &v_w_pt);
        v_w_pt.deinit();
        errdefer v_w.deinit();

        const o_w_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.attention.output.dense.weight", .{layer_idx}) catch unreachable;
        var o_w_pt = try loader.getTensor(o_w_name);
        var o_w = try ops.transpose(loader.allocator, &o_w_pt);
        o_w_pt.deinit();
        errdefer o_w.deinit();

        const q_b_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.attention.self.query.bias", .{layer_idx}) catch unreachable;
        var q_b = try loader.getTensor(q_b_name);
        errdefer q_b.deinit();

        const k_b_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.attention.self.key.bias", .{layer_idx}) catch unreachable;
        var k_b = try loader.getTensor(k_b_name);
        errdefer k_b.deinit();

        const v_b_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.attention.self.value.bias", .{layer_idx}) catch unreachable;
        var v_b = try loader.getTensor(v_b_name);
        errdefer v_b.deinit();

        const o_b_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.attention.output.dense.bias", .{layer_idx}) catch unreachable;
        var o_b = try loader.getTensor(o_b_name);
        errdefer o_b.deinit();

        const attn_ln_g_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.attention.output.LayerNorm.weight", .{layer_idx}) catch unreachable;
        var attn_ln_g = try loader.getTensor(attn_ln_g_name);
        errdefer attn_ln_g.deinit();

        const attn_ln_b_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.attention.output.LayerNorm.bias", .{layer_idx}) catch unreachable;
        var attn_ln_b = try loader.getTensor(attn_ln_b_name);
        errdefer attn_ln_b.deinit();

        const ff1_w_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.intermediate.dense.weight", .{layer_idx}) catch unreachable;
        var ff1_w_pt = try loader.getTensor(ff1_w_name);
        var ff1_w = try ops.transpose(loader.allocator, &ff1_w_pt);
        ff1_w_pt.deinit();
        errdefer ff1_w.deinit();

        const ff1_b_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.intermediate.dense.bias", .{layer_idx}) catch unreachable;
        var ff1_b = try loader.getTensor(ff1_b_name);
        errdefer ff1_b.deinit();

        const ff2_w_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.output.dense.weight", .{layer_idx}) catch unreachable;
        var ff2_w_pt = try loader.getTensor(ff2_w_name);
        var ff2_w = try ops.transpose(loader.allocator, &ff2_w_pt);
        ff2_w_pt.deinit();
        errdefer ff2_w.deinit();

        const ff2_b_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.output.dense.bias", .{layer_idx}) catch unreachable;
        var ff2_b = try loader.getTensor(ff2_b_name);
        errdefer ff2_b.deinit();

        const ff_ln_g_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.output.LayerNorm.weight", .{layer_idx}) catch unreachable;
        var ff_ln_g = try loader.getTensor(ff_ln_g_name);
        errdefer ff_ln_g.deinit();

        const ff_ln_b_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.output.LayerNorm.bias", .{layer_idx}) catch unreachable;
        var ff_ln_b = try loader.getTensor(ff_ln_b_name);
        errdefer ff_ln_b.deinit();

        return TransformerBlockWeights{
            .attention = AttentionWeights{
                .q_weight = q_w,
                .k_weight = k_w,
                .v_weight = v_w,
                .o_weight = o_w,
                .q_bias = q_b,
                .k_bias = k_b,
                .v_bias = v_b,
                .o_bias = o_b,
            },
            .attn_ln_gamma = attn_ln_g,
            .attn_ln_beta = attn_ln_b,
            .ff_linear1_weight = ff1_w,
            .ff_linear1_bias = ff1_b,
            .ff_linear2_weight = ff2_w,
            .ff_linear2_bias = ff2_b,
            .ff_ln_gamma = ff_ln_g,
            .ff_ln_beta = ff_ln_b,
        };
    }

    fn loadTransformerBlockQ8K(allocator: std.mem.Allocator, loader: *ModelLoader, layer_idx: usize) !TransformerBlockWeightsQ8K {
        var buf: [128]u8 = undefined;

        const q_w_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.attention.self.query.weight", .{layer_idx}) catch unreachable;
        var q_w = try loadQuantizedTransposedQ8K(allocator, loader, q_w_name);
        errdefer q_w.deinit();

        const k_w_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.attention.self.key.weight", .{layer_idx}) catch unreachable;
        var k_w = try loadQuantizedTransposedQ8K(allocator, loader, k_w_name);
        errdefer k_w.deinit();

        const v_w_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.attention.self.value.weight", .{layer_idx}) catch unreachable;
        var v_w = try loadQuantizedTransposedQ8K(allocator, loader, v_w_name);
        errdefer v_w.deinit();

        const o_w_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.attention.output.dense.weight", .{layer_idx}) catch unreachable;
        var o_w = try loadQuantizedTransposedQ8K(allocator, loader, o_w_name);
        errdefer o_w.deinit();

        const q_b_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.attention.self.query.bias", .{layer_idx}) catch unreachable;
        var q_b = try loader.getTensorDequantized(q_b_name);
        errdefer q_b.deinit();

        const k_b_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.attention.self.key.bias", .{layer_idx}) catch unreachable;
        var k_b = try loader.getTensorDequantized(k_b_name);
        errdefer k_b.deinit();

        const v_b_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.attention.self.value.bias", .{layer_idx}) catch unreachable;
        var v_b = try loader.getTensorDequantized(v_b_name);
        errdefer v_b.deinit();

        const o_b_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.attention.output.dense.bias", .{layer_idx}) catch unreachable;
        var o_b = try loader.getTensorDequantized(o_b_name);
        errdefer o_b.deinit();

        const attn_ln_g_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.attention.output.LayerNorm.weight", .{layer_idx}) catch unreachable;
        var attn_ln_g = try loader.getTensorDequantized(attn_ln_g_name);
        errdefer attn_ln_g.deinit();

        const attn_ln_b_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.attention.output.LayerNorm.bias", .{layer_idx}) catch unreachable;
        var attn_ln_b = try loader.getTensorDequantized(attn_ln_b_name);
        errdefer attn_ln_b.deinit();

        const ff1_w_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.intermediate.dense.weight", .{layer_idx}) catch unreachable;
        var ff1_w = try loadQuantizedTransposedQ8K(allocator, loader, ff1_w_name);
        errdefer ff1_w.deinit();

        const ff1_b_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.intermediate.dense.bias", .{layer_idx}) catch unreachable;
        var ff1_b = try loader.getTensorDequantized(ff1_b_name);
        errdefer ff1_b.deinit();

        const ff2_w_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.output.dense.weight", .{layer_idx}) catch unreachable;
        var ff2_w = try loadQuantizedTransposedQ8K(allocator, loader, ff2_w_name);
        errdefer ff2_w.deinit();

        const ff2_b_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.output.dense.bias", .{layer_idx}) catch unreachable;
        var ff2_b = try loader.getTensorDequantized(ff2_b_name);
        errdefer ff2_b.deinit();

        const ff_ln_g_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.output.LayerNorm.weight", .{layer_idx}) catch unreachable;
        var ff_ln_g = try loader.getTensorDequantized(ff_ln_g_name);
        errdefer ff_ln_g.deinit();

        const ff_ln_b_name = std.fmt.bufPrint(&buf, "encoder.layer.{d}.output.LayerNorm.bias", .{layer_idx}) catch unreachable;
        var ff_ln_b = try loader.getTensorDequantized(ff_ln_b_name);
        errdefer ff_ln_b.deinit();

        return TransformerBlockWeightsQ8K{
            .attention = AttentionWeightsQ8K{
                .q_weight = q_w,
                .k_weight = k_w,
                .v_weight = v_w,
                .o_weight = o_w,
                .q_bias = q_b,
                .k_bias = k_b,
                .v_bias = v_b,
                .o_bias = o_b,
            },
            .attn_ln_gamma = attn_ln_g,
            .attn_ln_beta = attn_ln_b,
            .ff_linear1_weight = ff1_w,
            .ff_linear1_bias = ff1_b,
            .ff_linear2_weight = ff2_w,
            .ff_linear2_bias = ff2_b,
            .ff_ln_gamma = ff_ln_g,
            .ff_ln_beta = ff_ln_b,
        };
    }
};

// =============================================================================
// Pooling and normalization ops
// =============================================================================

/// Mean pool hidden states over non-padding tokens
/// hidden: [seq_len, hidden_dim], mask: [seq_len] (1 = real token, 0 = padding)
fn meanPool(hidden: *const Tensor, attention_mask: []const u32, hidden_dim: usize) [384]f32 {
    var result: [384]f32 = [_]f32{0} ** 384;
    const seq_len = attention_mask.len;
    var count: f32 = 0;

    for (0..seq_len) |t| {
        if (attention_mask[t] == 1) {
            const row = hidden.data[t * hidden_dim .. (t + 1) * hidden_dim];
            for (0..hidden_dim) |d| {
                result[d] += row[d];
            }
            count += 1;
        }
    }

    if (count > 0) {
        for (0..hidden_dim) |d| {
            result[d] /= count;
        }
    }

    return result;
}

/// L2 normalize a vector in-place
fn l2Normalize(vec: *[384]f32) void {
    var sum_sq: f32 = 0;
    for (vec) |v| {
        sum_sq += v * v;
    }
    const norm = @sqrt(sum_sq);
    if (norm > 0) {
        for (vec) |*v| {
            v.* /= norm;
        }
    }
}

/// Load a Q8_K block-wise quantized tensor (pre-transposed in the .tl file).
fn loadQuantizedTransposedQ8K(allocator: std.mem.Allocator, loader: *ModelLoader, name: []const u8) !QuantizedTensorQ8K {
    _ = allocator;
    return loader.getQuantizedTensorQ8K(name);
}

// =============================================================================
// Combined Model (Model + Tokenizer)
// =============================================================================

pub const SentenceTransformer = struct {
    allocator: std.mem.Allocator,
    model: SentenceTransformerModel,
    tokenizer: Tokenizer,

    const Self = @This();

    /// Initialize from file paths
    pub fn init(allocator: std.mem.Allocator, model_path: []const u8, vocab_path: []const u8) !Self {
        var model = try SentenceTransformerModel.init(allocator, model_path);
        errdefer model.deinit();

        var tokenizer = try Tokenizer.init(allocator, vocab_path);
        errdefer tokenizer.deinit();

        return Self{
            .allocator = allocator,
            .model = model,
            .tokenizer = tokenizer,
        };
    }

    /// Initialize from embedded bytes (for WASM bundled builds)
    pub fn initFromBytes(allocator: std.mem.Allocator, model_bytes: []const u8, vocab_bytes: []const u8) !Self {
        var model = try SentenceTransformerModel.initFromBytes(allocator, model_bytes);
        errdefer model.deinit();

        var tokenizer = try Tokenizer.initFromString(allocator, vocab_bytes);
        errdefer tokenizer.deinit();

        return Self{
            .allocator = allocator,
            .model = model,
            .tokenizer = tokenizer,
        };
    }

    /// Embed a single text → normalized 384-dim vector
    pub fn embed(self: *Self, text: []const u8) ![384]f32 {
        return self.model.embed(&self.tokenizer, text);
    }

    /// Embed a batch of texts
    pub fn embedBatch(self: *Self, texts: []const []const u8) ![][384]f32 {
        var results = try self.allocator.alloc([384]f32, texts.len);
        errdefer self.allocator.free(results);

        for (texts, 0..) |text, i| {
            results[i] = try self.embed(text);
        }

        return results;
    }

    pub fn deinit(self: *Self) void {
        self.model.deinit();
        self.tokenizer.deinit();
    }
};

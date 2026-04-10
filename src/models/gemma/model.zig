// src/models/gemma/model.zig
// Gemma Decoder-Only Language Model
//
// Architecture: Token embedding → N × (RMSNorm → GQA → residual → RMSNorm → GeGLU FFN → residual) → RMSNorm → tied lm_head
//
// Supports float32 and dequantized (Q8_K, F16, Q8, Q4) weights.
// Gemma 2 variants add post-attention and post-FFN RMSNorm layers.
//
// Usage:
//   var model = try Gemma.init(allocator, "gemma_2b_q8.tl");
//   defer model.deinit();
//   const tokens = try model.generate(&[_]u32{2, 1596, 603}, 128, 0.0);
//   defer allocator.free(tokens);

const std = @import("std");
const tensor_mod = @import("tensor.zig");
const Tensor = tensor_mod.Tensor;
pub const ops = @import("ops.zig");
const loader_mod = @import("loader.zig");
const ModelLoader = loader_mod.ModelLoader;
const QuantFormat = loader_mod.QuantFormat;
const attention = @import("attention.zig");
const cache_mod = @import("cache.zig");
const KVCache = cache_mod.KVCache;

pub const config = @import("config.zig");
pub const tokenizer = @import("tokenizer.zig");
const GemmaConfig = config.GemmaConfig;
const GemmaVariant = config.GemmaVariant;
const GemmaTokens = config.GemmaTokens;
const GQAConfig = attention.GQAConfig;

// =============================================================================
// Weight Storage
// =============================================================================

/// Per-layer weights for a Gemma transformer block.
/// All cross-layer weights are stored as dequantized f32 (loader handles format dispatch).
pub const GemmaLayerWeights = struct {
    // Pre-attention RMSNorm
    input_norm: Tensor, // [hidden_dim]

    // GQA projections (dequantized to f32)
    q_proj: Tensor, // [hidden_dim, hidden_dim] pre-transposed
    k_proj: Tensor, // [hidden_dim, kv_dim] pre-transposed
    v_proj: Tensor, // [hidden_dim, kv_dim] pre-transposed
    o_proj: Tensor, // [hidden_dim, hidden_dim] pre-transposed

    // Post-attention RMSNorm (Gemma 2 only — zero-initialized for Gemma 1 and skipped)
    post_attn_norm: ?Tensor,

    // Pre-FFN RMSNorm
    pre_ffn_norm: Tensor, // [hidden_dim]

    // GeGLU FFN (3 weight matrices)
    gate_proj: Tensor, // [hidden_dim, intermediate_dim] pre-transposed
    up_proj: Tensor, // [hidden_dim, intermediate_dim] pre-transposed
    down_proj: Tensor, // [intermediate_dim, hidden_dim] pre-transposed

    // Post-FFN RMSNorm (Gemma 2 only)
    post_ffn_norm: ?Tensor,

    pub fn deinit(self: *GemmaLayerWeights) void {
        self.input_norm.deinit();
        self.q_proj.deinit();
        self.k_proj.deinit();
        self.v_proj.deinit();
        self.o_proj.deinit();
        if (self.post_attn_norm) |*t| t.deinit();
        self.pre_ffn_norm.deinit();
        self.gate_proj.deinit();
        self.up_proj.deinit();
        self.down_proj.deinit();
        if (self.post_ffn_norm) |*t| t.deinit();
    }
};

/// Complete Gemma model weights
pub const GemmaWeights = struct {
    token_embedding: Tensor, // [vocab_size, hidden_dim] — also used as lm_head (tied)
    layers: []GemmaLayerWeights,
    final_norm: Tensor, // [hidden_dim] — final RMSNorm gamma
    allocator: std.mem.Allocator,

    pub fn deinit(self: *GemmaWeights) void {
        self.token_embedding.deinit();
        for (self.layers) |*layer| layer.deinit();
        self.allocator.free(self.layers);
        self.final_norm.deinit();
    }
};

// =============================================================================
// Gemma Model
// =============================================================================

pub const Gemma = struct {
    allocator: std.mem.Allocator,
    cfg: GemmaConfig,
    weights: GemmaWeights,
    kv_cache: ?KVCache,

    const Self = @This();

    /// Load model from .tl file path
    pub fn init(allocator: std.mem.Allocator, model_path: []const u8) !Self {
        var loader = try ModelLoader.init(allocator, model_path);
        defer loader.deinit();

        const variant = config.inferVariantFromPath(model_path);
        const cfg = GemmaConfig.forVariant(variant);

        var weights = try loadWeights(allocator, &loader, cfg);
        errdefer weights.deinit();

        return Self{
            .allocator = allocator,
            .cfg = cfg,
            .weights = weights,
            .kv_cache = null,
        };
    }

    /// Initialize KV cache for autoregressive generation
    pub fn initCache(self: *Self, max_seq_len: usize) !void {
        if (self.kv_cache != null) return;
        self.kv_cache = try KVCache.init(
            self.allocator,
            self.cfg.num_layers,
            max_seq_len,
            self.cfg.kvDim(),
        );
    }

    /// Reset KV cache (for new generation session)
    pub fn resetCache(self: *Self) void {
        if (self.kv_cache) |*c| c.reset();
    }

    // =========================================================================
    // Forward Pass
    // =========================================================================

    /// Forward pass for a full sequence (no KV cache — used for prefill/prompt processing).
    /// input_ids: token IDs [seq_len]
    /// Returns: logits [seq_len, vocab_size] — caller owns memory
    pub fn forward(self: *Self, scratch: std.mem.Allocator, input_ids: []const u32) !Tensor {
        const cfg = self.cfg;
        const w = &self.weights;
        const hidden_dim = cfg.hidden_dim;
        _ = hidden_dim; // autofix
        const gqa_cfg = cfg.gqaConfig();

        // 1. Token embedding + scaling
        var hidden = try ops.embedding(scratch, input_ids, &w.token_embedding);
        defer hidden.deinit();
        ops.scaleInPlace(&hidden, cfg.embeddingScale());

        // 2. Apply RoPE and run through layers
        // Note: RoPE is applied inside the attention function after Q/K projection.
        // Here we process layer by layer.
        var x = hidden;
        hidden = undefined; // Transfer ownership

        for (w.layers) |*layer| {
            const new_x = try self.forwardLayer(scratch, &x, layer, gqa_cfg);
            x.deinit();
            x = new_x;
        }

        // 3. Final RMSNorm
        var x_norm = try ops.rmsNorm(scratch, &x, &w.final_norm, cfg.rms_norm_eps);
        x.deinit();

        // 4. Logits: x_norm @ token_embedding^T (tied weights)
        var token_emb_t = try ops.transpose(scratch, &w.token_embedding);
        defer token_emb_t.deinit();

        const logits = try ops.matmul(scratch, &x_norm, &token_emb_t);
        x_norm.deinit();

        return logits;
    }

    /// Forward pass for a single token using KV cache (autoregressive step).
    /// token_id: current token
    /// position: position in the sequence
    /// Returns: logits [1, vocab_size] — caller owns memory
    pub fn forwardStep(self: *Self, scratch: std.mem.Allocator, token_id: u32, position: usize) !Tensor {
        const cfg = self.cfg;
        const w = &self.weights;
        const hidden_dim = cfg.hidden_dim;
        _ = hidden_dim; // autofix
        const gqa_cfg = cfg.gqaConfig();

        var cache = &(self.kv_cache orelse return error.CacheNotInitialized);

        // 1. Embed single token
        const ids = [_]u32{token_id};
        var hidden = try ops.embedding(scratch, &ids, &w.token_embedding);
        defer hidden.deinit();
        ops.scaleInPlace(&hidden, cfg.embeddingScale());

        // 2. Process through layers with KV cache
        var x = hidden;
        hidden = undefined;

        for (w.layers, 0..) |*layer, layer_idx| {
            const new_x = try self.forwardLayerCached(scratch, &x, layer, gqa_cfg, cache, layer_idx, position);
            x.deinit();
            x = new_x;
        }

        // Increment cache length after processing all layers
        cache.current_length += 1;

        // 3. Final RMSNorm
        var x_norm = try ops.rmsNorm(scratch, &x, &w.final_norm, cfg.rms_norm_eps);
        x.deinit();

        // 4. Logits: x_norm @ token_embedding^T
        var token_emb_t = try ops.transpose(scratch, &w.token_embedding);
        defer token_emb_t.deinit();

        const logits = try ops.matmul(scratch, &x_norm, &token_emb_t);
        x_norm.deinit();

        return logits;
    }

    /// Forward through a single transformer layer (full sequence, no cache)
    fn forwardLayer(
        self: *const Self,
        allocator: std.mem.Allocator,
        x: *const Tensor,
        layer: *const GemmaLayerWeights,
        gqa_cfg: GQAConfig,
    ) !Tensor {
        const cfg = self.cfg;

        // 1. Pre-attention RMSNorm
        var attn_in = try ops.rmsNorm(allocator, x, &layer.input_norm, cfg.rms_norm_eps);
        defer attn_in.deinit();

        // 2. GQA self-attention (RoPE is applied to Q/K inside)
        // Build temporary GQA weights reference
        var gqa_weights = attention.GQAWeightsF32{
            .q_weight = layer.q_proj,
            .k_weight = layer.k_proj,
            .v_weight = layer.v_proj,
            .o_weight = layer.o_proj,
        };

        // Apply RoPE to Q and K after projection
        var attn_out = try self.gqaWithRope(allocator, &attn_in, &gqa_weights, gqa_cfg, 0);
        defer attn_out.deinit();

        // 3. Post-attention RMSNorm (Gemma 2 only)
        if (layer.post_attn_norm) |*norm_weight| {
            const normed = try ops.rmsNorm(allocator, &attn_out, norm_weight, cfg.rms_norm_eps);
            attn_out.deinit();
            attn_out = normed;
        }

        // 4. Residual connection
        var x1 = try ops.add(allocator, x, &attn_out);
        defer x1.deinit();

        // 5. Pre-FFN RMSNorm
        var ffn_in = try ops.rmsNorm(allocator, &x1, &layer.pre_ffn_norm, cfg.rms_norm_eps);
        defer ffn_in.deinit();

        // 6. GeGLU FFN: gate = SiLU(x @ gate_proj), up = x @ up_proj, out = (gate * up) @ down_proj
        var gate = try ops.matmul(allocator, &ffn_in, &layer.gate_proj);
        defer gate.deinit();
        ops.silu(&gate); // SiLU in-place

        var up = try ops.matmul(allocator, &ffn_in, &layer.up_proj);
        defer up.deinit();

        try ops.mulInPlace(&gate, &up); // gate = gate * up (element-wise)

        var ffn_out = try ops.matmul(allocator, &gate, &layer.down_proj);
        defer ffn_out.deinit();

        // 7. Post-FFN RMSNorm (Gemma 2 only)
        if (layer.post_ffn_norm) |*norm_weight| {
            const normed = try ops.rmsNorm(allocator, &ffn_out, norm_weight, cfg.rms_norm_eps);
            ffn_out.deinit();
            ffn_out = normed;
        }

        // 8. Residual connection
        return ops.add(allocator, &x1, &ffn_out);
    }

    /// Forward through a single layer with KV cache (single token)
    fn forwardLayerCached(
        self: *const Self,
        allocator: std.mem.Allocator,
        x: *const Tensor,
        layer: *const GemmaLayerWeights,
        gqa_cfg: GQAConfig,
        kv_cache: *KVCache,
        layer_idx: usize,
        position: usize,
    ) !Tensor {
        const cfg = self.cfg;

        // 1. Pre-attention RMSNorm
        var attn_in = try ops.rmsNorm(allocator, x, &layer.input_norm, cfg.rms_norm_eps);
        defer attn_in.deinit();

        // 2. GQA with KV cache
        var gqa_weights = attention.GQAWeightsF32{
            .q_weight = layer.q_proj,
            .k_weight = layer.k_proj,
            .v_weight = layer.v_proj,
            .o_weight = layer.o_proj,
        };

        var attn_out = try self.gqaCachedWithRope(allocator, &attn_in, &gqa_weights, gqa_cfg, kv_cache, layer_idx, position);
        defer attn_out.deinit();

        // 3. Post-attention RMSNorm (Gemma 2 only)
        if (layer.post_attn_norm) |*norm_weight| {
            const normed = try ops.rmsNorm(allocator, &attn_out, norm_weight, cfg.rms_norm_eps);
            attn_out.deinit();
            attn_out = normed;
        }

        // 4. Residual
        var x1 = try ops.add(allocator, x, &attn_out);
        defer x1.deinit();

        // 5. Pre-FFN RMSNorm
        var ffn_in = try ops.rmsNorm(allocator, &x1, &layer.pre_ffn_norm, cfg.rms_norm_eps);
        defer ffn_in.deinit();

        // 6. GeGLU FFN
        var gate = try ops.matmul(allocator, &ffn_in, &layer.gate_proj);
        defer gate.deinit();
        ops.silu(&gate);

        var up = try ops.matmul(allocator, &ffn_in, &layer.up_proj);
        defer up.deinit();

        try ops.mulInPlace(&gate, &up);

        var ffn_out = try ops.matmul(allocator, &gate, &layer.down_proj);
        defer ffn_out.deinit();

        // 7. Post-FFN RMSNorm (Gemma 2 only)
        if (layer.post_ffn_norm) |*norm_weight| {
            const normed = try ops.rmsNorm(allocator, &ffn_out, norm_weight, cfg.rms_norm_eps);
            ffn_out.deinit();
            ffn_out = normed;
        }

        // 8. Residual
        return ops.add(allocator, &x1, &ffn_out);
    }

    // =========================================================================
    // GQA + RoPE (applied after Q/K projection, before attention scores)
    // =========================================================================

    /// GQA with inline RoPE for full-sequence forward (no cache).
    /// Projects Q/K/V, applies RoPE to Q and K, then computes attention.
    fn gqaWithRope(
        self: *const Self,
        allocator: std.mem.Allocator,
        input: *const Tensor,
        weights: *const attention.GQAWeightsF32,
        gqa_cfg: GQAConfig,
        position_offset: usize,
    ) !Tensor {
        const seq_len = input.shape[0];
        const hidden_dim = gqa_cfg.hidden_dim;
        const head_dim = gqa_cfg.head_dim;
        const num_heads = gqa_cfg.num_heads;
        const num_kv_heads = gqa_cfg.num_kv_heads;
        const kv_dim = gqa_cfg.kv_dim();
        const heads_per_group = gqa_cfg.heads_per_group();
        const rope_base = self.cfg.rope_base;

        // Project Q: [seq_len, hidden_dim] -> [seq_len, hidden_dim]
        var q = try ops.matmul(allocator, input, &weights.q_weight);
        defer q.deinit();

        // Project K: [seq_len, hidden_dim] -> [seq_len, kv_dim]
        var k = try ops.matmul(allocator, input, &weights.k_weight);
        defer k.deinit();

        // Project V: [seq_len, hidden_dim] -> [seq_len, kv_dim]
        var v = try ops.matmul(allocator, input, &weights.v_weight);
        defer v.deinit();

        // Apply RoPE to Q and K
        ops.ropeInPlace(&q, &k, head_dim, num_heads, num_kv_heads, position_offset, rope_base);

        // Compute attention scores and weighted sum
        var out_shape = [_]usize{ seq_len, hidden_dim };
        var concat = try Tensor.init(allocator, &out_shape);
        errdefer concat.deinit();

        const scores_buf = try allocator.alloc(f32, seq_len * seq_len);
        defer allocator.free(scores_buf);

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
        const neg_inf = -std.math.inf(f32);

        for (0..num_heads) |h| {
            const q_head_offset = h * head_dim;
            const kv_head = h / heads_per_group;
            const kv_head_offset = kv_head * head_dim;

            for (0..seq_len) |i| {
                const q_base = i * hidden_dim + q_head_offset;
                for (0..seq_len) |j| {
                    if (j > i) {
                        scores_buf[i * seq_len + j] = neg_inf;
                        continue;
                    }
                    const k_base = j * kv_dim + kv_head_offset;
                    var dot: f32 = 0.0;
                    for (0..head_dim) |d| {
                        dot += q.data[q_base + d] * k.data[k_base + d];
                    }
                    scores_buf[i * seq_len + j] = dot * scale;
                }
            }

            // Softmax each row
            for (0..seq_len) |i| {
                const row = scores_buf[i * seq_len ..][0..seq_len];
                var max_val: f32 = row[0];
                for (row[1..]) |sv| {
                    if (sv > max_val) max_val = sv;
                }
                var sum_val: f32 = 0.0;
                for (row) |*sv| {
                    sv.* = @exp(sv.* - max_val);
                    sum_val += sv.*;
                }
                const inv_sum = 1.0 / sum_val;
                for (row) |*sv| {
                    sv.* *= inv_sum;
                }
            }

            // Output = scores @ V
            for (0..seq_len) |i| {
                const score_row = scores_buf[i * seq_len ..][0..seq_len];
                const out_base = i * hidden_dim + q_head_offset;
                for (0..head_dim) |d| {
                    concat.data[out_base + d] = 0.0;
                }
                for (0..seq_len) |j| {
                    const s = score_row[j];
                    if (s == 0.0) continue;
                    const v_base = j * kv_dim + kv_head_offset;
                    for (0..head_dim) |d| {
                        concat.data[out_base + d] += s * v.data[v_base + d];
                    }
                }
            }
        }

        // Output projection
        const output = try ops.matmul(allocator, &concat, &weights.o_weight);
        concat.deinit();
        return output;
    }

    /// GQA with RoPE for single-token cached decoding.
    fn gqaCachedWithRope(
        self: *const Self,
        allocator: std.mem.Allocator,
        input: *const Tensor,
        weights: *const attention.GQAWeightsF32,
        gqa_cfg: GQAConfig,
        kv_cache: *KVCache,
        layer: usize,
        position: usize,
    ) !Tensor {
        const hidden_dim = gqa_cfg.hidden_dim;
        const head_dim = gqa_cfg.head_dim;
        const num_heads = gqa_cfg.num_heads;
        const num_kv_heads = gqa_cfg.num_kv_heads;
        const kv_dim = gqa_cfg.kv_dim();
        const heads_per_group = gqa_cfg.heads_per_group();
        const rope_base = self.cfg.rope_base;

        // Project Q, K, V for current token
        var q = try ops.matmul(allocator, input, &weights.q_weight);
        defer q.deinit();

        var k_new = try ops.matmul(allocator, input, &weights.k_weight);
        defer k_new.deinit();

        var v_new = try ops.matmul(allocator, input, &weights.v_weight);
        defer v_new.deinit();

        // Apply RoPE at current position
        ops.ropeInPlace(&q, &k_new, head_dim, num_heads, num_kv_heads, position, rope_base);

        // Append to KV cache
        try kv_cache.append(layer, &k_new, &v_new);

        // Get all cached K/V up to current position
        const cache_len = position + 1;
        const all_k = kv_cache.keys[layer].data[0 .. cache_len * kv_dim];
        const all_v = kv_cache.values[layer].data[0 .. cache_len * kv_dim];

        // Allocate output [1, hidden_dim]
        var out_shape = [_]usize{ 1, hidden_dim };
        var concat = try Tensor.init(allocator, &out_shape);
        errdefer concat.deinit();

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
        const scores_buf = try allocator.alloc(f32, cache_len);
        defer allocator.free(scores_buf);

        for (0..num_heads) |h| {
            const q_head_offset = h * head_dim;
            const kv_head = h / heads_per_group;
            const kv_head_offset = kv_head * head_dim;

            // Compute scores: q_h dot each cached K
            for (0..cache_len) |j| {
                const k_base = j * kv_dim + kv_head_offset;
                var dot: f32 = 0.0;
                for (0..head_dim) |d| {
                    dot += q.data[q_head_offset + d] * all_k[k_base + d];
                }
                scores_buf[j] = dot * scale;
            }

            // Softmax
            var max_val: f32 = scores_buf[0];
            for (scores_buf[1..cache_len]) |sv| {
                if (sv > max_val) max_val = sv;
            }
            var sum_val: f32 = 0.0;
            for (scores_buf[0..cache_len]) |*sv| {
                sv.* = @exp(sv.* - max_val);
                sum_val += sv.*;
            }
            const inv_sum = 1.0 / sum_val;
            for (scores_buf[0..cache_len]) |*sv| {
                sv.* *= inv_sum;
            }

            // Weighted sum of V
            for (0..head_dim) |d| {
                concat.data[q_head_offset + d] = 0.0;
            }
            for (0..cache_len) |j| {
                const s = scores_buf[j];
                if (s == 0.0) continue;
                const v_base = j * kv_dim + kv_head_offset;
                for (0..head_dim) |d| {
                    concat.data[q_head_offset + d] += s * all_v[v_base + d];
                }
            }
        }

        // Output projection
        const output = try ops.matmul(allocator, &concat, &weights.o_weight);
        concat.deinit();
        return output;
    }

    // =========================================================================
    // Generation
    // =========================================================================

    /// Autoregressive text generation with greedy decoding.
    /// prompt_ids: input token IDs (should start with BOS)
    /// max_tokens: maximum tokens to generate (excluding prompt)
    /// temperature: sampling temperature (0.0 = greedy)
    /// Returns: array of all token IDs (prompt + generated), caller owns memory
    pub fn generate(self: *Self, prompt_ids: []const u32, max_tokens: usize, temperature: f32) ![]u32 {
        const allocator = self.allocator;
        const vocab_size = self.cfg.vocab_size;

        // Initialize KV cache if needed
        try self.initCache(prompt_ids.len + max_tokens);

        // Token output buffer
        var tokens: std.ArrayListUnmanaged(u32) = .{};
        errdefer tokens.deinit(allocator);
        try tokens.appendSlice(allocator, prompt_ids);

        // Step 1: Process prompt (full forward, no cache — populate cache for future steps)
        {
            var arena = std.heap.ArenaAllocator.init(allocator);
            defer arena.deinit();
            const scratch = arena.allocator();

            // Prefill: run prompt through all layers, populating KV cache
            var logits = try self.prefill(scratch, prompt_ids);

            // Get next token from last position's logits
            const last_logits_start = (prompt_ids.len - 1) * vocab_size;
            const last_logits = logits.data[last_logits_start..][0..vocab_size];
            const next_token = if (temperature == 0.0)
                argmax(last_logits)
            else
                sampleWithTemperature(last_logits, temperature);

            if (next_token == GemmaTokens.EOS) return tokens.toOwnedSlice(allocator);
            try tokens.append(allocator, next_token);
        }

        // Step 2: Autoregressive generation with KV cache
        var position = prompt_ids.len;
        while (tokens.items.len - prompt_ids.len < max_tokens) {
            var arena = std.heap.ArenaAllocator.init(allocator);
            defer arena.deinit();
            const scratch = arena.allocator();

            const current_token = tokens.items[tokens.items.len - 1];
            var logits = try self.forwardStep(scratch, current_token, position);

            const next_logits = logits.data[0..vocab_size];
            const next_token = if (temperature == 0.0)
                argmax(next_logits)
            else
                sampleWithTemperature(next_logits, temperature);

            if (next_token == GemmaTokens.EOS) break;
            try tokens.append(allocator, next_token);
            position += 1;
        }

        return tokens.toOwnedSlice(allocator);
    }

    /// Prefill: process the full prompt, populating KV cache.
    /// Returns logits for all positions [seq_len, vocab_size].
    fn prefill(self: *Self, scratch: std.mem.Allocator, input_ids: []const u32) !Tensor {
        const cfg = self.cfg;
        const w = &self.weights;
        const hidden_dim = cfg.hidden_dim;
        _ = hidden_dim; // autofix
        const gqa_cfg = cfg.gqaConfig();
        const kv_dim = cfg.kvDim();
        _ = kv_dim; // autofix
        var cache = &(self.kv_cache orelse return error.CacheNotInitialized);

        // 1. Embed
        var hidden = try ops.embedding(scratch, input_ids, &w.token_embedding);
        defer hidden.deinit();
        ops.scaleInPlace(&hidden, cfg.embeddingScale());

        var x = hidden;
        hidden = undefined;

        // 2. Process layers — manually handle KV cache population
        for (w.layers, 0..) |*layer, layer_idx| {
            var attn_in = try ops.rmsNorm(scratch, &x, &layer.input_norm, cfg.rms_norm_eps);
            defer attn_in.deinit();

            // GQA with RoPE and cache population
            var attn_out = try self.gqaPrefillWithRope(scratch, &attn_in, layer, gqa_cfg, cache, layer_idx);
            defer attn_out.deinit();

            if (layer.post_attn_norm) |*norm_weight| {
                const normed = try ops.rmsNorm(scratch, &attn_out, norm_weight, cfg.rms_norm_eps);
                attn_out.deinit();
                attn_out = normed;
            }

            var x1 = try ops.add(scratch, &x, &attn_out);
            defer x1.deinit();

            var ffn_in = try ops.rmsNorm(scratch, &x1, &layer.pre_ffn_norm, cfg.rms_norm_eps);
            defer ffn_in.deinit();

            var gate = try ops.matmul(scratch, &ffn_in, &layer.gate_proj);
            defer gate.deinit();
            ops.silu(&gate);

            var up = try ops.matmul(scratch, &ffn_in, &layer.up_proj);
            defer up.deinit();
            try ops.mulInPlace(&gate, &up);

            var ffn_out = try ops.matmul(scratch, &gate, &layer.down_proj);
            defer ffn_out.deinit();

            if (layer.post_ffn_norm) |*norm_weight| {
                const normed = try ops.rmsNorm(scratch, &ffn_out, norm_weight, cfg.rms_norm_eps);
                ffn_out.deinit();
                ffn_out = normed;
            }

            const new_x = try ops.add(scratch, &x1, &ffn_out);
            x.deinit();
            x = new_x;
        }

        // Set cache length after all layers processed
        cache.current_length = input_ids.len;

        // 3. Final RMSNorm + logits
        var x_norm = try ops.rmsNorm(scratch, &x, &w.final_norm, cfg.rms_norm_eps);
        x.deinit();

        var token_emb_t = try ops.transpose(scratch, &w.token_embedding);
        defer token_emb_t.deinit();

        const logits = try ops.matmul(scratch, &x_norm, &token_emb_t);
        x_norm.deinit();
        return logits;
    }

    /// GQA with RoPE for prefill — processes full sequence and populates KV cache.
    fn gqaPrefillWithRope(
        self: *const Self,
        allocator: std.mem.Allocator,
        input: *const Tensor,
        layer: *const GemmaLayerWeights,
        gqa_cfg: GQAConfig,
        kv_cache: *KVCache,
        layer_idx: usize,
    ) !Tensor {
        const seq_len = input.shape[0];
        const hidden_dim = gqa_cfg.hidden_dim;
        const head_dim = gqa_cfg.head_dim;
        const num_heads = gqa_cfg.num_heads;
        const num_kv_heads = gqa_cfg.num_kv_heads;
        const kv_dim = gqa_cfg.kv_dim();
        const heads_per_group = gqa_cfg.heads_per_group();
        const rope_base = self.cfg.rope_base;

        var q = try ops.matmul(allocator, input, &layer.q_proj);
        defer q.deinit();

        var k = try ops.matmul(allocator, input, &layer.k_proj);
        defer k.deinit();

        var v = try ops.matmul(allocator, input, &layer.v_proj);
        defer v.deinit();

        // Apply RoPE
        ops.ropeInPlace(&q, &k, head_dim, num_heads, num_kv_heads, 0, rope_base);

        // Store K/V into cache row-by-row
        const cache_k = kv_cache.keys[layer_idx].data;
        const cache_v = kv_cache.values[layer_idx].data;
        for (0..seq_len) |i| {
            @memcpy(cache_k[i * kv_dim ..][0..kv_dim], k.data[i * kv_dim ..][0..kv_dim]);
            @memcpy(cache_v[i * kv_dim ..][0..kv_dim], v.data[i * kv_dim ..][0..kv_dim]);
        }

        // Compute attention with causal mask
        var out_shape = [_]usize{ seq_len, hidden_dim };
        var concat = try Tensor.init(allocator, &out_shape);
        errdefer concat.deinit();

        const scores_buf = try allocator.alloc(f32, seq_len * seq_len);
        defer allocator.free(scores_buf);

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
        const neg_inf = -std.math.inf(f32);

        for (0..num_heads) |h| {
            const q_head_offset = h * head_dim;
            const kv_head = h / heads_per_group;
            const kv_head_offset = kv_head * head_dim;

            for (0..seq_len) |i| {
                const q_base = i * hidden_dim + q_head_offset;
                for (0..seq_len) |j| {
                    if (j > i) {
                        scores_buf[i * seq_len + j] = neg_inf;
                        continue;
                    }
                    const k_base = j * kv_dim + kv_head_offset;
                    var dot: f32 = 0.0;
                    for (0..head_dim) |d| {
                        dot += q.data[q_base + d] * k.data[k_base + d];
                    }
                    scores_buf[i * seq_len + j] = dot * scale;
                }
            }

            for (0..seq_len) |i| {
                const row = scores_buf[i * seq_len ..][0..seq_len];
                var max_val: f32 = row[0];
                for (row[1..]) |sv| {
                    if (sv > max_val) max_val = sv;
                }
                var sum_val: f32 = 0.0;
                for (row) |*sv| {
                    sv.* = @exp(sv.* - max_val);
                    sum_val += sv.*;
                }
                const inv_sum = 1.0 / sum_val;
                for (row) |*sv| {
                    sv.* *= inv_sum;
                }
            }

            for (0..seq_len) |i| {
                const score_row = scores_buf[i * seq_len ..][0..seq_len];
                const out_base = i * hidden_dim + q_head_offset;
                for (0..head_dim) |d| {
                    concat.data[out_base + d] = 0.0;
                }
                for (0..seq_len) |j| {
                    const s = score_row[j];
                    if (s == 0.0) continue;
                    const v_base = j * kv_dim + kv_head_offset;
                    for (0..head_dim) |d| {
                        concat.data[out_base + d] += s * v.data[v_base + d];
                    }
                }
            }
        }

        const output = try ops.matmul(allocator, &concat, &layer.o_proj);
        concat.deinit();
        return output;
    }

    // =========================================================================
    // Weight Loading
    // =========================================================================

    fn loadWeights(allocator: std.mem.Allocator, loader: *ModelLoader, cfg: GemmaConfig) !GemmaWeights {
        const is_gemma2 = cfg.isGemma2();

        // Token embedding (always dequantized for embedding lookup)
        var token_embedding = try loader.getTensorDequantized("token_embedding.weight");
        errdefer token_embedding.deinit();

        // Layer weights
        var layers = try allocator.alloc(GemmaLayerWeights, cfg.num_layers);
        var loaded: usize = 0;
        errdefer {
            for (layers[0..loaded]) |*l| l.deinit();
            allocator.free(layers);
        }

        var buf: [128]u8 = undefined;

        for (0..cfg.num_layers) |i| {
            var lw: GemmaLayerWeights = undefined;

            // Pre-attention RMSNorm
            const in_norm = try std.fmt.bufPrint(&buf, "layers.{d}.input_layernorm.weight", .{i});
            lw.input_norm = try loader.getTensorDequantized(in_norm);

            // Attention projections
            const q_w = try std.fmt.bufPrint(&buf, "layers.{d}.self_attn.q_proj.weight", .{i});
            lw.q_proj = try loader.getTensorDequantized(q_w);
            const k_w = try std.fmt.bufPrint(&buf, "layers.{d}.self_attn.k_proj.weight", .{i});
            lw.k_proj = try loader.getTensorDequantized(k_w);
            const v_w = try std.fmt.bufPrint(&buf, "layers.{d}.self_attn.v_proj.weight", .{i});
            lw.v_proj = try loader.getTensorDequantized(v_w);
            const o_w = try std.fmt.bufPrint(&buf, "layers.{d}.self_attn.o_proj.weight", .{i});
            lw.o_proj = try loader.getTensorDequantized(o_w);

            // Post-attention RMSNorm (Gemma 2 only)
            if (is_gemma2) {
                const pa_norm = try std.fmt.bufPrint(&buf, "layers.{d}.post_attention_layernorm.weight", .{i});
                lw.post_attn_norm = try loader.getTensorDequantized(pa_norm);
            } else {
                lw.post_attn_norm = null;
            }

            // Pre-FFN RMSNorm
            const ffn_norm = try std.fmt.bufPrint(&buf, "layers.{d}.post_attention_layernorm.weight", .{i});
            // NOTE: In Gemma 1 HuggingFace naming, post_attention_layernorm IS the pre-FFN norm.
            // In Gemma 2, we use a different key. Handle naming carefully in export script.
            if (is_gemma2) {
                const pre_ffn = try std.fmt.bufPrint(&buf, "layers.{d}.pre_feedforward_layernorm.weight", .{i});
                lw.pre_ffn_norm = try loader.getTensorDequantized(pre_ffn);
            } else {
                lw.pre_ffn_norm = try loader.getTensorDequantized(ffn_norm);
            }

            // FFN weights
            const gate_w = try std.fmt.bufPrint(&buf, "layers.{d}.mlp.gate_proj.weight", .{i});
            lw.gate_proj = try loader.getTensorDequantized(gate_w);
            const up_w = try std.fmt.bufPrint(&buf, "layers.{d}.mlp.up_proj.weight", .{i});
            lw.up_proj = try loader.getTensorDequantized(up_w);
            const down_w = try std.fmt.bufPrint(&buf, "layers.{d}.mlp.down_proj.weight", .{i});
            lw.down_proj = try loader.getTensorDequantized(down_w);

            // Post-FFN RMSNorm (Gemma 2 only)
            if (is_gemma2) {
                const pf_norm = try std.fmt.bufPrint(&buf, "layers.{d}.post_feedforward_layernorm.weight", .{i});
                lw.post_ffn_norm = try loader.getTensorDequantized(pf_norm);
            } else {
                lw.post_ffn_norm = null;
            }

            layers[i] = lw;
            loaded += 1;
        }

        // Final RMSNorm
        const final_norm = try loader.getTensorDequantized("norm.weight");

        return GemmaWeights{
            .token_embedding = token_embedding,
            .layers = layers,
            .final_norm = final_norm,
            .allocator = allocator,
        };
    }

    // =========================================================================
    // Cleanup
    // =========================================================================

    pub fn deinit(self: *Self) void {
        self.weights.deinit();
        if (self.kv_cache) |*c| c.deinit();
    }
};

// =============================================================================
// Sampling Utilities
// =============================================================================

/// Greedy selection: return index of maximum value
fn argmax(logits: []const f32) u32 {
    var max_idx: u32 = 0;
    var max_val = logits[0];
    for (logits[1..], 1..) |v, i| {
        if (v > max_val) {
            max_val = v;
            max_idx = @intCast(i);
        }
    }
    return max_idx;
}

/// Temperature-scaled sampling with softmax
fn sampleWithTemperature(logits: []const f32, temperature: f32) u32 {
    const len = logits.len;

    // Find max for numerical stability
    var max_val: f32 = logits[0];
    for (logits[1..]) |v| {
        if (v > max_val) max_val = v;
    }

    // Compute softmax with temperature
    var sum: f32 = 0.0;
    // We need a temporary buffer — use the stack for small vocab, but Gemma has 256k vocab.
    // For simplicity, do a two-pass approach with a running sum.

    // First pass: compute sum of exp
    for (logits) |v| {
        sum += @exp((v - max_val) / temperature);
    }

    // Second pass: sample
    var rng = std.Random.DefaultPrng.init(@intCast(std.time.nanoTimestamp()));
    var r = rng.random().float(f32) * sum;

    for (logits, 0..) |v, i| {
        r -= @exp((v - max_val) / temperature);
        if (r <= 0.0) return @intCast(i);
    }

    return @intCast(len - 1);
}

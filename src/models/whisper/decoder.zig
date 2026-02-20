// src/models/whisper/decoder.zig
// Whisper Text Decoder
//
// Architecture (per generation step):
//   Previous Token IDs
//       │
//       ▼
//   Token Embedding + Positional Embedding
//       │   [seq_len, n_text_state]
//       ▼
//   N × Decoder Blocks:
//       │
//       ├── Causal Self-Attention (can only see past tokens)
//       │
//       ├── Cross-Attention (attends to encoder output)
//       │
//       └── FFN
//       │
//       ▼
//   Final LayerNorm
//       │
//       ▼
//   Linear (project to vocabulary) using tied token embeddings
//       │
//       ▼
//   Logits [vocab_size]

const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const ops = @import("ops.zig");
const attention = @import("attention.zig");
const cache_mod = @import("cache.zig");
const config = @import("config.zig");
const loader_mod = @import("loader.zig");

const ModelLoader = loader_mod.ModelLoader;
const LoadError = loader_mod.LoadError;
const DecoderConfig = config.DecoderConfig;
const AttentionConfig = attention.AttentionConfig;
const CrossAttentionConfig = attention.CrossAttentionConfig;
const AttentionWeightsF32 = attention.AttentionWeightsF32;
const CrossAttentionWeightsF32 = attention.CrossAttentionWeightsF32;
const LayerKVCache = cache_mod.LayerKVCache;
const DecoderKVCache = cache_mod.DecoderKVCache;
const CachedCrossAttentionKV = attention.CachedCrossAttentionKV;
const SelfAttentionKVCache = attention.SelfAttentionKVCache;

/// Decoder block weights (causal self-attention + cross-attention + FFN)
pub const DecoderBlockWeights = struct {
    // Causal self-attention
    self_attn_ln_gamma: Tensor, // [n_text_state]
    self_attn_ln_beta: Tensor, // [n_text_state]
    self_attn: AttentionWeightsF32,

    // Cross-attention to encoder
    cross_attn_ln_gamma: Tensor, // [n_text_state]
    cross_attn_ln_beta: Tensor, // [n_text_state]
    cross_attn: CrossAttentionWeightsF32,

    // Feed-forward network
    ffn_ln_gamma: Tensor, // [n_text_state]
    ffn_ln_beta: Tensor, // [n_text_state]
    ffn_fc1_weight: Tensor, // [n_text_state, 4 * n_text_state] pre-transposed
    ffn_fc1_bias: Tensor, // [4 * n_text_state]
    ffn_fc2_weight: Tensor, // [4 * n_text_state, n_text_state] pre-transposed
    ffn_fc2_bias: Tensor, // [n_text_state]

    pub fn deinit(self: *DecoderBlockWeights) void {
        self.self_attn_ln_gamma.deinit();
        self.self_attn_ln_beta.deinit();
        self.self_attn.deinit();
        self.cross_attn_ln_gamma.deinit();
        self.cross_attn_ln_beta.deinit();
        self.cross_attn.deinit();
        self.ffn_ln_gamma.deinit();
        self.ffn_ln_beta.deinit();
        self.ffn_fc1_weight.deinit();
        self.ffn_fc1_bias.deinit();
        self.ffn_fc2_weight.deinit();
        self.ffn_fc2_bias.deinit();
    }
};

/// Complete Whisper decoder weights
pub const WhisperDecoderWeights = struct {
    // Token embedding (also used for output projection via tied weights)
    token_embedding: Tensor, // [n_vocab, n_text_state]

    // Positional embedding
    positional_embedding: Tensor, // [n_text_ctx, n_text_state]

    // Decoder blocks
    blocks: []DecoderBlockWeights,

    // Final layer norm
    ln_gamma: Tensor, // [n_text_state]
    ln_beta: Tensor, // [n_text_state]

    allocator: std.mem.Allocator,

    /// Load decoder weights from a ModelLoader
    pub fn loadFromLoader(allocator: std.mem.Allocator, loader: *ModelLoader, cfg: DecoderConfig) !*WhisperDecoderWeights {
        var weights = try allocator.create(WhisperDecoderWeights);
        errdefer allocator.destroy(weights);
        weights.allocator = allocator;

        // Load embeddings
        weights.token_embedding = try loader.getTensorDequantized("decoder.token_embedding");
        errdefer weights.token_embedding.deinit();
        weights.positional_embedding = try loader.getTensorDequantized("decoder.positional_embedding");
        errdefer weights.positional_embedding.deinit();

        // Load decoder blocks
        weights.blocks = try allocator.alloc(DecoderBlockWeights, cfg.n_text_layer);
        errdefer allocator.free(weights.blocks);

        var loaded_blocks: usize = 0;
        errdefer {
            for (weights.blocks[0..loaded_blocks]) |*block| block.deinit();
        }

        for (0..cfg.n_text_layer) |i| {
            var buf: [64]u8 = undefined;

            // Self-attention LayerNorm
            const attn_ln_w = try std.fmt.bufPrint(&buf, "decoder.blocks.{d}.attn_ln.weight", .{i});
            weights.blocks[i].self_attn_ln_gamma = try loader.getTensorDequantized(attn_ln_w);
            const attn_ln_b = try std.fmt.bufPrint(&buf, "decoder.blocks.{d}.attn_ln.bias", .{i});
            weights.blocks[i].self_attn_ln_beta = try loader.getTensorDequantized(attn_ln_b);

            // Self-attention Q, K, V, O projections
            const q_w = try std.fmt.bufPrint(&buf, "decoder.blocks.{d}.attn.q_weight", .{i});
            weights.blocks[i].self_attn.q_weight = try loader.getTensorDequantized(q_w);
            const q_b = try std.fmt.bufPrint(&buf, "decoder.blocks.{d}.attn.q_bias", .{i});
            weights.blocks[i].self_attn.q_bias = try loader.getTensorDequantized(q_b);
            const k_w = try std.fmt.bufPrint(&buf, "decoder.blocks.{d}.attn.k_weight", .{i});
            weights.blocks[i].self_attn.k_weight = try loader.getTensorDequantized(k_w);
            const k_b = try std.fmt.bufPrint(&buf, "decoder.blocks.{d}.attn.k_bias", .{i});
            weights.blocks[i].self_attn.k_bias = try loader.getTensorDequantized(k_b);
            const v_w = try std.fmt.bufPrint(&buf, "decoder.blocks.{d}.attn.v_weight", .{i});
            weights.blocks[i].self_attn.v_weight = try loader.getTensorDequantized(v_w);
            const v_b = try std.fmt.bufPrint(&buf, "decoder.blocks.{d}.attn.v_bias", .{i});
            weights.blocks[i].self_attn.v_bias = try loader.getTensorDequantized(v_b);
            const o_w = try std.fmt.bufPrint(&buf, "decoder.blocks.{d}.attn.o_weight", .{i});
            weights.blocks[i].self_attn.o_weight = try loader.getTensorDequantized(o_w);
            const o_b = try std.fmt.bufPrint(&buf, "decoder.blocks.{d}.attn.o_bias", .{i});
            weights.blocks[i].self_attn.o_bias = try loader.getTensorDequantized(o_b);

            // Cross-attention LayerNorm
            const cross_ln_w = try std.fmt.bufPrint(&buf, "decoder.blocks.{d}.cross_attn_ln.weight", .{i});
            weights.blocks[i].cross_attn_ln_gamma = try loader.getTensorDequantized(cross_ln_w);
            const cross_ln_b = try std.fmt.bufPrint(&buf, "decoder.blocks.{d}.cross_attn_ln.bias", .{i});
            weights.blocks[i].cross_attn_ln_beta = try loader.getTensorDequantized(cross_ln_b);

            // Cross-attention Q, K, V, O projections
            const cq_w = try std.fmt.bufPrint(&buf, "decoder.blocks.{d}.cross_attn.q_weight", .{i});
            weights.blocks[i].cross_attn.q_weight = try loader.getTensorDequantized(cq_w);
            const cq_b = try std.fmt.bufPrint(&buf, "decoder.blocks.{d}.cross_attn.q_bias", .{i});
            weights.blocks[i].cross_attn.q_bias = try loader.getTensorDequantized(cq_b);
            const ck_w = try std.fmt.bufPrint(&buf, "decoder.blocks.{d}.cross_attn.k_weight", .{i});
            weights.blocks[i].cross_attn.k_weight = try loader.getTensorDequantized(ck_w);
            const ck_b = try std.fmt.bufPrint(&buf, "decoder.blocks.{d}.cross_attn.k_bias", .{i});
            weights.blocks[i].cross_attn.k_bias = try loader.getTensorDequantized(ck_b);
            const cv_w = try std.fmt.bufPrint(&buf, "decoder.blocks.{d}.cross_attn.v_weight", .{i});
            weights.blocks[i].cross_attn.v_weight = try loader.getTensorDequantized(cv_w);
            const cv_b = try std.fmt.bufPrint(&buf, "decoder.blocks.{d}.cross_attn.v_bias", .{i});
            weights.blocks[i].cross_attn.v_bias = try loader.getTensorDequantized(cv_b);
            const co_w = try std.fmt.bufPrint(&buf, "decoder.blocks.{d}.cross_attn.o_weight", .{i});
            weights.blocks[i].cross_attn.o_weight = try loader.getTensorDequantized(co_w);
            const co_b = try std.fmt.bufPrint(&buf, "decoder.blocks.{d}.cross_attn.o_bias", .{i});
            weights.blocks[i].cross_attn.o_bias = try loader.getTensorDequantized(co_b);

            // FFN LayerNorm
            const mlp_ln_w = try std.fmt.bufPrint(&buf, "decoder.blocks.{d}.mlp_ln.weight", .{i});
            weights.blocks[i].ffn_ln_gamma = try loader.getTensorDequantized(mlp_ln_w);
            const mlp_ln_b = try std.fmt.bufPrint(&buf, "decoder.blocks.{d}.mlp_ln.bias", .{i});
            weights.blocks[i].ffn_ln_beta = try loader.getTensorDequantized(mlp_ln_b);

            // FFN fc1, fc2
            const fc1_w = try std.fmt.bufPrint(&buf, "decoder.blocks.{d}.mlp.fc1.weight", .{i});
            weights.blocks[i].ffn_fc1_weight = try loader.getTensorDequantized(fc1_w);
            const fc1_b = try std.fmt.bufPrint(&buf, "decoder.blocks.{d}.mlp.fc1.bias", .{i});
            weights.blocks[i].ffn_fc1_bias = try loader.getTensorDequantized(fc1_b);
            const fc2_w = try std.fmt.bufPrint(&buf, "decoder.blocks.{d}.mlp.fc2.weight", .{i});
            weights.blocks[i].ffn_fc2_weight = try loader.getTensorDequantized(fc2_w);
            const fc2_b = try std.fmt.bufPrint(&buf, "decoder.blocks.{d}.mlp.fc2.bias", .{i});
            weights.blocks[i].ffn_fc2_bias = try loader.getTensorDequantized(fc2_b);

            loaded_blocks += 1;
        }

        // Load final LayerNorm
        weights.ln_gamma = try loader.getTensorDequantized("decoder.ln.weight");
        errdefer weights.ln_gamma.deinit();
        weights.ln_beta = try loader.getTensorDequantized("decoder.ln.bias");

        return weights;
    }

    pub fn deinit(self: *WhisperDecoderWeights) void {
        self.token_embedding.deinit();
        self.positional_embedding.deinit();
        for (self.blocks) |*block| block.deinit();
        self.allocator.free(self.blocks);
        self.ln_gamma.deinit();
        self.ln_beta.deinit();
        // Free the struct itself (allocated by loadFromLoader)
        self.allocator.destroy(self);
    }
};

/// Whisper text decoder
pub const WhisperDecoder = struct {
    cfg: DecoderConfig,
    weights: *WhisperDecoderWeights,
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Create decoder with loaded weights
    pub fn init(
        allocator: std.mem.Allocator,
        cfg: DecoderConfig,
        weights: *WhisperDecoderWeights,
    ) Self {
        return .{
            .cfg = cfg,
            .weights = weights,
            .allocator = allocator,
        };
    }

    /// Embed token IDs with positional encoding
    /// tokens: array of token IDs
    /// position_offset: starting position (0 for first tokens)
    fn embedTokens(self: *const Self, tokens: []const u32, position_offset: usize) !Tensor {
        const allocator = self.allocator;
        const w = self.weights;

        // Token embedding lookup
        var token_emb = try ops.embedding(allocator, tokens, &w.token_embedding);
        errdefer token_emb.deinit();

        // Add positional embeddings for each position
        const seq_len = tokens.len;
        const hidden_dim = self.cfg.n_text_state;

        for (0..seq_len) |i| {
            const pos = position_offset + i;
            const row_start = i * hidden_dim;
            const pos_start = pos * hidden_dim;

            for (0..hidden_dim) |j| {
                token_emb.data[row_start + j] += w.positional_embedding.data[pos_start + j];
            }
        }

        return token_emb;
    }

    /// Forward pass through a single decoder block
    fn forwardBlock(
        self: *const Self,
        allocator: std.mem.Allocator,
        x: *const Tensor,
        encoder_output: *const Tensor,
        block: *const DecoderBlockWeights,
    ) !Tensor {
        const attn_config = AttentionConfig{
            .num_heads = self.cfg.n_text_head,
            .hidden_dim = self.cfg.n_text_state,
            .head_dim = self.cfg.headDim(),
        };

        const cross_config = CrossAttentionConfig{
            .num_heads = self.cfg.n_text_head,
            .decoder_hidden = self.cfg.n_text_state,
            .encoder_hidden = self.cfg.n_text_state, // Same as decoder for Whisper
            .head_dim = self.cfg.headDim(),
        };

        // 1. Pre-norm causal self-attention
        var self_attn_ln = try ops.layerNorm(
            allocator,
            x,
            &block.self_attn_ln_gamma,
            &block.self_attn_ln_beta,
            1e-5,
        );
        defer self_attn_ln.deinit();

        var self_attn_out = try attention.multiHeadCausalAttention(
            Tensor,
            allocator,
            &self_attn_ln,
            &block.self_attn,
            attn_config,
        );
        defer self_attn_out.deinit();

        // Residual
        var x1 = try ops.add(allocator, x, &self_attn_out);
        defer x1.deinit();

        // 2. Pre-norm cross-attention
        var cross_attn_ln = try ops.layerNorm(
            allocator,
            &x1,
            &block.cross_attn_ln_gamma,
            &block.cross_attn_ln_beta,
            1e-5,
        );
        defer cross_attn_ln.deinit();

        var cross_attn_out = try attention.multiHeadCrossAttention(
            Tensor,
            allocator,
            &cross_attn_ln,
            encoder_output,
            &block.cross_attn,
            cross_config,
        );
        defer cross_attn_out.deinit();

        // Residual
        var x2 = try ops.add(allocator, &x1, &cross_attn_out);
        defer x2.deinit();

        // 3. Pre-norm FFN
        var ffn_ln = try ops.layerNorm(allocator, &x2, &block.ffn_ln_gamma, &block.ffn_ln_beta, 1e-5);
        defer ffn_ln.deinit();

        // FFN: fc1 -> GELU (exact) -> fc2
        // Whisper uses exact GELU (approximate='none')
        var fc1 = try ops.matmul(allocator, &ffn_ln, &block.ffn_fc1_weight);
        defer fc1.deinit();
        try ops.addBiasInPlace(&fc1, &block.ffn_fc1_bias);
        ops.geluExact(&fc1);

        var fc2 = try ops.matmul(allocator, &fc1, &block.ffn_fc2_weight);
        try ops.addBiasInPlace(&fc2, &block.ffn_fc2_bias);

        // Residual
        const result = try ops.add(allocator, &x2, &fc2);
        fc2.deinit();

        return result;
    }

    /// Decode a sequence of tokens given encoder output
    /// Used for initial prompt processing (processes all tokens at once)
    /// tokens: [seq_len] array of token IDs
    /// encoder_output: [n_audio_ctx, n_audio_state] from encoder
    /// Returns: [seq_len, n_vocab] logits for each position
    pub fn forward(
        self: *const Self,
        tokens: []const u32,
        encoder_output: *const Tensor,
    ) !Tensor {
        const allocator = self.allocator;
        const w = self.weights;

        // Embed tokens with positions
        var x = try self.embedTokens(tokens, 0);

        // Process through decoder blocks
        for (w.blocks) |*block| {
            const new_x = try self.forwardBlock(allocator, &x, encoder_output, block);
            x.deinit();
            x = new_x;
        }

        // Final LayerNorm
        var x_norm = try ops.layerNorm(allocator, &x, &w.ln_gamma, &w.ln_beta, 1e-5);
        x.deinit();

        // Project to vocabulary using tied weights (token_embedding transposed)
        // logits = x_norm @ token_embedding^T
        var token_emb_t = try ops.transpose(allocator, &w.token_embedding);
        defer token_emb_t.deinit();

        const logits = try ops.matmul(allocator, &x_norm, &token_emb_t);
        x_norm.deinit();

        return logits;
    }

    /// Decode a single token (for autoregressive generation)
    /// Used during generation loop, one token at a time
    /// token_id: the current token to process
    /// position: position in the sequence (for positional embedding)
    /// encoder_output: [n_audio_ctx, n_audio_state] from encoder
    /// Returns: [1, n_vocab] logits for the next token
    pub fn decodeStep(
        self: *const Self,
        token_id: u32,
        position: usize,
        encoder_output: *const Tensor,
    ) !Tensor {
        const tokens = [_]u32{token_id};
        const allocator = self.allocator;
        const w = self.weights;

        // Embed single token with position
        var x = try self.embedTokens(&tokens, position);

        // Process through decoder blocks
        // Note: Without KV cache, this recomputes all previous tokens each time
        // For efficient generation, use decodeStepWithCache
        for (w.blocks) |*block| {
            const new_x = try self.forwardBlock(allocator, &x, encoder_output, block);
            x.deinit();
            x = new_x;
        }

        // Final LayerNorm
        var x_norm = try ops.layerNorm(allocator, &x, &w.ln_gamma, &w.ln_beta, 1e-5);
        x.deinit();

        // Project to vocabulary
        var token_emb_t = try ops.transpose(allocator, &w.token_embedding);
        defer token_emb_t.deinit();

        const logits = try ops.matmul(allocator, &x_norm, &token_emb_t);
        x_norm.deinit();

        return logits;
    }

    /// Get argmax of logits (greedy selection)
    fn argmax(logits: []const f32) u32 {
        var max_idx: u32 = 0;
        var max_val: f32 = logits[0];
        for (logits[1..], 1..) |v, i| {
            if (v > max_val) {
                max_val = v;
                max_idx = @intCast(i);
            }
        }
        return max_idx;
    }

    /// Greedy decoding: generate tokens until EOT or max_tokens
    /// encoder_output: [n_audio_ctx, n_audio_state] from encoder
    /// prompt_tokens: initial tokens (e.g., [SOT, LANG_EN, TRANSCRIBE, NO_TIMESTAMPS])
    /// max_tokens: maximum number of tokens to generate (default 224)
    /// Returns: slice of generated token IDs (caller owns memory)
    pub fn greedyDecode(
        self: *const Self,
        encoder_output: *const Tensor,
        prompt_tokens: []const u32,
        max_tokens: usize,
    ) ![]u32 {
        const allocator = self.allocator;
        const eot_token = config.WhisperTokens.EOT;

        // Allocate output buffer for all tokens (prompt + generated)
        var tokens = try allocator.alloc(u32, prompt_tokens.len + max_tokens);
        errdefer allocator.free(tokens);

        // Copy prompt tokens
        @memcpy(tokens[0..prompt_tokens.len], prompt_tokens);
        var num_tokens: usize = prompt_tokens.len;

        // Process prompt to get initial logits
        var logits = try self.forward(prompt_tokens, encoder_output);

        // Get first generated token from last position's logits
        const vocab_size = self.cfg.n_vocab;
        const last_logits = logits.data[(prompt_tokens.len - 1) * vocab_size ..][0..vocab_size];
        var next_token = argmax(last_logits);
        logits.deinit();

        // Generate tokens until EOT or max
        while (num_tokens < prompt_tokens.len + max_tokens) {
            // Check for end of transcription
            if (next_token == eot_token) {
                break;
            }

            // Add token to sequence
            tokens[num_tokens] = next_token;
            num_tokens += 1;

            // Get logits for next token
            // Note: Without KV cache, we reprocess the full sequence each time
            // This is inefficient but correct. KV cache optimization comes later.
            var step_logits = try self.forward(tokens[0..num_tokens], encoder_output);
            defer step_logits.deinit();

            // Get next token (last position)
            const step_last = step_logits.data[(num_tokens - 1) * vocab_size ..][0..vocab_size];
            next_token = argmax(step_last);
        }

        // Resize to actual length
        if (num_tokens < tokens.len) {
            const result = try allocator.realloc(tokens, num_tokens);
            return result;
        }
        return tokens;
    }

    /// Forward pass through a single decoder block WITH cached cross-attention K/V
    /// This avoids recomputing the encoder K/V projections every step
    fn forwardBlockWithCache(
        self: *const Self,
        allocator: std.mem.Allocator,
        x: *const Tensor,
        cached_cross_kv: *const CachedCrossAttentionKV,
        block: *const DecoderBlockWeights,
    ) !Tensor {
        const attn_config = AttentionConfig{
            .num_heads = self.cfg.n_text_head,
            .hidden_dim = self.cfg.n_text_state,
            .head_dim = self.cfg.headDim(),
        };

        const cross_config = CrossAttentionConfig{
            .num_heads = self.cfg.n_text_head,
            .decoder_hidden = self.cfg.n_text_state,
            .encoder_hidden = self.cfg.n_text_state,
            .head_dim = self.cfg.headDim(),
        };

        // 1. Pre-norm causal self-attention
        var self_attn_ln = try ops.layerNorm(
            allocator,
            x,
            &block.self_attn_ln_gamma,
            &block.self_attn_ln_beta,
            1e-5,
        );
        defer self_attn_ln.deinit();

        var self_attn_out = try attention.multiHeadCausalAttention(
            Tensor,
            allocator,
            &self_attn_ln,
            &block.self_attn,
            attn_config,
        );
        defer self_attn_out.deinit();

        // Residual
        var x1 = try ops.add(allocator, x, &self_attn_out);
        defer x1.deinit();

        // 2. Pre-norm cross-attention WITH CACHED K/V
        var cross_attn_ln = try ops.layerNorm(
            allocator,
            &x1,
            &block.cross_attn_ln_gamma,
            &block.cross_attn_ln_beta,
            1e-5,
        );
        defer cross_attn_ln.deinit();

        // Use cached cross-attention - only compute Q projection
        var cross_attn_out = try attention.multiHeadCrossAttentionWithCache(
            Tensor,
            allocator,
            &cross_attn_ln,
            cached_cross_kv,
            &block.cross_attn,
            cross_config,
        );
        defer cross_attn_out.deinit();

        // Residual
        var x2 = try ops.add(allocator, &x1, &cross_attn_out);
        defer x2.deinit();

        // 3. Pre-norm FFN
        var ffn_ln = try ops.layerNorm(allocator, &x2, &block.ffn_ln_gamma, &block.ffn_ln_beta, 1e-5);
        defer ffn_ln.deinit();

        // FFN: fc1 -> GELU (exact) -> fc2
        var fc1 = try ops.matmul(allocator, &ffn_ln, &block.ffn_fc1_weight);
        defer fc1.deinit();
        try ops.addBiasInPlace(&fc1, &block.ffn_fc1_bias);
        ops.geluExact(&fc1);

        var fc2 = try ops.matmul(allocator, &fc1, &block.ffn_fc2_weight);
        try ops.addBiasInPlace(&fc2, &block.ffn_fc2_bias);

        // Residual
        const result = try ops.add(allocator, &x2, &fc2);
        fc2.deinit();

        return result;
    }

    /// Forward with cached cross-attention K/V
    /// cached_cross_kvs: pre-computed K/V for each layer
    fn forwardWithCache(
        self: *const Self,
        tokens: []const u32,
        cached_cross_kvs: []const CachedCrossAttentionKV,
    ) !Tensor {
        const allocator = self.allocator;
        const w = self.weights;

        // Embed tokens with positions
        var x = try self.embedTokens(tokens, 0);

        // Process through decoder blocks with cached cross-attention
        for (w.blocks, 0..) |*block, i| {
            const new_x = try self.forwardBlockWithCache(allocator, &x, &cached_cross_kvs[i], block);
            x.deinit();
            x = new_x;
        }

        // Final LayerNorm
        var x_norm = try ops.layerNorm(allocator, &x, &w.ln_gamma, &w.ln_beta, 1e-5);
        x.deinit();

        // Project to vocabulary using tied weights
        var token_emb_t = try ops.transpose(allocator, &w.token_embedding);
        defer token_emb_t.deinit();

        const logits = try ops.matmul(allocator, &x_norm, &token_emb_t);
        x_norm.deinit();

        return logits;
    }

    /// Greedy decoding WITH KV cache - O(n) per token instead of O(n²)
    /// Pre-computes cross-attention K/V once and reuses for all steps
    pub fn greedyDecodeWithCache(
        self: *const Self,
        encoder_output: *const Tensor,
        prompt_tokens: []const u32,
        max_tokens: usize,
    ) ![]u32 {
        const allocator = self.allocator;
        const w = self.weights;
        const eot_token = config.WhisperTokens.EOT;
        const n_layers = w.blocks.len;

        // Pre-compute cross-attention K/V for all layers (computed ONCE)
        var cached_cross_kvs = try allocator.alloc(CachedCrossAttentionKV, n_layers);
        var cached_count: usize = 0;
        errdefer {
            for (cached_cross_kvs[0..cached_count]) |*c| c.deinit();
            allocator.free(cached_cross_kvs);
        }

        for (w.blocks, 0..) |*block, i| {
            cached_cross_kvs[i] = try attention.precomputeCrossAttentionKV(
                Tensor,
                allocator,
                encoder_output,
                &block.cross_attn,
            );
            cached_count += 1;
        }
        defer {
            for (cached_cross_kvs) |*c| c.deinit();
            allocator.free(cached_cross_kvs);
        }

        // Allocate output buffer
        var tokens = try allocator.alloc(u32, prompt_tokens.len + max_tokens);
        errdefer allocator.free(tokens);

        @memcpy(tokens[0..prompt_tokens.len], prompt_tokens);
        var num_tokens: usize = prompt_tokens.len;

        // Process prompt with cached cross-attention
        var logits = try self.forwardWithCache(prompt_tokens, cached_cross_kvs);

        const vocab_size = self.cfg.n_vocab;
        const last_logits = logits.data[(prompt_tokens.len - 1) * vocab_size ..][0..vocab_size];
        var next_token = argmax(last_logits);
        logits.deinit();

        // Generate tokens
        while (num_tokens < prompt_tokens.len + max_tokens) {
            if (next_token == eot_token) {
                break;
            }

            tokens[num_tokens] = next_token;
            num_tokens += 1;

            // Forward pass with cached cross-attention K/V
            // Note: Still O(n) for self-attention, but cross-attention is O(1) now
            var step_logits = try self.forwardWithCache(tokens[0..num_tokens], cached_cross_kvs);
            defer step_logits.deinit();

            const step_last = step_logits.data[(num_tokens - 1) * vocab_size ..][0..vocab_size];
            next_token = argmax(step_last);
        }

        // Resize to actual length
        if (num_tokens < tokens.len) {
            const result = try allocator.realloc(tokens, num_tokens);
            return result;
        }
        return tokens;
    }

    /// Forward single token through decoder block with both self-attn and cross-attn cache
    fn forwardBlockSingleToken(
        self: *const Self,
        allocator: std.mem.Allocator,
        x: *const Tensor, // [1, hidden_dim]
        self_kv_cache: *SelfAttentionKVCache,
        cached_cross_kv: *const CachedCrossAttentionKV,
        block: *const DecoderBlockWeights,
    ) !Tensor {
        const attn_config = AttentionConfig{
            .num_heads = self.cfg.n_text_head,
            .hidden_dim = self.cfg.n_text_state,
            .head_dim = self.cfg.headDim(),
        };

        const cross_config = CrossAttentionConfig{
            .num_heads = self.cfg.n_text_head,
            .decoder_hidden = self.cfg.n_text_state,
            .encoder_hidden = self.cfg.n_text_state,
            .head_dim = self.cfg.headDim(),
        };

        // 1. Pre-norm causal self-attention WITH KV CACHE
        var self_attn_ln = try ops.layerNorm(
            allocator,
            x,
            &block.self_attn_ln_gamma,
            &block.self_attn_ln_beta,
            1e-5,
        );
        defer self_attn_ln.deinit();

        var self_attn_out = try attention.multiHeadCausalAttentionWithCache(
            Tensor,
            allocator,
            &self_attn_ln,
            self_kv_cache,
            &block.self_attn,
            attn_config,
        );
        defer self_attn_out.deinit();

        // Residual
        var x1 = try ops.add(allocator, x, &self_attn_out);
        defer x1.deinit();

        // 2. Pre-norm cross-attention WITH CACHED K/V
        var cross_attn_ln = try ops.layerNorm(
            allocator,
            &x1,
            &block.cross_attn_ln_gamma,
            &block.cross_attn_ln_beta,
            1e-5,
        );
        defer cross_attn_ln.deinit();

        var cross_attn_out = try attention.multiHeadCrossAttentionWithCache(
            Tensor,
            allocator,
            &cross_attn_ln,
            cached_cross_kv,
            &block.cross_attn,
            cross_config,
        );
        defer cross_attn_out.deinit();

        // Residual
        var x2 = try ops.add(allocator, &x1, &cross_attn_out);
        defer x2.deinit();

        // 3. Pre-norm FFN
        var ffn_ln = try ops.layerNorm(allocator, &x2, &block.ffn_ln_gamma, &block.ffn_ln_beta, 1e-5);
        defer ffn_ln.deinit();

        var fc1 = try ops.matmul(allocator, &ffn_ln, &block.ffn_fc1_weight);
        defer fc1.deinit();
        try ops.addBiasInPlace(&fc1, &block.ffn_fc1_bias);
        ops.geluExact(&fc1);

        var fc2 = try ops.matmul(allocator, &fc1, &block.ffn_fc2_weight);
        try ops.addBiasInPlace(&fc2, &block.ffn_fc2_bias);

        const result = try ops.add(allocator, &x2, &fc2);
        fc2.deinit();

        return result;
    }

    /// Fully optimized greedy decode with BOTH self-attention AND cross-attention KV cache
    /// This achieves O(1) per token by caching all K/V
    pub fn greedyDecodeFullCache(
        self: *const Self,
        encoder_output: *const Tensor,
        prompt_tokens: []const u32,
        max_tokens: usize,
    ) ![]u32 {
        const allocator = self.allocator;
        const w = self.weights;
        const eot_token = config.WhisperTokens.EOT;
        const n_layers = w.blocks.len;
        const hidden_dim = self.cfg.n_text_state;
        const max_text_ctx = self.cfg.n_text_ctx;

        // Pre-compute cross-attention K/V for all layers (computed ONCE)
        var cached_cross_kvs = try allocator.alloc(CachedCrossAttentionKV, n_layers);
        var cross_count: usize = 0;
        errdefer {
            for (cached_cross_kvs[0..cross_count]) |*c| c.deinit();
            allocator.free(cached_cross_kvs);
        }

        for (w.blocks, 0..) |*block, i| {
            cached_cross_kvs[i] = try attention.precomputeCrossAttentionKV(
                Tensor,
                allocator,
                encoder_output,
                &block.cross_attn,
            );
            cross_count += 1;
        }
        defer {
            for (cached_cross_kvs) |*c| c.deinit();
            allocator.free(cached_cross_kvs);
        }

        // Initialize self-attention KV caches for all layers
        var self_kv_caches = try allocator.alloc(SelfAttentionKVCache, n_layers);
        var self_count: usize = 0;
        errdefer {
            for (self_kv_caches[0..self_count]) |*c| c.deinit();
            allocator.free(self_kv_caches);
        }

        for (0..n_layers) |i| {
            self_kv_caches[i] = try SelfAttentionKVCache.init(allocator, max_text_ctx, hidden_dim);
            self_count += 1;
        }
        defer {
            for (self_kv_caches) |*c| c.deinit();
            allocator.free(self_kv_caches);
        }

        // Allocate output buffer
        var tokens = try allocator.alloc(u32, prompt_tokens.len + max_tokens);
        errdefer allocator.free(tokens);

        @memcpy(tokens[0..prompt_tokens.len], prompt_tokens);
        var num_tokens: usize = prompt_tokens.len;

        // Process prompt tokens one-by-one to fill the self-attention KV cache
        // This ensures the cache state matches what we use for generation
        var final_hidden: Tensor = undefined;
        var final_hidden_valid = false;
        defer if (final_hidden_valid) final_hidden.deinit();

        for (prompt_tokens, 0..) |tok, pos| {
            const single_tok = [_]u32{tok};
            var emb = try self.embedTokens(&single_tok, pos);

            // Pass through each layer with proper KV cache update
            var x = emb;
            var x_owned = false;

            for (w.blocks, 0..) |*block, layer_i| {
                const new_x = try self.forwardBlockSingleToken(
                    allocator,
                    &x,
                    &self_kv_caches[layer_i],
                    &cached_cross_kvs[layer_i],
                    block,
                );
                if (x_owned) x.deinit() else emb.deinit();
                x = new_x;
                x_owned = true;
            }

            // Keep the final hidden state from the last prompt token
            if (final_hidden_valid) final_hidden.deinit();
            final_hidden = x;
            final_hidden_valid = true;
        }

        // Compute logits for first generated token from final prompt hidden state
        const vocab_size = self.cfg.n_vocab;
        var next_token: u32 = undefined;
        {
            var prompt_norm = try ops.layerNorm(allocator, &final_hidden, &w.ln_gamma, &w.ln_beta, 1e-5);
            defer prompt_norm.deinit();

            var prompt_emb_t = try ops.transpose(allocator, &w.token_embedding);
            defer prompt_emb_t.deinit();

            var prompt_logits = try ops.matmul(allocator, &prompt_norm, &prompt_emb_t);
            defer prompt_logits.deinit();

            next_token = argmax(prompt_logits.data[0..vocab_size]);
        }

        // Now generate new tokens using full KV cache
        while (num_tokens < prompt_tokens.len + max_tokens) {
            if (next_token == eot_token) {
                break;
            }

            tokens[num_tokens] = next_token;
            num_tokens += 1;

            // Embed single new token
            const single_tok = [_]u32{next_token};
            var emb = try self.embedTokens(&single_tok, num_tokens - 1);

            // Process through all layers with full KV cache
            var x = emb;
            var x_owned = false;

            for (w.blocks, 0..) |*block, i| {
                const new_x = try self.forwardBlockSingleToken(
                    allocator,
                    &x,
                    &self_kv_caches[i],
                    &cached_cross_kvs[i],
                    block,
                );
                if (x_owned) x.deinit() else emb.deinit();
                x = new_x;
                x_owned = true;
            }

            // Final LayerNorm
            var x_norm = try ops.layerNorm(allocator, &x, &w.ln_gamma, &w.ln_beta, 1e-5);
            x.deinit();

            // Project to vocabulary
            var token_emb_t = try ops.transpose(allocator, &w.token_embedding);
            defer token_emb_t.deinit();

            var step_logits = try ops.matmul(allocator, &x_norm, &token_emb_t);
            x_norm.deinit();
            defer step_logits.deinit();

            const step_last = step_logits.data[0..vocab_size];
            next_token = argmax(step_last);
        }

        // Resize to actual length
        if (num_tokens < tokens.len) {
            const result = try allocator.realloc(tokens, num_tokens);
            return result;
        }
        return tokens;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "decoder config" {
    const cfg = config.WhisperConfig.forVariant(.tiny);
    try std.testing.expectEqual(@as(usize, 384), cfg.decoder.n_text_state);
    try std.testing.expectEqual(@as(usize, 6), cfg.decoder.n_text_head);
    try std.testing.expectEqual(@as(usize, 64), cfg.decoder.headDim());
    try std.testing.expectEqual(@as(usize, 51865), cfg.decoder.n_vocab);
}

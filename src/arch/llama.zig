// src/arch/llama.zig
//
// Generic Llama-family decoder block.
// Used by: Llama-3 / Llama-2, Mistral, Qwen2, InkubaLM, N-ATLaS,
//          AfroLlama, and any other model that ships as a stack of:
//
//   RMSNorm → GQA (with full RoPE) → residual
//   RMSNorm → SwiGLU FFN          → residual
//   final RMSNorm → LM head (optionally tied to token embeddings)
//
// Single-token forward pass — autoregressive decode style. Multi-token
// prefill is the same loop run sequentially over the prompt; we'll add
// a fused prefill path later if it's a hot spot.
//
// Weights are held as ProjectionWeight (f32 or Q8K) so this code path
// works with both the f32 .tl format and quantised storage. GPU dispatch
// happens transparently through arch/common.projectMul.

const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const ops = @import("ops.zig");
const common = @import("common.zig");

pub const ProjectionWeight = common.ProjectionWeight;
const projectMul = common.projectMul;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Architecture-level config for a Llama-family model.
/// Mirrors the typed view in src/format/config_json.zig.LlamaConfig but is
/// re-declared here so arch/ has no dependency on format/ (format/ is
/// "load HF config into this struct"; arch/ is "execute this struct").
pub const LlamaConfig = struct {
    hidden_size: usize,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    vocab_size: usize,
    max_seq_len: usize,

    rms_norm_eps: f32 = 1e-5,
    rope_theta: f32 = 10000.0,

    /// Whether lm_head shares storage with embed_tokens (HF default true for
    /// Llama-3 small/base, false for Llama-3 8B+).
    tie_word_embeddings: bool = true,

    pub fn qDim(self: LlamaConfig) usize {
        return self.num_heads * self.head_dim;
    }
    pub fn kvDim(self: LlamaConfig) usize {
        return self.num_kv_heads * self.head_dim;
    }
    pub fn headsPerKv(self: LlamaConfig) usize {
        return self.num_heads / self.num_kv_heads;
    }
};

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

pub const LlamaLayerWeights = struct {
    input_layernorm: Tensor, // [hidden_size]
    post_attn_layernorm: Tensor, // [hidden_size]

    // Attention projections — row-major [out_dim, hidden_size].
    q_proj: ProjectionWeight, // out = q_dim
    k_proj: ProjectionWeight, // out = kv_dim
    v_proj: ProjectionWeight, // out = kv_dim
    o_proj: ProjectionWeight, // out = hidden_size, in = q_dim

    // SwiGLU FFN: down( silu(gate(x)) * up(x) )
    gate_proj: ProjectionWeight, // [intermediate, hidden]
    up_proj: ProjectionWeight, // [intermediate, hidden]
    down_proj: ProjectionWeight, // [hidden, intermediate]
};

pub const LlamaWeights = struct {
    embed_tokens: ProjectionWeight, // [vocab_size, hidden_size]
    final_norm: Tensor, // [hidden_size]
    /// LM head. If `LlamaConfig.tie_word_embeddings` is true, callers should
    /// pass the same ProjectionWeight as embed_tokens (no copy).
    lm_head: ProjectionWeight,

    layers: []LlamaLayerWeights,

    allocator: std.mem.Allocator,

    pub fn deinit(self: *LlamaWeights) void {
        // Tensor lifetimes are owned by the caller (model loader); we only
        // free the layers slice since we allocated it.
        self.allocator.free(self.layers);
    }
};

// ---------------------------------------------------------------------------
// KV cache
// ---------------------------------------------------------------------------

pub const LayerKVCache = struct {
    keys: []f32, // [max_len, kv_dim]
    values: []f32, // [max_len, kv_dim]
    length: usize,
    max_length: usize,
    kv_dim: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, max_length: usize, kv_dim: usize) !LayerKVCache {
        return .{
            .keys = try allocator.alloc(f32, max_length * kv_dim),
            .values = try allocator.alloc(f32, max_length * kv_dim),
            .length = 0,
            .max_length = max_length,
            .kv_dim = kv_dim,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *LayerKVCache) void {
        self.allocator.free(self.keys);
        self.allocator.free(self.values);
    }

    pub fn append(self: *LayerKVCache, k: []const f32, v: []const f32) void {
        std.debug.assert(self.length < self.max_length);
        const off = self.length * self.kv_dim;
        @memcpy(self.keys[off..][0..self.kv_dim], k);
        @memcpy(self.values[off..][0..self.kv_dim], v);
        self.length += 1;
    }

    pub fn reset(self: *LayerKVCache) void {
        self.length = 0;
    }
};

pub const LlamaCache = struct {
    layers: []LayerKVCache,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, cfg: LlamaConfig, max_seq_len: usize) !LlamaCache {
        const layers = try allocator.alloc(LayerKVCache, cfg.num_layers);
        var ok: usize = 0;
        errdefer {
            for (layers[0..ok]) |*c| c.deinit();
            allocator.free(layers);
        }
        const kv_dim = cfg.kvDim();
        for (0..cfg.num_layers) |i| {
            layers[i] = try LayerKVCache.init(allocator, max_seq_len, kv_dim);
            ok += 1;
        }
        return .{ .layers = layers, .allocator = allocator };
    }

    pub fn reset(self: *LlamaCache) void {
        for (self.layers) |*c| c.reset();
    }

    pub fn deinit(self: *LlamaCache) void {
        for (self.layers) |*c| c.deinit();
        self.allocator.free(self.layers);
    }
};

// ---------------------------------------------------------------------------
// Scratch
// ---------------------------------------------------------------------------

pub const Scratch = struct {
    hidden: []f32, // [hidden]
    norm_out: []f32, // [hidden]
    residual: []f32, // [hidden]
    q: []f32, // [q_dim]
    k: []f32, // [kv_dim]
    v: []f32, // [kv_dim]
    attn_out: []f32, // [q_dim]
    attn_scores: []f32, // [num_heads, max_seq_len]
    ffn_gate: []f32, // [intermediate]
    ffn_up: []f32, // [intermediate]
    logits: []f32, // [vocab_size]
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, cfg: LlamaConfig, max_seq_len: usize) !Scratch {
        return .{
            .hidden = try allocator.alloc(f32, cfg.hidden_size),
            .norm_out = try allocator.alloc(f32, cfg.hidden_size),
            .residual = try allocator.alloc(f32, cfg.hidden_size),
            .q = try allocator.alloc(f32, cfg.qDim()),
            .k = try allocator.alloc(f32, cfg.kvDim()),
            .v = try allocator.alloc(f32, cfg.kvDim()),
            .attn_out = try allocator.alloc(f32, cfg.qDim()),
            .attn_scores = try allocator.alloc(f32, cfg.num_heads * max_seq_len),
            .ffn_gate = try allocator.alloc(f32, cfg.intermediate_size),
            .ffn_up = try allocator.alloc(f32, cfg.intermediate_size),
            .logits = try allocator.alloc(f32, cfg.vocab_size),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Scratch) void {
        const a = self.allocator;
        a.free(self.hidden);
        a.free(self.norm_out);
        a.free(self.residual);
        a.free(self.q);
        a.free(self.k);
        a.free(self.v);
        a.free(self.attn_out);
        a.free(self.attn_scores);
        a.free(self.ffn_gate);
        a.free(self.ffn_up);
        a.free(self.logits);
    }
};

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

/// Embed a single token id into `out` (length hidden_size).
/// embed_tokens layout: [vocab_size, hidden_size] row-major.
fn embedToken(out: []f32, w: ProjectionWeight, token_id: u32, hidden: usize) void {
    const row: usize = @intCast(token_id);
    switch (w) {
        .f32 => |t| {
            const src = t.data[row * hidden ..][0..hidden];
            @memcpy(out, src);
        },
        .q8k => |q| {
            // Dequantize one row. Q8K is per-block (block_size=32) with f32 scales.
            const block_size: usize = 32;
            std.debug.assert(hidden % block_size == 0);
            const blocks_per_row = hidden / block_size;
            const data_off = row * hidden;
            const scale_off = row * blocks_per_row;
            for (0..blocks_per_row) |b| {
                const s = q.scales[scale_off + b];
                const src = q.data[data_off + b * block_size ..][0..block_size];
                const dst = out[b * block_size ..][0..block_size];
                for (src, dst) |i8v, *fv| fv.* = @as(f32, @floatFromInt(i8v)) * s;
            }
        },
    }
}

/// One causal GQA attention step. The new (k, v) for this position are
/// already appended to `cache`; `q` is for the new position only.
/// Writes attention output to `out` (length q_dim).
fn attentionStep(
    out: []f32,
    q: []const f32,
    cache: *const LayerKVCache,
    scores_scratch: []f32,
    cfg: LlamaConfig,
) void {
    const num_heads = cfg.num_heads;
    const num_kv = cfg.num_kv_heads;
    const head_dim = cfg.head_dim;
    const heads_per_kv = num_heads / num_kv;
    const seq_len = cache.length;
    const inv_sqrt_d: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    @memset(out, 0.0);

    for (0..num_heads) |h| {
        const kv_h = h / heads_per_kv;
        const q_off = h * head_dim;
        const q_head = q[q_off..][0..head_dim];

        // scores[t] = q · k_t  (for t = 0..seq_len-1)
        const scores = scores_scratch[h * seq_len ..][0..seq_len];
        for (0..seq_len) |t| {
            const k_off = t * cfg.kvDim() + kv_h * head_dim;
            const k_t = cache.keys[k_off..][0..head_dim];
            var s: f32 = 0.0;
            for (0..head_dim) |d| s += q_head[d] * k_t[d];
            scores[t] = s * inv_sqrt_d;
        }

        // softmax (numerically stable)
        var max_s: f32 = scores[0];
        for (scores[1..]) |s| {
            if (s > max_s) max_s = s;
        }
        var denom: f32 = 0.0;
        for (scores) |*s| {
            s.* = @exp(s.* - max_s);
            denom += s.*;
        }
        const inv_denom: f32 = 1.0 / denom;
        for (scores) |*s| s.* *= inv_denom;

        // out_head = Σ_t scores[t] * v_t
        const out_head = out[q_off..][0..head_dim];
        for (0..seq_len) |t| {
            const v_off = t * cfg.kvDim() + kv_h * head_dim;
            const v_t = cache.values[v_off..][0..head_dim];
            const w = scores[t];
            for (0..head_dim) |d| out_head[d] += w * v_t[d];
        }
    }
}

/// Forward one token. Reads token_id at `position`, returns logits via scratch.logits.
/// Caller is responsible for sampling.
pub fn forwardToken(
    cfg: LlamaConfig,
    weights: *const LlamaWeights,
    cache: *LlamaCache,
    scratch: *Scratch,
    token_id: u32,
    position: usize,
) void {
    std.debug.assert(weights.layers.len == cfg.num_layers);
    std.debug.assert(cache.layers.len == cfg.num_layers);

    const hidden = cfg.hidden_size;
    const q_dim = cfg.qDim();
    const kv_dim = cfg.kvDim();

    embedToken(scratch.hidden, weights.embed_tokens, token_id, hidden);

    for (weights.layers, 0..) |*lw, layer_idx| {
        @memcpy(scratch.residual, scratch.hidden);

        // ---- Attention block ----
        @memcpy(scratch.norm_out, scratch.hidden);
        ops.rmsNorm1DInPlace(scratch.norm_out, lw.input_layernorm.data, hidden, cfg.rms_norm_eps);

        projectMul(scratch.q, lw.q_proj, scratch.norm_out, q_dim, hidden);
        projectMul(scratch.k, lw.k_proj, scratch.norm_out, kv_dim, hidden);
        projectMul(scratch.v, lw.v_proj, scratch.norm_out, kv_dim, hidden);

        ops.ropeInPlace(
            scratch.q,
            scratch.k,
            cfg.num_heads,
            cfg.num_kv_heads,
            cfg.head_dim,
            cfg.head_dim, // full RoPE
            position,
            cfg.rope_theta,
        );

        var layer_cache = &cache.layers[layer_idx];
        layer_cache.append(scratch.k, scratch.v);

        attentionStep(scratch.attn_out, scratch.q, layer_cache, scratch.attn_scores, cfg);

        // residual += o_proj(attn_out)
        // Use scratch.norm_out as a temp buffer for the projection result.
        projectMul(scratch.norm_out, lw.o_proj, scratch.attn_out, hidden, q_dim);
        for (scratch.residual, scratch.norm_out) |*r, p| r.* += p;

        // ---- FFN block ----
        @memcpy(scratch.hidden, scratch.residual);
        @memcpy(scratch.norm_out, scratch.hidden);
        ops.rmsNorm1DInPlace(scratch.norm_out, lw.post_attn_layernorm.data, hidden, cfg.rms_norm_eps);

        projectMul(scratch.ffn_gate, lw.gate_proj, scratch.norm_out, cfg.intermediate_size, hidden);
        projectMul(scratch.ffn_up, lw.up_proj, scratch.norm_out, cfg.intermediate_size, hidden);

        ops.siluSliceInPlace(scratch.ffn_gate);
        for (scratch.ffn_gate, scratch.ffn_up) |*g, u| g.* *= u;

        // residual += down_proj(gate*up). Reuse scratch.norm_out for projection output.
        projectMul(scratch.norm_out, lw.down_proj, scratch.ffn_gate, hidden, cfg.intermediate_size);
        for (scratch.hidden, scratch.residual, scratch.norm_out) |*h_out, r, p| h_out.* = r + p;
    }

    // Final norm + LM head
    ops.rmsNorm1DInPlace(scratch.hidden, weights.final_norm.data, hidden, cfg.rms_norm_eps);
    projectMul(scratch.logits, weights.lm_head, scratch.hidden, cfg.vocab_size, hidden);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

// Build a tiny random Llama with f32 weights, drive one token through the
// forward pass, and assert the result is finite and the shapes match.
// Validates wiring (RoPE, GQA, SwiGLU, residuals) without real weights.
test "forwardToken: tiny 2-layer Llama, no NaNs, expected logit shape" {
    const allocator = testing.allocator;

    const cfg = LlamaConfig{
        .hidden_size = 16,
        .num_layers = 2,
        .num_heads = 4,
        .num_kv_heads = 2, // GQA: 2 heads per kv group
        .head_dim = 4,
        .intermediate_size = 32,
        .vocab_size = 24,
        .max_seq_len = 8,
        .rms_norm_eps = 1e-5,
        .rope_theta = 10000.0,
        .tie_word_embeddings = true,
    };

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    // Helper: allocate an f32 ProjectionWeight with random values in [-0.1, 0.1].
    const Helper = struct {
        fn randTensor(a: std.mem.Allocator, r: std.Random, shape: []const usize) !Tensor {
            const t = try Tensor.init(a, shape);
            for (t.data) |*v| v.* = (r.float(f32) - 0.5) * 0.2;
            return t;
        }
        fn randProj(a: std.mem.Allocator, r: std.Random, rows: usize, cols: usize) !ProjectionWeight {
            const t = try randTensor(a, r, &[_]usize{ rows, cols });
            return ProjectionWeight{ .f32 = t };
        }
    };

    // Allocate weights. We'll keep a flat list of every Tensor we
    // allocate so we can free them at the end.
    var owned_tensors: std.ArrayList(Tensor) = .{};
    defer {
        for (owned_tensors.items) |*t| t.deinit();
        owned_tensors.deinit(allocator);
    }

    const trackTensor = struct {
        fn call(list: *std.ArrayList(Tensor), a: std.mem.Allocator, t: Tensor) !void {
            try list.append(a, t);
        }
    }.call;

    const embed_t = try Helper.randTensor(allocator, rng, &[_]usize{ cfg.vocab_size, cfg.hidden_size });
    try trackTensor(&owned_tensors, allocator, embed_t);
    const embed_w = ProjectionWeight{ .f32 = embed_t };

    const final_norm = try Helper.randTensor(allocator, rng, &[_]usize{cfg.hidden_size});
    try trackTensor(&owned_tensors, allocator, final_norm);

    var layers = try allocator.alloc(LlamaLayerWeights, cfg.num_layers);
    defer allocator.free(layers);

    const q_dim = cfg.qDim();
    const kv_dim = cfg.kvDim();

    for (0..cfg.num_layers) |i| {
        const in_ln = try Helper.randTensor(allocator, rng, &[_]usize{cfg.hidden_size});
        try trackTensor(&owned_tensors, allocator, in_ln);
        const post_ln = try Helper.randTensor(allocator, rng, &[_]usize{cfg.hidden_size});
        try trackTensor(&owned_tensors, allocator, post_ln);

        const q_p = try Helper.randProj(allocator, rng, q_dim, cfg.hidden_size);
        const k_p = try Helper.randProj(allocator, rng, kv_dim, cfg.hidden_size);
        const v_p = try Helper.randProj(allocator, rng, kv_dim, cfg.hidden_size);
        const o_p = try Helper.randProj(allocator, rng, cfg.hidden_size, q_dim);
        const g_p = try Helper.randProj(allocator, rng, cfg.intermediate_size, cfg.hidden_size);
        const u_p = try Helper.randProj(allocator, rng, cfg.intermediate_size, cfg.hidden_size);
        const d_p = try Helper.randProj(allocator, rng, cfg.hidden_size, cfg.intermediate_size);
        try trackTensor(&owned_tensors, allocator, q_p.f32);
        try trackTensor(&owned_tensors, allocator, k_p.f32);
        try trackTensor(&owned_tensors, allocator, v_p.f32);
        try trackTensor(&owned_tensors, allocator, o_p.f32);
        try trackTensor(&owned_tensors, allocator, g_p.f32);
        try trackTensor(&owned_tensors, allocator, u_p.f32);
        try trackTensor(&owned_tensors, allocator, d_p.f32);

        layers[i] = .{
            .input_layernorm = in_ln,
            .post_attn_layernorm = post_ln,
            .q_proj = q_p,
            .k_proj = k_p,
            .v_proj = v_p,
            .o_proj = o_p,
            .gate_proj = g_p,
            .up_proj = u_p,
            .down_proj = d_p,
        };
    }

    const weights = LlamaWeights{
        .embed_tokens = embed_w,
        .final_norm = final_norm,
        .lm_head = embed_w, // tied
        .layers = layers,
        .allocator = allocator,
    };
    // Don't call weights.deinit() — layers is freed via the explicit defer above.

    var cache = try LlamaCache.init(allocator, cfg, cfg.max_seq_len);
    defer cache.deinit();

    var scratch = try Scratch.init(allocator, cfg, cfg.max_seq_len);
    defer scratch.deinit();

    // Drive 3 tokens.
    forwardToken(cfg, &weights, &cache, &scratch, 0, 0);
    forwardToken(cfg, &weights, &cache, &scratch, 1, 1);
    forwardToken(cfg, &weights, &cache, &scratch, 2, 2);

    try testing.expectEqual(@as(usize, 3), cache.layers[0].length);
    try testing.expectEqual(cfg.vocab_size, scratch.logits.len);

    for (scratch.logits) |v| {
        try testing.expect(std.math.isFinite(v));
    }
}

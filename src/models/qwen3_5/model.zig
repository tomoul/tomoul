// src/models/qwen3_5/model.zig
// Qwen3.5-0.8B Model — Hybrid Gated DeltaNet + Attention
//
// 24-layer decoder with [L L L A] × 6 pattern:
//   18× Gated DeltaNet (linear-time recurrence, O(1) per token)
//   6× Full GQA Attention (8Q / 2KV, head_dim=256, partial RoPE)
//
// Usage:
//   const model = try Qwen3_5.load(allocator, "qwen3_5_0.8b.tl");
//   defer model.deinit();
//   const output = model.generate(tokenizer, "Hello", 100, .{});

const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
pub const ops = @import("ops.zig");
const loader_mod = @import("loader.zig");
const ModelLoader = loader_mod.ModelLoader;
const QuantFormat = loader_mod.QuantFormat;
const quantization = @import("quantization.zig");
const QuantizedTensorQ8K = quantization.QuantizedTensorQ8K;
const cache_mod = @import("cache.zig");
const KVCache = cache_mod.KVCache;

pub const config = @import("config.zig");
pub const deltanet = @import("deltanet.zig");
pub const tokenizer_mod = @import("tokenizer.zig");

const Qwen3_5Config = config.Qwen3_5Config;
const LayerType = config.LayerType;
const SpecialTokens = config.SpecialTokens;
const DeltaNetWeights = deltanet.DeltaNetWeights;
const DeltaNetState = deltanet.DeltaNetState;
const DeltaNetScratch = deltanet.DeltaNetScratch;

// ============================================================================
// Projection Weight (F32 or Q8K)
// ============================================================================

/// A weight matrix that can be either F32 or Q8K quantized.
/// Used for all large projection matrices in the model.
pub const ProjectionWeight = union(enum) {
    f32: Tensor,
    q8k: QuantizedTensorQ8K,
};

/// GPU dispatch function type: (result, weight_ptr, input, rows, cols) → handled
pub const GpuMatvecFn = *const fn ([]f32, usize, []const f32, usize, usize) bool;

/// GPU fused FFN dispatch: (hidden, residual, layer_idx) → handled
pub const GpuFfnFn = *const fn ([]f32, []const f32, usize) bool;

/// GPU fused DeltaNet input proj: (hidden, layer_idx, qkv_out, z_out, b_out, a_out) → handled
pub const GpuDnInputFn = *const fn ([]const f32, usize, []f32, []f32, []f32, []f32) bool;

/// GPU fused attention input proj: (hidden, layer_idx, q_out, k_out, v_out) → handled
pub const GpuAttnInputFn = *const fn ([]const f32, usize, []f32, []f32, []f32) bool;

/// GPU fused o_proj + post-attn residual + FFN: (attn_out, pre_attn_residual, layer_idx, hidden_out) → handled
/// Replaces 2 round-trips (separate o_proj SGEMV + separate FFN dispatch) with 1 submit.
pub const GpuOprojFfnFn = *const fn ([]const f32, []const f32, usize, []f32) bool;

// ---- Phase 2B: GPU-resident hidden state ----
/// Upload initial hidden state once per token.
pub const GpuTokenBeginFn = *const fn ([]const f32) bool;
/// Read back final hidden state once per token.
pub const GpuTokenEndFn = *const fn ([]f32) bool;
/// DeltaNet input projections with GPU-resident hidden (snapshot residual + RMSNorm + projects).
pub const GpuLayerStartDnFn = *const fn (usize, []f32, []f32, []f32, []f32) bool;
/// Full-attention input projections with GPU-resident hidden.
pub const GpuLayerStartFaFn = *const fn (usize, []f32, []f32, []f32) bool;
/// o_proj + post-attn residual + FFN with GPU-resident hidden (no readback).
pub const GpuOprojFfnResidentFn = *const fn ([]const f32, usize) bool;

// ---- Phase 4: fused final RMSNorm + LM head ----
/// Final RMSNorm + LM head in one GPU submit (replaces CPU final_norm + gpuMatvec).
pub const GpuLmHeadFn = *const fn ([]f32) bool;

/// Global GPU matvec dispatch (null = CPU-only mode).
var gpu_matvec_fn: ?GpuMatvecFn = null;

/// Global GPU fused FFN dispatch (null = CPU-only mode).
var gpu_ffn_fn: ?GpuFfnFn = null;

/// Global GPU fused DeltaNet input projections dispatch.
var gpu_dn_input_fn: ?GpuDnInputFn = null;

/// Global GPU fused attention input projections dispatch.
var gpu_attn_input_fn: ?GpuAttnInputFn = null;

/// Global GPU fused o_proj + post-attn residual + FFN dispatch (null = use separate o_proj/FFN paths).
var gpu_oproj_ffn_fn: ?GpuOprojFfnFn = null;

// ---- Phase 2B globals ----
var gpu_token_begin_fn: ?GpuTokenBeginFn = null;
var gpu_token_end_fn: ?GpuTokenEndFn = null;
var gpu_layer_start_dn_fn: ?GpuLayerStartDnFn = null;
var gpu_layer_start_fa_fn: ?GpuLayerStartFaFn = null;
var gpu_oproj_ffn_resident_fn: ?GpuOprojFfnResidentFn = null;
var gpu_lm_head_fn: ?GpuLmHeadFn = null;

/// Enable GPU-accelerated matrix-vector multiply for all projectMul calls.
pub fn setGpuMatvec(f: ?GpuMatvecFn) void {
    gpu_matvec_fn = f;
}

/// Enable GPU-fused FFN (RMSNorm + gate/up/SiLU/down + residual in one dispatch).
pub fn setGpuFfn(f: ?GpuFfnFn) void {
    gpu_ffn_fn = f;
}

/// Enable GPU-fused DeltaNet input projections (RMSNorm + qkv/z/b/a in one dispatch).
pub fn setGpuDnInput(f: ?GpuDnInputFn) void {
    gpu_dn_input_fn = f;
}

/// Enable GPU-fused attention input projections (RMSNorm + q/k/v in one dispatch).
pub fn setGpuAttnInput(f: ?GpuAttnInputFn) void {
    gpu_attn_input_fn = f;
}

/// Enable GPU-fused o_proj + post-attn residual + FFN (1 submit replaces 2 round-trips).
pub fn setGpuOprojFfn(f: ?GpuOprojFfnFn) void {
    gpu_oproj_ffn_fn = f;
}

// ---- Phase 2B setters ----
pub fn setGpuTokenBegin(f: ?GpuTokenBeginFn) void { gpu_token_begin_fn = f; }
pub fn setGpuTokenEnd(f: ?GpuTokenEndFn) void { gpu_token_end_fn = f; }
pub fn setGpuLayerStartDn(f: ?GpuLayerStartDnFn) void { gpu_layer_start_dn_fn = f; }
pub fn setGpuLayerStartFa(f: ?GpuLayerStartFaFn) void { gpu_layer_start_fa_fn = f; }
pub fn setGpuOprojFfnResident(f: ?GpuOprojFfnResidentFn) void { gpu_oproj_ffn_resident_fn = f; }
pub fn setGpuLmHead(f: ?GpuLmHeadFn) void { gpu_lm_head_fn = f; }

/// Matrix-vector multiply dispatching on weight format (and GPU if available).
pub fn projectMul(result: []f32, w: ProjectionWeight, vec: []const f32, rows: usize, cols: usize) void {
    if (gpu_matvec_fn) |dispatch_fn| {
        const ptr: usize = switch (w) {
            .f32 => |t| @intFromPtr(t.data.ptr),
            .q8k => |q| @intFromPtr(q.data.ptr),
        };
        if (dispatch_fn(result, ptr, vec, rows, cols)) return;
    }
    switch (w) {
        .f32 => |t| ops.matvecMul(result, t.data, vec, rows, cols),
        .q8k => |q| ops.matvecMulQ8K(result, q.data, q.scales, vec, rows, cols),
    }
}

// ============================================================================
// Weight Structures
// ============================================================================

pub const FullAttentionWeights = struct {
    q_proj: ProjectionWeight, // [q_proj_dim, hidden_size] = [4096, 1024] (includes gate when attn_output_gate)
    k_proj: ProjectionWeight, // [kv_dim, hidden_size] = [512, 1024]
    v_proj: ProjectionWeight, // [kv_dim, hidden_size] = [512, 1024]
    o_proj: ProjectionWeight, // [hidden_size, q_dim] = [1024, 2048]
    q_norm: Tensor, // [head_dim] = [256]
    k_norm: Tensor, // [head_dim] = [256]
};

pub const LayerWeights = struct {
    layer_type: LayerType,
    input_layernorm: Tensor, // [hidden_size]
    post_attn_layernorm: Tensor, // [hidden_size]

    // Attention (mutually exclusive based on layer_type)
    full_attn: ?FullAttentionWeights,
    deltanet_w: ?DeltaNetWeights,

    // FFN (SwiGLU, all layers)
    gate_proj: ProjectionWeight, // [intermediate, hidden_size] = [3584, 1024]
    up_proj: ProjectionWeight, // [intermediate, hidden_size] = [3584, 1024]
    down_proj: ProjectionWeight, // [hidden_size, intermediate] = [1024, 3584]
};

pub const Qwen3_5Weights = struct {
    embed_tokens: ProjectionWeight, // [vocab_size, hidden_size] = [248320, 1024]
    final_norm: Tensor, // [hidden_size] = [1024]
    layers: []LayerWeights,

    allocator: std.mem.Allocator,

    pub fn deinit(self: *Qwen3_5Weights) void {
        self.allocator.free(self.layers);
    }
};

// ============================================================================
// Cache
// ============================================================================

pub const Qwen3_5Cache = struct {
    /// DeltaNet recurrent states (18 layers)
    deltanet_states: []DeltaNetState,
    /// KV caches for full attention layers (6 layers)
    kv_caches: []AttentionKVCache,

    allocator: std.mem.Allocator,

    pub const AttentionKVCache = struct {
        /// K cache: [max_len, kv_dim] where kv_dim = num_kv_heads * head_dim
        keys: []f32,
        /// V cache: [max_len, kv_dim]
        values: []f32,
        length: usize,
        max_length: usize,
        kv_dim: usize,
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, max_length: usize, kv_dim: usize) !AttentionKVCache {
            return .{
                .keys = try allocator.alloc(f32, max_length * kv_dim),
                .values = try allocator.alloc(f32, max_length * kv_dim),
                .length = 0,
                .max_length = max_length,
                .kv_dim = kv_dim,
                .allocator = allocator,
            };
        }

        pub fn append(self: *AttentionKVCache, k: []const f32, v: []const f32) void {
            const offset = self.length * self.kv_dim;
            @memcpy(self.keys[offset..][0..self.kv_dim], k);
            @memcpy(self.values[offset..][0..self.kv_dim], v);
            self.length += 1;
        }

        pub fn reset(self: *AttentionKVCache) void {
            self.length = 0;
        }

        pub fn deinit(self: *AttentionKVCache) void {
            self.allocator.free(self.keys);
            self.allocator.free(self.values);
        }
    };

    pub fn init(allocator: std.mem.Allocator, cfg: Qwen3_5Config, max_cache_len: usize) !Qwen3_5Cache {
        const num_dn = cfg.numDeltaNetLayers();
        const num_fa = cfg.numFullAttentionLayers();
        const kv_dim = cfg.fullAttentionKvDim();

        var dn_states = try allocator.alloc(DeltaNetState, num_dn);
        errdefer allocator.free(dn_states);
        var dn_init_count: usize = 0;
        errdefer for (dn_states[0..dn_init_count]) |*s| s.deinit();

        for (0..num_dn) |i| {
            dn_states[i] = try DeltaNetState.init(allocator, cfg);
            dn_init_count += 1;
        }

        var kv_caches = try allocator.alloc(AttentionKVCache, num_fa);
        errdefer allocator.free(kv_caches);
        var kv_init_count: usize = 0;
        errdefer for (kv_caches[0..kv_init_count]) |*c| c.deinit();

        for (0..num_fa) |i| {
            kv_caches[i] = try AttentionKVCache.init(allocator, max_cache_len, kv_dim);
            kv_init_count += 1;
        }

        return .{
            .deltanet_states = dn_states,
            .kv_caches = kv_caches,
            .allocator = allocator,
        };
    }

    pub fn reset(self: *Qwen3_5Cache) void {
        for (self.deltanet_states) |*s| s.reset();
        for (self.kv_caches) |*c| c.reset();
    }

    pub fn deinit(self: *Qwen3_5Cache) void {
        for (self.deltanet_states) |*s| s.deinit();
        for (self.kv_caches) |*c| c.deinit();
        self.allocator.free(self.deltanet_states);
        self.allocator.free(self.kv_caches);
    }
};

// ============================================================================
// Scratch Buffers (pre-allocated per-step temporaries)
// ============================================================================

pub const Scratch = struct {
    /// [hidden_size] current hidden state
    hidden: []f32,
    /// [hidden_size] residual buffer
    residual: []f32,
    /// [hidden_size] norm output
    norm_out: []f32,
    /// [intermediate_size] for FFN gate projection
    ffn_gate: []f32,
    /// [intermediate_size] for FFN up projection
    ffn_up: []f32,
    /// DeltaNet scratch
    dn_scratch: DeltaNetScratch,
    /// Full attention scratch
    attn_q: []f32, // [q_proj_dim] — first q_dim are Q, rest is gate (when attn_output_gate)
    attn_k: []f32, // [kv_dim]
    attn_v: []f32, // [kv_dim]
    attn_out: []f32, // [q_dim]
    attn_scores: []f32, // [num_heads, max_cache_len] — scratch for attention scores
    /// [vocab_size] logits output
    logits: []f32,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, cfg: Qwen3_5Config, max_cache_len: usize) !Scratch {
        return .{
            .hidden = try allocator.alloc(f32, cfg.hidden_size),
            .residual = try allocator.alloc(f32, cfg.hidden_size),
            .norm_out = try allocator.alloc(f32, cfg.hidden_size),
            .ffn_gate = try allocator.alloc(f32, cfg.intermediate_size),
            .ffn_up = try allocator.alloc(f32, cfg.intermediate_size),
            .dn_scratch = try DeltaNetScratch.init(allocator, cfg),
            .attn_q = try allocator.alloc(f32, cfg.fullAttentionQProjDim()),
            .attn_k = try allocator.alloc(f32, cfg.fullAttentionKvDim()),
            .attn_v = try allocator.alloc(f32, cfg.fullAttentionKvDim()),
            .attn_out = try allocator.alloc(f32, cfg.fullAttentionQDim()),
            .attn_scores = try allocator.alloc(f32, cfg.num_attention_heads * max_cache_len),
            .logits = try allocator.alloc(f32, cfg.vocab_size),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Scratch) void {
        self.allocator.free(self.hidden);
        self.allocator.free(self.residual);
        self.allocator.free(self.norm_out);
        self.allocator.free(self.ffn_gate);
        self.allocator.free(self.ffn_up);
        self.dn_scratch.deinit();
        self.allocator.free(self.attn_q);
        self.allocator.free(self.attn_k);
        self.allocator.free(self.attn_v);
        self.allocator.free(self.attn_out);
        self.allocator.free(self.attn_scores);
        self.allocator.free(self.logits);
    }
};

// ============================================================================
// Model
// ============================================================================

pub const GenerateOptions = struct {
    max_tokens: usize = 256,
    temperature: f32 = 0.0, // 0 = greedy
    stop_on_eos: bool = true,
};

pub const Qwen3_5 = struct {
    cfg: Qwen3_5Config,
    weights: Qwen3_5Weights,
    cache: Qwen3_5Cache,
    scratch: Scratch,
    allocator: std.mem.Allocator,
    position: usize,

    const Self = @This();

    pub fn load(allocator: std.mem.Allocator, loader: *ModelLoader) !Self {
        return loadWithConfig(allocator, loader, Qwen3_5Config.default, 4096);
    }

    pub fn loadWithConfig(
        allocator: std.mem.Allocator,
        loader: *ModelLoader,
        cfg: Qwen3_5Config,
        max_cache_len: usize,
    ) !Self {
        var weights = try loadWeights(allocator, loader, cfg);
        errdefer weights.deinit();

        var model_cache = try Qwen3_5Cache.init(allocator, cfg, max_cache_len);
        errdefer model_cache.deinit();

        var scratch = try Scratch.init(allocator, cfg, max_cache_len);
        errdefer scratch.deinit();

        return .{
            .cfg = cfg,
            .weights = weights,
            .cache = model_cache,
            .scratch = scratch,
            .allocator = allocator,
            .position = 0,
        };
    }

    /// Run a single token through all 24 layers. Returns logits [vocab_size].
    /// The returned slice is valid until the next call to forwardStep.
    pub fn forwardStep(self: *Self, token_id: u32) []const f32 {
        const cfg = self.cfg;
        const hidden_size = cfg.hidden_size;

        // Embedding lookup
        const embed_offset = @as(usize, token_id) * hidden_size;
        switch (self.weights.embed_tokens) {
            .f32 => |t| @memcpy(self.scratch.hidden, t.data[embed_offset..][0..hidden_size]),
            .q8k => |q| ops.dequantRowQ8K(self.scratch.hidden, q.data, q.scales, token_id, hidden_size),
        }

        // Phase 2B: fully GPU-resident hidden state path. All four hooks must be
        // present (token begin/end + layer start + oproj+ffn resident).
        const phase2b = (gpu_token_begin_fn != null and
            gpu_token_end_fn != null and
            gpu_layer_start_dn_fn != null and
            gpu_layer_start_fa_fn != null and
            gpu_oproj_ffn_resident_fn != null);

        if (phase2b) {
            // Upload hidden once at start of token. From here until gpuTokenEnd,
            // the canonical hidden state lives in hidden_staging on the GPU and
            // pre_attn_residual_staging is updated each layer via vkCmdCopyBuffer.
            if (!gpu_token_begin_fn.?(self.scratch.hidden)) @panic("gpu_token_begin_fn failed");
        }

        var dn_idx: usize = 0;
        var fa_idx: usize = 0;

        for (0..cfg.num_hidden_layers) |layer_idx| {
            const lw = &self.weights.layers[layer_idx];

            if (phase2b) {
                // ---------------- Phase 2B per-layer path ----------------
                // gpuLayerStart{Dn,Fa} performs (in one command buffer):
                //   1. cmdCopyBuffer(hidden_staging → pre_attn_residual_staging)
                //   2. RMSNorm(input_layernorm) on hidden_staging in place
                //   3. input projections (qkv/z/b/a or q/k/v) — read back to CPU
                // No CPU residual snapshot or input RMSNorm needed.
                var attn_out_buf: []const f32 = &.{};

                switch (lw.layer_type) {
                    .linear_attention => {
                        if (!gpu_layer_start_dn_fn.?(
                            layer_idx,
                            self.scratch.dn_scratch.mixed_qkv,
                            self.scratch.dn_scratch.z,
                            self.scratch.dn_scratch.beta,
                            self.scratch.dn_scratch.alpha,
                        )) @panic("gpu_layer_start_dn_fn failed");

                        // CPU: conv1d + recurrence + gate/norm — no inner o_proj (do_o_proj=false)
                        deltanet.deltaNetStepPostProjections(
                            &self.cache.deltanet_states[dn_idx],
                            &lw.deltanet_w.?,
                            &self.scratch.dn_scratch,
                            cfg,
                            false,
                        );
                        const value_dim = cfg.linearValueDim();
                        attn_out_buf = self.scratch.dn_scratch.output[0..value_dim];
                        dn_idx += 1;
                    },
                    .full_attention => {
                        if (!gpu_layer_start_fa_fn.?(
                            layer_idx,
                            self.scratch.attn_q[0..cfg.fullAttentionQProjDim()],
                            self.scratch.attn_k,
                            self.scratch.attn_v,
                        )) @panic("gpu_layer_start_fa_fn failed");

                        self.fullAttentionStepPostProjections(lw, &self.cache.kv_caches[fa_idx], false);
                        const q_dim = cfg.fullAttentionQDim();
                        attn_out_buf = self.scratch.attn_out[0..q_dim];
                        fa_idx += 1;
                    },
                }

                // GPU: o_proj + add_dup(pre_attn_residual already on GPU) + RMSNorm + FFN
                // hidden_staging is updated in place; no readback.
                if (!gpu_oproj_ffn_resident_fn.?(attn_out_buf, layer_idx)) {
                    @panic("gpu_oproj_ffn_resident_fn failed");
                }
                continue;
            }

            // -------------- Legacy / partial-GPU path (Phase 2A and earlier) --------------
            // Save residual
            @memcpy(self.scratch.residual, self.scratch.hidden);

            // When the GPU-fused o_proj+FFN dispatch is available, we skip the inner
            // o_proj/out_proj inside attention (it becomes the first dispatch in the
            // fused command buffer). Otherwise the attention step does its own o_proj.
            const fused_oproj_ffn_available = (gpu_oproj_ffn_fn != null);
            const do_inner_oproj = !fused_oproj_ffn_available;

            // Attn-output buffer captured for the fused-dispatch path below.
            var attn_out_buf: []const f32 = &.{};

            // Attention (DeltaNet or Full GQA)
            switch (lw.layer_type) {
                .linear_attention => {
                    // CPU RMSNorm first for bit-exact precision, then GPU batches projections
                    ops.rmsNorm1DInPlace(self.scratch.hidden, lw.input_layernorm.data, hidden_size, cfg.rms_norm_eps);

                    // Try GPU-fused input projections (qkv/z/b/a SGEMVs in one dispatch)
                    const gpu_handled_input = if (gpu_dn_input_fn) |fn_ptr|
                        fn_ptr(
                            self.scratch.hidden,
                            layer_idx,
                            self.scratch.dn_scratch.mixed_qkv,
                            self.scratch.dn_scratch.z,
                            self.scratch.dn_scratch.beta,
                            self.scratch.dn_scratch.alpha,
                        )
                    else
                        false;

                    if (gpu_handled_input) {
                        // GPU did projections; continue from conv1d
                        deltanet.deltaNetStepPostProjections(
                            &self.cache.deltanet_states[dn_idx],
                            &lw.deltanet_w.?,
                            &self.scratch.dn_scratch,
                            cfg,
                            do_inner_oproj,
                        );
                    } else {
                        // CPU fallback: full DeltaNet step (hidden already RMSNorm'd)
                        deltanet.deltaNetStep(
                            self.scratch.hidden,
                            &self.cache.deltanet_states[dn_idx],
                            &lw.deltanet_w.?,
                            &self.scratch.dn_scratch,
                            cfg,
                            do_inner_oproj,
                        );
                    }

                    if (do_inner_oproj) {
                        // Copy result to hidden (out_proj already wrote to dn_scratch.result)
                        @memcpy(self.scratch.hidden, self.scratch.dn_scratch.result);
                    } else {
                        // Capture attention output for fused o_proj+FFN
                        const value_dim = cfg.linearValueDim();
                        attn_out_buf = self.scratch.dn_scratch.output[0..value_dim];
                    }
                    dn_idx += 1;
                },
                .full_attention => {
                    // Try GPU-fused input projections (RMSNorm on CPU + q/k/v SGEMVs in one dispatch)
                    // CPU RMSNorm first for bit-exact precision, then GPU batches projections
                    ops.rmsNorm1DInPlace(self.scratch.hidden, lw.input_layernorm.data, hidden_size, cfg.rms_norm_eps);

                    const gpu_handled_input = if (gpu_attn_input_fn) |fn_ptr|
                        fn_ptr(
                            self.scratch.hidden,
                            layer_idx,
                            self.scratch.attn_q[0..cfg.fullAttentionQProjDim()],
                            self.scratch.attn_k,
                            self.scratch.attn_v,
                        )
                    else
                        false;

                    if (gpu_handled_input) {
                        // GPU did q/k/v projections; continue from de-interleave
                        self.fullAttentionStepPostProjections(lw, &self.cache.kv_caches[fa_idx], do_inner_oproj);
                    } else {
                        // CPU fallback: full attention step (hidden already RMSNorm'd)
                        self.fullAttentionStep(lw, &self.cache.kv_caches[fa_idx], do_inner_oproj);
                    }

                    if (!do_inner_oproj) {
                        // Capture attention output for fused o_proj+FFN
                        const q_dim = cfg.fullAttentionQDim();
                        attn_out_buf = self.scratch.attn_out[0..q_dim];
                    }
                    fa_idx += 1;
                },
            }

            // ----------------------------------------------------------------
            // Post-attention path: either fused o_proj+FFN (GPU) or CPU/separate.
            // ----------------------------------------------------------------
            if (fused_oproj_ffn_available) {
                // Single GPU command buffer:
                //   o_proj \u2192 add_dup(post-attn residual) \u2192 RMSNorm \u2192 gate/up \u2192 silu_mul \u2192 down \u2192 ffn residual_add

                const handled = gpu_oproj_ffn_fn.?(
                    attn_out_buf,
                    self.scratch.residual,
                    layer_idx,
                    self.scratch.hidden,
                );
                if (!handled) @panic("gpu_oproj_ffn_fn failed unexpectedly");
            } else {
                // Legacy path: o_proj already done by inner attention step.
                // Residual connection
                for (0..hidden_size) |i| {
                    self.scratch.hidden[i] += self.scratch.residual[i];
                }

                // Save residual for FFN
                @memcpy(self.scratch.residual, self.scratch.hidden);

                // Try GPU-fused FFN (RMSNorm + gate/up/SiLU\u00d7mul/down + residual add in one dispatch)
                const gpu_handled_ffn = if (gpu_ffn_fn) |ffn_fn|
                    ffn_fn(self.scratch.hidden, self.scratch.residual, layer_idx)
                else
                    false;

                if (!gpu_handled_ffn) {
                    // CPU fallback: RMSNorm + FFN + residual add
                    ops.rmsNorm1DInPlace(self.scratch.hidden, lw.post_attn_layernorm.data, hidden_size, cfg.rms_norm_eps);
                    self.ffnStep(lw);
                    for (0..hidden_size) |i| {
                        self.scratch.hidden[i] += self.scratch.residual[i];
                    }
                }
            }
        }

        if (phase2b) {
            if (gpu_lm_head_fn) |lm_fn| {
                // Fused final RMSNorm + LM head on GPU (1 submit, reads back logits directly).
                if (!lm_fn(self.scratch.logits)) @panic("gpu_lm_head_fn failed");
            } else {
                // Fallback: read back hidden, then CPU final_norm + standalone matvec.
                if (!gpu_token_end_fn.?(self.scratch.hidden)) @panic("gpu_token_end_fn failed");
                ops.rmsNorm1DInPlace(self.scratch.hidden, self.weights.final_norm.data, hidden_size, cfg.rms_norm_eps);
                projectMul(self.scratch.logits, self.weights.embed_tokens, self.scratch.hidden, cfg.vocab_size, hidden_size);
            }
        } else {
            // Non-resident path: hidden is already on CPU.
            ops.rmsNorm1DInPlace(self.scratch.hidden, self.weights.final_norm.data, hidden_size, cfg.rms_norm_eps);
            projectMul(self.scratch.logits, self.weights.embed_tokens, self.scratch.hidden, cfg.vocab_size, hidden_size);
        }

        self.position += 1;
        return self.scratch.logits;
    }

    /// Full GQA attention step (single token, with KV cache).
    /// do_o_proj: when true, write final o_proj into self.scratch.hidden;
    ///            when false, leave attention output (post-gate) in self.scratch.attn_out
    ///            so caller can run o_proj as part of a fused dispatch.
    fn fullAttentionStep(self: *Self, lw: *const LayerWeights, kv_cache: *Qwen3_5Cache.AttentionKVCache, do_o_proj: bool) void {
        const cfg = self.cfg;
        const hidden_size = cfg.hidden_size;
        const q_proj_dim = cfg.fullAttentionQProjDim();
        const kv_dim = cfg.fullAttentionKvDim();

        const attn = &lw.full_attn.?;

        // Q projection: output is [q_proj_dim] in interleaved layout [Q0 G0 Q1 G1 ...] per head
        projectMul(self.scratch.attn_q[0..q_proj_dim], attn.q_proj, self.scratch.hidden, q_proj_dim, hidden_size);
        projectMul(self.scratch.attn_k, attn.k_proj, self.scratch.hidden, kv_dim, hidden_size);
        projectMul(self.scratch.attn_v, attn.v_proj, self.scratch.hidden, kv_dim, hidden_size);

        // Continue with de-interleave, norms, RoPE, attention, o_proj
        self.fullAttentionStepPostProjections(lw, kv_cache, do_o_proj);
    }

    /// Continue full attention step after Q/K/V projections are already computed.
    /// Assumes attn_q[0..q_proj_dim], attn_k[0..kv_dim], attn_v[0..kv_dim] are filled.
    /// do_o_proj: see fullAttentionStep doc.
    fn fullAttentionStepPostProjections(self: *Self, lw: *const LayerWeights, kv_cache: *Qwen3_5Cache.AttentionKVCache, do_o_proj: bool) void {
        const cfg = self.cfg;
        const hidden_size = cfg.hidden_size;
        const q_dim = cfg.fullAttentionQDim();
        const kv_dim = cfg.fullAttentionKvDim();
        const head_dim = cfg.head_dim;
        const num_heads = cfg.num_attention_heads;
        const num_kv_heads = cfg.num_key_value_heads;
        const partial_dim = cfg.ropePartialDim();
        const gqa_ratio = cfg.gqaRatio();

        const attn = &lw.full_attn.?;

        // De-interleave Q and gate: [Q0(hd) G0(hd) Q1(hd) G1(hd) ...] → [Q0 Q1 ... | G0 G1 ...]
        // Step 1: Extract Q parts to attn_out (temp buffer)
        for (0..num_heads) |h| {
            const src = self.scratch.attn_q[h * head_dim * 2 ..][0..head_dim];
            const dst = self.scratch.attn_out[h * head_dim ..][0..head_dim];
            @memcpy(dst, src);
        }
        // Step 2: Extract gate parts (iterate backward to avoid overwrites)
        {
            var h: usize = num_heads;
            while (h > 0) {
                h -= 1;
                const src_off = h * head_dim * 2 + head_dim;
                const dst_off = q_dim + h * head_dim;
                if (src_off != dst_off) {
                    @memcpy(self.scratch.attn_q[dst_off..][0..head_dim], self.scratch.attn_q[src_off..][0..head_dim]);
                }
            }
        }
        // Step 3: Copy Q back from temp
        @memcpy(self.scratch.attn_q[0..q_dim], self.scratch.attn_out[0..q_dim]);

        // Q/K RMSNorm normalization (per head, using per-head norm weights)
        // PyTorch uses Qwen3_5RMSNorm: x * (1+weight) / sqrt(mean(x²) + eps)
        // Export adds +1 to weights, so we just use: x * weight / sqrt(mean(x²) + eps)
        for (0..num_heads) |h| {
            const q_h = self.scratch.attn_q[h * head_dim ..][0..head_dim];
            var sum_sq: f32 = 0.0;
            for (q_h) |v| sum_sq += v * v;
            const inv_rms: f32 = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(head_dim)) + cfg.rms_norm_eps);
            for (0..head_dim) |d| {
                q_h[d] = q_h[d] * attn.q_norm.data[d] * inv_rms;
            }
        }
        for (0..num_kv_heads) |h| {
            const k_h = self.scratch.attn_k[h * head_dim ..][0..head_dim];
            var sum_sq: f32 = 0.0;
            for (k_h) |v| sum_sq += v * v;
            const inv_rms: f32 = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(head_dim)) + cfg.rms_norm_eps);
            for (0..head_dim) |d| {
                k_h[d] = k_h[d] * attn.k_norm.data[d] * inv_rms;
            }
        }

        // Partial RoPE (only first partial_dim dimensions per head, only Q part)
        ops.ropeInPlace(
            self.scratch.attn_q[0..q_dim],
            self.scratch.attn_k,
            num_heads,
            num_kv_heads,
            head_dim,
            partial_dim,
            self.position,
            cfg.rope_theta,
        );

        // Append K, V to cache
        kv_cache.append(self.scratch.attn_k, self.scratch.attn_v);

        // GQA attention: each Q head attends to its corresponding KV head
        const seq_len = kv_cache.length;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        for (0..num_heads) |h| {
            const kv_h = h / gqa_ratio; // Which KV head this Q head uses
            const q_h = self.scratch.attn_q[h * head_dim ..][0..head_dim];
            const scores = self.scratch.attn_scores[h * seq_len ..][0..seq_len];

            // Compute attention scores: Q @ K^T / sqrt(d)
            for (0..seq_len) |t| {
                const k_t = kv_cache.keys[t * kv_dim + kv_h * head_dim ..][0..head_dim];
                var dot: f32 = 0.0;
                for (0..head_dim) |d| {
                    dot += q_h[d] * k_t[d];
                }
                scores[t] = dot * scale;
            }

            // Causal mask: no masking needed since we only attend to cached positions (all valid)

            // Softmax
            var max_val: f32 = scores[0];
            for (scores[1..]) |s| if (s > max_val) {
                max_val = s;
            };
            var sum_exp: f32 = 0.0;
            for (scores) |*s| {
                s.* = @exp(s.* - max_val);
                sum_exp += s.*;
            }
            for (scores) |*s| s.* /= sum_exp;

            // Weighted sum of values
            const o_h = self.scratch.attn_out[h * head_dim ..][0..head_dim];
            @memset(o_h, 0);
            for (0..seq_len) |t| {
                const v_t = kv_cache.values[t * kv_dim + kv_h * head_dim ..][0..head_dim];
                const w = scores[t];
                for (0..head_dim) |d| {
                    o_h[d] += w * v_t[d];
                }
            }
        }

        // Apply output gate: attn_out = sigmoid(gate) * attn_out
        if (cfg.attn_output_gate) {
            const gate = self.scratch.attn_q[q_dim..][0..q_dim]; // second half of q_proj output
            for (0..q_dim) |i| {
                const g = 1.0 / (1.0 + @exp(-gate[i])); // sigmoid
                self.scratch.attn_out[i] *= g;
            }
        }

        // Output projection: hidden = W_o @ attn_out (optional — caller may skip to fuse on GPU)
        if (do_o_proj) {
            projectMul(self.scratch.hidden, attn.o_proj, self.scratch.attn_out, hidden_size, q_dim);
        }
    }

    /// SwiGLU FFN step (all layers share same structure).
    fn ffnStep(self: *Self, lw: *const LayerWeights) void {
        const hidden_size = self.cfg.hidden_size;
        const intermediate = self.cfg.intermediate_size;

        // gate = W_gate @ hidden
        projectMul(self.scratch.ffn_gate, lw.gate_proj, self.scratch.hidden, intermediate, hidden_size);

        // up = W_up @ hidden
        projectMul(self.scratch.ffn_up, lw.up_proj, self.scratch.hidden, intermediate, hidden_size);

        // gate = silu(gate) * up
        for (0..intermediate) |i| {
            const g = self.scratch.ffn_gate[i];
            self.scratch.ffn_gate[i] = (g / (1.0 + @exp(-g))) * self.scratch.ffn_up[i];
        }

        // hidden = W_down @ gate
        projectMul(self.scratch.hidden, lw.down_proj, self.scratch.ffn_gate, hidden_size, intermediate);
    }

    /// Generate tokens autoregressively.
    /// prompt_ids: tokenized prompt
    /// Returns: generated token IDs (caller must free)
    pub fn generate(self: *Self, prompt_ids: []const u32, opts: GenerateOptions) ![]u32 {
        var tokens = std.ArrayListUnmanaged(u32){};
        errdefer tokens.deinit(self.allocator);

        // Reset state for new generation
        self.cache.reset();
        self.position = 0;

        // Prefill: process prompt tokens sequentially
        var last_logits: []const f32 = undefined;
        for (prompt_ids) |token_id| {
            last_logits = self.forwardStep(token_id);
        }

        // Decode loop
        for (0..opts.max_tokens) |_| {
            const next_token = if (opts.temperature <= 0.0)
                argmax(last_logits)
            else
                sampleWithTemperature(last_logits, opts.temperature);

            try tokens.append(self.allocator, next_token);

            if (opts.stop_on_eos and next_token == SpecialTokens.EOS) {
                break;
            }

            last_logits = self.forwardStep(next_token);
        }

        return tokens.toOwnedSlice(self.allocator);
    }

    pub fn reset(self: *Self) void {
        self.cache.reset();
        self.position = 0;
    }

    pub fn deinit(self: *Self) void {
        self.scratch.deinit();
        self.cache.deinit();
        self.weights.deinit();
    }
};

// ============================================================================
// Weight Loading
// ============================================================================

fn loadWeights(allocator: std.mem.Allocator, loader: *ModelLoader, cfg: Qwen3_5Config) !Qwen3_5Weights {
    const is_q8k = loader.quant_format == .q8_k;

    // Helpers to load projection (Q8K or F32) and small tensors (always F32)
    const Helper = struct {
        fn proj(ld: *ModelLoader, name: []const u8, q8k: bool) !ProjectionWeight {
            if (q8k) {
                return .{ .q8k = try ld.getQuantizedTensorQ8K(name) };
            } else {
                return .{ .f32 = try ld.getTensor(name) };
            }
        }
        fn small(ld: *ModelLoader, name: []const u8, q8k: bool) !Tensor {
            if (q8k) {
                return ld.getQ8KAsF32(name);
            } else {
                return ld.getTensor(name);
            }
        }
    };

    const embed_tokens = try Helper.proj(loader, "embed_tokens.weight", is_q8k);
    const final_norm = try Helper.small(loader, "norm.weight", is_q8k);

    var layers = try allocator.alloc(LayerWeights, cfg.num_hidden_layers);
    errdefer allocator.free(layers);

    var name_buf: [128]u8 = undefined;

    for (0..cfg.num_hidden_layers) |i| {
        const lt = cfg.getLayerType(i);

        // Common weights
        const input_ln = try Helper.small(loader, layerName(&name_buf, i, "input_layernorm.weight"), is_q8k);
        const post_ln = try Helper.small(loader, layerName(&name_buf, i, "post_attn_layernorm.weight"), is_q8k);
        const gate = try Helper.proj(loader, layerName(&name_buf, i, "mlp.gate_proj.weight"), is_q8k);
        const up = try Helper.proj(loader, layerName(&name_buf, i, "mlp.up_proj.weight"), is_q8k);
        const down = try Helper.proj(loader, layerName(&name_buf, i, "mlp.down_proj.weight"), is_q8k);

        var full_attn: ?FullAttentionWeights = null;
        var deltanet_w: ?DeltaNetWeights = null;

        switch (lt) {
            .full_attention => {
                full_attn = .{
                    .q_proj = try Helper.proj(loader, layerName(&name_buf, i, "self_attn.q_proj.weight"), is_q8k),
                    .k_proj = try Helper.proj(loader, layerName(&name_buf, i, "self_attn.k_proj.weight"), is_q8k),
                    .v_proj = try Helper.proj(loader, layerName(&name_buf, i, "self_attn.v_proj.weight"), is_q8k),
                    .o_proj = try Helper.proj(loader, layerName(&name_buf, i, "self_attn.o_proj.weight"), is_q8k),
                    .q_norm = try Helper.small(loader, layerName(&name_buf, i, "self_attn.q_norm.weight"), is_q8k),
                    .k_norm = try Helper.small(loader, layerName(&name_buf, i, "self_attn.k_norm.weight"), is_q8k),
                };
            },
            .linear_attention => {
                deltanet_w = .{
                    .in_proj_qkv = try Helper.proj(loader, layerName(&name_buf, i, "deltanet.in_proj_qkv.weight"), is_q8k),
                    .in_proj_z = try Helper.proj(loader, layerName(&name_buf, i, "deltanet.in_proj_z.weight"), is_q8k),
                    .in_proj_b = try Helper.proj(loader, layerName(&name_buf, i, "deltanet.in_proj_b.weight"), is_q8k),
                    .in_proj_a = try Helper.proj(loader, layerName(&name_buf, i, "deltanet.in_proj_a.weight"), is_q8k),
                    .conv1d_weight = try Helper.small(loader, layerName(&name_buf, i, "deltanet.conv1d.weight"), is_q8k),
                    .conv1d_bias = try Helper.small(loader, layerName(&name_buf, i, "deltanet.conv1d.bias"), is_q8k),
                    .A_log = try Helper.small(loader, layerName(&name_buf, i, "deltanet.A_log"), is_q8k),
                    .dt_bias = try Helper.small(loader, layerName(&name_buf, i, "deltanet.dt_bias"), is_q8k),
                    .out_proj = try Helper.proj(loader, layerName(&name_buf, i, "deltanet.out_proj.weight"), is_q8k),
                    .norm_weight = try Helper.small(loader, layerName(&name_buf, i, "deltanet.norm.weight"), is_q8k),
                };
            },
        }

        layers[i] = .{
            .layer_type = lt,
            .input_layernorm = input_ln,
            .post_attn_layernorm = post_ln,
            .full_attn = full_attn,
            .deltanet_w = deltanet_w,
            .gate_proj = gate,
            .up_proj = up,
            .down_proj = down,
        };
    }

    return .{
        .embed_tokens = embed_tokens,
        .final_norm = final_norm,
        .layers = layers,
        .allocator = allocator,
    };
}

fn layerName(buf: *[128]u8, layer_idx: usize, suffix: []const u8) []const u8 {
    const result = std.fmt.bufPrint(buf, "layers.{d}.{s}", .{ layer_idx, suffix }) catch unreachable;
    return result;
}

// ============================================================================
// Sampling
// ============================================================================

fn argmax(logits: []const f32) u32 {
    var max_idx: u32 = 0;
    var max_val = logits[0];
    for (logits[1..], 1..) |val, i| {
        if (val > max_val) {
            max_val = val;
            max_idx = @intCast(i);
        }
    }
    return max_idx;
}

fn sampleWithTemperature(logits: []const f32, temperature: f32) u32 {
    // For now, greedy (temperature sampling requires PRNG)
    _ = temperature;
    return argmax(logits);
}

// ============================================================================
// Tests
// ============================================================================

test "argmax" {
    const logits = [_]f32{ 0.1, 0.5, 0.2, 0.8, 0.3 };
    try std.testing.expectEqual(@as(u32, 3), argmax(&logits));
}

test "layer_name_format" {
    var buf: [128]u8 = undefined;
    const name = layerName(&buf, 3, "self_attn.q_proj.weight");
    try std.testing.expectEqualStrings("layers.3.self_attn.q_proj.weight", name);
}

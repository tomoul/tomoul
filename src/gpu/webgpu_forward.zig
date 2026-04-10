// src/gpu/webgpu_forward.zig
//
// GPU Forward Pass for Transformer Encoder (WebGPU Backend)
//
// Dispatches full encoder pipeline via WebGPU compute shaders (WGSL).
// Weight data uploaded once at init; only token IDs transferred per inference.
//
// Dispatch sequence per forward():
//   1. Embedding lookup + sum       (embedding_lookup.wgsl)
//   2. Embedding LayerNorm           (layernorm.wgsl)
//   3. For each of N layers:
//      a. Q/K/V projections ×3       (sgemm_bias.wgsl)
//      b. Multi-head attention        (attention.wgsl)
//      c. Output projection           (sgemm_bias.wgsl)
//      d. Residual add                (residual_add.wgsl)
//      e. Attention LayerNorm         (layernorm.wgsl)
//      f. FFN up projection           (sgemm_bias.wgsl)
//      g. GELU activation             (gelu.wgsl)
//      h. FFN down projection         (sgemm_bias.wgsl)
//      i. Residual add                (residual_add.wgsl)
//      j. FFN LayerNorm               (layernorm.wgsl)
//   4. Mean pooling + L2 normalize   (pool_normalize.wgsl)
//
// All dispatches recorded as sequential compute passes in a single submission.

const std = @import("std");
const gpu = @import("webgpu");

// Embedded WGSL shaders (compiled at build time via @embedFile)
const wgsl_sgemm_bias = @embedFile("shaders/webgpu/sgemm_bias.wgsl");
const wgsl_layernorm = @embedFile("shaders/webgpu/layernorm.wgsl");
const wgsl_gelu = @embedFile("shaders/webgpu/gelu.wgsl");
const wgsl_residual_add = @embedFile("shaders/webgpu/residual_add.wgsl");
const wgsl_attention = @embedFile("shaders/webgpu/attention.wgsl");
const wgsl_embedding_lookup = @embedFile("shaders/webgpu/embedding_lookup.wgsl");
const wgsl_pool_normalize = @embedFile("shaders/webgpu/pool_normalize.wgsl");
const wgsl_attention_batch = @embedFile("shaders/webgpu/attention_batch.wgsl");
const wgsl_pool_normalize_batch = @embedFile("shaders/webgpu/pool_normalize_batch.wgsl");

// ============================================================================
// Uniform Structs (must match WGSL shader layouts exactly)
// ============================================================================

const SgemmBiasPC = extern struct { m: u32, n: u32, k: u32 };
const LayerNormPC = extern struct { rows: u32, cols: u32, eps: f32 };
const ElementPC = extern struct { count: u32 };
const AttentionPC = extern struct { seq_len: u32, num_heads: u32, head_dim: u32, scale: f32 };
const AttentionBatchPC = extern struct { total_tokens: u32, num_heads: u32, head_dim: u32, scale: f32, batch_size: u32 };
const EmbeddingPC = extern struct { seq_len: u32, hidden_dim: u32 };
const PoolNormPC = extern struct { seq_len: u32, hidden_dim: u32 };
const PoolNormBatchPC = extern struct { batch_size: u32, hidden_dim: u32 };

// ============================================================================
// Public Configuration Types (identical layout to other backends)
// ============================================================================

pub const GpuConfig = struct {
    hidden_dim: u32,
    num_heads: u32,
    head_dim: u32,
    ffn_dim: u32,
    num_layers: u32,
    vocab_size: u32,
    max_seq_len: u32,
    max_batch_tokens: u32 = 0,
};

pub const EmbeddingData = struct {
    word_emb: []const f32,
    pos_emb: []const f32,
    type_emb: []const f32,
    ln_gamma: []const f32,
    ln_beta: []const f32,
};

pub const LayerData = struct {
    q_weight: []const f32,
    q_bias: []const f32,
    k_weight: []const f32,
    k_bias: []const f32,
    v_weight: []const f32,
    v_bias: []const f32,
    o_weight: []const f32,
    o_bias: []const f32,
    ff1_weight: []const f32,
    ff1_bias: []const f32,
    ff2_weight: []const f32,
    ff2_bias: []const f32,
    attn_ln_gamma: []const f32,
    attn_ln_beta: []const f32,
    ff_ln_gamma: []const f32,
    ff_ln_beta: []const f32,
};

// ============================================================================
// Internal Types
// ============================================================================

const LayerGpuWeights = struct {
    q_weight: gpu.GpuBuffer,
    q_bias: gpu.GpuBuffer,
    k_weight: gpu.GpuBuffer,
    k_bias: gpu.GpuBuffer,
    v_weight: gpu.GpuBuffer,
    v_bias: gpu.GpuBuffer,
    o_weight: gpu.GpuBuffer,
    o_bias: gpu.GpuBuffer,
    ff1_weight: gpu.GpuBuffer,
    ff1_bias: gpu.GpuBuffer,
    ff2_weight: gpu.GpuBuffer,
    ff2_bias: gpu.GpuBuffer,
    attn_ln_gamma: gpu.GpuBuffer,
    attn_ln_beta: gpu.GpuBuffer,
    ff_ln_gamma: gpu.GpuBuffer,
    ff_ln_beta: gpu.GpuBuffer,
};

pub const WebGpuForwardError = error{
    WebGpuInitFailed,
    BufferCreationFailed,
    PipelineCreationFailed,
    UploadFailed,
    ReadbackFailed,
    OutOfMemory,
    BatchTooLarge,
};

// ============================================================================
// WebGPU Forward Pass Context
// ============================================================================

pub const GpuForward = struct {
    allocator: std.mem.Allocator,
    ctx: gpu.WebGpuContext,
    config: GpuConfig,

    // Compute pipelines (one per shader, reused across dispatches)
    sgemm_bias_pipe: gpu.ComputePipeline,
    layernorm_pipe: gpu.ComputePipeline,
    gelu_pipe: gpu.ComputePipeline,
    residual_add_pipe: gpu.ComputePipeline,
    attention_pipe: gpu.ComputePipeline,
    embedding_pipe: gpu.ComputePipeline,
    pool_normalize_pipe: gpu.ComputePipeline,
    attention_batch_pipe: gpu.ComputePipeline,
    pool_normalize_batch_pipe: gpu.ComputePipeline,

    // Embedding weights (persistent on GPU)
    word_emb_buf: gpu.GpuBuffer,
    pos_emb_buf: gpu.GpuBuffer,
    type_emb_buf: gpu.GpuBuffer,
    embed_ln_gamma_buf: gpu.GpuBuffer,
    embed_ln_beta_buf: gpu.GpuBuffer,

    // Per-layer weights (persistent on GPU)
    layers: []LayerGpuWeights,

    // Activation buffers (reused across layers)
    buf_a: gpu.GpuBuffer,
    buf_b: gpu.GpuBuffer,
    q_buf: gpu.GpuBuffer,
    k_buf: gpu.GpuBuffer,
    v_buf: gpu.GpuBuffer,
    ffn_buf: gpu.GpuBuffer,
    ids_buf: gpu.GpuBuffer,
    output_buf: gpu.GpuBuffer,

    // Batch-specific buffers
    offsets_buf: gpu.GpuBuffer,
    lengths_buf: gpu.GpuBuffer,
    batch_output_buf: gpu.GpuBuffer,
    max_batch_tokens: u32,

    const Self = @This();

    // ====================================================================
    // Init
    // ====================================================================

    pub fn init(
        allocator: std.mem.Allocator,
        config: GpuConfig,
        embeddings: EmbeddingData,
        layer_weights: []const LayerData,
    ) WebGpuForwardError!Self {
        var self: Self = undefined;
        self.allocator = allocator;
        self.config = config;

        // 1. Init WebGPU context (JS bridge)
        self.ctx = gpu.WebGpuContext.init() catch {
            return WebGpuForwardError.WebGpuInitFailed;
        };

        // 2. Create compute pipelines from embedded WGSL
        //    Args: (source, entry_point, num_storage_bindings, has_uniform)
        self.sgemm_bias_pipe = self.ctx.createComputePipeline(wgsl_sgemm_bias, "main", 4, true) catch return WebGpuForwardError.PipelineCreationFailed;
        self.layernorm_pipe = self.ctx.createComputePipeline(wgsl_layernorm, "main", 3, true) catch return WebGpuForwardError.PipelineCreationFailed;
        self.gelu_pipe = self.ctx.createComputePipeline(wgsl_gelu, "main", 1, true) catch return WebGpuForwardError.PipelineCreationFailed;
        self.residual_add_pipe = self.ctx.createComputePipeline(wgsl_residual_add, "main", 2, true) catch return WebGpuForwardError.PipelineCreationFailed;
        self.attention_pipe = self.ctx.createComputePipeline(wgsl_attention, "main", 4, true) catch return WebGpuForwardError.PipelineCreationFailed;
        self.embedding_pipe = self.ctx.createComputePipeline(wgsl_embedding_lookup, "main", 5, true) catch return WebGpuForwardError.PipelineCreationFailed;
        self.pool_normalize_pipe = self.ctx.createComputePipeline(wgsl_pool_normalize, "main", 2, true) catch return WebGpuForwardError.PipelineCreationFailed;
        self.attention_batch_pipe = self.ctx.createComputePipeline(wgsl_attention_batch, "main", 5, true) catch return WebGpuForwardError.PipelineCreationFailed;
        self.pool_normalize_batch_pipe = self.ctx.createComputePipeline(wgsl_pool_normalize_batch, "main", 4, true) catch return WebGpuForwardError.PipelineCreationFailed;

        // 3. Upload embedding weights
        self.word_emb_buf = try uploadF32(&self.ctx, embeddings.word_emb);
        self.pos_emb_buf = try uploadF32(&self.ctx, embeddings.pos_emb);
        self.type_emb_buf = try uploadF32(&self.ctx, embeddings.type_emb);
        self.embed_ln_gamma_buf = try uploadF32(&self.ctx, embeddings.ln_gamma);
        self.embed_ln_beta_buf = try uploadF32(&self.ctx, embeddings.ln_beta);

        // 4. Upload per-layer weights
        self.layers = allocator.alloc(LayerGpuWeights, config.num_layers) catch return WebGpuForwardError.OutOfMemory;
        for (layer_weights, 0..) |lw, i| {
            self.layers[i] = LayerGpuWeights{
                .q_weight = try uploadF32(&self.ctx, lw.q_weight),
                .q_bias = try uploadF32(&self.ctx, lw.q_bias),
                .k_weight = try uploadF32(&self.ctx, lw.k_weight),
                .k_bias = try uploadF32(&self.ctx, lw.k_bias),
                .v_weight = try uploadF32(&self.ctx, lw.v_weight),
                .v_bias = try uploadF32(&self.ctx, lw.v_bias),
                .o_weight = try uploadF32(&self.ctx, lw.o_weight),
                .o_bias = try uploadF32(&self.ctx, lw.o_bias),
                .ff1_weight = try uploadF32(&self.ctx, lw.ff1_weight),
                .ff1_bias = try uploadF32(&self.ctx, lw.ff1_bias),
                .ff2_weight = try uploadF32(&self.ctx, lw.ff2_weight),
                .ff2_bias = try uploadF32(&self.ctx, lw.ff2_bias),
                .attn_ln_gamma = try uploadF32(&self.ctx, lw.attn_ln_gamma),
                .attn_ln_beta = try uploadF32(&self.ctx, lw.attn_ln_beta),
                .ff_ln_gamma = try uploadF32(&self.ctx, lw.ff_ln_gamma),
                .ff_ln_beta = try uploadF32(&self.ctx, lw.ff_ln_beta),
            };
        }

        // 5. Allocate activation buffers
        const max_tokens = if (config.max_batch_tokens > 0) config.max_batch_tokens else config.max_seq_len;
        self.max_batch_tokens = max_tokens;
        const hidden = config.hidden_dim;
        const ffn = config.ffn_dim;
        const act_size = max_tokens * hidden * @sizeOf(f32);

        self.buf_a = self.ctx.createStorageBuffer(act_size, true) catch return WebGpuForwardError.BufferCreationFailed;
        self.buf_b = self.ctx.createStorageBuffer(act_size, true) catch return WebGpuForwardError.BufferCreationFailed;
        self.q_buf = self.ctx.createStorageBuffer(act_size, true) catch return WebGpuForwardError.BufferCreationFailed;
        self.k_buf = self.ctx.createStorageBuffer(act_size, true) catch return WebGpuForwardError.BufferCreationFailed;
        self.v_buf = self.ctx.createStorageBuffer(act_size, true) catch return WebGpuForwardError.BufferCreationFailed;
        self.ffn_buf = self.ctx.createStorageBuffer(max_tokens * ffn * @sizeOf(f32), true) catch return WebGpuForwardError.BufferCreationFailed;
        self.ids_buf = self.ctx.createStorageBuffer(3 * max_tokens * @sizeOf(u32), true) catch return WebGpuForwardError.BufferCreationFailed;
        self.output_buf = self.ctx.createStorageBuffer(hidden * @sizeOf(f32), true) catch return WebGpuForwardError.BufferCreationFailed;

        // Batch-specific buffers
        const max_batch: u32 = 64;
        self.offsets_buf = self.ctx.createStorageBuffer(max_batch * @sizeOf(u32), true) catch return WebGpuForwardError.BufferCreationFailed;
        self.lengths_buf = self.ctx.createStorageBuffer(max_batch * @sizeOf(u32), true) catch return WebGpuForwardError.BufferCreationFailed;
        self.batch_output_buf = self.ctx.createStorageBuffer(max_batch * hidden * @sizeOf(f32), true) catch return WebGpuForwardError.BufferCreationFailed;

        return self;
    }

    // ====================================================================
    // Forward Pass (single sentence)
    // ====================================================================

    pub fn forward(self: *Self, token_ids: []const u32, output: []f32) WebGpuForwardError!void {
        const seq_len: u32 = @intCast(token_ids.len);
        const hidden = self.config.hidden_dim;

        // Prepare IDs buffer: [token_ids | position_ids (0..N-1) | type_ids (all 0)]
        const ids = self.allocator.alloc(u32, seq_len * 3) catch return WebGpuForwardError.OutOfMemory;
        defer self.allocator.free(ids);
        @memcpy(ids[0..seq_len], token_ids);
        for (0..seq_len) |i| ids[seq_len + i] = @intCast(i);
        @memset(ids[2 * seq_len .. 3 * seq_len], 0);
        self.ctx.uploadToBuffer(&self.ids_buf, std.mem.sliceAsBytes(ids)) catch return WebGpuForwardError.UploadFailed;

        // Begin command buffer recording
        self.ctx.beginCommandBuffer();

        // --- Embedding lookup ---
        {
            const pc = EmbeddingPC{ .seq_len = seq_len, .hidden_dim = hidden };
            self.ctx.cmdDispatch(
                self.embedding_pipe,
                &[_]gpu.GpuBuffer{ self.ids_buf, self.word_emb_buf, self.pos_emb_buf, self.type_emb_buf, self.buf_a },
                std.mem.asBytes(&pc),
                (seq_len * hidden + 255) / 256, 1, 1,
            );
        }

        // --- Embedding LayerNorm ---
        self.recordLayerNorm(seq_len, self.buf_a, self.embed_ln_gamma_buf, self.embed_ln_beta_buf);

        // --- Transformer Layers ---
        for (0..self.config.num_layers) |i| {
            self.recordLayer(@intCast(i), seq_len);
        }

        // --- Pool + Normalize ---
        {
            const pc = PoolNormPC{ .seq_len = seq_len, .hidden_dim = hidden };
            self.ctx.cmdDispatch(
                self.pool_normalize_pipe,
                &[_]gpu.GpuBuffer{ self.buf_a, self.output_buf },
                std.mem.asBytes(&pc),
                1, 1, 1,
            );
        }

        // Submit
        self.ctx.submit();

        // Readback (JS host ensures GPU work is complete before this returns)
        self.ctx.readbackFromBuffer(&self.output_buf, std.mem.sliceAsBytes(output[0..hidden])) catch return WebGpuForwardError.ReadbackFailed;
    }

    // ====================================================================
    // Batched Forward Pass
    // ====================================================================

    pub fn forwardBatch(self: *Self, batch_ids: []const []const u32, output: []f32) WebGpuForwardError!void {
        const batch_size: u32 = @intCast(batch_ids.len);
        const hidden = self.config.hidden_dim;

        // Compute total tokens and per-sentence offsets/lengths
        var total_tokens: u32 = 0;
        const offsets = self.allocator.alloc(u32, batch_size) catch return WebGpuForwardError.OutOfMemory;
        defer self.allocator.free(offsets);
        const lengths = self.allocator.alloc(u32, batch_size) catch return WebGpuForwardError.OutOfMemory;
        defer self.allocator.free(lengths);

        for (batch_ids, 0..) |ids, i| {
            offsets[i] = total_tokens;
            lengths[i] = @intCast(ids.len);
            total_tokens += @intCast(ids.len);
        }

        if (total_tokens > self.max_batch_tokens) {
            return WebGpuForwardError.BatchTooLarge;
        }

        // Pack all token IDs
        const all_ids = self.allocator.alloc(u32, total_tokens * 3) catch return WebGpuForwardError.OutOfMemory;
        defer self.allocator.free(all_ids);
        var pos: u32 = 0;
        for (batch_ids) |ids| {
            const len: u32 = @intCast(ids.len);
            @memcpy(all_ids[pos .. pos + len], ids);
            for (0..len) |j| all_ids[total_tokens + pos + j] = @intCast(j);
            @memset(all_ids[2 * total_tokens + pos .. 2 * total_tokens + pos + len], 0);
            pos += len;
        }
        self.ctx.uploadToBuffer(&self.ids_buf, std.mem.sliceAsBytes(all_ids)) catch return WebGpuForwardError.UploadFailed;
        self.ctx.uploadToBuffer(&self.offsets_buf, std.mem.sliceAsBytes(offsets)) catch return WebGpuForwardError.UploadFailed;
        self.ctx.uploadToBuffer(&self.lengths_buf, std.mem.sliceAsBytes(lengths)) catch return WebGpuForwardError.UploadFailed;

        // Record
        self.ctx.beginCommandBuffer();

        // --- Embedding lookup ---
        {
            const pc = EmbeddingPC{ .seq_len = total_tokens, .hidden_dim = hidden };
            self.ctx.cmdDispatch(
                self.embedding_pipe,
                &[_]gpu.GpuBuffer{ self.ids_buf, self.word_emb_buf, self.pos_emb_buf, self.type_emb_buf, self.buf_a },
                std.mem.asBytes(&pc),
                (total_tokens * hidden + 255) / 256, 1, 1,
            );
        }

        // --- Embedding LayerNorm ---
        self.recordLayerNorm(total_tokens, self.buf_a, self.embed_ln_gamma_buf, self.embed_ln_beta_buf);

        // --- Transformer Layers (batch-aware attention) ---
        for (0..self.config.num_layers) |i| {
            self.recordLayerBatch(@intCast(i), total_tokens, batch_size);
        }

        // --- Batched Pool + Normalize ---
        {
            const pc = PoolNormBatchPC{ .batch_size = batch_size, .hidden_dim = hidden };
            self.ctx.cmdDispatch(
                self.pool_normalize_batch_pipe,
                &[_]gpu.GpuBuffer{ self.buf_a, self.batch_output_buf, self.offsets_buf, self.lengths_buf },
                std.mem.asBytes(&pc),
                batch_size, 1, 1,
            );
        }

        // Submit and readback
        self.ctx.submit();
        self.ctx.readbackFromBuffer(&self.batch_output_buf, std.mem.sliceAsBytes(output[0 .. batch_size * hidden])) catch return WebGpuForwardError.ReadbackFailed;
    }

    // ====================================================================
    // Per-Layer Dispatch
    // ====================================================================

    fn recordLayer(self: *Self, layer_idx: u32, seq_len: u32) void {
        const lw = &self.layers[layer_idx];
        const hidden = self.config.hidden_dim;
        const ffn_dim = self.config.ffn_dim;
        const elem_count = seq_len * hidden;

        // Q, K, V projections
        self.recordGemm(self.buf_a, lw.q_weight, self.q_buf, lw.q_bias, seq_len, hidden, hidden);
        self.recordGemm(self.buf_a, lw.k_weight, self.k_buf, lw.k_bias, seq_len, hidden, hidden);
        self.recordGemm(self.buf_a, lw.v_weight, self.v_buf, lw.v_bias, seq_len, hidden, hidden);

        // Multi-head attention
        {
            const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.config.head_dim)));
            const pc = AttentionPC{
                .seq_len = seq_len,
                .num_heads = self.config.num_heads,
                .head_dim = self.config.head_dim,
                .scale = scale,
            };
            self.ctx.cmdDispatch(
                self.attention_pipe,
                &[_]gpu.GpuBuffer{ self.q_buf, self.k_buf, self.v_buf, self.buf_b },
                std.mem.asBytes(&pc),
                self.config.num_heads, seq_len, 1,
            );
        }

        // Output projection
        self.recordGemm(self.buf_b, lw.o_weight, self.q_buf, lw.o_bias, seq_len, hidden, hidden);

        // Residual + LayerNorm
        self.recordAdd(self.q_buf, self.buf_a, elem_count);
        self.recordLayerNorm(seq_len, self.q_buf, lw.attn_ln_gamma, lw.attn_ln_beta);

        // FFN up + GELU
        self.recordGemm(self.q_buf, lw.ff1_weight, self.ffn_buf, lw.ff1_bias, seq_len, ffn_dim, hidden);
        {
            const pc = ElementPC{ .count = seq_len * ffn_dim };
            self.ctx.cmdDispatch(
                self.gelu_pipe,
                &[_]gpu.GpuBuffer{self.ffn_buf},
                std.mem.asBytes(&pc),
                (pc.count + 255) / 256, 1, 1,
            );
        }

        // FFN down + Residual + LayerNorm
        self.recordGemm(self.ffn_buf, lw.ff2_weight, self.buf_a, lw.ff2_bias, seq_len, hidden, ffn_dim);
        self.recordAdd(self.buf_a, self.q_buf, elem_count);
        self.recordLayerNorm(seq_len, self.buf_a, lw.ff_ln_gamma, lw.ff_ln_beta);
    }

    fn recordLayerBatch(self: *Self, layer_idx: u32, total_tokens: u32, batch_size: u32) void {
        const lw = &self.layers[layer_idx];
        const hidden = self.config.hidden_dim;
        const ffn_dim = self.config.ffn_dim;
        const elem_count = total_tokens * hidden;

        // Q, K, V
        self.recordGemm(self.buf_a, lw.q_weight, self.q_buf, lw.q_bias, total_tokens, hidden, hidden);
        self.recordGemm(self.buf_a, lw.k_weight, self.k_buf, lw.k_bias, total_tokens, hidden, hidden);
        self.recordGemm(self.buf_a, lw.v_weight, self.v_buf, lw.v_bias, total_tokens, hidden, hidden);

        // Batch-aware attention
        {
            const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.config.head_dim)));
            const pc = AttentionBatchPC{
                .total_tokens = total_tokens,
                .num_heads = self.config.num_heads,
                .head_dim = self.config.head_dim,
                .scale = scale,
                .batch_size = batch_size,
            };
            self.ctx.cmdDispatch(
                self.attention_batch_pipe,
                &[_]gpu.GpuBuffer{ self.q_buf, self.k_buf, self.v_buf, self.buf_b, self.offsets_buf },
                std.mem.asBytes(&pc),
                self.config.num_heads, total_tokens, 1,
            );
        }

        // Output proj + residual + LN + FFN + GELU + FFN + residual + LN
        self.recordGemm(self.buf_b, lw.o_weight, self.q_buf, lw.o_bias, total_tokens, hidden, hidden);
        self.recordAdd(self.q_buf, self.buf_a, elem_count);
        self.recordLayerNorm(total_tokens, self.q_buf, lw.attn_ln_gamma, lw.attn_ln_beta);
        self.recordGemm(self.q_buf, lw.ff1_weight, self.ffn_buf, lw.ff1_bias, total_tokens, ffn_dim, hidden);
        {
            const pc = ElementPC{ .count = total_tokens * ffn_dim };
            self.ctx.cmdDispatch(
                self.gelu_pipe,
                &[_]gpu.GpuBuffer{self.ffn_buf},
                std.mem.asBytes(&pc),
                (pc.count + 255) / 256, 1, 1,
            );
        }
        self.recordGemm(self.ffn_buf, lw.ff2_weight, self.buf_a, lw.ff2_bias, total_tokens, hidden, ffn_dim);
        self.recordAdd(self.buf_a, self.q_buf, elem_count);
        self.recordLayerNorm(total_tokens, self.buf_a, lw.ff_ln_gamma, lw.ff_ln_beta);
    }

    // ====================================================================
    // Dispatch Helpers
    // ====================================================================

    fn recordGemm(self: *Self, a: gpu.GpuBuffer, b: gpu.GpuBuffer, c: gpu.GpuBuffer, bias: gpu.GpuBuffer, m: u32, n: u32, k: u32) void {
        const pc = SgemmBiasPC{ .m = m, .n = n, .k = k };
        self.ctx.cmdDispatch(
            self.sgemm_bias_pipe,
            &[_]gpu.GpuBuffer{ a, b, c, bias },
            std.mem.asBytes(&pc),
            (n + 63) / 64, (m + 63) / 64, 1,
        );
    }

    fn recordLayerNorm(self: *Self, seq_len: u32, data: gpu.GpuBuffer, gamma_buf: gpu.GpuBuffer, beta_buf: gpu.GpuBuffer) void {
        const pc = LayerNormPC{ .rows = seq_len, .cols = self.config.hidden_dim, .eps = 1e-12 };
        self.ctx.cmdDispatch(
            self.layernorm_pipe,
            &[_]gpu.GpuBuffer{ data, gamma_buf, beta_buf },
            std.mem.asBytes(&pc),
            seq_len, 1, 1,
        );
    }

    fn recordAdd(self: *Self, a: gpu.GpuBuffer, b: gpu.GpuBuffer, count: u32) void {
        const pc = ElementPC{ .count = count };
        self.ctx.cmdDispatch(
            self.residual_add_pipe,
            &[_]gpu.GpuBuffer{ a, b },
            std.mem.asBytes(&pc),
            (count + 255) / 256, 1, 1,
        );
    }

    // ====================================================================
    // Cleanup
    // ====================================================================

    pub fn deinit(self: *Self) void {
        // Batch-specific buffers
        self.ctx.destroyBuffer(&self.batch_output_buf);
        self.ctx.destroyBuffer(&self.lengths_buf);
        self.ctx.destroyBuffer(&self.offsets_buf);

        // Activation buffers
        self.ctx.destroyBuffer(&self.output_buf);
        self.ctx.destroyBuffer(&self.ids_buf);
        self.ctx.destroyBuffer(&self.ffn_buf);
        self.ctx.destroyBuffer(&self.v_buf);
        self.ctx.destroyBuffer(&self.k_buf);
        self.ctx.destroyBuffer(&self.q_buf);
        self.ctx.destroyBuffer(&self.buf_b);
        self.ctx.destroyBuffer(&self.buf_a);

        // Layer weights
        for (self.layers) |*lw| {
            inline for (std.meta.fields(LayerGpuWeights)) |field| {
                self.ctx.destroyBuffer(&@field(lw, field.name));
            }
        }
        self.allocator.free(self.layers);

        // Embedding weights
        self.ctx.destroyBuffer(&self.embed_ln_beta_buf);
        self.ctx.destroyBuffer(&self.embed_ln_gamma_buf);
        self.ctx.destroyBuffer(&self.type_emb_buf);
        self.ctx.destroyBuffer(&self.pos_emb_buf);
        self.ctx.destroyBuffer(&self.word_emb_buf);

        // Pipelines
        self.ctx.destroyPipeline(&self.pool_normalize_batch_pipe);
        self.ctx.destroyPipeline(&self.attention_batch_pipe);
        self.ctx.destroyPipeline(&self.pool_normalize_pipe);
        self.ctx.destroyPipeline(&self.embedding_pipe);
        self.ctx.destroyPipeline(&self.attention_pipe);
        self.ctx.destroyPipeline(&self.residual_add_pipe);
        self.ctx.destroyPipeline(&self.gelu_pipe);
        self.ctx.destroyPipeline(&self.layernorm_pipe);
        self.ctx.destroyPipeline(&self.sgemm_bias_pipe);

        // Context
        self.ctx.deinit();
    }
};

// ============================================================================
// Helpers
// ============================================================================

fn uploadF32(ctx: *gpu.WebGpuContext, data: []const f32) WebGpuForwardError!gpu.GpuBuffer {
    const size = data.len * @sizeOf(f32);
    const buf = ctx.createStorageBuffer(size, false) catch return WebGpuForwardError.BufferCreationFailed;
    ctx.uploadToBuffer(&buf, std.mem.sliceAsBytes(data)) catch return WebGpuForwardError.UploadFailed;
    return buf;
}

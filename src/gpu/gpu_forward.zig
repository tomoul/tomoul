// src/gpu/gpu_forward.zig
//
// GPU Forward Pass for Transformer Encoder
//
// Dispatches full encoder pipeline via Vulkan compute shaders.
// Weight data uploaded once at init; only token IDs transferred per inference.
//
// Dispatch sequence per forward():
//   1. Embedding lookup + sum       (embedding_lookup.comp)
//   2. Embedding LayerNorm           (layernorm.comp)
//   3. For each of N layers:
//      a. Q/K/V projections ×3       (sgemm_bias.comp)
//      b. Multi-head attention        (attention.comp)
//      c. Output projection           (sgemm_bias.comp)
//      d. Residual add                (residual_add.comp)
//      e. Attention LayerNorm         (layernorm.comp)
//      f. FFN up projection           (sgemm_bias.comp)
//      g. GELU activation             (gelu.comp)
//      h. FFN down projection         (sgemm_bias.comp)
//      i. Residual add                (residual_add.comp)
//      j. FFN LayerNorm               (layernorm.comp)
//   4. Mean pooling + L2 normalize   (pool_normalize.comp)
//
// All dispatches batched into a single command buffer with memory barriers.

const std = @import("std");
const gpu = @import("vulkan");

// Embedded SPIR-V shaders (compiled at build time, no external files needed)
const spv_sgemm_bias = @embedFile("shaders/vulkan/sgemm_bias.spv");
const spv_layernorm = @embedFile("shaders/vulkan/layernorm.spv");
const spv_gelu = @embedFile("shaders/vulkan/gelu.spv");
const spv_residual_add = @embedFile("shaders/vulkan/residual_add.spv");
const spv_attention = @embedFile("shaders/vulkan/attention.spv");
const spv_embedding_lookup = @embedFile("shaders/vulkan/embedding_lookup.spv");
const spv_pool_normalize = @embedFile("shaders/vulkan/pool_normalize.spv");
const spv_attention_batch = @embedFile("shaders/vulkan/attention_batch.spv");
const spv_pool_normalize_batch = @embedFile("shaders/vulkan/pool_normalize_batch.spv");

// ============================================================================
// Push Constant Structs (must match GLSL shader layouts exactly)
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
// Public Configuration Types
// ============================================================================

pub const GpuConfig = struct {
    hidden_dim: u32,
    num_heads: u32,
    head_dim: u32,
    ffn_dim: u32,
    num_layers: u32,
    vocab_size: u32,
    max_seq_len: u32,
    /// Max total tokens for batch forward (default = max_seq_len).
    /// Set higher for batched inference, e.g. 32 * avg_seq_len.
    max_batch_tokens: u32 = 0, // 0 means use max_seq_len
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

pub const GpuForwardError = error{
    VulkanInitFailed,
} || gpu.VulkanError || std.mem.Allocator.Error;

// ============================================================================
// GPU Forward Pass Context
// ============================================================================

pub const GpuForward = struct {
    allocator: std.mem.Allocator,
    ctx: gpu.VulkanContext,
    config: GpuConfig,

    // Compute pipelines (one per shader type, reused across dispatches)
    sgemm_bias_pipe: gpu.ComputePipeline,
    layernorm_pipe: gpu.ComputePipeline,
    gelu_pipe: gpu.ComputePipeline,
    residual_add_pipe: gpu.ComputePipeline,
    attention_pipe: gpu.ComputePipeline,
    embedding_pipe: gpu.ComputePipeline,
    pool_normalize_pipe: gpu.ComputePipeline,
    // Batch-specific pipelines
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
    buf_a: gpu.GpuBuffer, // [max_seq, hidden] — primary I/O
    buf_b: gpu.GpuBuffer, // [max_seq, hidden] — attention output
    q_buf: gpu.GpuBuffer, // [max_seq, hidden] — Q projection / output proj / workspace
    k_buf: gpu.GpuBuffer, // [max_seq, hidden] — K projection
    v_buf: gpu.GpuBuffer, // [max_seq, hidden] — V projection
    ffn_buf: gpu.GpuBuffer, // [max_seq, ffn_dim] — FFN intermediate
    ids_buf: gpu.GpuBuffer, // [3 * max_seq] u32 — token/position/type IDs
    output_buf: gpu.GpuBuffer, // [hidden] f32 — final pooled+normalized output
    // Batch-specific buffers
    offsets_buf: gpu.GpuBuffer, // [max_batch] u32 — per-sentence start offsets
    lengths_buf: gpu.GpuBuffer, // [max_batch] u32 — per-sentence token counts
    batch_output_buf: gpu.GpuBuffer, // [max_batch * hidden] f32 — batched output
    max_batch_tokens: u32, // max total tokens across all sentences in a batch

    const Self = @This();

    // ====================================================================
    // Init
    // ====================================================================

    pub fn init(
        allocator: std.mem.Allocator,
        config: GpuConfig,
        embeddings: EmbeddingData,
        layer_weights: []const LayerData,
    ) GpuForwardError!Self {
        var self: Self = undefined;
        self.allocator = allocator;
        self.config = config;

        // 1. Init Vulkan context
        self.ctx = gpu.VulkanContext.init(allocator) catch |e| {
            std.debug.print("GPU: Vulkan init failed: {}\n", .{e});
            return e;
        };
        errdefer self.ctx.deinit();

        std.debug.print("GPU: Using device: {s}\n", .{self.ctx.getDeviceName()});

        // 2. Create compute pipelines from embedded SPIR-V shaders (no external files)
        self.sgemm_bias_pipe = try self.ctx.createComputePipeline(spv_sgemm_bias, 4, @sizeOf(SgemmBiasPC));
        self.layernorm_pipe = try self.ctx.createComputePipeline(spv_layernorm, 3, @sizeOf(LayerNormPC));
        self.gelu_pipe = try self.ctx.createComputePipeline(spv_gelu, 1, @sizeOf(ElementPC));
        self.residual_add_pipe = try self.ctx.createComputePipeline(spv_residual_add, 2, @sizeOf(ElementPC));
        self.attention_pipe = try self.ctx.createComputePipeline(spv_attention, 4, @sizeOf(AttentionPC));
        self.embedding_pipe = try self.ctx.createComputePipeline(spv_embedding_lookup, 5, @sizeOf(EmbeddingPC));
        self.pool_normalize_pipe = try self.ctx.createComputePipeline(spv_pool_normalize, 2, @sizeOf(PoolNormPC));
        // Batch-specific pipelines
        self.attention_batch_pipe = try self.ctx.createComputePipeline(spv_attention_batch, 5, @sizeOf(AttentionBatchPC));
        self.pool_normalize_batch_pipe = try self.ctx.createComputePipeline(spv_pool_normalize_batch, 4, @sizeOf(PoolNormBatchPC));

        // 3. Upload embedding weights
        self.word_emb_buf = try uploadF32(&self.ctx, embeddings.word_emb);
        self.pos_emb_buf = try uploadF32(&self.ctx, embeddings.pos_emb);
        self.type_emb_buf = try uploadF32(&self.ctx, embeddings.type_emb);
        self.embed_ln_gamma_buf = try uploadF32(&self.ctx, embeddings.ln_gamma);
        self.embed_ln_beta_buf = try uploadF32(&self.ctx, embeddings.ln_beta);

        // 4. Upload per-layer weights
        self.layers = try allocator.alloc(LayerGpuWeights, config.num_layers);
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

        // 5. Allocate activation buffers (sized for batch if max_batch_tokens > max_seq_len)
        const max_tokens = if (config.max_batch_tokens > 0) config.max_batch_tokens else config.max_seq_len;
        self.max_batch_tokens = max_tokens;
        const hidden = config.hidden_dim;
        const ffn = config.ffn_dim;
        const act_size = max_tokens * hidden * @sizeOf(f32);

        self.buf_a = try self.ctx.createStorageBuffer(act_size, true);
        self.buf_b = try self.ctx.createStorageBuffer(act_size, true);
        self.q_buf = try self.ctx.createStorageBuffer(act_size, true);
        self.k_buf = try self.ctx.createStorageBuffer(act_size, true);
        self.v_buf = try self.ctx.createStorageBuffer(act_size, true);
        self.ffn_buf = try self.ctx.createStorageBuffer(max_tokens * ffn * @sizeOf(f32), true);
        self.ids_buf = try self.ctx.createStorageBuffer(3 * max_tokens * @sizeOf(u32), true);
        self.output_buf = try self.ctx.createStorageBuffer(hidden * @sizeOf(f32), true);

        // Batch-specific buffers
        const max_batch: u32 = 64; // max batch size
        self.offsets_buf = try self.ctx.createStorageBuffer(max_batch * @sizeOf(u32), true);
        self.lengths_buf = try self.ctx.createStorageBuffer(max_batch * @sizeOf(u32), true);
        self.batch_output_buf = try self.ctx.createStorageBuffer(max_batch * hidden * @sizeOf(f32), true);

        const weight_count = 5 + config.num_layers * 16;
        std.debug.print("GPU: Uploaded {} weight buffers, {} activation buffers\n", .{ weight_count, 8 });

        return self;
    }

    // ====================================================================
    // Forward Pass
    // ====================================================================

    pub fn forward(self: *Self, token_ids: []const u32, output: []f32) GpuForwardError!void {
        const seq_len: u32 = @intCast(token_ids.len);
        const hidden = self.config.hidden_dim;

        // Prepare IDs buffer: [token_ids | position_ids (0..N-1) | type_ids (all 0)]
        const ids = try self.allocator.alloc(u32, seq_len * 3);
        defer self.allocator.free(ids);
        @memcpy(ids[0..seq_len], token_ids);
        for (0..seq_len) |i| ids[seq_len + i] = @intCast(i);
        @memset(ids[2 * seq_len .. 3 * seq_len], 0);
        try self.ctx.uploadToBuffer(&self.ids_buf, std.mem.sliceAsBytes(ids));

        // Reset descriptor pool (frees all sets from previous forward call)
        self.ctx.resetDescriptorPool();

        // Begin batched command buffer recording
        try self.ctx.beginCommandBuffer();

        // --- Embedding lookup ---
        {
            const desc = try self.ctx.allocateDescriptorSet(&self.embedding_pipe);
            try self.ctx.bindBuffers(desc, &[_]gpu.GpuBuffer{
                self.ids_buf,      self.word_emb_buf, self.pos_emb_buf,
                self.type_emb_buf, self.buf_a,
            });
            const pc = EmbeddingPC{ .seq_len = seq_len, .hidden_dim = hidden };
            self.ctx.cmdDispatch(&self.embedding_pipe, desc, (seq_len * hidden + 255) / 256, 1, 1, std.mem.asBytes(&pc));
        }

        // --- Embedding LayerNorm ---
        try self.recordLayerNorm(seq_len, self.buf_a, self.embed_ln_gamma_buf, self.embed_ln_beta_buf);

        // --- Transformer Layers ---
        for (0..self.config.num_layers) |i| {
            try self.recordLayer(@intCast(i), seq_len);
        }

        // --- Pool + Normalize ---
        {
            const desc = try self.ctx.allocateDescriptorSet(&self.pool_normalize_pipe);
            try self.ctx.bindBuffers(desc, &[_]gpu.GpuBuffer{ self.buf_a, self.output_buf });
            const pc = PoolNormPC{ .seq_len = seq_len, .hidden_dim = hidden };
            self.ctx.cmdDispatch(&self.pool_normalize_pipe, desc, 1, 1, 1, std.mem.asBytes(&pc));
        }

        // Submit all dispatches and wait
        try self.ctx.submitAndWait();

        // Readback final embedding
        try self.ctx.readbackFromBuffer(&self.output_buf, std.mem.sliceAsBytes(output[0..hidden]));
    }

    // ====================================================================
    // Batched Forward Pass (multiple sentences, single command buffer)
    // ====================================================================

    /// Process a batch of tokenized sentences in a single GPU submission.
    /// `batch_ids` is an array of token ID slices (one per sentence).
    /// `output` must be batch_ids.len * hidden_dim floats.
    pub fn forwardBatch(self: *Self, batch_ids: []const []const u32, output: []f32) GpuForwardError!void {
        const batch_size: u32 = @intCast(batch_ids.len);
        const hidden = self.config.hidden_dim;

        // Compute total tokens and per-sentence offsets/lengths
        var total_tokens: u32 = 0;
        const offsets = try self.allocator.alloc(u32, batch_size);
        defer self.allocator.free(offsets);
        const lengths = try self.allocator.alloc(u32, batch_size);
        defer self.allocator.free(lengths);

        for (batch_ids, 0..) |ids, i| {
            offsets[i] = total_tokens;
            lengths[i] = @intCast(ids.len);
            total_tokens += @intCast(ids.len);
        }

        if (total_tokens > self.max_batch_tokens) {
            std.debug.print("GPU: batch total tokens {} exceeds max {}\n", .{ total_tokens, self.max_batch_tokens });
            return error.VulkanInitFailed;
        }

        // Pack all token IDs with position and type IDs
        const all_ids = try self.allocator.alloc(u32, total_tokens * 3);
        defer self.allocator.free(all_ids);
        var pos: u32 = 0;
        for (batch_ids, 0..) |ids, i| {
            const len: u32 = @intCast(ids.len);
            @memcpy(all_ids[pos .. pos + len], ids);
            // Position IDs: 0..len-1 per sentence (reset per sentence)
            for (0..len) |j| all_ids[total_tokens + pos + j] = @intCast(j);
            // Type IDs: all 0
            @memset(all_ids[2 * total_tokens + pos .. 2 * total_tokens + pos + len], 0);
            _ = i;
            pos += len;
        }
        try self.ctx.uploadToBuffer(&self.ids_buf, std.mem.sliceAsBytes(all_ids));

        // Upload offsets and lengths for batch-aware shaders
        try self.ctx.uploadToBuffer(&self.offsets_buf, std.mem.sliceAsBytes(offsets));
        try self.ctx.uploadToBuffer(&self.lengths_buf, std.mem.sliceAsBytes(lengths));

        // Reset descriptor pool and begin recording
        self.ctx.resetDescriptorPool();
        try self.ctx.beginCommandBuffer();

        // --- Embedding lookup (works on packed tokens directly) ---
        {
            const desc = try self.ctx.allocateDescriptorSet(&self.embedding_pipe);
            try self.ctx.bindBuffers(desc, &[_]gpu.GpuBuffer{
                self.ids_buf,      self.word_emb_buf, self.pos_emb_buf,
                self.type_emb_buf, self.buf_a,
            });
            const pc = EmbeddingPC{ .seq_len = total_tokens, .hidden_dim = hidden };
            self.ctx.cmdDispatch(&self.embedding_pipe, desc, (total_tokens * hidden + 255) / 256, 1, 1, std.mem.asBytes(&pc));
        }

        // --- Embedding LayerNorm (element-wise, works on packed) ---
        try self.recordLayerNorm(total_tokens, self.buf_a, self.embed_ln_gamma_buf, self.embed_ln_beta_buf);

        // --- Transformer Layers (with batch-aware attention) ---
        for (0..self.config.num_layers) |i| {
            try self.recordLayerBatch(@intCast(i), total_tokens, batch_size);
        }

        // --- Batched Pool + Normalize ---
        {
            const desc = try self.ctx.allocateDescriptorSet(&self.pool_normalize_batch_pipe);
            try self.ctx.bindBuffers(desc, &[_]gpu.GpuBuffer{
                self.buf_a, self.batch_output_buf, self.offsets_buf, self.lengths_buf,
            });
            const pc = PoolNormBatchPC{ .batch_size = batch_size, .hidden_dim = hidden };
            self.ctx.cmdDispatch(&self.pool_normalize_batch_pipe, desc, batch_size, 1, 1, std.mem.asBytes(&pc));
        }

        // Submit and wait
        try self.ctx.submitAndWait();

        // Readback all embeddings
        const out_bytes = batch_size * hidden * @sizeOf(f32);
        try self.ctx.readbackFromBuffer(&self.batch_output_buf, std.mem.sliceAsBytes(output[0 .. batch_size * hidden]));
        _ = out_bytes;
    }

    // ====================================================================
    // Per-Layer Dispatch
    // ====================================================================

    fn recordLayer(self: *Self, layer_idx: u32, seq_len: u32) GpuForwardError!void {
        const lw = &self.layers[layer_idx];
        const hidden = self.config.hidden_dim;
        const ffn_dim = self.config.ffn_dim;
        const elem_count = seq_len * hidden;

        // Q, K, V projections: buf_a @ W + b → q_buf, k_buf, v_buf
        try self.recordGemm(self.buf_a, lw.q_weight, self.q_buf, lw.q_bias, seq_len, hidden, hidden);
        try self.recordGemm(self.buf_a, lw.k_weight, self.k_buf, lw.k_bias, seq_len, hidden, hidden);
        try self.recordGemm(self.buf_a, lw.v_weight, self.v_buf, lw.v_bias, seq_len, hidden, hidden);

        // Multi-head attention: q_buf, k_buf, v_buf → buf_b
        {
            const desc = try self.ctx.allocateDescriptorSet(&self.attention_pipe);
            try self.ctx.bindBuffers(desc, &[_]gpu.GpuBuffer{
                self.q_buf, self.k_buf, self.v_buf, self.buf_b,
            });
            const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.config.head_dim)));
            const pc = AttentionPC{
                .seq_len = seq_len,
                .num_heads = self.config.num_heads,
                .head_dim = self.config.head_dim,
                .scale = scale,
            };
            self.ctx.cmdDispatch(&self.attention_pipe, desc, self.config.num_heads, seq_len, 1, std.mem.asBytes(&pc));
        }

        // Output projection: buf_b @ W_o + b_o → q_buf (reuse)
        try self.recordGemm(self.buf_b, lw.o_weight, self.q_buf, lw.o_bias, seq_len, hidden, hidden);

        // Residual: q_buf += buf_a
        try self.recordAdd(self.q_buf, self.buf_a, elem_count);

        // Attention LayerNorm: q_buf in-place
        try self.recordLayerNorm(seq_len, self.q_buf, lw.attn_ln_gamma, lw.attn_ln_beta);

        // FFN up: q_buf @ W_up + b_up → ffn_buf
        try self.recordGemm(self.q_buf, lw.ff1_weight, self.ffn_buf, lw.ff1_bias, seq_len, ffn_dim, hidden);

        // GELU: ffn_buf in-place
        {
            const desc = try self.ctx.allocateDescriptorSet(&self.gelu_pipe);
            try self.ctx.bindBuffers(desc, &[_]gpu.GpuBuffer{self.ffn_buf});
            const pc = ElementPC{ .count = seq_len * ffn_dim };
            self.ctx.cmdDispatch(&self.gelu_pipe, desc, (pc.count + 255) / 256, 1, 1, std.mem.asBytes(&pc));
        }

        // FFN down: ffn_buf @ W_down + b_down → buf_a
        try self.recordGemm(self.ffn_buf, lw.ff2_weight, self.buf_a, lw.ff2_bias, seq_len, hidden, ffn_dim);

        // Residual: buf_a += q_buf
        try self.recordAdd(self.buf_a, self.q_buf, elem_count);

        // FFN LayerNorm: buf_a in-place
        try self.recordLayerNorm(seq_len, self.buf_a, lw.ff_ln_gamma, lw.ff_ln_beta);
    }

    /// Per-layer dispatch for batched forward — same as recordLayer but uses
    /// batch-aware attention that respects sentence boundaries.
    fn recordLayerBatch(self: *Self, layer_idx: u32, total_tokens: u32, batch_size: u32) GpuForwardError!void {
        const lw = &self.layers[layer_idx];
        const hidden = self.config.hidden_dim;
        const ffn_dim = self.config.ffn_dim;
        const elem_count = total_tokens * hidden;

        // Q, K, V projections (GEMM treats packed tokens as rows — transparent)
        try self.recordGemm(self.buf_a, lw.q_weight, self.q_buf, lw.q_bias, total_tokens, hidden, hidden);
        try self.recordGemm(self.buf_a, lw.k_weight, self.k_buf, lw.k_bias, total_tokens, hidden, hidden);
        try self.recordGemm(self.buf_a, lw.v_weight, self.v_buf, lw.v_bias, total_tokens, hidden, hidden);

        // Batch-aware multi-head attention (uses offsets to prevent cross-sentence attention)
        {
            const desc = try self.ctx.allocateDescriptorSet(&self.attention_batch_pipe);
            try self.ctx.bindBuffers(desc, &[_]gpu.GpuBuffer{
                self.q_buf, self.k_buf, self.v_buf, self.buf_b, self.offsets_buf,
            });
            const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.config.head_dim)));
            const pc = AttentionBatchPC{
                .total_tokens = total_tokens,
                .num_heads = self.config.num_heads,
                .head_dim = self.config.head_dim,
                .scale = scale,
                .batch_size = batch_size,
            };
            self.ctx.cmdDispatch(&self.attention_batch_pipe, desc, self.config.num_heads, total_tokens, 1, std.mem.asBytes(&pc));
        }

        // Output projection, residual, LayerNorm, FFN — all element-wise/per-row, work on packed
        try self.recordGemm(self.buf_b, lw.o_weight, self.q_buf, lw.o_bias, total_tokens, hidden, hidden);
        try self.recordAdd(self.q_buf, self.buf_a, elem_count);
        try self.recordLayerNorm(total_tokens, self.q_buf, lw.attn_ln_gamma, lw.attn_ln_beta);
        try self.recordGemm(self.q_buf, lw.ff1_weight, self.ffn_buf, lw.ff1_bias, total_tokens, ffn_dim, hidden);
        {
            const desc = try self.ctx.allocateDescriptorSet(&self.gelu_pipe);
            try self.ctx.bindBuffers(desc, &[_]gpu.GpuBuffer{self.ffn_buf});
            const pc = ElementPC{ .count = total_tokens * ffn_dim };
            self.ctx.cmdDispatch(&self.gelu_pipe, desc, (pc.count + 255) / 256, 1, 1, std.mem.asBytes(&pc));
        }
        try self.recordGemm(self.ffn_buf, lw.ff2_weight, self.buf_a, lw.ff2_bias, total_tokens, hidden, ffn_dim);
        try self.recordAdd(self.buf_a, self.q_buf, elem_count);
        try self.recordLayerNorm(total_tokens, self.buf_a, lw.ff_ln_gamma, lw.ff_ln_beta);
    }

    // ====================================================================
    // Dispatch Helpers
    // ====================================================================

    fn recordGemm(self: *Self, a: gpu.GpuBuffer, b: gpu.GpuBuffer, c: gpu.GpuBuffer, bias: gpu.GpuBuffer, m: u32, n: u32, k: u32) GpuForwardError!void {
        const desc = try self.ctx.allocateDescriptorSet(&self.sgemm_bias_pipe);
        try self.ctx.bindBuffers(desc, &[_]gpu.GpuBuffer{ a, b, c, bias });
        const pc = SgemmBiasPC{ .m = m, .n = n, .k = k };
        self.ctx.cmdDispatch(&self.sgemm_bias_pipe, desc, (n + 63) / 64, (m + 63) / 64, 1, std.mem.asBytes(&pc));
    }

    fn recordLayerNorm(self: *Self, seq_len: u32, data: gpu.GpuBuffer, gamma: gpu.GpuBuffer, beta: gpu.GpuBuffer) GpuForwardError!void {
        const desc = try self.ctx.allocateDescriptorSet(&self.layernorm_pipe);
        try self.ctx.bindBuffers(desc, &[_]gpu.GpuBuffer{ data, gamma, beta });
        const pc = LayerNormPC{ .rows = seq_len, .cols = self.config.hidden_dim, .eps = 1e-12 };
        self.ctx.cmdDispatch(&self.layernorm_pipe, desc, seq_len, 1, 1, std.mem.asBytes(&pc));
    }

    fn recordAdd(self: *Self, a: gpu.GpuBuffer, b: gpu.GpuBuffer, count: u32) GpuForwardError!void {
        const desc = try self.ctx.allocateDescriptorSet(&self.residual_add_pipe);
        try self.ctx.bindBuffers(desc, &[_]gpu.GpuBuffer{ a, b });
        const pc = ElementPC{ .count = count };
        self.ctx.cmdDispatch(&self.residual_add_pipe, desc, (count + 255) / 256, 1, 1, std.mem.asBytes(&pc));
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

        // Vulkan context
        self.ctx.deinit();
    }
};

// ============================================================================
// File Helpers
// ============================================================================

fn uploadF32(ctx: *gpu.VulkanContext, data: []const f32) !gpu.GpuBuffer {
    const size = data.len * @sizeOf(f32);
    var buf = try ctx.createStorageBuffer(size, true);
    try ctx.uploadToBuffer(&buf, std.mem.sliceAsBytes(data));
    return buf;
}

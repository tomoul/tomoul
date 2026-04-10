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

// ============================================================================
// Push Constant Structs (must match GLSL shader layouts exactly)
// ============================================================================

const SgemmBiasPC = extern struct { m: u32, n: u32, k: u32 };
const LayerNormPC = extern struct { rows: u32, cols: u32, eps: f32 };
const ElementPC = extern struct { count: u32 };
const AttentionPC = extern struct { seq_len: u32, num_heads: u32, head_dim: u32, scale: f32 };
const EmbeddingPC = extern struct { seq_len: u32, hidden_dim: u32 };
const PoolNormPC = extern struct { seq_len: u32, hidden_dim: u32 };

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
    ShaderLoadFailed,
    VulkanInitFailed,
} || gpu.VulkanError || std.mem.Allocator.Error || std.fs.File.OpenError || std.fs.File.ReadError || std.fs.File.StatError;

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

        // 2. Create compute pipelines from SPIR-V shaders
        self.sgemm_bias_pipe = try loadPipeline(&self.ctx, allocator, "src/gpu/shaders/sgemm_bias.spv", 4, @sizeOf(SgemmBiasPC));
        self.layernorm_pipe = try loadPipeline(&self.ctx, allocator, "src/gpu/shaders/layernorm.spv", 3, @sizeOf(LayerNormPC));
        self.gelu_pipe = try loadPipeline(&self.ctx, allocator, "src/gpu/shaders/gelu.spv", 1, @sizeOf(ElementPC));
        self.residual_add_pipe = try loadPipeline(&self.ctx, allocator, "src/gpu/shaders/residual_add.spv", 2, @sizeOf(ElementPC));
        self.attention_pipe = try loadPipeline(&self.ctx, allocator, "src/gpu/shaders/attention.spv", 4, @sizeOf(AttentionPC));
        self.embedding_pipe = try loadPipeline(&self.ctx, allocator, "src/gpu/shaders/embedding_lookup.spv", 5, @sizeOf(EmbeddingPC));
        self.pool_normalize_pipe = try loadPipeline(&self.ctx, allocator, "src/gpu/shaders/pool_normalize.spv", 2, @sizeOf(PoolNormPC));

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

        // 5. Allocate activation buffers
        const max_seq = config.max_seq_len;
        const hidden = config.hidden_dim;
        const ffn = config.ffn_dim;
        const act_size = max_seq * hidden * @sizeOf(f32);

        self.buf_a = try self.ctx.createStorageBuffer(act_size, true);
        self.buf_b = try self.ctx.createStorageBuffer(act_size, true);
        self.q_buf = try self.ctx.createStorageBuffer(act_size, true);
        self.k_buf = try self.ctx.createStorageBuffer(act_size, true);
        self.v_buf = try self.ctx.createStorageBuffer(act_size, true);
        self.ffn_buf = try self.ctx.createStorageBuffer(max_seq * ffn * @sizeOf(f32), true);
        self.ids_buf = try self.ctx.createStorageBuffer(3 * max_seq * @sizeOf(u32), true);
        self.output_buf = try self.ctx.createStorageBuffer(hidden * @sizeOf(f32), true);

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
                self.ids_buf, self.word_emb_buf, self.pos_emb_buf,
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

fn loadPipeline(
    ctx: *gpu.VulkanContext,
    allocator: std.mem.Allocator,
    path: []const u8,
    num_buffers: u32,
    pc_size: u32,
) !gpu.ComputePipeline {
    const file = std.fs.cwd().openFile(path, .{}) catch |e| {
        std.debug.print("GPU: Failed to open shader: {s}: {}\n", .{ path, e });
        return e;
    };
    defer file.close();
    const stat = try file.stat();
    const data = try allocator.alloc(u8, stat.size);
    defer allocator.free(data);
    const bytes = try file.readAll(data);
    return ctx.createComputePipeline(data[0..bytes], num_buffers, pc_size);
}

fn uploadF32(ctx: *gpu.VulkanContext, data: []const f32) !gpu.GpuBuffer {
    const size = data.len * @sizeOf(f32);
    var buf = try ctx.createStorageBuffer(size, true);
    try ctx.uploadToBuffer(&buf, std.mem.sliceAsBytes(data));
    return buf;
}

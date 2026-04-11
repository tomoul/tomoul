// src/gpu/metal_forward.zig
//
// GPU Forward Pass for Transformer Encoder (Metal Backend)
//
// Dispatches full encoder pipeline via Metal compute shaders.
// Weight data uploaded once at init; only token IDs transferred per inference.
//
// Dispatch sequence per forward():
//   1. Embedding lookup + sum       (embedding_lookup)
//   2. Embedding LayerNorm           (layernorm)
//   3. For each of N layers:
//      a. Q/K/V projections ×3       (sgemm_bias)
//      b. Multi-head attention        (attention)
//      c. Output projection           (sgemm_bias)
//      d. Residual add                (residual_add)
//      e. Attention LayerNorm         (layernorm)
//      f. FFN up projection           (sgemm_bias)
//      g. GELU activation             (gelu)
//      h. FFN down projection         (sgemm_bias)
//      i. Residual add                (residual_add)
//      j. FFN LayerNorm               (layernorm)
//   4. Mean pooling + L2 normalize   (pool_normalize)
//
// All dispatches batched into a single command buffer with memory barriers.

const std = @import("std");
const mtl = @import("metal");

// Embedded MSL shader source (compiled at runtime, no external files needed)
const msl_source = @embedFile("shaders/metal/kernels.metal");

// ============================================================================
// Push Constant Structs (must match MSL shader parameter structs exactly)
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
// Public Configuration Types (identical layout to gpu_forward.zig)
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
    q_weight: mtl.MetalBuffer,
    q_bias: mtl.MetalBuffer,
    k_weight: mtl.MetalBuffer,
    k_bias: mtl.MetalBuffer,
    v_weight: mtl.MetalBuffer,
    v_bias: mtl.MetalBuffer,
    o_weight: mtl.MetalBuffer,
    o_bias: mtl.MetalBuffer,
    ff1_weight: mtl.MetalBuffer,
    ff1_bias: mtl.MetalBuffer,
    ff2_weight: mtl.MetalBuffer,
    ff2_bias: mtl.MetalBuffer,
    attn_ln_gamma: mtl.MetalBuffer,
    attn_ln_beta: mtl.MetalBuffer,
    ff_ln_gamma: mtl.MetalBuffer,
    ff_ln_beta: mtl.MetalBuffer,
};

pub const MetalForwardError = error{
    MetalNotAvailable,
    DeviceNotFound,
    CommandQueueCreationFailed,
    LibraryCompilationFailed,
    FunctionNotFound,
    PipelineCreationFailed,
    BufferCreationFailed,
    CommandBufferCreationFailed,
    EncoderCreationFailed,
    NotRecording,
    OutOfMemory,
    BatchTooLarge,
};

// ============================================================================
// Metal Forward Pass Context
// ============================================================================

pub const MetalForward = struct {
    allocator: std.mem.Allocator,
    ctx: mtl.MetalContext,
    config: GpuConfig,

    // Compute pipelines (one per kernel function, reused across dispatches)
    sgemm_bias_pipe: mtl.MetalPipeline,
    layernorm_pipe: mtl.MetalPipeline,
    gelu_pipe: mtl.MetalPipeline,
    residual_add_pipe: mtl.MetalPipeline,
    attention_pipe: mtl.MetalPipeline,
    embedding_pipe: mtl.MetalPipeline,
    pool_normalize_pipe: mtl.MetalPipeline,
    attention_batch_pipe: mtl.MetalPipeline,
    pool_normalize_batch_pipe: mtl.MetalPipeline,

    // Embedding weights (persistent on GPU)
    word_emb_buf: mtl.MetalBuffer,
    pos_emb_buf: mtl.MetalBuffer,
    type_emb_buf: mtl.MetalBuffer,
    embed_ln_gamma_buf: mtl.MetalBuffer,
    embed_ln_beta_buf: mtl.MetalBuffer,

    // Per-layer weights (persistent on GPU)
    layers: []LayerGpuWeights,

    // Activation buffers (reused across layers)
    buf_a: mtl.MetalBuffer, // [max_seq, hidden] — primary I/O
    buf_b: mtl.MetalBuffer, // [max_seq, hidden] — attention output
    q_buf: mtl.MetalBuffer, // [max_seq, hidden] — Q projection / workspace
    k_buf: mtl.MetalBuffer, // [max_seq, hidden] — K projection
    v_buf: mtl.MetalBuffer, // [max_seq, hidden] — V projection
    ffn_buf: mtl.MetalBuffer, // [max_seq, ffn_dim] — FFN intermediate
    ids_buf: mtl.MetalBuffer, // [3 * max_seq] u32 — token/position/type IDs
    output_buf: mtl.MetalBuffer, // [hidden] f32 — final pooled+normalized output

    // Batch-specific buffers
    offsets_buf: mtl.MetalBuffer, // [max_batch] u32
    lengths_buf: mtl.MetalBuffer, // [max_batch] u32
    batch_output_buf: mtl.MetalBuffer, // [max_batch * hidden] f32
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
    ) MetalForwardError!Self {
        var self: Self = undefined;
        self.allocator = allocator;
        self.config = config;

        // 1. Init Metal context (dlopen Metal.framework, get device, compile shaders)
        self.ctx = mtl.MetalContext.init(msl_source) catch |e| {
            std.debug.print("GPU: Metal init failed: {}\n", .{e});
            return @errorCast(e);
        };
        errdefer self.ctx.deinit();

        std.debug.print("GPU: Using device: {s}\n", .{self.ctx.getDeviceName()});

        // 2. Create compute pipelines for each kernel
        self.sgemm_bias_pipe = try self.ctx.createComputePipeline("sgemm_bias");
        self.layernorm_pipe = try self.ctx.createComputePipeline("layernorm");
        self.gelu_pipe = try self.ctx.createComputePipeline("gelu");
        self.residual_add_pipe = try self.ctx.createComputePipeline("residual_add");
        self.attention_pipe = try self.ctx.createComputePipeline("attention");
        self.embedding_pipe = try self.ctx.createComputePipeline("embedding_lookup");
        self.pool_normalize_pipe = try self.ctx.createComputePipeline("pool_normalize");
        self.attention_batch_pipe = try self.ctx.createComputePipeline("attention_batch");
        self.pool_normalize_batch_pipe = try self.ctx.createComputePipeline("pool_normalize_batch");

        // 3. Upload embedding weights
        self.word_emb_buf = try uploadF32(&self.ctx, embeddings.word_emb);
        self.pos_emb_buf = try uploadF32(&self.ctx, embeddings.pos_emb);
        self.type_emb_buf = try uploadF32(&self.ctx, embeddings.type_emb);
        self.embed_ln_gamma_buf = try uploadF32(&self.ctx, embeddings.ln_gamma);
        self.embed_ln_beta_buf = try uploadF32(&self.ctx, embeddings.ln_beta);

        // 4. Upload per-layer weights
        self.layers = allocator.alloc(LayerGpuWeights, config.num_layers) catch return MetalForwardError.OutOfMemory;
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

        self.buf_a = try self.ctx.createBuffer(act_size);
        self.buf_b = try self.ctx.createBuffer(act_size);
        self.q_buf = try self.ctx.createBuffer(act_size);
        self.k_buf = try self.ctx.createBuffer(act_size);
        self.v_buf = try self.ctx.createBuffer(act_size);
        self.ffn_buf = try self.ctx.createBuffer(max_tokens * ffn * @sizeOf(f32));
        self.ids_buf = try self.ctx.createBuffer(3 * max_tokens * @sizeOf(u32));
        self.output_buf = try self.ctx.createBuffer(hidden * @sizeOf(f32));

        // Batch-specific buffers
        const max_batch: u32 = 64;
        self.offsets_buf = try self.ctx.createBuffer(max_batch * @sizeOf(u32));
        self.lengths_buf = try self.ctx.createBuffer(max_batch * @sizeOf(u32));
        self.batch_output_buf = try self.ctx.createBuffer(max_batch * hidden * @sizeOf(f32));

        const weight_count = 5 + config.num_layers * 16;
        std.debug.print("GPU: Uploaded {} weight buffers, {} activation buffers\n", .{ weight_count, 8 });

        return self;
    }

    // ====================================================================
    // Forward Pass
    // ====================================================================

    pub fn forward(self: *Self, token_ids: []const u32, output: []f32) MetalForwardError!void {
        const seq_len: u32 = @intCast(token_ids.len);
        const hidden = self.config.hidden_dim;

        // Prepare IDs buffer: [token_ids | position_ids (0..N-1) | type_ids (all 0)]
        const ids = self.allocator.alloc(u32, seq_len * 3) catch return MetalForwardError.OutOfMemory;
        defer self.allocator.free(ids);
        @memcpy(ids[0..seq_len], token_ids);
        for (0..seq_len) |i| ids[seq_len + i] = @intCast(i);
        @memset(ids[2 * seq_len .. 3 * seq_len], 0);
        self.ctx.uploadToBuffer(&self.ids_buf, std.mem.sliceAsBytes(ids));

        // Begin command buffer recording
        try self.ctx.beginCommandBuffer();

        // --- Embedding lookup ---
        {
            const pc = EmbeddingPC{ .seq_len = seq_len, .hidden_dim = hidden };
            self.ctx.cmdDispatch(
                &self.embedding_pipe,
                &[_]mtl.MetalBuffer{ self.ids_buf, self.word_emb_buf, self.pos_emb_buf, self.type_emb_buf, self.buf_a },
                std.mem.asBytes(&pc),
                .{ .width = (seq_len * hidden + 255) / 256, .height = 1, .depth = 1 },
                .{ .width = 256, .height = 1, .depth = 1 },
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
                &self.pool_normalize_pipe,
                &[_]mtl.MetalBuffer{ self.buf_a, self.output_buf },
                std.mem.asBytes(&pc),
                .{ .width = 1, .height = 1, .depth = 1 },
                .{ .width = 256, .height = 1, .depth = 1 },
            );
        }

        // Submit all dispatches and wait
        self.ctx.submitAndWait();

        // Readback final embedding
        self.ctx.readbackFromBuffer(&self.output_buf, std.mem.sliceAsBytes(output[0..hidden]));
    }

    // ====================================================================
    // Batched Forward Pass
    // ====================================================================

    pub fn forwardBatch(self: *Self, batch_ids: []const []const u32, output: []f32) MetalForwardError!void {
        const batch_size: u32 = @intCast(batch_ids.len);
        const hidden = self.config.hidden_dim;

        // Compute total tokens and per-sentence offsets/lengths
        var total_tokens: u32 = 0;
        const offsets = self.allocator.alloc(u32, batch_size) catch return MetalForwardError.OutOfMemory;
        defer self.allocator.free(offsets);
        const lengths = self.allocator.alloc(u32, batch_size) catch return MetalForwardError.OutOfMemory;
        defer self.allocator.free(lengths);

        for (batch_ids, 0..) |ids, i| {
            offsets[i] = total_tokens;
            lengths[i] = @intCast(ids.len);
            total_tokens += @intCast(ids.len);
        }

        if (total_tokens > self.max_batch_tokens) {
            std.debug.print("GPU: batch total tokens {} exceeds max {}\n", .{ total_tokens, self.max_batch_tokens });
            return MetalForwardError.BatchTooLarge;
        }

        // Pack all token IDs with position and type IDs
        const all_ids = self.allocator.alloc(u32, total_tokens * 3) catch return MetalForwardError.OutOfMemory;
        defer self.allocator.free(all_ids);
        var pos: u32 = 0;
        for (batch_ids) |ids| {
            const len: u32 = @intCast(ids.len);
            @memcpy(all_ids[pos .. pos + len], ids);
            for (0..len) |j| all_ids[total_tokens + pos + j] = @intCast(j);
            @memset(all_ids[2 * total_tokens + pos .. 2 * total_tokens + pos + len], 0);
            pos += len;
        }
        self.ctx.uploadToBuffer(&self.ids_buf, std.mem.sliceAsBytes(all_ids));
        self.ctx.uploadToBuffer(&self.offsets_buf, std.mem.sliceAsBytes(offsets));
        self.ctx.uploadToBuffer(&self.lengths_buf, std.mem.sliceAsBytes(lengths));

        // Begin recording
        try self.ctx.beginCommandBuffer();

        // --- Embedding lookup ---
        {
            const pc = EmbeddingPC{ .seq_len = total_tokens, .hidden_dim = hidden };
            self.ctx.cmdDispatch(
                &self.embedding_pipe,
                &[_]mtl.MetalBuffer{ self.ids_buf, self.word_emb_buf, self.pos_emb_buf, self.type_emb_buf, self.buf_a },
                std.mem.asBytes(&pc),
                .{ .width = (total_tokens * hidden + 255) / 256, .height = 1, .depth = 1 },
                .{ .width = 256, .height = 1, .depth = 1 },
            );
        }

        // --- Embedding LayerNorm ---
        self.recordLayerNorm(total_tokens, self.buf_a, self.embed_ln_gamma_buf, self.embed_ln_beta_buf);

        // --- Transformer Layers (with batch-aware attention) ---
        for (0..self.config.num_layers) |i| {
            self.recordLayerBatch(@intCast(i), total_tokens, batch_size);
        }

        // --- Batched Pool + Normalize ---
        {
            const pc = PoolNormBatchPC{ .batch_size = batch_size, .hidden_dim = hidden };
            self.ctx.cmdDispatch(
                &self.pool_normalize_batch_pipe,
                &[_]mtl.MetalBuffer{ self.buf_a, self.batch_output_buf, self.offsets_buf, self.lengths_buf },
                std.mem.asBytes(&pc),
                .{ .width = batch_size, .height = 1, .depth = 1 },
                .{ .width = 256, .height = 1, .depth = 1 },
            );
        }

        // Submit and wait
        self.ctx.submitAndWait();

        // Readback all embeddings
        self.ctx.readbackFromBuffer(&self.batch_output_buf, std.mem.sliceAsBytes(output[0 .. batch_size * hidden]));
    }

    // ====================================================================
    // Per-Layer Dispatch
    // ====================================================================

    fn recordLayer(self: *Self, layer_idx: u32, seq_len: u32) void {
        const lw = &self.layers[layer_idx];
        const hidden = self.config.hidden_dim;
        const ffn_dim = self.config.ffn_dim;
        const elem_count = seq_len * hidden;

        // Q, K, V projections: buf_a @ W + b → q_buf, k_buf, v_buf
        // These are independent (all read buf_a, write separate buffers) — skip barriers on Q and K
        self.recordGemmNoBarrier(self.buf_a, lw.q_weight, self.q_buf, lw.q_bias, seq_len, hidden, hidden);
        self.recordGemmNoBarrier(self.buf_a, lw.k_weight, self.k_buf, lw.k_bias, seq_len, hidden, hidden);
        self.recordGemm(self.buf_a, lw.v_weight, self.v_buf, lw.v_bias, seq_len, hidden, hidden);

        // Multi-head attention: q_buf, k_buf, v_buf → buf_b
        {
            const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.config.head_dim)));
            const pc = AttentionPC{
                .seq_len = seq_len,
                .num_heads = self.config.num_heads,
                .head_dim = self.config.head_dim,
                .scale = scale,
            };
            self.ctx.cmdDispatch(
                &self.attention_pipe,
                &[_]mtl.MetalBuffer{ self.q_buf, self.k_buf, self.v_buf, self.buf_b },
                std.mem.asBytes(&pc),
                .{ .width = self.config.num_heads, .height = seq_len, .depth = 1 },
                .{ .width = 256, .height = 1, .depth = 1 },
            );
        }

        // Output projection: buf_b @ W_o + b_o → q_buf (reuse)
        self.recordGemm(self.buf_b, lw.o_weight, self.q_buf, lw.o_bias, seq_len, hidden, hidden);

        // Residual: q_buf += buf_a
        self.recordAdd(self.q_buf, self.buf_a, elem_count);

        // Attention LayerNorm: q_buf in-place
        self.recordLayerNorm(seq_len, self.q_buf, lw.attn_ln_gamma, lw.attn_ln_beta);

        // FFN up: q_buf @ W_up + b_up → ffn_buf
        self.recordGemm(self.q_buf, lw.ff1_weight, self.ffn_buf, lw.ff1_bias, seq_len, ffn_dim, hidden);

        // GELU: ffn_buf in-place
        {
            const pc = ElementPC{ .count = seq_len * ffn_dim };
            self.ctx.cmdDispatch(
                &self.gelu_pipe,
                &[_]mtl.MetalBuffer{self.ffn_buf},
                std.mem.asBytes(&pc),
                .{ .width = (pc.count + 255) / 256, .height = 1, .depth = 1 },
                .{ .width = 256, .height = 1, .depth = 1 },
            );
        }

        // FFN down: ffn_buf @ W_down + b_down → buf_a
        self.recordGemm(self.ffn_buf, lw.ff2_weight, self.buf_a, lw.ff2_bias, seq_len, hidden, ffn_dim);

        // Residual: buf_a += q_buf
        self.recordAdd(self.buf_a, self.q_buf, elem_count);

        // FFN LayerNorm: buf_a in-place
        self.recordLayerNorm(seq_len, self.buf_a, lw.ff_ln_gamma, lw.ff_ln_beta);
    }

    fn recordLayerBatch(self: *Self, layer_idx: u32, total_tokens: u32, batch_size: u32) void {
        const lw = &self.layers[layer_idx];
        const hidden = self.config.hidden_dim;
        const ffn_dim = self.config.ffn_dim;
        const elem_count = total_tokens * hidden;

        // Q, K, V projections — independent, skip barriers on Q and K
        self.recordGemmNoBarrier(self.buf_a, lw.q_weight, self.q_buf, lw.q_bias, total_tokens, hidden, hidden);
        self.recordGemmNoBarrier(self.buf_a, lw.k_weight, self.k_buf, lw.k_bias, total_tokens, hidden, hidden);
        self.recordGemm(self.buf_a, lw.v_weight, self.v_buf, lw.v_bias, total_tokens, hidden, hidden);

        // Batch-aware multi-head attention
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
                &self.attention_batch_pipe,
                &[_]mtl.MetalBuffer{ self.q_buf, self.k_buf, self.v_buf, self.buf_b, self.offsets_buf },
                std.mem.asBytes(&pc),
                .{ .width = self.config.num_heads, .height = total_tokens, .depth = 1 },
                .{ .width = 1, .height = 1, .depth = 1 },
            );
        }

        // Output projection, residual, LayerNorm, FFN
        self.recordGemm(self.buf_b, lw.o_weight, self.q_buf, lw.o_bias, total_tokens, hidden, hidden);
        self.recordAdd(self.q_buf, self.buf_a, elem_count);
        self.recordLayerNorm(total_tokens, self.q_buf, lw.attn_ln_gamma, lw.attn_ln_beta);
        self.recordGemm(self.q_buf, lw.ff1_weight, self.ffn_buf, lw.ff1_bias, total_tokens, ffn_dim, hidden);
        {
            const pc = ElementPC{ .count = total_tokens * ffn_dim };
            self.ctx.cmdDispatch(
                &self.gelu_pipe,
                &[_]mtl.MetalBuffer{self.ffn_buf},
                std.mem.asBytes(&pc),
                .{ .width = (pc.count + 255) / 256, .height = 1, .depth = 1 },
                .{ .width = 256, .height = 1, .depth = 1 },
            );
        }
        self.recordGemm(self.ffn_buf, lw.ff2_weight, self.buf_a, lw.ff2_bias, total_tokens, hidden, ffn_dim);
        self.recordAdd(self.buf_a, self.q_buf, elem_count);
        self.recordLayerNorm(total_tokens, self.buf_a, lw.ff_ln_gamma, lw.ff_ln_beta);
    }

    // ====================================================================
    // Dispatch Helpers
    // ====================================================================

    fn recordGemm(self: *Self, a: mtl.MetalBuffer, b: mtl.MetalBuffer, c: mtl.MetalBuffer, bias: mtl.MetalBuffer, m: u32, n: u32, k: u32) void {
        const pc = SgemmBiasPC{ .m = m, .n = n, .k = k };
        self.ctx.cmdDispatch(
            &self.sgemm_bias_pipe,
            &[_]mtl.MetalBuffer{ a, b, c, bias },
            std.mem.asBytes(&pc),
            .{ .width = (n + 63) / 64, .height = (m + 63) / 64, .depth = 1 },
            .{ .width = 16, .height = 16, .depth = 1 },
        );
    }

    /// GEMM dispatch without trailing barrier — for independent parallel projections
    fn recordGemmNoBarrier(self: *Self, a: mtl.MetalBuffer, b: mtl.MetalBuffer, c: mtl.MetalBuffer, bias: mtl.MetalBuffer, m: u32, n: u32, k: u32) void {
        const pc = SgemmBiasPC{ .m = m, .n = n, .k = k };
        self.ctx.cmdDispatchNoBarrier(
            &self.sgemm_bias_pipe,
            &[_]mtl.MetalBuffer{ a, b, c, bias },
            std.mem.asBytes(&pc),
            .{ .width = (n + 63) / 64, .height = (m + 63) / 64, .depth = 1 },
            .{ .width = 16, .height = 16, .depth = 1 },
        );
    }

    fn recordLayerNorm(self: *Self, seq_len: u32, data: mtl.MetalBuffer, gamma: mtl.MetalBuffer, beta: mtl.MetalBuffer) void {
        const pc = LayerNormPC{ .rows = seq_len, .cols = self.config.hidden_dim, .eps = 1e-12 };
        self.ctx.cmdDispatch(
            &self.layernorm_pipe,
            &[_]mtl.MetalBuffer{ data, gamma, beta },
            std.mem.asBytes(&pc),
            .{ .width = seq_len, .height = 1, .depth = 1 },
            .{ .width = 256, .height = 1, .depth = 1 },
        );
    }

    fn recordAdd(self: *Self, a: mtl.MetalBuffer, b: mtl.MetalBuffer, count: u32) void {
        const pc = ElementPC{ .count = count };
        self.ctx.cmdDispatch(
            &self.residual_add_pipe,
            &[_]mtl.MetalBuffer{ a, b },
            std.mem.asBytes(&pc),
            .{ .width = (count + 255) / 256, .height = 1, .depth = 1 },
            .{ .width = 256, .height = 1, .depth = 1 },
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

        // Metal context
        self.ctx.deinit();
    }
};

// ============================================================================
// Helpers
// ============================================================================

fn uploadF32(ctx: *mtl.MetalContext, data: []const f32) MetalForwardError!mtl.MetalBuffer {
    const size = data.len * @sizeOf(f32);
    var buf = try ctx.createBuffer(size);
    ctx.uploadToBuffer(&buf, std.mem.sliceAsBytes(data));
    return buf;
}

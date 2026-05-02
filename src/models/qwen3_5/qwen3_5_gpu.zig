// src/models/qwen3_5/qwen3_5_gpu.zig
// GPU-Accelerated Matrix-Vector Projections for Qwen3.5
//
// Provides GPU-accelerated matvec via Vulkan compute shaders.
// All large weight matrices are dequantized to F32 and uploaded to device-local
// GPU VRAM at initialization. During inference, each projectMul call is dispatched
// to the SGEMV compute shader on the GPU.
//
// Architecture:
//   - Weight buffers: DEVICE_LOCAL VRAM (uploaded via staging at init)
//   - Input/output staging: HOST_VISIBLE (for CPU ↔ GPU vector transfers)
//   - One descriptor set per weight (pre-allocated at init)
//   - Single SGEMV pipeline shared across all weights
//
// Usage (from cli.zig):
//   var gpu = try GpuAccelerator.initFromWeights(allocator, &model.weights, cfg);
//   defer gpu.deinit();
//   setGlobalInstance(&gpu);
//   model_mod.setGpuMatvec(&GpuAccelerator.dispatch);
//   // ... run inference, all projectMul calls now go through GPU ...

const std = @import("std");
const gpu = @import("vulkan");
const vkl = @import("vk_loader");

/// SPIR-V shader bytecode (embedded at compile time)
const sgemv_spirv = @embedFile("shaders/sgemv.spv");
const sgemv_q8k_spirv = @embedFile("shaders/sgemv_q8k.spv");
const rmsnorm_spirv = @embedFile("shaders/rmsnorm.spv");
const silu_mul_spirv = @embedFile("shaders/silu_mul.spv");
const residual_add_spirv = @embedFile("shaders/residual_add.spv");
const add_dup_spirv = @embedFile("shaders/add_dup.spv");

/// Push constants for SGEMV shaders. Both F32 and Q8K variants use this layout
/// (the F32 shader has an unused third field for size compatibility).
const SgemvPC = extern struct {
    M: u32, // output rows
    N: u32, // input cols
    blocks_per_row: u32, // = N / 32 (Q8K only; ignored by F32 shader)
};

/// Quantization kind of a registered weight matrix.
pub const WeightKind = enum { f32_dense, q8k_packed };

const RmsNormPC = extern struct {
    N: u32,
    eps: f32,
};

const SiLuMulPC = extern struct {
    N: u32,
};

const ResidualAddPC = extern struct {
    count: u32,
};

/// Global instance pointer for the static dispatch function.
var global_instance: ?*GpuAccelerator = null;

pub fn setGlobalInstance(accel: ?*GpuAccelerator) void {
    global_instance = accel;
}

/// Per-layer FFN info needed to initialize the fused FFN pipeline.
pub const FfnLayerInfo = struct {
    norm_weight_data: []const f32,
    gate_weight_key: usize,
    up_weight_key: usize,
    down_weight_key: usize,
    /// Output projection from attention/DeltaNet (W_o for full attn, out_proj for DeltaNet).
    /// Used to fuse o_proj + post-attn residual into the FFN command buffer.
    o_proj_weight_key: usize,
};

/// Per-layer info needed to initialize fused input projections.
pub const InputProjLayerInfo = struct {
    is_deltanet: bool, // true = DeltaNet, false = Full Attention
    input_norm_data: []const f32, // input_layernorm weights
    // DeltaNet keys (only valid when is_deltanet == true)
    dn_qkv_key: usize,
    dn_z_key: usize,
    dn_b_key: usize,
    dn_a_key: usize,
    // Full attention keys (only valid when is_deltanet == false)
    fa_q_key: usize,
    fa_k_key: usize,
    fa_v_key: usize,
};

/// A registered weight matrix on the GPU.
const WeightEntry = struct {
    kind: WeightKind,
    /// F32: device-local f32 buffer of M*N floats.
    /// Q8K: device-local uint32 buffer of (M*N)/4 packed int8 weights.
    primary_buf: gpu.GpuBuffer,
    /// Q8K only: device-local f32 buffer of (M*N)/32 scales. Undefined for F32.
    scales_buf: gpu.GpuBuffer,
    /// Pre-bound descriptor set. Layout matches the kind's pipeline:
    /// F32: [primary, input, output] (3 bindings)
    /// Q8K: [primary, scales, input, output] (4 bindings)
    desc_set: vkl.VkDescriptorSet,
    rows: u32,
    cols: u32,
    blocks_per_row: u32, // Q8K: cols / 32; F32: 0
};

/// GPU-accelerated matrix-vector multiply engine for Qwen3.5.
pub const GpuAccelerator = struct {
    allocator: std.mem.Allocator,
    ctx: gpu.VulkanContext,
    sgemv_pipeline: gpu.ComputePipeline,        // F32 SGEMV (3 bindings)
    sgemv_q8k_pipeline: gpu.ComputePipeline,    // Q8K-direct SGEMV (4 bindings)

    // Weight storage
    weights: std.ArrayListUnmanaged(WeightEntry),
    // Map: CPU data pointer address → weight index
    weight_map: std.AutoHashMapUnmanaged(usize, u32),

    // Host-visible staging buffers for input/output vectors
    input_staging: gpu.GpuBuffer,
    output_staging: gpu.GpuBuffer,

    // Fused FFN state (initialized after all weights registered via initFusedFfn)
    fused_ffn: ?FusedFfn = null,

    // Fused input projections state (initialized after fused FFN via initFusedInputProj)
    fused_input_proj: ?FusedInputProj = null,

    const Self = @This();

    /// Per-projection binding: pipeline + descriptor set + push-constant cache.
    /// Used inside fused command buffers so we can pick the right pipeline (F32 vs Q8K).
    const ProjBinding = struct {
        pipeline: *const gpu.ComputePipeline,
        desc_set: vkl.VkDescriptorSet,
        pc: SgemvPC,
    };

    const FfnLayerDescs = struct {
        rmsnorm_desc: vkl.VkDescriptorSet,
        oproj: ProjBinding, // attention output projection (W_o or DN out_proj)
        gate: ProjBinding,
        up: ProjBinding,
        down: ProjBinding,
    };

    const InputProjDescs = struct {
        rmsnorm_desc: vkl.VkDescriptorSet,
        proj: [4]ProjBinding, // qkv/q, z/k, b/v, a/unused
        num_projs: u32, // 4 for DeltaNet, 3 for Full Attention
    };

    const FusedFfn = struct {
        rmsnorm_pipeline: gpu.ComputePipeline,
        silu_mul_pipeline: gpu.ComputePipeline,
        residual_add_pipeline: gpu.ComputePipeline,
        add_dup_pipeline: gpu.ComputePipeline,

        hidden_staging: gpu.GpuBuffer,
        residual_staging: gpu.GpuBuffer,         // FFN residual (snapshot of hidden after post-attn add)
        ffn_gate_gpu: gpu.GpuBuffer,
        ffn_up_gpu: gpu.GpuBuffer,

        // Phase 2A: fused o_proj + post-attn residual support
        attn_out_staging: gpu.GpuBuffer,         // host-visible: input to o_proj (size = max attn_out_dim)
        pre_attn_residual_staging: gpu.GpuBuffer, // host-visible: pre-attention residual (added to o_proj output)
        oproj_out_gpu: gpu.GpuBuffer,            // device-local: o_proj output (size = hidden_size)
        attn_out_capacity: u32,                  // floats reserved in attn_out_staging

        // Phase 4: fused final RMSNorm + LM head (single submit at token end)
        final_norm_buf: gpu.GpuBuffer,           // device-local: final_norm weights
        final_norm_desc: vkl.VkDescriptorSet,    // [hidden_staging, final_norm_buf]
        lm_head_proj: ProjBinding,               // embed_tokens weight, input=hidden_staging, output=output_staging

        norm_weight_bufs: []gpu.GpuBuffer,
        layer_descs: []FfnLayerDescs,
        silu_mul_desc: vkl.VkDescriptorSet,
        residual_add_desc: vkl.VkDescriptorSet,
        add_dup_desc: vkl.VkDescriptorSet,       // shared: writes hidden + residual = oproj_out + pre_attn_residual

        hidden_size: u32,
        intermediate_size: u32,
        rms_norm_eps: f32,
        num_layers: u32,
    };

    const FusedInputProj = struct {
        // Output staging buffers (host-visible, shared across layers)
        // Sized for max: [0]=6144, [1]=2048, [2]=512, [3]=16 floats
        proj_out: [4]gpu.GpuBuffer,
        // Per-layer input norm weights (device-local)
        input_norm_bufs: []gpu.GpuBuffer,
        // Per-layer descriptor sets (indexed by global layer_idx 0..23)
        layer_descs: []InputProjDescs,
        hidden_size: u32,
        rms_norm_eps: f32,
        num_layers: u32,
    };

    /// Initialize the GPU accelerator (Vulkan context + SGEMV pipeline + staging buffers).
    pub fn init(allocator: std.mem.Allocator) !Self {
        var ctx = gpu.VulkanContext.init(allocator) catch |err| {
            std.debug.print("GPU init failed: {}\n", .{err});
            return err;
        };
        errdefer ctx.deinit();

        std.debug.print("GPU: {s}\n", .{ctx.getDeviceName()});
        std.debug.print("  Max workgroup count: {d}x{d}x{d}\n", .{
            ctx.max_compute_work_group_count[0],
            ctx.max_compute_work_group_count[1],
            ctx.max_compute_work_group_count[2],
        });

        // Create SGEMV pipeline (3 storage buffers: weight, input, output)
        var sgemv_pipeline = try ctx.createComputePipeline(
            sgemv_spirv,
            3,
            @sizeOf(SgemvPC),
        );
        errdefer ctx.destroyPipeline(&sgemv_pipeline);

        // Create Q8K-direct SGEMV pipeline (4 storage buffers: packed_weight, scales, input, output)
        var sgemv_q8k_pipeline = try ctx.createComputePipeline(
            sgemv_q8k_spirv,
            4,
            @sizeOf(SgemvPC),
        );
        errdefer ctx.destroyPipeline(&sgemv_q8k_pipeline);

        // Staging buffers: sized for largest vectors (vocab=248320 for output)
        const max_input_floats: usize = 4096; // max input dimension across all projections
        const max_output_floats: usize = 248320; // vocab_size for LM head

        var input_staging = try ctx.createStorageBuffer(max_input_floats * @sizeOf(f32), true);
        errdefer ctx.destroyBuffer(&input_staging);
        var output_staging = try ctx.createStorageBuffer(max_output_floats * @sizeOf(f32), true);
        errdefer ctx.destroyBuffer(&output_staging);

        return Self{
            .allocator = allocator,
            .ctx = ctx,
            .sgemv_pipeline = sgemv_pipeline,
            .sgemv_q8k_pipeline = sgemv_q8k_pipeline,
            .weights = .{},
            .weight_map = .{},
            .input_staging = input_staging,
            .output_staging = output_staging,
        };
    }

    /// Register an F32 weight matrix: upload to device-local VRAM and pre-allocate descriptor set.
    pub fn registerF32(self: *Self, cpu_key: usize, data: []const f32, rows: usize, cols: usize) !void {
        const byte_size = data.len * @sizeOf(f32);
        var dev_buf = try self.ctx.uploadToDeviceBuffer(byte_size, std.mem.sliceAsBytes(data));
        errdefer self.ctx.destroyBuffer(&dev_buf);

        // Pre-allocate descriptor set bound to [weight, input_staging, output_staging]
        const desc_set = try self.ctx.allocateDescriptorSet(&self.sgemv_pipeline);
        try self.ctx.bindBuffers(desc_set, &[_]gpu.GpuBuffer{
            dev_buf,
            self.input_staging,
            self.output_staging,
        });

        const idx: u32 = @intCast(self.weights.items.len);
        try self.weights.append(self.allocator, .{
            .kind = .f32_dense,
            .primary_buf = dev_buf,
            .scales_buf = undefined,
            .desc_set = desc_set,
            .rows = @intCast(rows),
            .cols = @intCast(cols),
            .blocks_per_row = 0,
        });
        try self.weight_map.put(self.allocator, cpu_key, idx);
    }

    /// Register a Q8K weight matrix: upload packed int8 data + scales DIRECTLY to VRAM.
    /// The Q8K-aware compute shader dequantizes on the fly during SGEMV, reading
    /// 1 byte/weight instead of 4 bytes/weight (3.5× less memory bandwidth).
    /// Requires cols % 32 == 0 (always true for Qwen3.5 dimensions).
    pub fn registerQ8K(
        self: *Self,
        cpu_key: usize,
        q_data: []const i8,
        scales: []const f32,
        rows: usize,
        cols: usize,
        block_size: usize,
    ) !void {
        std.debug.assert(block_size == 32);
        std.debug.assert(cols % 32 == 0);
        const total = rows * cols;
        std.debug.assert(q_data.len == total);
        std.debug.assert(scales.len == (total + 31) / 32);

        // Upload packed int8 weights as raw bytes (interpreted as uint32[] in shader)
        var packed_buf = try self.ctx.uploadToDeviceBuffer(total, std.mem.sliceAsBytes(q_data));
        errdefer self.ctx.destroyBuffer(&packed_buf);

        // Upload scales as f32 buffer
        var scales_buf = try self.ctx.uploadToDeviceBuffer(
            scales.len * @sizeOf(f32),
            std.mem.sliceAsBytes(scales),
        );
        errdefer self.ctx.destroyBuffer(&scales_buf);

        // Pre-allocate descriptor set bound to [packed_weight, scales, input_staging, output_staging]
        const desc_set = try self.ctx.allocateDescriptorSet(&self.sgemv_q8k_pipeline);
        try self.ctx.bindBuffers(desc_set, &[_]gpu.GpuBuffer{
            packed_buf,
            scales_buf,
            self.input_staging,
            self.output_staging,
        });

        const idx: u32 = @intCast(self.weights.items.len);
        try self.weights.append(self.allocator, .{
            .kind = .q8k_packed,
            .primary_buf = packed_buf,
            .scales_buf = scales_buf,
            .desc_set = desc_set,
            .rows = @intCast(rows),
            .cols = @intCast(cols),
            .blocks_per_row = @intCast(cols / 32),
        });
        try self.weight_map.put(self.allocator, cpu_key, idx);
    }

    /// Pick the appropriate SGEMV pipeline for a given weight kind.
    fn pipelineFor(self: *Self, kind: WeightKind) *const gpu.ComputePipeline {
        return switch (kind) {
            .f32_dense => &self.sgemv_pipeline,
            .q8k_packed => &self.sgemv_q8k_pipeline,
        };
    }

    /// Allocate and bind a fused-pipeline descriptor set for a registered weight,
    /// using the right pipeline + buffer layout for the weight's quantization kind.
    /// F32 layout: [weight, input, output]
    /// Q8K layout: [packed_weight, scales, input, output]
    fn bindProj(
        self: *Self,
        weight_idx: u32,
        input_buf: gpu.GpuBuffer,
        output_buf: gpu.GpuBuffer,
    ) !ProjBinding {
        const entry = self.weights.items[weight_idx];
        const pipeline = self.pipelineFor(entry.kind);
        const desc = try self.ctx.allocateDescriptorSet(pipeline);
        switch (entry.kind) {
            .f32_dense => try self.ctx.bindBuffers(desc, &[_]gpu.GpuBuffer{
                entry.primary_buf, input_buf, output_buf,
            }),
            .q8k_packed => try self.ctx.bindBuffers(desc, &[_]gpu.GpuBuffer{
                entry.primary_buf, entry.scales_buf, input_buf, output_buf,
            }),
        }
        return .{
            .pipeline = pipeline,
            .desc_set = desc,
            .pc = .{
                .M = entry.rows,
                .N = entry.cols,
                .blocks_per_row = entry.blocks_per_row,
            },
        };
    }

    /// Initialize fused FFN pipeline: creates pipelines, staging/intermediate buffers,
    /// uploads per-layer norm weights, and pre-allocates all descriptor sets.
    /// Must be called AFTER all projection weights are registered via registerF32/registerQ8K.
    /// `attn_out_capacity` = max attention output dim across layers (q_dim for full attn,
    /// value_dim for DeltaNet) — sizes the o_proj input staging buffer.
    pub fn initFusedFfn(
        self: *Self,
        layers_info: []const FfnLayerInfo,
        hidden_size: u32,
        intermediate_size: u32,
        rms_norm_eps: f32,
        attn_out_capacity: u32,
    ) !void {
        const num_layers: u32 = @intCast(layers_info.len);

        // Create pipelines for RMSNorm (2 buffers), SiLU×mul (2 buffers), ResidualAdd (2 buffers), AddDup (4 buffers)
        var rmsnorm_pipeline = try self.ctx.createComputePipeline(rmsnorm_spirv, 2, @sizeOf(RmsNormPC));
        errdefer self.ctx.destroyPipeline(&rmsnorm_pipeline);

        var silu_mul_pipeline = try self.ctx.createComputePipeline(silu_mul_spirv, 2, @sizeOf(SiLuMulPC));
        errdefer self.ctx.destroyPipeline(&silu_mul_pipeline);

        var residual_add_pipeline = try self.ctx.createComputePipeline(residual_add_spirv, 2, @sizeOf(ResidualAddPC));
        errdefer self.ctx.destroyPipeline(&residual_add_pipeline);

        var add_dup_pipeline = try self.ctx.createComputePipeline(add_dup_spirv, 4, @sizeOf(ResidualAddPC));
        errdefer self.ctx.destroyPipeline(&add_dup_pipeline);

        // Host-visible staging for hidden state + residual (CPU ↔ GPU I/O)
        var hidden_staging = try self.ctx.createStorageBuffer(@as(usize, hidden_size) * @sizeOf(f32), true);
        errdefer self.ctx.destroyBuffer(&hidden_staging);

        var residual_staging = try self.ctx.createStorageBuffer(@as(usize, hidden_size) * @sizeOf(f32), true);
        errdefer self.ctx.destroyBuffer(&residual_staging);

        // Device-local intermediates (gate/up projections stay on GPU)
        var ffn_gate_gpu = try self.ctx.createStorageBuffer(@as(usize, intermediate_size) * @sizeOf(f32), false);
        errdefer self.ctx.destroyBuffer(&ffn_gate_gpu);

        var ffn_up_gpu = try self.ctx.createStorageBuffer(@as(usize, intermediate_size) * @sizeOf(f32), false);
        errdefer self.ctx.destroyBuffer(&ffn_up_gpu);

        // Phase 2A buffers: attn output staging (input to o_proj), pre-attn residual staging,
        // and device-local oproj output buffer.
        var attn_out_staging = try self.ctx.createStorageBuffer(@as(usize, attn_out_capacity) * @sizeOf(f32), true);
        errdefer self.ctx.destroyBuffer(&attn_out_staging);

        var pre_attn_residual_staging = try self.ctx.createStorageBuffer(@as(usize, hidden_size) * @sizeOf(f32), true);
        errdefer self.ctx.destroyBuffer(&pre_attn_residual_staging);

        var oproj_out_gpu = try self.ctx.createStorageBuffer(@as(usize, hidden_size) * @sizeOf(f32), false);
        errdefer self.ctx.destroyBuffer(&oproj_out_gpu);

        // Upload per-layer norm weights to device-local VRAM
        const norm_weight_bufs = try self.allocator.alloc(gpu.GpuBuffer, num_layers);
        var norm_uploaded: usize = 0;
        errdefer {
            for (norm_weight_bufs[0..norm_uploaded]) |*buf| self.ctx.destroyBuffer(buf);
            self.allocator.free(norm_weight_bufs);
        }

        for (layers_info, 0..) |info, i| {
            norm_weight_bufs[i] = try self.ctx.uploadToDeviceBuffer(
                info.norm_weight_data.len * @sizeOf(f32),
                std.mem.sliceAsBytes(info.norm_weight_data),
            );
            norm_uploaded += 1;
        }

        // Pre-allocate descriptor sets per layer
        const layer_descs = try self.allocator.alloc(FfnLayerDescs, num_layers);
        errdefer self.allocator.free(layer_descs);

        for (layers_info, 0..) |info, i| {
            const oproj_idx = self.weight_map.get(info.o_proj_weight_key) orelse return error.OutOfMemory;
            const gate_idx = self.weight_map.get(info.gate_weight_key) orelse return error.OutOfMemory;
            const up_idx = self.weight_map.get(info.up_weight_key) orelse return error.OutOfMemory;
            const down_idx = self.weight_map.get(info.down_weight_key) orelse return error.OutOfMemory;

            // RMSNorm: binding 0 = hidden (r/w in-place), binding 1 = norm weight
            const rmsnorm_desc = try self.ctx.allocateDescriptorSet(&rmsnorm_pipeline);
            try self.ctx.bindBuffers(rmsnorm_desc, &[_]gpu.GpuBuffer{ hidden_staging, norm_weight_bufs[i] });

            // o_proj: input=attn_out_staging, output=oproj_out_gpu (device-local)
            // Gate / Up / Down: bind via the kind-aware helper. Output buffers:
            //   gate input=hidden, output=ffn_gate_gpu
            //   up   input=hidden, output=ffn_up_gpu
            //   down input=ffn_gate_gpu (after silu*mul), output=hidden_staging
            const oproj = try self.bindProj(oproj_idx, attn_out_staging, oproj_out_gpu);
            const gate = try self.bindProj(gate_idx, hidden_staging, ffn_gate_gpu);
            const up = try self.bindProj(up_idx, hidden_staging, ffn_up_gpu);
            const down = try self.bindProj(down_idx, ffn_gate_gpu, hidden_staging);

            layer_descs[i] = .{
                .rmsnorm_desc = rmsnorm_desc,
                .oproj = oproj,
                .gate = gate,
                .up = up,
                .down = down,
            };
        }

        // Shared descriptor sets (same buffers for all layers)
        const silu_mul_desc = try self.ctx.allocateDescriptorSet(&silu_mul_pipeline);
        try self.ctx.bindBuffers(silu_mul_desc, &[_]gpu.GpuBuffer{ ffn_gate_gpu, ffn_up_gpu });

        const residual_add_desc = try self.ctx.allocateDescriptorSet(&residual_add_pipeline);
        try self.ctx.bindBuffers(residual_add_desc, &[_]gpu.GpuBuffer{ hidden_staging, residual_staging });

        // add_dup: hidden = oproj_out + pre_attn_residual; residual = same
        // Bindings: [hidden, oproj_out, pre_attn_residual, residual]
        const add_dup_desc = try self.ctx.allocateDescriptorSet(&add_dup_pipeline);
        try self.ctx.bindBuffers(add_dup_desc, &[_]gpu.GpuBuffer{
            hidden_staging,
            oproj_out_gpu,
            pre_attn_residual_staging,
            residual_staging,
        });

        self.fused_ffn = FusedFfn{
            .rmsnorm_pipeline = rmsnorm_pipeline,
            .silu_mul_pipeline = silu_mul_pipeline,
            .residual_add_pipeline = residual_add_pipeline,
            .add_dup_pipeline = add_dup_pipeline,
            .hidden_staging = hidden_staging,
            .residual_staging = residual_staging,
            .ffn_gate_gpu = ffn_gate_gpu,
            .ffn_up_gpu = ffn_up_gpu,
            .attn_out_staging = attn_out_staging,
            .pre_attn_residual_staging = pre_attn_residual_staging,
            .oproj_out_gpu = oproj_out_gpu,
            .attn_out_capacity = attn_out_capacity,
            .norm_weight_bufs = norm_weight_bufs,
            .layer_descs = layer_descs,
            .silu_mul_desc = silu_mul_desc,
            .residual_add_desc = residual_add_desc,
            .add_dup_desc = add_dup_desc,
            .final_norm_buf = undefined,
            .final_norm_desc = undefined,
            .lm_head_proj = undefined,
            .hidden_size = hidden_size,
            .intermediate_size = intermediate_size,
            .rms_norm_eps = rms_norm_eps,
            .num_layers = num_layers,
        };

        std.debug.print("GPU: Fused FFN initialized ({d} layers)\n", .{num_layers});
    }

    /// Initialize fused input projections: for each layer, fuse RMSNorm + input SGEMVs into
    /// a single command buffer. Must be called AFTER initFusedFfn (shares hidden_staging and
    /// rmsnorm_pipeline). Must be called AFTER all projection weights are registered.
    pub fn initFusedInputProj(
        self: *Self,
        layers_info: []const InputProjLayerInfo,
        hidden_size: u32,
        rms_norm_eps: f32,
        dn_qkv_dim: u32, // DeltaNet: 6144
        dn_value_dim: u32, // DeltaNet z: 2048
        dn_num_heads: u32, // DeltaNet b/a: 16
        fa_q_proj_dim: u32, // Full attn q: 4096
        fa_kv_dim: u32, // Full attn k/v: 512
    ) !void {
        const ffn = self.fused_ffn orelse return error.OutOfMemory; // need hidden_staging
        const num_layers: u32 = @intCast(layers_info.len);

        // Output staging buffers (host-visible) — sized for max across layer types
        const out_sizes = [4]usize{
            @max(@as(usize, dn_qkv_dim), @as(usize, fa_q_proj_dim)), // qkv/q
            @max(@as(usize, dn_value_dim), @as(usize, fa_kv_dim)), // z/k
            @max(@as(usize, dn_num_heads), @as(usize, fa_kv_dim)), // b/v
            @as(usize, dn_num_heads), // a (DN only)
        };

        var proj_out: [4]gpu.GpuBuffer = undefined;
        var proj_created: usize = 0;
        errdefer for (proj_out[0..proj_created]) |*buf| self.ctx.destroyBuffer(buf);

        for (0..4) |oi| {
            proj_out[oi] = try self.ctx.createStorageBuffer(out_sizes[oi] * @sizeOf(f32), true);
            proj_created += 1;
        }

        // Upload per-layer input norm weights to device-local VRAM
        const input_norm_bufs = try self.allocator.alloc(gpu.GpuBuffer, num_layers);
        var norm_uploaded: usize = 0;
        errdefer {
            for (input_norm_bufs[0..norm_uploaded]) |*buf| self.ctx.destroyBuffer(buf);
            self.allocator.free(input_norm_bufs);
        }

        for (layers_info, 0..) |info, i| {
            input_norm_bufs[i] = try self.ctx.uploadToDeviceBuffer(
                info.input_norm_data.len * @sizeOf(f32),
                std.mem.sliceAsBytes(info.input_norm_data),
            );
            norm_uploaded += 1;
        }

        // Pre-allocate descriptor sets per layer
        const layer_descs = try self.allocator.alloc(InputProjDescs, num_layers);
        errdefer self.allocator.free(layer_descs);

        for (layers_info, 0..) |info, i| {
            // RMSNorm: binding 0 = hidden (r/w), binding 1 = norm weight
            const rmsnorm_desc = try self.ctx.allocateDescriptorSet(&ffn.rmsnorm_pipeline);
            try self.ctx.bindBuffers(rmsnorm_desc, &[_]gpu.GpuBuffer{ ffn.hidden_staging, input_norm_bufs[i] });

            var proj: [4]ProjBinding = undefined;
            var num_projs: u32 = 0;

            if (info.is_deltanet) {
                // DeltaNet: qkv, z, b, a — all read from hidden_staging
                const keys = [4]usize{ info.dn_qkv_key, info.dn_z_key, info.dn_b_key, info.dn_a_key };
                for (0..4) |pi| {
                    const w_idx = self.weight_map.get(keys[pi]) orelse return error.OutOfMemory;
                    proj[pi] = try self.bindProj(w_idx, ffn.hidden_staging, proj_out[pi]);
                }
                num_projs = 4;
            } else {
                // Full attention: q, k, v — all read from hidden_staging
                const keys = [3]usize{ info.fa_q_key, info.fa_k_key, info.fa_v_key };
                for (0..3) |pi| {
                    const w_idx = self.weight_map.get(keys[pi]) orelse return error.OutOfMemory;
                    proj[pi] = try self.bindProj(w_idx, ffn.hidden_staging, proj_out[pi]);
                }
                proj[3] = undefined;
                num_projs = 3;
            }

            layer_descs[i] = .{
                .rmsnorm_desc = rmsnorm_desc,
                .proj = proj,
                .num_projs = num_projs,
            };
        }

        self.fused_input_proj = FusedInputProj{
            .proj_out = proj_out,
            .input_norm_bufs = input_norm_bufs,
            .layer_descs = layer_descs,
            .hidden_size = hidden_size,
            .rms_norm_eps = rms_norm_eps,
            .num_layers = num_layers,
        };

        std.debug.print("GPU: Fused input projections initialized ({d} layers)\n", .{num_layers});
    }

    /// Phase 4: Initialize fused final RMSNorm + LM head (single submit at token end).
    /// Replaces the CPU final_norm + standalone gpuMatvec LM head with one fused CB.
    pub fn initFusedLmHead(
        self: *Self,
        final_norm_data: []const f32,
        lm_head_weight_key: usize,
    ) !void {
        const ffn = if (self.fused_ffn) |*f| f else return error.OutOfMemory;

        var final_norm_buf = try self.ctx.uploadToDeviceBuffer(
            final_norm_data.len * @sizeOf(f32),
            std.mem.sliceAsBytes(final_norm_data),
        );
        errdefer self.ctx.destroyBuffer(&final_norm_buf);

        const final_norm_desc = try self.ctx.allocateDescriptorSet(&ffn.rmsnorm_pipeline);
        try self.ctx.bindBuffers(final_norm_desc, &[_]gpu.GpuBuffer{ ffn.hidden_staging, final_norm_buf });

        const lm_idx = self.weight_map.get(lm_head_weight_key) orelse return error.OutOfMemory;
        const lm_proj = try self.bindProj(lm_idx, ffn.hidden_staging, self.output_staging);

        ffn.final_norm_buf = final_norm_buf;
        ffn.final_norm_desc = final_norm_desc;
        ffn.lm_head_proj = lm_proj;
    }

    /// Phase 4: Execute final RMSNorm + LM head in a single GPU submit.
    /// After this, logits are in output_staging and read back to caller.
    pub fn gpuLmHead(self: *Self, logits: []f32) !void {
        const ffn = self.fused_ffn orelse return error.OutOfMemory;

        try self.ctx.beginCommandBuffer();

        const rms_pc = RmsNormPC{ .N = ffn.hidden_size, .eps = ffn.rms_norm_eps };
        self.ctx.cmdDispatch(
            &ffn.rmsnorm_pipeline, ffn.final_norm_desc,
            1, 1, 1, std.mem.asBytes(&rms_pc),
        );
        self.ctx.cmdDispatch(
            ffn.lm_head_proj.pipeline, ffn.lm_head_proj.desc_set,
            ffn.lm_head_proj.pc.M, 1, 1, std.mem.asBytes(&ffn.lm_head_proj.pc),
        );

        try self.ctx.submitAndWait();
        try self.ctx.readbackFromBuffer(&self.output_staging, std.mem.sliceAsBytes(logits));
    }
    pub fn dispatchLmHead(logits: []f32) bool {
        const self = global_instance orelse return false;
        self.gpuLmHead(logits) catch return false;
        return true;
    }

    /// Execute a single GPU matrix-vector multiply: result = weight[buf_idx] @ input
    pub fn gpuMatvec(
        self: *Self,
        result: []f32,
        buf_idx: u32,
        input: []const f32,
        rows: usize,
        cols: usize,
    ) !void {
        const entry = self.weights.items[buf_idx];

        // Upload input vector to staging buffer
        try self.ctx.uploadToBuffer(&self.input_staging, std.mem.sliceAsBytes(input[0..cols]));

        // Push constants — Q8K shader uses blocks_per_row, F32 shader ignores it
        const pc = SgemvPC{
            .M = @intCast(rows),
            .N = @intCast(cols),
            .blocks_per_row = entry.blocks_per_row,
        };

        try self.ctx.dispatch(
            self.pipelineFor(entry.kind),
            entry.desc_set,
            @intCast(rows), // one workgroup per output row
            1,
            1,
            std.mem.asBytes(&pc),
        );

        // Readback result from staging buffer
        try self.ctx.readbackFromBuffer(
            &self.output_staging,
            std.mem.sliceAsBytes(result[0..rows]),
        );
    }

    /// Static dispatch function matching the GpuMatvecFn signature in model.zig.
    /// Returns true if handled by GPU, false to fall back to CPU.
    pub fn dispatch(result: []f32, weight_ptr: usize, input: []const f32, rows: usize, cols: usize) bool {
        const self = global_instance orelse return false;
        const buf_idx = self.weight_map.get(weight_ptr) orelse return false;
        self.gpuMatvec(result, buf_idx, input, rows, cols) catch return false;
        return true;
    }

    /// Execute fused FFN on GPU: RMSNorm → gate/up SGEMV → SiLU×mul → down SGEMV → residual add.
    /// All 6 operations recorded into a single command buffer with 1 submit+wait.
    pub fn gpuFfnFused(self: *Self, hidden: []f32, residual: []const f32, layer_idx: usize) !void {
        const ffn = self.fused_ffn orelse return error.OutOfMemory;
        const descs = ffn.layer_descs[layer_idx];
        const h: usize = ffn.hidden_size;

        // Upload hidden + residual to host-visible staging
        try self.ctx.uploadToBuffer(&ffn.hidden_staging, std.mem.sliceAsBytes(hidden[0..h]));
        try self.ctx.uploadToBuffer(&ffn.residual_staging, std.mem.sliceAsBytes(residual[0..h]));

        // Record fused command buffer
        try self.ctx.beginCommandBuffer();

        // 1. RMSNorm: hidden = hidden * w / sqrt(mean(x²) + eps)
        const rmsnorm_pc = RmsNormPC{ .N = ffn.hidden_size, .eps = ffn.rms_norm_eps };
        self.ctx.cmdDispatch(
            &ffn.rmsnorm_pipeline,
            descs.rmsnorm_desc,
            1, 1, 1,
            std.mem.asBytes(&rmsnorm_pc),
        );

        // 2. Gate projection: ffn_gate = gate_weight @ hidden (barrier from cmdDispatch)
        self.ctx.cmdDispatch(
            descs.gate.pipeline,
            descs.gate.desc_set,
            descs.gate.pc.M, 1, 1,
            std.mem.asBytes(&descs.gate.pc),
        );

        // 3. Up projection: ffn_up = up_weight @ hidden (NO barrier — parallel with gate)
        self.ctx.cmdDispatchNoBarrier(
            descs.up.pipeline,
            descs.up.desc_set,
            descs.up.pc.M, 1, 1,
            std.mem.asBytes(&descs.up.pc),
        );

        // 4. SiLU×mul: ffn_gate = silu(ffn_gate) * ffn_up (barrier ensures gate+up complete)
        const silu_pc = SiLuMulPC{ .N = ffn.intermediate_size };
        self.ctx.cmdDispatch(
            &ffn.silu_mul_pipeline,
            ffn.silu_mul_desc,
            (ffn.intermediate_size + 255) / 256, 1, 1,
            std.mem.asBytes(&silu_pc),
        );

        // 5. Down projection: hidden = down_weight @ ffn_gate (barrier ensures SiLU complete)
        self.ctx.cmdDispatch(
            descs.down.pipeline,
            descs.down.desc_set,
            descs.down.pc.M, 1, 1,
            std.mem.asBytes(&descs.down.pc),
        );

        // 6. Residual add: hidden += residual (barrier ensures down complete)
        const res_pc = ResidualAddPC{ .count = ffn.hidden_size };
        self.ctx.cmdDispatch(
            &ffn.residual_add_pipeline,
            ffn.residual_add_desc,
            (ffn.hidden_size + 255) / 256, 1, 1,
            std.mem.asBytes(&res_pc),
        );

        // Single submit + wait
        try self.ctx.submitAndWait();

        // Readback hidden state
        try self.ctx.readbackFromBuffer(
            &ffn.hidden_staging,
            std.mem.sliceAsBytes(hidden[0..h]),
        );
    }

    /// Static dispatch for fused FFN matching GpuFfnFn signature in model.zig.
    pub fn dispatchFfn(hidden: []f32, residual: []const f32, layer_idx: usize) bool {
        const self = global_instance orelse return false;
        self.gpuFfnFused(hidden, residual, layer_idx) catch return false;
        return true;
    }

    /// Phase 2A: Execute fused o_proj + post-attn-residual + FFN on GPU.
    /// Single command buffer (1 submit) replaces the previous 2 round-trips
    /// (separate o_proj SGEMV + separate FFN dispatch).
    ///
    /// Sequence:
    ///   1. o_proj SGEMV: oproj_out_gpu = W_o @ attn_out_staging
    ///   2. add_dup: hidden_staging = residual_staging = oproj_out_gpu + pre_attn_residual_staging
    ///   3. RMSNorm hidden_staging in-place
    ///   4. gate / up SGEMVs (parallel)
    ///   5. SiLU×mul
    ///   6. down SGEMV: hidden_staging = W_down @ ffn_gate_gpu
    ///   7. residual_add: hidden_staging += residual_staging
    ///
    /// Inputs:
    ///   attn_out          — attention output (q_dim for full attn, value_dim for DeltaNet)
    ///   pre_attn_residual — hidden state before attention (residual to add after o_proj)
    ///   layer_idx         — layer index for descriptor lookup
    ///   hidden_out        — final output (FFN result), size = hidden_size
    pub fn gpuOprojFfnFused(
        self: *Self,
        attn_out: []const f32,
        pre_attn_residual: []const f32,
        layer_idx: usize,
        hidden_out: []f32,
    ) !void {
        const ffn = self.fused_ffn orelse return error.OutOfMemory;
        const descs = ffn.layer_descs[layer_idx];
        const h: usize = ffn.hidden_size;

        // Sanity: o_proj input dim must fit
        std.debug.assert(attn_out.len <= ffn.attn_out_capacity);
        std.debug.assert(pre_attn_residual.len == h);
        std.debug.assert(hidden_out.len == h);

        // Upload o_proj input + pre-attention residual to host-visible staging
        try self.ctx.uploadToBuffer(&ffn.attn_out_staging, std.mem.sliceAsBytes(attn_out));
        try self.ctx.uploadToBuffer(&ffn.pre_attn_residual_staging, std.mem.sliceAsBytes(pre_attn_residual));

        try self.ctx.beginCommandBuffer();

        // 1. o_proj SGEMV: oproj_out_gpu = W_o @ attn_out_staging
        self.ctx.cmdDispatch(
            descs.oproj.pipeline,
            descs.oproj.desc_set,
            descs.oproj.pc.M, 1, 1,
            std.mem.asBytes(&descs.oproj.pc),
        );

        // 2. add_dup: hidden_staging = oproj_out_gpu + pre_attn_residual_staging,
        //    AND simultaneously residual_staging = same value (saves the FFN residual snapshot)
        const add_dup_pc = ResidualAddPC{ .count = ffn.hidden_size };
        self.ctx.cmdDispatch(
            &ffn.add_dup_pipeline,
            ffn.add_dup_desc,
            (ffn.hidden_size + 255) / 256, 1, 1,
            std.mem.asBytes(&add_dup_pc),
        );

        // 3. RMSNorm: hidden_staging = hidden_staging * w / sqrt(mean(x²) + eps)
        const rmsnorm_pc = RmsNormPC{ .N = ffn.hidden_size, .eps = ffn.rms_norm_eps };
        self.ctx.cmdDispatch(
            &ffn.rmsnorm_pipeline,
            descs.rmsnorm_desc,
            1, 1, 1,
            std.mem.asBytes(&rmsnorm_pc),
        );

        // 4. Gate projection (barrier from cmdDispatch)
        self.ctx.cmdDispatch(
            descs.gate.pipeline,
            descs.gate.desc_set,
            descs.gate.pc.M, 1, 1,
            std.mem.asBytes(&descs.gate.pc),
        );

        // 5. Up projection (NO barrier — parallel with gate)
        self.ctx.cmdDispatchNoBarrier(
            descs.up.pipeline,
            descs.up.desc_set,
            descs.up.pc.M, 1, 1,
            std.mem.asBytes(&descs.up.pc),
        );

        // 6. SiLU×mul
        const silu_pc = SiLuMulPC{ .N = ffn.intermediate_size };
        self.ctx.cmdDispatch(
            &ffn.silu_mul_pipeline,
            ffn.silu_mul_desc,
            (ffn.intermediate_size + 255) / 256, 1, 1,
            std.mem.asBytes(&silu_pc),
        );

        // 7. Down projection: hidden_staging = W_down @ ffn_gate_gpu
        self.ctx.cmdDispatch(
            descs.down.pipeline,
            descs.down.desc_set,
            descs.down.pc.M, 1, 1,
            std.mem.asBytes(&descs.down.pc),
        );

        // 8. FFN residual_add: hidden_staging += residual_staging (the snapshot from step 2)
        const res_pc = ResidualAddPC{ .count = ffn.hidden_size };
        self.ctx.cmdDispatch(
            &ffn.residual_add_pipeline,
            ffn.residual_add_desc,
            (ffn.hidden_size + 255) / 256, 1, 1,
            std.mem.asBytes(&res_pc),
        );

        try self.ctx.submitAndWait();

        // Readback final hidden state
        try self.ctx.readbackFromBuffer(
            &ffn.hidden_staging,
            std.mem.sliceAsBytes(hidden_out),
        );
    }

    /// Static dispatch for fused o_proj+FFN matching GpuOprojFfnFn in model.zig.
    pub fn dispatchOprojFfn(
        attn_out: []const f32,
        pre_attn_residual: []const f32,
        layer_idx: usize,
        hidden_out: []f32,
    ) bool {
        const self = global_instance orelse return false;
        self.gpuOprojFfnFused(attn_out, pre_attn_residual, layer_idx, hidden_out) catch return false;
        return true;
    }

    /// Execute fused DeltaNet input projections on GPU:
    /// qkv/z/b/a SGEMVs (parallel) in single command buffer.
    /// Hidden must already be RMSNorm'd by the caller.
    pub fn gpuDnInputFused(
        self: *Self,
        hidden: []const f32,
        layer_idx: usize,
        qkv: []f32,
        z: []f32,
        b: []f32,
        a: []f32,
    ) !void {
        const fip = self.fused_input_proj orelse return error.OutOfMemory;
        const ffn = self.fused_ffn orelse return error.OutOfMemory;
        const descs = fip.layer_descs[layer_idx];
        const h: usize = fip.hidden_size;

        // Upload RMSNorm'd hidden to hidden_staging (shared with fused FFN)
        try self.ctx.uploadToBuffer(&ffn.hidden_staging, std.mem.sliceAsBytes(hidden[0..h]));

        // Record fused command buffer (no RMSNorm — already done on CPU)
        try self.ctx.beginCommandBuffer();

        // 1. qkv SGEMV (first dispatch needs barrier for upload visibility)
        self.ctx.cmdDispatch(
            descs.proj[0].pipeline,
            descs.proj[0].desc_set,
            descs.proj[0].pc.M, 1, 1,
            std.mem.asBytes(&descs.proj[0].pc),
        );

        // 2. z SGEMV (NO barrier — parallel with qkv, both read hidden_staging)
        self.ctx.cmdDispatchNoBarrier(
            descs.proj[1].pipeline,
            descs.proj[1].desc_set,
            descs.proj[1].pc.M, 1, 1,
            std.mem.asBytes(&descs.proj[1].pc),
        );

        // 3. b SGEMV (NO barrier — parallel)
        self.ctx.cmdDispatchNoBarrier(
            descs.proj[2].pipeline,
            descs.proj[2].desc_set,
            descs.proj[2].pc.M, 1, 1,
            std.mem.asBytes(&descs.proj[2].pc),
        );

        // 4. a SGEMV (NO barrier — parallel)
        self.ctx.cmdDispatchNoBarrier(
            descs.proj[3].pipeline,
            descs.proj[3].desc_set,
            descs.proj[3].pc.M, 1, 1,
            std.mem.asBytes(&descs.proj[3].pc),
        );

        // Single submit + wait
        try self.ctx.submitAndWait();

        // Readback projection outputs
        try self.ctx.readbackFromBuffer(&fip.proj_out[0], std.mem.sliceAsBytes(qkv));
        try self.ctx.readbackFromBuffer(&fip.proj_out[1], std.mem.sliceAsBytes(z));
        try self.ctx.readbackFromBuffer(&fip.proj_out[2], std.mem.sliceAsBytes(b));
        try self.ctx.readbackFromBuffer(&fip.proj_out[3], std.mem.sliceAsBytes(a));
    }

    /// Execute fused full attention input projections on GPU:
    /// q/k/v SGEMVs (parallel) in single command buffer.
    /// Hidden must already be RMSNorm'd by the caller.
    pub fn gpuAttnInputFused(
        self: *Self,
        hidden: []const f32,
        layer_idx: usize,
        q: []f32,
        k: []f32,
        v: []f32,
    ) !void {
        const fip = self.fused_input_proj orelse return error.OutOfMemory;
        const ffn = self.fused_ffn orelse return error.OutOfMemory;
        const descs = fip.layer_descs[layer_idx];
        const h: usize = fip.hidden_size;

        // Upload RMSNorm'd hidden to hidden_staging (shared with fused FFN)
        try self.ctx.uploadToBuffer(&ffn.hidden_staging, std.mem.sliceAsBytes(hidden[0..h]));

        // Record fused command buffer (no RMSNorm — already done on CPU)
        try self.ctx.beginCommandBuffer();

        // 1. q_proj SGEMV (first dispatch needs barrier for upload visibility)
        self.ctx.cmdDispatch(
            descs.proj[0].pipeline,
            descs.proj[0].desc_set,
            descs.proj[0].pc.M, 1, 1,
            std.mem.asBytes(&descs.proj[0].pc),
        );

        // 2. k_proj SGEMV (NO barrier — parallel with q)
        self.ctx.cmdDispatchNoBarrier(
            descs.proj[1].pipeline,
            descs.proj[1].desc_set,
            descs.proj[1].pc.M, 1, 1,
            std.mem.asBytes(&descs.proj[1].pc),
        );

        // 3. v_proj SGEMV (NO barrier — parallel with q, k)
        self.ctx.cmdDispatchNoBarrier(
            descs.proj[2].pipeline,
            descs.proj[2].desc_set,
            descs.proj[2].pc.M, 1, 1,
            std.mem.asBytes(&descs.proj[2].pc),
        );

        // Single submit + wait
        try self.ctx.submitAndWait();

        // Readback projection outputs
        try self.ctx.readbackFromBuffer(&fip.proj_out[0], std.mem.sliceAsBytes(q));
        try self.ctx.readbackFromBuffer(&fip.proj_out[1], std.mem.sliceAsBytes(k));
        try self.ctx.readbackFromBuffer(&fip.proj_out[2], std.mem.sliceAsBytes(v));
    }

    /// Static dispatch for fused DeltaNet input matching GpuDnInputFn in model.zig.
    pub fn dispatchDnInput(hidden: []const f32, layer_idx: usize, qkv: []f32, z: []f32, b: []f32, a: []f32) bool {
        const self = global_instance orelse return false;
        self.gpuDnInputFused(hidden, layer_idx, qkv, z, b, a) catch return false;
        return true;
    }

    // =========================================================================
    // Phase 2B: GPU-resident hidden state (no per-layer readback/upload)
    //
    // The hidden state lives in `fused_ffn.hidden_staging` between layers.
    // Per-token cost drops from ~5 transfers/layer to ~3.5 (no hidden upload
    // at input proj, no hidden readback at FFN end, no pre_attn_residual upload).
    // Plus the input RMSNorm moves from CPU to GPU (folded into the input-proj
    // command buffer), which is bit-exact thanks to the 1.0/sqrt fix in
    // rmsnorm.comp.
    // =========================================================================

    /// Phase 2B: Upload the initial hidden state to `hidden_staging` once per token.
    /// All subsequent per-layer dispatches keep `hidden_staging` GPU-resident.
    pub fn gpuTokenBegin(self: *Self, hidden: []const f32) !void {
        const ffn = self.fused_ffn orelse return error.OutOfMemory;
        std.debug.assert(hidden.len == ffn.hidden_size);
        try self.ctx.uploadToBuffer(&ffn.hidden_staging, std.mem.sliceAsBytes(hidden));
    }
    pub fn dispatchTokenBegin(hidden: []const f32) bool {
        const self = global_instance orelse return false;
        self.gpuTokenBegin(hidden) catch return false;
        return true;
    }

    /// Phase 2B: Read back the final hidden state from `hidden_staging` once per token.
    pub fn gpuTokenEnd(self: *Self, hidden_out: []f32) !void {
        const ffn = self.fused_ffn orelse return error.OutOfMemory;
        std.debug.assert(hidden_out.len == ffn.hidden_size);
        try self.ctx.readbackFromBuffer(&ffn.hidden_staging, std.mem.sliceAsBytes(hidden_out));
    }
    pub fn dispatchTokenEnd(hidden_out: []f32) bool {
        const self = global_instance orelse return false;
        self.gpuTokenEnd(hidden_out) catch return false;
        return true;
    }

    /// Phase 2B: Fused DeltaNet input projections with GPU-resident hidden state.
    /// Sequence in single command buffer:
    ///   1. cmdCopyBuffer(hidden_staging → pre_attn_residual_staging)  ← snapshot
    ///   2. RMSNorm(hidden_staging) in place                            ← GPU
    ///   3. qkv/z/b/a SGEMVs (parallel, all read hidden_staging)
    /// Then readback the four projection outputs.
    pub fn gpuLayerStartDn(
        self: *Self,
        layer_idx: usize,
        qkv: []f32,
        z: []f32,
        b: []f32,
        a: []f32,
    ) !void {
        const ffn = self.fused_ffn orelse return error.OutOfMemory;
        const fip = self.fused_input_proj orelse return error.OutOfMemory;
        const descs = fip.layer_descs[layer_idx];
        const h: usize = ffn.hidden_size;

        try self.ctx.beginCommandBuffer();

        // 1. Snapshot pre-attention hidden state on the GPU (residual for after o_proj).
        self.ctx.cmdCopyBuffer(&ffn.hidden_staging, &ffn.pre_attn_residual_staging, h * @sizeOf(f32));

        // 2. GPU RMSNorm (bit-exact 1.0/sqrt match to CPU)
        const rmsnorm_pc = RmsNormPC{ .N = ffn.hidden_size, .eps = ffn.rms_norm_eps };
        self.ctx.cmdDispatch(
            &ffn.rmsnorm_pipeline,
            descs.rmsnorm_desc,
            1, 1, 1,
            std.mem.asBytes(&rmsnorm_pc),
        );

        // 3a. qkv (barrier from cmdDispatch ensures RMSNorm finished)
        self.ctx.cmdDispatch(
            descs.proj[0].pipeline,
            descs.proj[0].desc_set,
            descs.proj[0].pc.M, 1, 1,
            std.mem.asBytes(&descs.proj[0].pc),
        );
        // 3b-d. z, b, a — parallel with qkv (all read hidden_staging)
        self.ctx.cmdDispatchNoBarrier(
            descs.proj[1].pipeline, descs.proj[1].desc_set,
            descs.proj[1].pc.M, 1, 1, std.mem.asBytes(&descs.proj[1].pc),
        );
        self.ctx.cmdDispatchNoBarrier(
            descs.proj[2].pipeline, descs.proj[2].desc_set,
            descs.proj[2].pc.M, 1, 1, std.mem.asBytes(&descs.proj[2].pc),
        );
        self.ctx.cmdDispatchNoBarrier(
            descs.proj[3].pipeline, descs.proj[3].desc_set,
            descs.proj[3].pc.M, 1, 1, std.mem.asBytes(&descs.proj[3].pc),
        );

        try self.ctx.submitAndWait();

        // Readback projections only (hidden_staging stays GPU-resident, but is now
        // RMSNorm'd — it'll be overwritten by oproj_ffn before being used as input).
        try self.ctx.readbackFromBuffer(&fip.proj_out[0], std.mem.sliceAsBytes(qkv));
        try self.ctx.readbackFromBuffer(&fip.proj_out[1], std.mem.sliceAsBytes(z));
        try self.ctx.readbackFromBuffer(&fip.proj_out[2], std.mem.sliceAsBytes(b));
        try self.ctx.readbackFromBuffer(&fip.proj_out[3], std.mem.sliceAsBytes(a));
    }
    pub fn dispatchLayerStartDn(layer_idx: usize, qkv: []f32, z: []f32, b: []f32, a: []f32) bool {
        const self = global_instance orelse return false;
        self.gpuLayerStartDn(layer_idx, qkv, z, b, a) catch return false;
        return true;
    }

    /// Phase 2B: Fused full-attention input projections with GPU-resident hidden state.
    /// Same shape as gpuLayerStartDn but with q/k/v projections.
    pub fn gpuLayerStartFa(
        self: *Self,
        layer_idx: usize,
        q: []f32,
        k: []f32,
        v: []f32,
    ) !void {
        const ffn = self.fused_ffn orelse return error.OutOfMemory;
        const fip = self.fused_input_proj orelse return error.OutOfMemory;
        const descs = fip.layer_descs[layer_idx];
        const h: usize = ffn.hidden_size;

        try self.ctx.beginCommandBuffer();
        self.ctx.cmdCopyBuffer(&ffn.hidden_staging, &ffn.pre_attn_residual_staging, h * @sizeOf(f32));

        const rmsnorm_pc = RmsNormPC{ .N = ffn.hidden_size, .eps = ffn.rms_norm_eps };
        self.ctx.cmdDispatch(
            &ffn.rmsnorm_pipeline,
            descs.rmsnorm_desc,
            1, 1, 1,
            std.mem.asBytes(&rmsnorm_pc),
        );

        self.ctx.cmdDispatch(
            descs.proj[0].pipeline, descs.proj[0].desc_set,
            descs.proj[0].pc.M, 1, 1, std.mem.asBytes(&descs.proj[0].pc),
        );
        self.ctx.cmdDispatchNoBarrier(
            descs.proj[1].pipeline, descs.proj[1].desc_set,
            descs.proj[1].pc.M, 1, 1, std.mem.asBytes(&descs.proj[1].pc),
        );
        self.ctx.cmdDispatchNoBarrier(
            descs.proj[2].pipeline, descs.proj[2].desc_set,
            descs.proj[2].pc.M, 1, 1, std.mem.asBytes(&descs.proj[2].pc),
        );

        try self.ctx.submitAndWait();

        try self.ctx.readbackFromBuffer(&fip.proj_out[0], std.mem.sliceAsBytes(q));
        try self.ctx.readbackFromBuffer(&fip.proj_out[1], std.mem.sliceAsBytes(k));
        try self.ctx.readbackFromBuffer(&fip.proj_out[2], std.mem.sliceAsBytes(v));
    }
    pub fn dispatchLayerStartFa(layer_idx: usize, q: []f32, k: []f32, v: []f32) bool {
        const self = global_instance orelse return false;
        self.gpuLayerStartFa(layer_idx, q, k, v) catch return false;
        return true;
    }

    /// Phase 2B: Fused o_proj + post-attn residual + FFN with GPU-resident hidden state.
    /// Caller must have invoked gpuLayerStart{Dn,Fa} first this layer (which snapshotted
    /// the pre-attention hidden into pre_attn_residual_staging).
    /// Sequence:
    ///   1. o_proj SGEMV: oproj_out_gpu = W_o @ attn_out_staging
    ///   2. add_dup: hidden_staging = residual_staging = oproj_out_gpu + pre_attn_residual_staging
    ///   3. RMSNorm hidden_staging in place (post_attn_layernorm)
    ///   4. gate / up SGEMVs (parallel)
    ///   5. SiLU×mul
    ///   6. down SGEMV: hidden_staging = W_down @ ffn_gate_gpu
    ///   7. residual_add: hidden_staging += residual_staging
    /// Hidden_staging remains GPU-resident — no readback. Next layer's gpuLayerStart*
    /// (or gpuTokenEnd at end of token) handles the next step.
    pub fn gpuOprojFfnFusedResident(
        self: *Self,
        attn_out: []const f32,
        layer_idx: usize,
    ) !void {
        const ffn = self.fused_ffn orelse return error.OutOfMemory;
        const descs = ffn.layer_descs[layer_idx];

        std.debug.assert(attn_out.len <= ffn.attn_out_capacity);

        // Only the o_proj input still needs uploading (attention runs on CPU).
        try self.ctx.uploadToBuffer(&ffn.attn_out_staging, std.mem.sliceAsBytes(attn_out));

        try self.ctx.beginCommandBuffer();

        // 1. o_proj
        self.ctx.cmdDispatch(
            descs.oproj.pipeline, descs.oproj.desc_set,
            descs.oproj.pc.M, 1, 1, std.mem.asBytes(&descs.oproj.pc),
        );

        // 2. add_dup: hidden = oproj_out + pre_attn_residual; residual = same (FFN snapshot)
        const add_dup_pc = ResidualAddPC{ .count = ffn.hidden_size };
        self.ctx.cmdDispatch(
            &ffn.add_dup_pipeline, ffn.add_dup_desc,
            (ffn.hidden_size + 255) / 256, 1, 1, std.mem.asBytes(&add_dup_pc),
        );

        // 3. RMSNorm
        const rmsnorm_pc = RmsNormPC{ .N = ffn.hidden_size, .eps = ffn.rms_norm_eps };
        self.ctx.cmdDispatch(
            &ffn.rmsnorm_pipeline, descs.rmsnorm_desc,
            1, 1, 1, std.mem.asBytes(&rmsnorm_pc),
        );

        // 4. gate / up
        self.ctx.cmdDispatch(
            descs.gate.pipeline, descs.gate.desc_set,
            descs.gate.pc.M, 1, 1, std.mem.asBytes(&descs.gate.pc),
        );
        self.ctx.cmdDispatchNoBarrier(
            descs.up.pipeline, descs.up.desc_set,
            descs.up.pc.M, 1, 1, std.mem.asBytes(&descs.up.pc),
        );

        // 5. SiLU×mul
        const silu_pc = SiLuMulPC{ .N = ffn.intermediate_size };
        self.ctx.cmdDispatch(
            &ffn.silu_mul_pipeline, ffn.silu_mul_desc,
            (ffn.intermediate_size + 255) / 256, 1, 1, std.mem.asBytes(&silu_pc),
        );

        // 6. down
        self.ctx.cmdDispatch(
            descs.down.pipeline, descs.down.desc_set,
            descs.down.pc.M, 1, 1, std.mem.asBytes(&descs.down.pc),
        );

        // 7. FFN residual add
        const res_pc = ResidualAddPC{ .count = ffn.hidden_size };
        self.ctx.cmdDispatch(
            &ffn.residual_add_pipeline, ffn.residual_add_desc,
            (ffn.hidden_size + 255) / 256, 1, 1, std.mem.asBytes(&res_pc),
        );

        try self.ctx.submitAndWait();
        // No readback — hidden_staging stays GPU-resident for next layer.
    }
    pub fn dispatchOprojFfnResident(attn_out: []const f32, layer_idx: usize) bool {
        const self = global_instance orelse return false;
        self.gpuOprojFfnFusedResident(attn_out, layer_idx) catch return false;
        return true;
    }

    /// Static dispatch for fused attention input matching GpuAttnInputFn in model.zig.
    pub fn dispatchAttnInput(hidden: []const f32, layer_idx: usize, q: []f32, k: []f32, v: []f32) bool {
        const self = global_instance orelse return false;
        self.gpuAttnInputFused(hidden, layer_idx, q, k, v) catch return false;
        return true;
    }

    /// Report number of registered weights and total VRAM usage.
    pub fn printStats(self: *const Self) void {
        var total_bytes: usize = 0;
        for (self.weights.items) |entry| {
            const elems: usize = @as(usize, entry.rows) * @as(usize, entry.cols);
            switch (entry.kind) {
                .f32_dense => total_bytes += elems * @sizeOf(f32),
                .q8k_packed => total_bytes += elems + (elems / 32) * @sizeOf(f32),
            }
        }
        std.debug.print("GPU: {d} weight matrices uploaded ({d:.1} MB VRAM)\n", .{
            self.weights.items.len,
            @as(f64, @floatFromInt(total_bytes)) / (1024.0 * 1024.0),
        });
    }

    pub fn deinit(self: *Self) void {
        // Clean up fused input projections resources
        if (self.fused_input_proj) |*fip| {
            for (fip.input_norm_bufs) |*buf| self.ctx.destroyBuffer(buf);
            self.allocator.free(fip.input_norm_bufs);
            self.allocator.free(fip.layer_descs);
            for (&fip.proj_out) |*buf| self.ctx.destroyBuffer(buf);
            self.fused_input_proj = null;
        }

        // Clean up fused FFN resources
        if (self.fused_ffn) |*ffn| {
            for (ffn.norm_weight_bufs) |*buf| self.ctx.destroyBuffer(buf);
            self.allocator.free(ffn.norm_weight_bufs);
            self.allocator.free(ffn.layer_descs);
            self.ctx.destroyBuffer(&ffn.hidden_staging);
            self.ctx.destroyBuffer(&ffn.residual_staging);
            self.ctx.destroyBuffer(&ffn.ffn_gate_gpu);
            self.ctx.destroyBuffer(&ffn.ffn_up_gpu);
            self.ctx.destroyBuffer(&ffn.attn_out_staging);
            self.ctx.destroyBuffer(&ffn.pre_attn_residual_staging);
            self.ctx.destroyBuffer(&ffn.oproj_out_gpu);
            self.ctx.destroyBuffer(&ffn.final_norm_buf);
            self.ctx.destroyPipeline(&ffn.rmsnorm_pipeline);
            self.ctx.destroyPipeline(&ffn.silu_mul_pipeline);
            self.ctx.destroyPipeline(&ffn.residual_add_pipeline);
            self.ctx.destroyPipeline(&ffn.add_dup_pipeline);
            self.fused_ffn = null;
        }

        for (self.weights.items) |*entry| {
            self.ctx.destroyBuffer(&entry.primary_buf);
            if (entry.kind == .q8k_packed) {
                self.ctx.destroyBuffer(&entry.scales_buf);
            }
        }
        self.weights.deinit(self.allocator);
        self.weight_map.deinit(self.allocator);

        self.ctx.destroyBuffer(&self.input_staging);
        self.ctx.destroyBuffer(&self.output_staging);
        self.ctx.destroyPipeline(&self.sgemv_pipeline);
        self.ctx.destroyPipeline(&self.sgemv_q8k_pipeline);
        self.ctx.deinit();

        if (global_instance == self) global_instance = null;
    }
};

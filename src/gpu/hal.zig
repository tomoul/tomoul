// src/gpu/hal.zig
//
// GPU Hardware Abstraction Layer (HAL)
//
// Defines the backend-agnostic interface for GPU compute acceleration.
// All GPU backends (Vulkan, Metal, CUDA) and the CPU fallback implement
// this interface, enabling transparent GPU acceleration:
//
//   const backend = try GpuBackend.init(allocator);  // auto-selects best available
//   try backend.uploadWeights(embeddings, layers);
//   try backend.forward(input_ids, output);
//   backend.deinit();
//
// Platform dispatch (comptime):
//   Linux/Windows/Android → Vulkan (dlopen libvulkan)
//   macOS/iOS             → Metal  (Obj-C runtime)
//   Fallback              → CPU    (no-op backend, runs existing CPU path)
//
// Runtime: If the native GPU library fails to load or no device is found,
// silently falls back to CPU. The user never sees an error.

const std = @import("std");
const builtin = @import("builtin");

// ============================================================================
// Backend Buffer Handle
// ============================================================================

/// Opaque GPU buffer handle. Backend-specific data stored internally.
/// Users only hold handles; the backend manages all memory.
pub const BufferHandle = struct {
    id: u32,
};

// ============================================================================
// Transformer Weight/Config Types (backend-agnostic)
// ============================================================================

pub const HalConfig = struct {
    hidden_dim: u32,
    num_heads: u32,
    head_dim: u32,
    ffn_dim: u32,
    num_layers: u32,
    vocab_size: u32,
    max_seq_len: u32,
    max_batch_tokens: u32,
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
// Backend Interface
// ============================================================================

pub const BackendType = enum {
    vulkan,
    metal,
    cpu,
};

pub const BackendError = error{
    InitFailed,
    DeviceNotFound,
    ShaderLoadFailed,
    BufferCreationFailed,
    UploadFailed,
    DispatchFailed,
    ReadbackFailed,
    OutOfMemory,
};

/// The GPU backend interface. All backends expose these operations.
/// This is a tagged union — holds exactly one backend implementation.
pub const GpuBackend = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    backend_type: BackendType,

    pub const VTable = struct {
        /// Run forward pass for a single sentence (token IDs → embedding vector)
        forward: *const fn (ptr: *anyopaque, input_ids: []const u32, output: []f32) anyerror!void,

        /// Run batched forward pass (multiple sentences → flat output buffer)
        forwardBatch: *const fn (ptr: *anyopaque, batch_ids: []const []const u32, output: []f32) anyerror!void,

        /// Get the device name for logging
        getDeviceName: *const fn (ptr: *anyopaque) []const u8,

        /// Release all GPU resources
        deinit: *const fn (ptr: *anyopaque) void,
    };

    /// Forward pass: token IDs → normalized embedding vector
    pub fn forward(self: GpuBackend, input_ids: []const u32, output: []f32) !void {
        return self.vtable.forward(self.ptr, input_ids, output);
    }

    /// Batched forward pass: multiple sentences → flat output buffer
    pub fn forwardBatch(self: GpuBackend, batch_ids: []const []const u32, output: []f32) !void {
        return self.vtable.forwardBatch(self.ptr, batch_ids, output);
    }

    /// Get device name (e.g. "NVIDIA GeForce RTX 4090", "Apple M2", "CPU fallback")
    pub fn getDeviceName(self: GpuBackend) []const u8 {
        return self.vtable.getDeviceName(self.ptr);
    }

    /// Release all resources
    pub fn deinit(self: GpuBackend) void {
        self.vtable.deinit(self.ptr);
    }
};

// ============================================================================
// Auto-detection: try GPU, fall back to CPU
// ============================================================================

/// Try to initialize the best available backend for this platform.
/// Returns a GPU backend if available, otherwise CPU fallback.
/// Never fails — always returns a usable backend.
pub fn autoDetect(
    allocator: std.mem.Allocator,
    config: HalConfig,
    embeddings: EmbeddingData,
    layers: []const LayerData,
) GpuBackend {
    // Try platform-native GPU backend first
    if (tryGpuBackend(allocator, config, embeddings, layers)) |backend| {
        return backend;
    }

    // Fall back to CPU — this never fails
    return cpuFallback();
}

/// Try to initialize the GPU backend for this platform.
/// Returns null if GPU is unavailable (no driver, no device, etc.)
fn tryGpuBackend(
    allocator: std.mem.Allocator,
    config: HalConfig,
    embeddings: EmbeddingData,
    layers: []const LayerData,
) ?GpuBackend {
    switch (comptime builtin.os.tag) {
        .linux, .windows => {
            // Vulkan backend (no circular dependency — vulkan_backend doesn't import hal)
            const VulkanBackend = @import("vulkan_backend");
            const gpu_fwd = @import("vulkan_forward");

            // HAL types have identical layout to gpu_fwd types, safe to reinterpret
            const gpu_config = gpu_fwd.GpuConfig{
                .hidden_dim = config.hidden_dim,
                .num_heads = config.num_heads,
                .head_dim = config.head_dim,
                .ffn_dim = config.ffn_dim,
                .num_layers = config.num_layers,
                .vocab_size = config.vocab_size,
                .max_seq_len = config.max_seq_len,
                .max_batch_tokens = config.max_batch_tokens,
            };

            const gpu_embeddings = gpu_fwd.EmbeddingData{
                .word_emb = embeddings.word_emb,
                .pos_emb = embeddings.pos_emb,
                .type_emb = embeddings.type_emb,
                .ln_gamma = embeddings.ln_gamma,
                .ln_beta = embeddings.ln_beta,
            };

            const gpu_layers: []const gpu_fwd.LayerData = @ptrCast(layers);

            const backend = VulkanBackend.init(allocator, gpu_config, gpu_embeddings, gpu_layers) catch |err| {
                std.log.info("GPU: Vulkan unavailable ({s}), using CPU fallback", .{@errorName(err)});
                return null;
            };

            return GpuBackend{
                .ptr = @ptrCast(backend),
                .vtable = &vulkan_vtable,
                .backend_type = .vulkan,
            };
        },
        .macos, .ios => {
            // Metal backend (no circular dependency — metal_backend doesn't import hal)
            const MetalBackend = @import("metal_backend");
            const metal_fwd = @import("metal_forward");

            const metal_config = metal_fwd.GpuConfig{
                .hidden_dim = config.hidden_dim,
                .num_heads = config.num_heads,
                .head_dim = config.head_dim,
                .ffn_dim = config.ffn_dim,
                .num_layers = config.num_layers,
                .vocab_size = config.vocab_size,
                .max_seq_len = config.max_seq_len,
                .max_batch_tokens = config.max_batch_tokens,
            };

            const metal_embeddings = metal_fwd.EmbeddingData{
                .word_emb = embeddings.word_emb,
                .pos_emb = embeddings.pos_emb,
                .type_emb = embeddings.type_emb,
                .ln_gamma = embeddings.ln_gamma,
                .ln_beta = embeddings.ln_beta,
            };

            const metal_layers: []const metal_fwd.LayerData = @ptrCast(layers);

            const backend = MetalBackend.init(allocator, metal_config, metal_embeddings, metal_layers) catch |err| {
                std.log.info("GPU: Metal unavailable ({s}), using CPU fallback", .{@errorName(err)});
                return null;
            };

            return GpuBackend{
                .ptr = @ptrCast(backend),
                .vtable = &metal_vtable,
                .backend_type = .metal,
            };
        },
        else => return null,
    }
}

// Vulkan vtable — bridges VulkanBackend methods to GpuBackend interface
fn vulkanForward(ptr: *anyopaque, input_ids: []const u32, output: []f32) anyerror!void {
    const VulkanBackend = @import("vulkan_backend");
    const self: *VulkanBackend = @ptrCast(@alignCast(ptr));
    try self.forward(input_ids, output);
}

fn vulkanForwardBatch(ptr: *anyopaque, batch_ids: []const []const u32, output: []f32) anyerror!void {
    const VulkanBackend = @import("vulkan_backend");
    const self: *VulkanBackend = @ptrCast(@alignCast(ptr));
    try self.forwardBatch(batch_ids, output);
}

fn vulkanGetDeviceName(ptr: *anyopaque) []const u8 {
    const VulkanBackend = @import("vulkan_backend");
    const self: *const VulkanBackend = @ptrCast(@alignCast(ptr));
    return self.getDeviceName();
}

fn vulkanDeinit(ptr: *anyopaque) void {
    const VulkanBackend = @import("vulkan_backend");
    const self: *VulkanBackend = @ptrCast(@alignCast(ptr));
    self.deinit();
}

const vulkan_vtable: GpuBackend.VTable = .{
    .forward = &vulkanForward,
    .forwardBatch = &vulkanForwardBatch,
    .getDeviceName = &vulkanGetDeviceName,
    .deinit = &vulkanDeinit,
};

// Metal vtable — bridges MetalBackend methods to GpuBackend interface
fn metalForward(ptr: *anyopaque, input_ids: []const u32, output: []f32) anyerror!void {
    const MetalBackend = @import("metal_backend");
    const self: *MetalBackend = @ptrCast(@alignCast(ptr));
    try self.forward(input_ids, output);
}

fn metalForwardBatch(ptr: *anyopaque, batch_ids: []const []const u32, output: []f32) anyerror!void {
    const MetalBackend = @import("metal_backend");
    const self: *MetalBackend = @ptrCast(@alignCast(ptr));
    try self.forwardBatch(batch_ids, output);
}

fn metalGetDeviceName(ptr: *anyopaque) []const u8 {
    const MetalBackend = @import("metal_backend");
    const self: *const MetalBackend = @ptrCast(@alignCast(ptr));
    return self.getDeviceName();
}

fn metalDeinit(ptr: *anyopaque) void {
    const MetalBackend = @import("metal_backend");
    const self: *MetalBackend = @ptrCast(@alignCast(ptr));
    self.deinit();
}

const metal_vtable: GpuBackend.VTable = .{
    .forward = &metalForward,
    .forwardBatch = &metalForwardBatch,
    .getDeviceName = &metalGetDeviceName,
    .deinit = &metalDeinit,
};

/// CPU fallback — wraps the existing CPU inference path behind the HAL interface.
/// This is a no-op backend that signals the caller to use the CPU path.
fn cpuFallback() GpuBackend {
    return GpuBackend{
        .ptr = undefined,
        .vtable = &cpu_vtable,
        .backend_type = .cpu,
    };
}

fn cpuForward(_: *anyopaque, _: []const u32, _: []f32) anyerror!void {
    return error.InitFailed; // signals caller to use CPU path
}

fn cpuForwardBatch(_: *anyopaque, _: []const []const u32, _: []f32) anyerror!void {
    return error.InitFailed;
}

fn cpuGetDeviceName(_: *anyopaque) []const u8 {
    return "CPU fallback";
}

fn cpuDeinit(_: *anyopaque) void {}

const cpu_vtable: GpuBackend.VTable = .{
    .forward = &cpuForward,
    .forwardBatch = &cpuForwardBatch,
    .getDeviceName = &cpuGetDeviceName,
    .deinit = &cpuDeinit,
};

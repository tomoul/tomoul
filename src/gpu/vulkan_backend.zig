// src/gpu/vulkan_backend.zig
//
// Vulkan HAL Backend
//
// Wraps GpuForward (Vulkan compute shaders) and provides forward/forwardBatch
// methods that hal.zig bridges into the GpuBackend vtable.
//
// This module has NO dependency on hal.zig — avoiding circular imports.
// hal.zig imports this module and wraps it behind the GpuBackend interface.
//
// Platforms: Linux, Windows, Android, WSL2 (via dzn/D3D12)

const std = @import("std");
const gpu_fwd = @import("vulkan_forward");

const Self = @This();

allocator: std.mem.Allocator,
fwd: gpu_fwd.GpuForward,
device_name_buf: [256]u8,
device_name_len: usize,

/// Initialize Vulkan backend: create device, upload weights, compile shaders.
/// Config/Embedding/Layer types are from vulkan_forward (no HAL dependency).
pub fn init(
    allocator: std.mem.Allocator,
    config: gpu_fwd.GpuConfig,
    embeddings: gpu_fwd.EmbeddingData,
    layers: []const gpu_fwd.LayerData,
) !*Self {
    const fwd = try gpu_fwd.GpuForward.init(allocator, config, embeddings, layers);

    const self = try allocator.create(Self);
    self.* = Self{
        .allocator = allocator,
        .fwd = fwd,
        .device_name_buf = undefined,
        .device_name_len = 0,
    };

    // Cache device name
    const name = self.fwd.ctx.getDeviceName();
    self.device_name_len = name.len;
    @memcpy(self.device_name_buf[0..name.len], name);

    return self;
}

pub fn forward(self: *Self, input_ids: []const u32, output: []f32) !void {
    try self.fwd.forward(input_ids, output);
}

pub fn forwardBatch(self: *Self, batch_ids: []const []const u32, output: []f32) !void {
    try self.fwd.forwardBatch(batch_ids, output);
}

pub fn getDeviceName(self: *const Self) []const u8 {
    return self.device_name_buf[0..self.device_name_len];
}

pub fn deinit(self: *Self) void {
    self.fwd.deinit();
    self.allocator.destroy(self);
}

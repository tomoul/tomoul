// src/gpu/webgpu_backend.zig
//
// WebGPU HAL Backend
//
// Wraps GpuForward (WebGPU compute shaders) and provides forward/forwardBatch
// methods that hal.zig bridges into the GpuBackend vtable.
//
// This module has NO dependency on hal.zig — avoiding circular imports.
// hal.zig imports this module and wraps it behind the GpuBackend interface.
//
// Platform: wasm32 (browser with WebGPU support)

const std = @import("std");
const gpu_fwd = @import("webgpu_forward");

const Self = @This();

allocator: std.mem.Allocator,
fwd: gpu_fwd.GpuForward,
device_name_buf: [256]u8,
device_name_len: usize,

/// Initialize WebGPU backend: create device, upload weights, compile shaders.
/// Config/Embedding/Layer types are from webgpu_forward (no HAL dependency).
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
    const copy_len = @min(name.len, 255);
    self.device_name_len = copy_len;
    @memcpy(self.device_name_buf[0..copy_len], name[0..copy_len]);

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

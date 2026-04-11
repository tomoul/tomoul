// src/gpu/metal_backend.zig
//
// Metal HAL Backend
//
// Wraps MetalForward (Metal compute shaders) and provides forward/forwardBatch
// methods that hal.zig bridges into the GpuBackend vtable.
//
// This module has NO dependency on hal.zig — avoiding circular imports.
// hal.zig imports this module and wraps it behind the GpuBackend interface.
//
// Platforms: macOS, iOS (Apple Silicon with Metal support)

const std = @import("std");
const metal_fwd = @import("metal_forward");

const Self = @This();

allocator: std.mem.Allocator,
fwd: metal_fwd.MetalForward,
device_name_buf: [256]u8,
device_name_len: usize,

/// Initialize Metal backend: create device, upload weights, compile shaders.
/// Config/Embedding/Layer types are from metal_forward (no HAL dependency).
pub fn init(
    allocator: std.mem.Allocator,
    config: metal_fwd.GpuConfig,
    embeddings: metal_fwd.EmbeddingData,
    layers: []const metal_fwd.LayerData,
) !*Self {
    const fwd = try metal_fwd.MetalForward.init(allocator, config, embeddings, layers);

    const self = try allocator.create(Self);
    self.* = Self{
        .allocator = allocator,
        .fwd = fwd,
        .device_name_buf = undefined,
        .device_name_len = 0,
    };

    // Cache device name (null-terminated for C API)
    const name = self.fwd.ctx.getDeviceName();
    const copy_len = @min(name.len, 255);
    self.device_name_len = copy_len + 1; // include null terminator
    @memcpy(self.device_name_buf[0..copy_len], name[0..copy_len]);
    self.device_name_buf[copy_len] = 0;

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

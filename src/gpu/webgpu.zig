// src/gpu/webgpu.zig
//
// WebGPU Bridge for WASM
//
// Provides a Zig-friendly interface to WebGPU operations via extern imports.
// The JavaScript host implements these functions using the browser's WebGPU API.
//
// Design: WASM calls extern JS functions to create buffers, upload data,
// compile pipelines, record dispatches, and submit work. The JS host
// handles async GPU operations (mapAsync, onSubmittedWorkDone) outside
// the WASM call boundary.
//
// Buffer handles and pipeline handles are opaque u32 IDs managed by JS.
//
// Platform: wasm32-freestanding only (comptime-guarded)

const std = @import("std");
const builtin = @import("builtin");

// ============================================================================
// Handle Types
// ============================================================================

/// Opaque GPU buffer handle (managed by JS host)
pub const GpuBuffer = u32;

/// Opaque compute pipeline handle (managed by JS host)
pub const ComputePipeline = u32;

/// Sentinel value for invalid handles
pub const INVALID_HANDLE: u32 = 0xFFFFFFFF;

// ============================================================================
// Error Types
// ============================================================================

pub const WebGpuError = error{
    InitFailed,
    BufferCreationFailed,
    PipelineCreationFailed,
    DispatchFailed,
    UploadFailed,
    ReadbackFailed,
    OutOfMemory,
};

// ============================================================================
// Extern Imports (implemented by JS host)
// ============================================================================

extern "webgpu" fn wgpuInit() u32;

/// Create a GPU buffer. Returns handle or INVALID_HANDLE on failure.
/// writable: 1 = STORAGE (read_write), 0 = STORAGE (read-only)
extern "webgpu" fn wgpuCreateBuffer(size: u32, writable: u32) u32;

/// Upload data from WASM memory to a GPU buffer.
extern "webgpu" fn wgpuUploadToBuffer(handle: u32, ptr: [*]const u8, size: u32) void;

/// Read data from a GPU buffer into WASM memory.
/// NOTE: JS host must ensure GPU work is complete before this is called.
extern "webgpu" fn wgpuReadbackFromBuffer(handle: u32, ptr: [*]u8, size: u32) void;

/// Destroy a GPU buffer.
extern "webgpu" fn wgpuDestroyBuffer(handle: u32) void;

/// Create a compute pipeline from WGSL source.
/// wgsl_ptr/len: pointer to WGSL source in WASM memory
/// entry_ptr/len: entry point function name
/// num_storage_bindings: number of storage buffer bindings
/// has_uniform: 1 if pipeline uses a uniform buffer (last binding)
/// Returns pipeline handle or INVALID_HANDLE on failure.
extern "webgpu" fn wgpuCreatePipeline(
    wgsl_ptr: [*]const u8,
    wgsl_len: u32,
    entry_ptr: [*]const u8,
    entry_len: u32,
    num_storage_bindings: u32,
    has_uniform: u32,
) u32;

/// Destroy a compute pipeline.
extern "webgpu" fn wgpuDestroyPipeline(handle: u32) void;

/// Begin recording a command buffer. Clears any previous recording.
extern "webgpu" fn wgpuBeginCommandBuffer() void;

/// Set a storage buffer binding for the next dispatch.
extern "webgpu" fn wgpuSetBinding(slot: u32, handle: u32) void;

/// Set the uniform buffer data for the next dispatch.
/// Copies from WASM memory. Size must match the pipeline's uniform binding.
extern "webgpu" fn wgpuSetUniform(ptr: [*]const u8, size: u32) void;

/// Record a compute dispatch with the given pipeline and workgroup counts.
/// Uses the bindings/uniform set by prior wgpuSetBinding/wgpuSetUniform calls.
extern "webgpu" fn wgpuDispatch(pipeline: u32, groups_x: u32, groups_y: u32, groups_z: u32) void;

/// Submit all recorded commands and signal the JS host to execute them.
/// From WASM perspective this returns immediately; actual GPU work is async.
extern "webgpu" fn wgpuSubmit() void;

/// Get the GPU device name. Writes to ptr, returns actual length.
extern "webgpu" fn wgpuGetDeviceName(ptr: [*]u8, max_len: u32) u32;

// ============================================================================
// Zig-Friendly Wrapper
// ============================================================================

pub const WebGpuContext = struct {
    initialized: bool,

    const Self = @This();

    pub fn init() WebGpuError!Self {
        if (wgpuInit() == 0) {
            return WebGpuError.InitFailed;
        }
        return Self{ .initialized = true };
    }

    pub fn deinit(self: *Self) void {
        self.initialized = false;
    }

    pub fn createStorageBuffer(self: *const Self, size: usize, writable: bool) WebGpuError!GpuBuffer {
        _ = self;
        const handle = wgpuCreateBuffer(@intCast(size), if (writable) @as(u32, 1) else @as(u32, 0));
        if (handle == INVALID_HANDLE) return WebGpuError.BufferCreationFailed;
        return handle;
    }

    pub fn uploadToBuffer(self: *const Self, buf: *const GpuBuffer, data: []const u8) WebGpuError!void {
        _ = self;
        wgpuUploadToBuffer(buf.*, data.ptr, @intCast(data.len));
    }

    pub fn readbackFromBuffer(self: *const Self, buf: *const GpuBuffer, data: []u8) WebGpuError!void {
        _ = self;
        wgpuReadbackFromBuffer(buf.*, data.ptr, @intCast(data.len));
    }

    pub fn destroyBuffer(self: *const Self, buf: *const GpuBuffer) void {
        _ = self;
        if (buf.* != INVALID_HANDLE) {
            wgpuDestroyBuffer(buf.*);
        }
    }

    pub fn createComputePipeline(
        self: *const Self,
        wgsl_source: []const u8,
        entry_point: []const u8,
        num_storage_bindings: u32,
        has_uniform: bool,
    ) WebGpuError!ComputePipeline {
        _ = self;
        const handle = wgpuCreatePipeline(
            wgsl_source.ptr,
            @intCast(wgsl_source.len),
            entry_point.ptr,
            @intCast(entry_point.len),
            num_storage_bindings,
            if (has_uniform) @as(u32, 1) else @as(u32, 0),
        );
        if (handle == INVALID_HANDLE) return WebGpuError.PipelineCreationFailed;
        return handle;
    }

    pub fn destroyPipeline(self: *const Self, pipe: *const ComputePipeline) void {
        _ = self;
        if (pipe.* != INVALID_HANDLE) {
            wgpuDestroyPipeline(pipe.*);
        }
    }

    pub fn beginCommandBuffer(self: *const Self) void {
        _ = self;
        wgpuBeginCommandBuffer();
    }

    pub fn bindBuffers(self: *const Self, buffers: []const GpuBuffer) void {
        _ = self;
        for (buffers, 0..) |buf, i| {
            wgpuSetBinding(@intCast(i), buf);
        }
    }

    pub fn setUniform(self: *const Self, data: []const u8) void {
        _ = self;
        wgpuSetUniform(data.ptr, @intCast(data.len));
    }

    pub fn cmdDispatch(
        self: *const Self,
        pipeline: ComputePipeline,
        buffers: []const GpuBuffer,
        uniform_data: []const u8,
        groups_x: u32,
        groups_y: u32,
        groups_z: u32,
    ) void {
        _ = self;
        // Set bindings
        for (buffers, 0..) |buf, i| {
            wgpuSetBinding(@intCast(i), buf);
        }
        // Set uniform (params)
        if (uniform_data.len > 0) {
            wgpuSetUniform(uniform_data.ptr, @intCast(uniform_data.len));
        }
        // Dispatch
        wgpuDispatch(pipeline, groups_x, groups_y, groups_z);
    }

    pub fn submit(self: *const Self) void {
        _ = self;
        wgpuSubmit();
    }

    pub fn getDeviceName(self: *const Self) []const u8 {
        _ = self;
        const S = struct {
            var name_buf: [256]u8 = undefined;
            var name_len: u32 = 0;
        };
        if (S.name_len == 0) {
            S.name_len = wgpuGetDeviceName(&S.name_buf, 256);
        }
        return S.name_buf[0..S.name_len];
    }
};

// src/gpu/metal.zig
//
// Metal Compute Backend for Tomoul
//
// Provides GPU-accelerated compute via Apple Metal compute shaders.
// Used for SGEMM, attention, and full transformer forward passes.
//
// ZERO link-time Metal dependency: Metal.framework is loaded at runtime via dlopen.
// Obj-C runtime (libobjc, always available on macOS) dispatches all Metal API calls.
// If Metal.framework is not found or no GPU device exists, init() returns error.
//
// Architecture:
//   init() → dlopen Metal.framework → MTLCreateSystemDefaultDevice → command queue
//   createBuffer() → MTLDevice.newBufferWithLength (shared memory, CPU+GPU visible)
//   uploadToBuffer() → memcpy to buffer.contents (unified memory, no staging needed)
//   compilePipeline() → newLibraryWithSource → newFunctionWithName → newComputePipelineState
//   beginCommandBuffer() → commandBuffer + computeCommandEncoder
//   cmdDispatch() → set pipeline, bind buffers, set params, dispatch, barrier
//   submitAndWait() → endEncoding + commit + waitUntilCompleted
//   readbackFromBuffer() → memcpy from buffer.contents
//   deinit() → release all Metal objects

const std = @import("std");

pub const MetalError = error{
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
};

// ============================================================================
// Obj-C Runtime Types
// ============================================================================

const ObjcId = *anyopaque;
const ObjcSEL = *anyopaque;
const NSUInteger = u64;

/// MTLSize — matches Metal's struct definition (3 × NSUInteger = 24 bytes)
pub const MTLSize = extern struct {
    width: u64,
    height: u64,
    depth: u64,
};

/// MTLBarrierScope.buffers = 1 << 0
const MTLBarrierScopeBuffers: NSUInteger = 1;

/// MTLResourceStorageModeShared = 0 (CPU + GPU accessible on unified memory)
const MTLResourceStorageModeShared: NSUInteger = 0;

// ============================================================================
// Obj-C Runtime Externals (from libobjc, always available via libc on macOS)
// ============================================================================

extern "c" fn objc_getClass(name: [*:0]const u8) ?*anyopaque;
extern "c" fn sel_registerName(name: [*:0]const u8) ObjcSEL;
extern "c" fn objc_msgSend() void;

// ============================================================================
// Typed msg_send Wrappers
//
// objc_msgSend is a variadic trampoline. We cast its address to the exact
// function pointer type matching each Obj-C method signature. On ARM64 this
// is safe — the AAPCS64 calling convention handles all argument passing
// through general/float registers consistently.
// ============================================================================

/// (self, _cmd) → id?
inline fn msg0(obj: ObjcId, sel: ObjcSEL) ?*anyopaque {
    const F = *const fn (ObjcId, ObjcSEL) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(&objc_msgSend))(obj, sel);
}

/// (self, _cmd) → void
inline fn msg0_void(obj: ObjcId, sel: ObjcSEL) void {
    const F = *const fn (ObjcId, ObjcSEL) callconv(.c) void;
    @as(F, @ptrCast(&objc_msgSend))(obj, sel);
}

/// (self, _cmd) → NSUInteger
inline fn msg0_uint(obj: ObjcId, sel: ObjcSEL) NSUInteger {
    const F = *const fn (ObjcId, ObjcSEL) callconv(.c) NSUInteger;
    return @as(F, @ptrCast(&objc_msgSend))(obj, sel);
}

/// (self, _cmd) → [*:0]const u8
inline fn msg0_cstr(obj: ObjcId, sel: ObjcSEL) [*:0]const u8 {
    const F = *const fn (ObjcId, ObjcSEL) callconv(.c) [*:0]const u8;
    return @as(F, @ptrCast(&objc_msgSend))(obj, sel);
}

/// (self, _cmd) → *anyopaque (non-null pointer, e.g. buffer.contents)
inline fn msg0_ptr(obj: ObjcId, sel: ObjcSEL) *anyopaque {
    const F = *const fn (ObjcId, ObjcSEL) callconv(.c) *anyopaque;
    return @as(F, @ptrCast(&objc_msgSend))(obj, sel);
}

/// (self, _cmd, id) → id?
inline fn msg1_id(obj: ObjcId, sel: ObjcSEL, arg: ObjcId) ?*anyopaque {
    const F = *const fn (ObjcId, ObjcSEL, ObjcId) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(&objc_msgSend))(obj, sel, arg);
}

/// (self, _cmd, id) → void
inline fn msg1_id_void(obj: ObjcId, sel: ObjcSEL, arg: ObjcId) void {
    const F = *const fn (ObjcId, ObjcSEL, ObjcId) callconv(.c) void;
    @as(F, @ptrCast(&objc_msgSend))(obj, sel, arg);
}

/// (self, _cmd, NSUInteger) → void
inline fn msg1_uint_void(obj: ObjcId, sel: ObjcSEL, arg: NSUInteger) void {
    const F = *const fn (ObjcId, ObjcSEL, NSUInteger) callconv(.c) void;
    @as(F, @ptrCast(&objc_msgSend))(obj, sel, arg);
}

/// (self, _cmd, [*:0]const u8) → id?   [class method: +stringWithUTF8String:]
inline fn msg1_cstr(obj: ObjcId, sel: ObjcSEL, arg: [*:0]const u8) ?*anyopaque {
    const F = *const fn (ObjcId, ObjcSEL, [*:0]const u8) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(&objc_msgSend))(obj, sel, arg);
}

/// (self, _cmd, NSUInteger, NSUInteger) → id?  [newBufferWithLength:options:]
inline fn msg2_uint(obj: ObjcId, sel: ObjcSEL, a: NSUInteger, b: NSUInteger) ?*anyopaque {
    const F = *const fn (ObjcId, ObjcSEL, NSUInteger, NSUInteger) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(&objc_msgSend))(obj, sel, a, b);
}

/// (self, _cmd, id, NSUInteger, NSUInteger) → void  [setBuffer:offset:atIndex:]
inline fn msg3_id_uint_uint(obj: ObjcId, sel: ObjcSEL, a: ObjcId, b: NSUInteger, c: NSUInteger) void {
    const F = *const fn (ObjcId, ObjcSEL, ObjcId, NSUInteger, NSUInteger) callconv(.c) void;
    @as(F, @ptrCast(&objc_msgSend))(obj, sel, a, b, c);
}

/// (self, _cmd, *const anyopaque, NSUInteger, NSUInteger) → void  [setBytes:length:atIndex:]
inline fn msg3_ptr_uint_uint(obj: ObjcId, sel: ObjcSEL, a: *const anyopaque, b: NSUInteger, c: NSUInteger) void {
    const F = *const fn (ObjcId, ObjcSEL, *const anyopaque, NSUInteger, NSUInteger) callconv(.c) void;
    @as(F, @ptrCast(&objc_msgSend))(obj, sel, a, b, c);
}

/// (self, _cmd, *const anyopaque, NSUInteger, NSUInteger) → id?  [newBufferWithBytes:length:options:]
inline fn msg3_ptr_uint_uint_id(obj: ObjcId, sel: ObjcSEL, a: *const anyopaque, b: NSUInteger, c: NSUInteger) ?*anyopaque {
    const F = *const fn (ObjcId, ObjcSEL, *const anyopaque, NSUInteger, NSUInteger) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(&objc_msgSend))(obj, sel, a, b, c);
}

/// (self, _cmd, id, ?id, *?id) → id?  [newLibraryWithSource:options:error:]
inline fn msg3_id_id_errptr(obj: ObjcId, sel: ObjcSEL, a: ObjcId, b: ?*anyopaque, c: *?*anyopaque) ?*anyopaque {
    const F = *const fn (ObjcId, ObjcSEL, ObjcId, ?*anyopaque, *?*anyopaque) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(&objc_msgSend))(obj, sel, a, b, c);
}

/// (self, _cmd, id, *?id) → id?  [newComputePipelineStateWithFunction:error:]
inline fn msg2_id_errptr(obj: ObjcId, sel: ObjcSEL, a: ObjcId, b: *?*anyopaque) ?*anyopaque {
    const F = *const fn (ObjcId, ObjcSEL, ObjcId, *?*anyopaque) callconv(.c) ?*anyopaque;
    return @as(F, @ptrCast(&objc_msgSend))(obj, sel, a, b);
}

/// (self, _cmd, MTLSize, MTLSize) → void  [dispatchThreadgroups:threadsPerThreadgroup:]
inline fn msg2_sizes(obj: ObjcId, sel: ObjcSEL, a: MTLSize, b: MTLSize) void {
    const F = *const fn (ObjcId, ObjcSEL, MTLSize, MTLSize) callconv(.c) void;
    @as(F, @ptrCast(&objc_msgSend))(obj, sel, a, b);
}

// ============================================================================
// Public Types
// ============================================================================

/// GPU buffer handle (wraps MTLBuffer)
pub const MetalBuffer = struct {
    obj: ObjcId, // MTLBuffer
    size: usize,
};

/// Compute pipeline handle (wraps MTLComputePipelineState)
pub const MetalPipeline = struct {
    state: ObjcId, // MTLComputePipelineState
};

// ============================================================================
// Metal Compute Context
// ============================================================================

pub const MetalContext = struct {
    device: ObjcId, // MTLDevice
    command_queue: ObjcId, // MTLCommandQueue
    library: ObjcId, // MTLLibrary (compiled MSL kernels)
    metal_lib_handle: *anyopaque, // dlopen handle for Metal.framework

    // Command recording state
    command_buffer: ?ObjcId, // current MTLCommandBuffer (null when not recording)
    encoder: ?ObjcId, // current MTLComputeCommandEncoder

    // Cached selectors (avoid repeated sel_registerName calls)
    sel_newBufferWithLength: ObjcSEL,
    sel_newBufferWithBytes: ObjcSEL,
    sel_contents: ObjcSEL,
    sel_release: ObjcSEL,
    sel_commandBuffer: ObjcSEL,
    sel_computeCommandEncoder: ObjcSEL,
    sel_setComputePipelineState: ObjcSEL,
    sel_setBuffer_offset_atIndex: ObjcSEL,
    sel_setBytes_length_atIndex: ObjcSEL,
    sel_dispatchThreadgroups: ObjcSEL,
    sel_memoryBarrierWithScope: ObjcSEL,
    sel_endEncoding: ObjcSEL,
    sel_commit: ObjcSEL,
    sel_waitUntilCompleted: ObjcSEL,
    sel_newFunctionWithName: ObjcSEL,
    sel_newComputePipelineState: ObjcSEL,

    // Device info
    device_name: [256]u8,
    device_name_len: usize,

    const Self = @This();

    // ====================================================================
    // Initialization
    // ====================================================================

    /// Initialize Metal: dlopen Metal.framework → get default device → create command queue
    /// → compile MSL shader library. Returns MetalNotAvailable if Metal is not present.
    pub fn init(msl_source: [*:0]const u8) MetalError!Self {
        var self: Self = undefined;

        // 1. dlopen Metal.framework (zero link-time dependency)
        const metal_path = "/System/Library/Frameworks/Metal.framework/Metal";
        self.metal_lib_handle = std.c.dlopen(metal_path, .{ .LAZY = true }) orelse {
            std.log.info("GPU: Metal.framework not found", .{});
            return MetalError.MetalNotAvailable;
        };
        errdefer _ = std.c.dlclose(self.metal_lib_handle);

        // 2. Get MTLCreateSystemDefaultDevice function
        const create_device_fn = @as(
            ?*const fn () callconv(.c) ?*anyopaque,
            @ptrCast(@alignCast(std.c.dlsym(self.metal_lib_handle, "MTLCreateSystemDefaultDevice"))),
        ) orelse {
            std.log.info("GPU: MTLCreateSystemDefaultDevice not found", .{});
            return MetalError.MetalNotAvailable;
        };

        // 3. Get default Metal device
        self.device = create_device_fn() orelse {
            std.log.info("GPU: No Metal device found", .{});
            return MetalError.DeviceNotFound;
        };

        // 4. Cache all selectors
        self.sel_newBufferWithLength = sel_registerName("newBufferWithLength:options:");
        self.sel_newBufferWithBytes = sel_registerName("newBufferWithBytes:length:options:");
        self.sel_contents = sel_registerName("contents");
        self.sel_release = sel_registerName("release");
        self.sel_commandBuffer = sel_registerName("commandBuffer");
        self.sel_computeCommandEncoder = sel_registerName("computeCommandEncoder");
        self.sel_setComputePipelineState = sel_registerName("setComputePipelineState:");
        self.sel_setBuffer_offset_atIndex = sel_registerName("setBuffer:offset:atIndex:");
        self.sel_setBytes_length_atIndex = sel_registerName("setBytes:length:atIndex:");
        self.sel_dispatchThreadgroups = sel_registerName("dispatchThreadgroups:threadsPerThreadgroup:");
        self.sel_memoryBarrierWithScope = sel_registerName("memoryBarrierWithScope:");
        self.sel_endEncoding = sel_registerName("endEncoding");
        self.sel_commit = sel_registerName("commit");
        self.sel_waitUntilCompleted = sel_registerName("waitUntilCompleted");
        self.sel_newFunctionWithName = sel_registerName("newFunctionWithName:");
        self.sel_newComputePipelineState = sel_registerName("newComputePipelineStateWithFunction:error:");

        // 5. Create command queue
        const sel_newCommandQueue = sel_registerName("newCommandQueue");
        self.command_queue = msg0(self.device, sel_newCommandQueue) orelse {
            std.log.info("GPU: Failed to create Metal command queue", .{});
            return MetalError.CommandQueueCreationFailed;
        };

        // 6. Get device name
        const sel_name = sel_registerName("name");
        const name_nsstring = msg0(self.device, sel_name);
        if (name_nsstring) |ns| {
            const sel_utf8 = sel_registerName("UTF8String");
            const cstr = msg0_cstr(ns, sel_utf8);
            const name_slice = std.mem.sliceTo(cstr, 0);
            self.device_name_len = @min(name_slice.len, 256);
            @memcpy(self.device_name[0..self.device_name_len], name_slice[0..self.device_name_len]);
        } else {
            const fallback = "Unknown Metal Device";
            self.device_name_len = fallback.len;
            @memcpy(self.device_name[0..fallback.len], fallback);
        }

        std.debug.print("GPU: Using Metal device: {s}\n", .{self.device_name[0..self.device_name_len]});

        // 7. Compile MSL shader library from embedded source
        self.library = try self.compileLibrary(msl_source);

        // 8. Init command recording state
        self.command_buffer = null;
        self.encoder = null;

        return self;
    }

    // ====================================================================
    // Shader Compilation
    // ====================================================================

    fn compileLibrary(self: *Self, source: [*:0]const u8) MetalError!ObjcId {
        // Create NSString from MSL source
        const NSString_class = objc_getClass("NSString") orelse return MetalError.LibraryCompilationFailed;
        const sel_stringWithUTF8 = sel_registerName("stringWithUTF8String:");
        const msl_nsstring = msg1_cstr(NSString_class, sel_stringWithUTF8, source) orelse {
            std.log.err("GPU: Failed to create NSString from MSL source", .{});
            return MetalError.LibraryCompilationFailed;
        };

        // Compile: [device newLibraryWithSource:options:error:]
        const sel_newLib = sel_registerName("newLibraryWithSource:options:error:");
        var err_obj: ?*anyopaque = null;
        const library = msg3_id_id_errptr(self.device, sel_newLib, msl_nsstring, null, &err_obj) orelse {
            // Extract error description for debugging
            if (err_obj) |e| {
                const sel_desc = sel_registerName("localizedDescription");
                if (msg0(e, sel_desc)) |desc_ns| {
                    const sel_utf8 = sel_registerName("UTF8String");
                    const desc_cstr = msg0_cstr(desc_ns, sel_utf8);
                    std.log.err("GPU: Metal shader compilation failed: {s}", .{desc_cstr});
                }
            }
            return MetalError.LibraryCompilationFailed;
        };

        return library;
    }

    // ====================================================================
    // Pipeline Creation
    // ====================================================================

    /// Create a compute pipeline for a named kernel function in the compiled library.
    pub fn createComputePipeline(self: *Self, kernel_name: [*:0]const u8) MetalError!MetalPipeline {
        // Create NSString for function name
        const NSString_class = objc_getClass("NSString") orelse return MetalError.FunctionNotFound;
        const sel_stringWithUTF8 = sel_registerName("stringWithUTF8String:");
        const name_nsstring = msg1_cstr(NSString_class, sel_stringWithUTF8, kernel_name) orelse {
            return MetalError.FunctionNotFound;
        };

        // Get function from library
        const function = msg1_id(self.library, self.sel_newFunctionWithName, name_nsstring) orelse {
            std.log.err("GPU: Metal kernel function '{s}' not found in library", .{kernel_name});
            return MetalError.FunctionNotFound;
        };
        defer msg0_void(function, self.sel_release);

        // Create pipeline state
        var err_obj: ?*anyopaque = null;
        const pipeline_state = msg2_id_errptr(self.device, self.sel_newComputePipelineState, function, &err_obj) orelse {
            if (err_obj) |e| {
                const sel_desc = sel_registerName("localizedDescription");
                if (msg0(e, sel_desc)) |desc_ns| {
                    const sel_utf8 = sel_registerName("UTF8String");
                    const desc_cstr = msg0_cstr(desc_ns, sel_utf8);
                    std.log.err("GPU: Pipeline creation failed for '{s}': {s}", .{ kernel_name, desc_cstr });
                }
            }
            return MetalError.PipelineCreationFailed;
        };

        return MetalPipeline{ .state = pipeline_state };
    }

    // ====================================================================
    // Buffer Management
    // ====================================================================

    /// Create a GPU buffer with shared storage (CPU+GPU accessible on unified memory).
    pub fn createBuffer(self: *Self, size: usize) MetalError!MetalBuffer {
        const buf = msg2_uint(self.device, self.sel_newBufferWithLength, @intCast(size), MTLResourceStorageModeShared) orelse {
            return MetalError.BufferCreationFailed;
        };
        return MetalBuffer{ .obj = buf, .size = size };
    }

    /// Upload data to a buffer (memcpy — unified memory, no staging needed).
    pub fn uploadToBuffer(self: *Self, buf: *const MetalBuffer, data: []const u8) void {
        _ = self;
        const dst = msg0_ptr(buf.obj, sel_registerName("contents"));
        const dst_slice: [*]u8 = @ptrCast(dst);
        @memcpy(dst_slice[0..data.len], data);
    }

    /// Read data back from a buffer (memcpy — unified memory).
    pub fn readbackFromBuffer(self: *Self, buf: *const MetalBuffer, output: []u8) void {
        _ = self;
        const src = msg0_ptr(buf.obj, sel_registerName("contents"));
        const src_slice: [*]const u8 = @ptrCast(src);
        @memcpy(output, src_slice[0..output.len]);
    }

    /// Release a Metal buffer.
    pub fn destroyBuffer(self: *Self, buf: *MetalBuffer) void {
        msg0_void(buf.obj, self.sel_release);
    }

    /// Release a compute pipeline.
    pub fn destroyPipeline(self: *Self, pipe: *MetalPipeline) void {
        msg0_void(pipe.state, self.sel_release);
    }

    // ====================================================================
    // Command Recording
    // ====================================================================

    /// Begin recording a new command buffer with a compute encoder.
    /// All subsequent cmdDispatch calls are recorded into this encoder.
    pub fn beginCommandBuffer(self: *Self) MetalError!void {
        self.command_buffer = msg0(self.command_queue, self.sel_commandBuffer) orelse {
            return MetalError.CommandBufferCreationFailed;
        };
        self.encoder = msg0(self.command_buffer.?, self.sel_computeCommandEncoder) orelse {
            return MetalError.EncoderCreationFailed;
        };
    }

    /// Record a compute dispatch: set pipeline → bind buffers → set params → dispatch → barrier.
    /// Buffers are bound at indices 0..N-1, params bytes at index N.
    pub fn cmdDispatch(
        self: *Self,
        pipeline: *const MetalPipeline,
        buffers: []const MetalBuffer,
        params: []const u8,
        groups: MTLSize,
        threads_per_group: MTLSize,
    ) void {
        self.cmdDispatchNoBarrier(pipeline, buffers, params, groups, threads_per_group);

        // Memory barrier between dispatches (ensures write visibility)
        const enc = self.encoder orelse return;
        msg1_uint_void(enc, self.sel_memoryBarrierWithScope, MTLBarrierScopeBuffers);
    }

    /// Record a compute dispatch WITHOUT a trailing memory barrier.
    /// Use when the next dispatch does NOT read from buffers written by this one
    /// (e.g. independent Q/K/V projections that all read the same input).
    /// Caller must issue cmdBarrier() before any dispatch that reads the output.
    pub fn cmdDispatchNoBarrier(
        self: *Self,
        pipeline: *const MetalPipeline,
        buffers: []const MetalBuffer,
        params: []const u8,
        groups: MTLSize,
        threads_per_group: MTLSize,
    ) void {
        const enc = self.encoder orelse return;

        // Set pipeline state
        msg1_id_void(enc, self.sel_setComputePipelineState, pipeline.state);

        // Bind storage buffers at indices 0..N-1
        for (buffers, 0..) |buf, i| {
            msg3_id_uint_uint(enc, self.sel_setBuffer_offset_atIndex, buf.obj, 0, @intCast(i));
        }

        // Set params bytes at index N (after all buffers)
        if (params.len > 0) {
            msg3_ptr_uint_uint(enc, self.sel_setBytes_length_atIndex, params.ptr, @intCast(params.len), @intCast(buffers.len));
        }

        // Dispatch threadgroups
        msg2_sizes(enc, self.sel_dispatchThreadgroups, groups, threads_per_group);
    }

    /// Issue an explicit memory barrier (buffer scope).
    /// Call after a group of independent dispatches that write to separate buffers,
    /// before any dispatch that needs to read those outputs.
    pub fn cmdBarrier(self: *Self) void {
        const enc = self.encoder orelse return;
        msg1_uint_void(enc, self.sel_memoryBarrierWithScope, MTLBarrierScopeBuffers);
    }

    /// End recording, submit the command buffer, and block until completion.
    pub fn submitAndWait(self: *Self) void {
        if (self.encoder) |enc| {
            msg0_void(enc, self.sel_endEncoding);
            self.encoder = null;
        }
        if (self.command_buffer) |cb| {
            msg0_void(cb, self.sel_commit);
            msg0_void(cb, self.sel_waitUntilCompleted);
            self.command_buffer = null;
        }
    }

    // ====================================================================
    // Device Info
    // ====================================================================

    pub fn getDeviceName(self: *const Self) []const u8 {
        return self.device_name[0..self.device_name_len];
    }

    // ====================================================================
    // Cleanup
    // ====================================================================

    pub fn deinit(self: *Self) void {
        msg0_void(self.library, self.sel_release);
        msg0_void(self.command_queue, self.sel_release);
        msg0_void(self.device, self.sel_release);
        _ = std.c.dlclose(self.metal_lib_handle);
    }
};

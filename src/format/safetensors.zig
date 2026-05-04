// src/format/safetensors.zig
//
// Reader for the HuggingFace safetensors format.
//
// Layout:
//   [u64 LE: header_len] [header_len bytes: JSON header] [tensor data...]
//
// The JSON header maps tensor name → { dtype, shape, data_offsets: [start, end] }.
// `data_offsets` are relative to the start of the data section
// (i.e. byte (8 + header_len) of the file).
//
// This module only does header parsing and tensor-byte slicing. Conversion
// from raw bytes to Tomoul Tensors (with bf16/f16 → f32 dequantization)
// happens at the model-loading layer that knows the target precision.

const std = @import("std");
const dtype_mod = @import("dtype.zig");
const Dtype = dtype_mod.Dtype;

pub const Error = error{
    FileTooSmall,
    HeaderTooLarge,
    InvalidJson,
    MissingField,
    UnknownDtype,
    InvalidOffsets,
    TensorNotFound,
    OutOfBounds,
    UnsupportedDtype,
    ShapeMismatch,
};

/// Metadata for one tensor, as parsed from the JSON header.
pub const TensorInfo = struct {
    name: []const u8,
    dtype: Dtype,
    shape: []const usize,
    /// Absolute byte offset into the file where this tensor's data starts.
    file_offset: u64,
    /// Number of bytes occupied by this tensor's data.
    byte_len: u64,
};

/// In-memory view of a safetensors file.
///
/// Owns: the file bytes (when opened via `init`), parsed JSON arena,
/// the tensor map, and any allocated shape slices.
pub const SafetensorsFile = struct {
    allocator: std.mem.Allocator,
    file_bytes: []const u8,
    owns_file_bytes: bool,
    /// Arena that backs the std.json.Value tree. Kept alive for the
    /// lifetime of the file because TensorInfo.name borrows from it.
    json_arena: std.heap.ArenaAllocator,
    tensors: std.StringHashMap(TensorInfo),
    /// Insertion-ordered tensor names, for iteration.
    order: std.ArrayList([]const u8),
    /// Offset of the data section relative to the file start.
    data_section_offset: u64,

    const Self = @This();

    /// Open and parse a safetensors file from disk.
    pub fn init(allocator: std.mem.Allocator, path: []const u8) !Self {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const size = try file.getEndPos();
        const buf = try allocator.alloc(u8, size);
        errdefer allocator.free(buf);

        const n = try file.readAll(buf);
        if (n != size) return Error.FileTooSmall;

        return parseInternal(allocator, buf, true);
    }

    /// Parse a safetensors blob already in memory. Does not take ownership of `bytes`.
    pub fn initFromBytes(allocator: std.mem.Allocator, bytes: []const u8) !Self {
        return parseInternal(allocator, bytes, false);
    }

    fn parseInternal(allocator: std.mem.Allocator, bytes: []const u8, owns: bool) !Self {
        if (bytes.len < 8) return Error.FileTooSmall;

        const header_len = std.mem.readInt(u64, bytes[0..8], .little);
        if (header_len > bytes.len - 8) return Error.HeaderTooLarge;

        const header_bytes = bytes[8 .. 8 + header_len];
        const data_section_offset: u64 = 8 + header_len;

        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();

        const parsed = std.json.parseFromSliceLeaky(
            std.json.Value,
            arena.allocator(),
            header_bytes,
            .{},
        ) catch return Error.InvalidJson;

        if (parsed != .object) return Error.InvalidJson;

        var tensors = std.StringHashMap(TensorInfo).init(allocator);
        errdefer tensors.deinit();

        var order: std.ArrayList([]const u8) = .{};
        errdefer order.deinit(allocator);

        var it = parsed.object.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            // Skip the optional metadata field.
            if (std.mem.eql(u8, name, "__metadata__")) continue;

            const obj = entry.value_ptr.*;
            if (obj != .object) return Error.InvalidJson;

            const dtype_field = obj.object.get("dtype") orelse return Error.MissingField;
            if (dtype_field != .string) return Error.MissingField;
            const dtype = Dtype.fromString(dtype_field.string) orelse return Error.UnknownDtype;

            const shape_field = obj.object.get("shape") orelse return Error.MissingField;
            if (shape_field != .array) return Error.MissingField;

            const shape = try arena.allocator().alloc(usize, shape_field.array.items.len);
            for (shape_field.array.items, shape) |v, *out| {
                if (v != .integer or v.integer < 0) return Error.InvalidJson;
                out.* = @intCast(v.integer);
            }

            const offsets_field = obj.object.get("data_offsets") orelse return Error.MissingField;
            if (offsets_field != .array or offsets_field.array.items.len != 2) {
                return Error.MissingField;
            }
            const start_v = offsets_field.array.items[0];
            const end_v = offsets_field.array.items[1];
            if (start_v != .integer or end_v != .integer) return Error.InvalidJson;
            if (start_v.integer < 0 or end_v.integer < start_v.integer) {
                return Error.InvalidOffsets;
            }
            const start: u64 = @intCast(start_v.integer);
            const end: u64 = @intCast(end_v.integer);

            const file_offset = data_section_offset + start;
            const byte_len = end - start;
            if (file_offset + byte_len > bytes.len) return Error.OutOfBounds;

            // Sanity: shape product * dtype size == byte_len
            var elem_count: u64 = 1;
            for (shape) |d| elem_count *= @as(u64, d);
            const expected = elem_count * @as(u64, dtype.byteSize());
            if (expected != byte_len) return Error.InvalidOffsets;

            const info = TensorInfo{
                .name = name, // borrowed from arena (json.Value owns the key string)
                .dtype = dtype,
                .shape = shape,
                .file_offset = file_offset,
                .byte_len = byte_len,
            };
            try tensors.put(name, info);
            try order.append(allocator, name);
        }

        return Self{
            .allocator = allocator,
            .file_bytes = bytes,
            .owns_file_bytes = owns,
            .json_arena = arena,
            .tensors = tensors,
            .order = order,
            .data_section_offset = data_section_offset,
        };
    }

    pub fn deinit(self: *Self) void {
        self.order.deinit(self.allocator);
        self.tensors.deinit();
        self.json_arena.deinit();
        if (self.owns_file_bytes) self.allocator.free(self.file_bytes);
    }

    pub fn getInfo(self: *const Self, name: []const u8) ?TensorInfo {
        return self.tensors.get(name);
    }

    /// Borrow the raw bytes for a tensor. Lifetime tied to this file's `file_bytes`.
    pub fn getBytes(self: *const Self, name: []const u8) ![]const u8 {
        const info = self.tensors.get(name) orelse return Error.TensorNotFound;
        const start: usize = @intCast(info.file_offset);
        const end: usize = @intCast(info.file_offset + info.byte_len);
        return self.file_bytes[start..end];
    }

    pub fn count(self: *const Self) usize {
        return self.order.items.len;
    }

    pub fn names(self: *const Self) []const []const u8 {
        return self.order.items;
    }

    /// Read a tensor as f32, converting from bf16/f16/f32 source. Allocates
    /// a new owned slice; caller frees with `allocator.free(slice)`.
    /// Element count = product of shape dims.
    pub fn readF32(
        self: *const Self,
        allocator: std.mem.Allocator,
        name: []const u8,
    ) ![]f32 {
        const info = self.tensors.get(name) orelse return Error.TensorNotFound;
        const bytes = try self.getBytes(name);

        var elem_count: usize = 1;
        for (info.shape) |d| elem_count *= d;

        const out = try allocator.alloc(f32, elem_count);
        errdefer allocator.free(out);

        switch (info.dtype) {
            .f32 => {
                // Memcpy via byte view; underlying buffer alignment is unknown.
                const dst_bytes = std.mem.sliceAsBytes(out);
                @memcpy(dst_bytes, bytes);
            },
            .bf16 => {
                std.debug.assert(bytes.len == elem_count * 2);
                for (0..elem_count) |i| {
                    const lo: u16 = bytes[i * 2];
                    const hi: u16 = bytes[i * 2 + 1];
                    const bits: u16 = lo | (hi << 8);
                    out[i] = dtype_mod.bf16ToF32(bits);
                }
            },
            .f16 => {
                std.debug.assert(bytes.len == elem_count * 2);
                for (0..elem_count) |i| {
                    const lo: u16 = bytes[i * 2];
                    const hi: u16 = bytes[i * 2 + 1];
                    const bits: u16 = lo | (hi << 8);
                    out[i] = dtype_mod.f16ToF32(bits);
                }
            },
            else => return Error.UnsupportedDtype,
        }
        return out;
    }

    /// Read a tensor and verify its shape matches `expected`.
    /// Wraps `readF32` with a shape sanity check.
    pub fn readF32Checked(
        self: *const Self,
        allocator: std.mem.Allocator,
        name: []const u8,
        expected: []const usize,
    ) ![]f32 {
        const info = self.tensors.get(name) orelse return Error.TensorNotFound;
        if (info.shape.len != expected.len) return Error.ShapeMismatch;
        for (info.shape, expected) |a, b| {
            if (a != b) return Error.ShapeMismatch;
        }
        return self.readF32(allocator, name);
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

/// Build a tiny synthetic safetensors blob: one F32 tensor named "w" of shape [2, 2].
fn buildSyntheticBlob(allocator: std.mem.Allocator) ![]u8 {
    const header =
        \\{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}
    ;
    const data_bytes: usize = 16; // 4 f32 values
    const total = 8 + header.len + data_bytes;
    const buf = try allocator.alloc(u8, total);

    std.mem.writeInt(u64, buf[0..8], header.len, .little);
    @memcpy(buf[8 .. 8 + header.len], header);

    const data_start = 8 + header.len;
    const values = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    @memcpy(buf[data_start..][0..16], std.mem.sliceAsBytes(&values));

    return buf;
}

test "parse synthetic safetensors blob" {
    const allocator = testing.allocator;
    const blob = try buildSyntheticBlob(allocator);
    defer allocator.free(blob);

    var st = try SafetensorsFile.initFromBytes(allocator, blob);
    defer st.deinit();

    try testing.expectEqual(@as(usize, 1), st.count());

    const info = st.getInfo("w").?;
    try testing.expectEqual(Dtype.f32, info.dtype);
    try testing.expectEqualSlices(usize, &[_]usize{ 2, 2 }, info.shape);
    try testing.expectEqual(@as(u64, 16), info.byte_len);

    const bytes = try st.getBytes("w");
    try testing.expectEqual(@as(usize, 16), bytes.len);

    // Read f32 values without assuming buffer alignment.
    var v0_buf: [4]u8 = undefined;
    var v3_buf: [4]u8 = undefined;
    @memcpy(&v0_buf, bytes[0..4]);
    @memcpy(&v3_buf, bytes[12..16]);
    const v0: f32 = @bitCast(v0_buf);
    const v3: f32 = @bitCast(v3_buf);
    try testing.expectEqual(@as(f32, 1.0), v0);
    try testing.expectEqual(@as(f32, 4.0), v3);
}

test "rejects truncated header" {
    const allocator = testing.allocator;
    var blob = [_]u8{ 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, '{' };
    try testing.expectError(Error.HeaderTooLarge, SafetensorsFile.initFromBytes(allocator, &blob));
}

test "readF32: f32 source round-trip" {
    const allocator = testing.allocator;
    const blob = try buildSyntheticBlob(allocator);
    defer allocator.free(blob);

    var st = try SafetensorsFile.initFromBytes(allocator, blob);
    defer st.deinit();

    const values = try st.readF32(allocator, "w");
    defer allocator.free(values);

    try testing.expectEqual(@as(usize, 4), values.len);
    try testing.expectEqual(@as(f32, 1.0), values[0]);
    try testing.expectEqual(@as(f32, 4.0), values[3]);
}

test "readF32: bf16 source converts to f32" {
    const allocator = testing.allocator;
    // Header for one bf16 tensor "w" of shape [3], byte_len = 6.
    const header =
        \\{"w":{"dtype":"BF16","shape":[3],"data_offsets":[0,6]}}
    ;
    const total = 8 + header.len + 6;
    const buf = try allocator.alloc(u8, total);
    defer allocator.free(buf);
    std.mem.writeInt(u64, buf[0..8], header.len, .little);
    @memcpy(buf[8 .. 8 + header.len], header);

    // bf16 bit patterns for 1.0, -2.0, 0.5
    const data_start = 8 + header.len;
    std.mem.writeInt(u16, buf[data_start..][0..2], 0x3F80, .little); // 1.0
    std.mem.writeInt(u16, buf[data_start + 2 ..][0..2], 0xC000, .little); // -2.0
    std.mem.writeInt(u16, buf[data_start + 4 ..][0..2], 0x3F00, .little); // 0.5

    var st = try SafetensorsFile.initFromBytes(allocator, buf);
    defer st.deinit();

    const values = try st.readF32(allocator, "w");
    defer allocator.free(values);

    try testing.expectEqual(@as(usize, 3), values.len);
    try testing.expectEqual(@as(f32, 1.0), values[0]);
    try testing.expectEqual(@as(f32, -2.0), values[1]);
    try testing.expectEqual(@as(f32, 0.5), values[2]);
}

test "readF32Checked: shape mismatch" {
    const allocator = testing.allocator;
    const blob = try buildSyntheticBlob(allocator);
    defer allocator.free(blob);

    var st = try SafetensorsFile.initFromBytes(allocator, blob);
    defer st.deinit();

    try testing.expectError(
        Error.ShapeMismatch,
        st.readF32Checked(allocator, "w", &[_]usize{ 4, 1 }),
    );
}

test "rejects bad json" {
    const allocator = testing.allocator;
    const header = "not json";
    var buf: [8 + 8]u8 = undefined;
    std.mem.writeInt(u64, buf[0..8], header.len, .little);
    @memcpy(buf[8..][0..header.len], header);
    try testing.expectError(Error.InvalidJson, SafetensorsFile.initFromBytes(allocator, &buf));
}

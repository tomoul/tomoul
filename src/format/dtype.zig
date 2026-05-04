// src/format/dtype.zig
//
// Source-format dtypes (HuggingFace safetensors / PyTorch / numpy naming).
// Distinct from the Tomoul-internal storage formats in core/quantization.zig.

const std = @import("std");

pub const Dtype = enum {
    f32,
    f16,
    bf16,
    i64,
    i32,
    i16,
    i8,
    u8,
    bool,

    pub fn fromString(s: []const u8) ?Dtype {
        const map = .{
            .{ "F32", .f32 },     .{ "F16", .f16 }, .{ "BF16", .bf16 },
            .{ "I64", .i64 },     .{ "I32", .i32 }, .{ "I16", .i16 },
            .{ "I8", .i8 },       .{ "U8", .u8 },   .{ "BOOL", .bool },
            .{ "float32", .f32 }, .{ "float16", .f16 }, .{ "bfloat16", .bf16 },
        };
        inline for (map) |entry| {
            if (std.mem.eql(u8, s, entry[0])) return entry[1];
        }
        return null;
    }

    pub fn byteSize(self: Dtype) usize {
        return switch (self) {
            .f32, .i32 => 4,
            .f16, .bf16, .i16 => 2,
            .i8, .u8, .bool => 1,
            .i64 => 8,
        };
    }
};

/// Convert one bf16 value (stored as u16) to f32.
/// bf16 = top 16 bits of an IEEE-754 f32. Reconstruct by left-shifting.
pub inline fn bf16ToF32(bits: u16) f32 {
    const u: u32 = @as(u32, bits) << 16;
    return @bitCast(u);
}

/// Convert a slice of bf16 values (raw u16 bit-pattern) to a pre-allocated f32 slice.
pub fn bf16ToF32Slice(src: []const u16, dst: []f32) void {
    std.debug.assert(src.len == dst.len);
    for (src, dst) |b, *d| d.* = bf16ToF32(b);
}

/// Convert one f16 value (stored as u16) to f32. Wraps std lib for symmetry with bf16.
pub inline fn f16ToF32(bits: u16) f32 {
    const h: f16 = @bitCast(bits);
    return @floatCast(h);
}

pub fn f16ToF32Slice(src: []const u16, dst: []f32) void {
    std.debug.assert(src.len == dst.len);
    for (src, dst) |b, *d| d.* = f16ToF32(b);
}

test "bf16 round-trip basic" {
    // bf16(1.0) = 0x3F80
    try std.testing.expectEqual(@as(f32, 1.0), bf16ToF32(0x3F80));
    try std.testing.expectEqual(@as(f32, -2.0), bf16ToF32(0xC000));
    try std.testing.expectEqual(@as(f32, 0.0), bf16ToF32(0x0000));
}

test "Dtype.fromString" {
    try std.testing.expectEqual(Dtype.f32, Dtype.fromString("F32").?);
    try std.testing.expectEqual(Dtype.bf16, Dtype.fromString("BF16").?);
    try std.testing.expectEqual(@as(?Dtype, null), Dtype.fromString("FP8"));
}

test "Dtype.byteSize" {
    try std.testing.expectEqual(@as(usize, 4), Dtype.f32.byteSize());
    try std.testing.expectEqual(@as(usize, 2), Dtype.bf16.byteSize());
    try std.testing.expectEqual(@as(usize, 8), Dtype.i64.byteSize());
}

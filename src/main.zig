const std = @import("std");

// Core modules
pub const tensor = @import("core/tensor.zig");
pub const ops = @import("core/ops.zig");

// Re-export main types
pub const Tensor = tensor.Tensor;
pub const TensorError = tensor.TensorError;

pub fn main() !void {
    const print = std.debug.print;

    print("\n", .{});
    print("============================================================\n", .{});
    print("                    TOMOUL v0.1.0                           \n", .{});
    print("         Minimalist AI Inference Engine in Zig              \n", .{});
    print("============================================================\n", .{});
    print("\n", .{});

    // Use a general purpose allocator for the demo
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Demo: Create and manipulate tensors
    print("=== Phase 1: Tensor Core Demo ===\n\n", .{});

    // Create a 2x3 tensor
    print("1. Creating a 2x3 tensor...\n", .{});
    var shape = [_]usize{ 2, 3 };
    var a = try Tensor.init(allocator, &shape);
    defer a.deinit();

    // Fill with values
    for (a.data, 0..) |*val, i| {
        val.* = @floatFromInt(i + 1);
    }

    print("   Tensor A (2x3):\n", .{});
    print("   [ ", .{});
    for (a.data[0..3]) |v| print("{d:.1} ", .{v});
    print("]\n", .{});
    print("   [ ", .{});
    for (a.data[3..6]) |v| print("{d:.1} ", .{v});
    print("]\n\n", .{});

    // Create another tensor for operations
    print("2. Creating tensor B with same shape, filled with 10.0...\n", .{});
    var b = try Tensor.init(allocator, &shape);
    defer b.deinit();
    b.fill(10.0);

    print("   Tensor B (2x3):\n", .{});
    print("   [ ", .{});
    for (b.data[0..3]) |v| print("{d:.1} ", .{v});
    print("]\n", .{});
    print("   [ ", .{});
    for (b.data[3..6]) |v| print("{d:.1} ", .{v});
    print("]\n\n", .{});

    // Element-wise addition
    print("3. Element-wise addition: C = A + B\n", .{});
    var c = try ops.add(allocator, &a, &b);
    defer c.deinit();

    print("   Result C (2x3):\n", .{});
    print("   [ ", .{});
    for (c.data[0..3]) |v| print("{d:.1} ", .{v});
    print("]\n", .{});
    print("   [ ", .{});
    for (c.data[3..6]) |v| print("{d:.1} ", .{v});
    print("]\n\n", .{});

    // Element-wise multiplication
    print("4. Element-wise multiplication: D = A * B\n", .{});
    var d = try ops.mul(allocator, &a, &b);
    defer d.deinit();

    print("   Result D (2x3):\n", .{});
    print("   [ ", .{});
    for (d.data[0..3]) |v| print("{d:.1} ", .{v});
    print("]\n", .{});
    print("   [ ", .{});
    for (d.data[3..6]) |v| print("{d:.1} ", .{v});
    print("]\n\n", .{});

    // Scalar operations
    print("5. Scalar multiplication: E = A * 2.0\n", .{});
    var e = try ops.scale(allocator, &a, 2.0);
    defer e.deinit();

    print("   Result E (2x3):\n", .{});
    print("   [ ", .{});
    for (e.data[0..3]) |v| print("{d:.1} ", .{v});
    print("]\n", .{});
    print("   [ ", .{});
    for (e.data[3..6]) |v| print("{d:.1} ", .{v});
    print("]\n\n", .{});

    // Reduction operations
    print("6. Reduction operations on tensor A:\n", .{});
    print("   Sum:  {d:.1}\n", .{ops.sum(&a)});
    print("   Mean: {d:.2}\n", .{ops.mean(&a)});
    print("   Max:  {d:.1}\n", .{ops.max(&a)});
    print("   Min:  {d:.1}\n", .{ops.min(&a)});

    print("\n=== Phase 1 Complete! ===\n", .{});
    print("Tensor core is operational. Ready for Phase 2 (Matrix Multiplication).\n\n", .{});
}

// Pull in tests from submodules
test {
    _ = @import("core/tensor.zig");
    _ = @import("core/ops.zig");
}
